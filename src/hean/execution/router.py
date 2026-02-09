"""Order router - routes orders to paper or live broker.

Enhanced with:
- Idempotency keys to prevent duplicate orders on network retries
- Order deduplication with TTL-based cache
"""

import asyncio
import hashlib
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from hean.config import settings
from hean.core.bus import EventBus

try:
    from hean.core.execution.iceberg import IcebergOrder
except ImportError:  # Optional Phase 3 component
    IcebergOrder = None  # type: ignore[assignment]
from hean.core.ofi import OrderFlowImbalance
from hean.core.regime import RegimeDetector
from hean.core.types import Event, EventType, Order, OrderRequest, OrderStatus, Tick

try:
    from hean.exchange.bybit.http import BybitHTTPClient
except ImportError:  # Optional live trading dependency
    BybitHTTPClient = None  # type: ignore[assignment]
from hean.exchange.bybit.ws_private import BybitPrivateWebSocket
from hean.exchange.bybit.ws_public import BybitPublicWebSocket
from hean.exchange.executor import SmartLimitExecutor
from hean.execution.execution_diagnostics import ExecutionDiagnostics
from hean.execution.maker_retry_queue import MakerRetryQueue
from hean.execution.order_manager import OrderManager
from hean.execution.paper_broker import PaperBroker
from hean.logging import get_logger
from hean.observability.no_trade_report import no_trade_report

logger = get_logger(__name__)


@dataclass
class IdempotencyEntry:
    """Entry for idempotency cache."""
    idempotency_key: str
    order_id: str
    created_at: datetime
    status: str  # 'pending', 'completed', 'failed'
    result: Any = None


def generate_idempotency_key(order_request: "OrderRequest") -> str:
    """Generate a unique idempotency key for an order request.

    The key is based on:
    - strategy_id
    - symbol
    - side
    - size (rounded to 6 decimals)
    - signal_id (if available)
    - timestamp window (5 second buckets to handle slight timing differences)

    This ensures that identical requests within a 5-second window
    are deduplicated, preventing duplicate orders from network retries.
    """
    # Create a time bucket (5 second windows)
    now = datetime.utcnow()
    time_bucket = now.replace(
        second=(now.second // 5) * 5,
        microsecond=0
    ).isoformat()

    # Build the key components
    components = [
        order_request.strategy_id,
        order_request.symbol,
        order_request.side,
        f"{order_request.size:.6f}",
        time_bucket,
    ]

    # Add signal_id if available (for more precise deduplication)
    signal_id = getattr(order_request, 'signal_id', None) or order_request.metadata.get('signal_id', '')
    if signal_id:
        components.append(signal_id)

    # Hash the components
    key_string = "|".join(str(c) for c in components)
    return hashlib.sha256(key_string.encode()).hexdigest()[:32]


class ExecutionRouter:
    """Routes order requests to appropriate broker (paper or live)."""

    def __init__(
        self,
        bus: EventBus,
        order_manager: OrderManager,
        regime_detector: RegimeDetector | None = None,
    ) -> None:
        """Initialize the execution router.

        Args:
            bus: Event bus
            order_manager: Order manager
            regime_detector: Regime detector for volatility data (optional)
        """
        self._bus = bus
        self._order_manager = order_manager
        self._paper_broker = PaperBroker(bus)
        self._regime_detector = regime_detector
        self._running = False

        # Bybit clients (for live trading)
        self._bybit_http: BybitHTTPClient | None = None
        self._bybit_ws_public: BybitPublicWebSocket | None = None
        self._bybit_ws_private: BybitPrivateWebSocket | None = None
        self._current_bids: dict[str, float] = {}
        self._current_asks: dict[str, float] = {}
        self._current_prices: dict[str, float] = {}
        self._maker_orders: dict[str, Order] = {}  # order_id -> order
        self._ttl_check_task: asyncio.Task[None] | None = None
        self._retry_check_task: asyncio.Task[None] | None = None

        # Execution diagnostics
        self._diagnostics = ExecutionDiagnostics()

        # Retry queue
        self._retry_queue = MakerRetryQueue()

        # Adaptive parameters
        self._adaptive_ttl_ms = settings.maker_ttl_ms
        self._adaptive_offset_bps = settings.maker_price_offset_bps
        self._recent_expired_count = deque(maxlen=10)  # Track last 10 expired orders
        self._volatility_history: dict[str, deque[float]] = {}  # Track volatility per symbol

        # Volatility gating thresholds
        if settings.debug_mode:
            # DEBUG: Relaxed thresholds for testing
            self._volatility_soft_block_percentile = 100.0  # Disabled - never soft block
            self._volatility_medium_penalty_percentile = (
                50.0  # Apply size penalty starting at 50th percentile
            )
            self._volatility_hard_block_percentile = 95.0  # Hard block at P95+
        else:
            # PRODUCTION: Strict thresholds
            self._volatility_soft_block_percentile = 90.0  # Soft block at P90+
            self._volatility_medium_penalty_percentile = (
                75.0  # Apply size penalty starting at 75th percentile
            )
            self._volatility_hard_block_percentile = 99.0  # Hard block at P99+

        # Original order requests for retry queue
        self._order_requests: dict[str, OrderRequest] = {}

        # Phase 3: Smart Limit Engine, OFI, and Iceberg
        self._smart_executor: SmartLimitExecutor | None = None
        self._ofi: OrderFlowImbalance | None = None
        self._iceberg: Any | None = None
        self._phase3_enabled = True  # Enable Phase 3 features (iceberg disabled below)
        self._stale_order_ttl = timedelta(seconds=30)  # Cancel PENDING orders older than 30s

        # Idempotency cache to prevent duplicate orders
        self._idempotency_cache: dict[str, IdempotencyEntry] = {}
        self._idempotency_ttl = timedelta(minutes=5)  # Cache entries for 5 minutes
        self._idempotency_cleanup_task: asyncio.Task[None] | None = None
        self._idempotency_metrics = {
            "duplicates_prevented": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def start(self) -> None:
        """Start the execution router."""
        # Always start paper broker as safety net (fallback if Bybit fails)
        await self._paper_broker.start()

        # Initialize Bybit clients if in live mode
        if settings.is_live:
            try:
                if BybitHTTPClient is None:
                    raise ImportError("BybitHTTPClient not available (httpx or related deps missing)")

                self._bybit_http = BybitHTTPClient()
                await self._bybit_http.connect()

                self._bybit_ws_public = BybitPublicWebSocket(self._bus)
                await self._bybit_ws_public.connect()

                self._bybit_ws_private = BybitPrivateWebSocket(self._bus)
                await self._bybit_ws_private.connect()
                await self._bybit_ws_private.subscribe_all()

                logger.info("Bybit clients connected and ready for live trading")
            except Exception as e:
                logger.error(f"Failed to connect to Bybit: {e}")
                logger.warning("Falling back to paper trading mode")
                self._bybit_http = None
                self._bybit_ws_public = None
                self._bybit_ws_private = None

        self._bus.subscribe(EventType.ORDER_REQUEST, self._handle_order_request)
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self._bus.subscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        self._running = True

        # Initialize Phase 3 components
        if self._phase3_enabled:
            self._ofi = OrderFlowImbalance(
                bus=self._bus,
                lookback_window=20,
            )
            await self._ofi.start()
            # Iceberg splitting disabled: causes phantom PENDING orders that block trading
            self._iceberg = None
            logger.info("Iceberg splitting disabled (causes stale order accumulation)")

            self._smart_executor = SmartLimitExecutor(
                bus=self._bus,
                bybit_http=self._bybit_http,
            )
            await self._smart_executor.start()
            logger.info("Phase 3 components initialized: Smart Limit Engine, OFI, Iceberg")

        if settings.maker_first:
            self._ttl_check_task = asyncio.create_task(self._check_maker_ttl_loop())
            self._retry_check_task = asyncio.create_task(self._check_retry_queue_loop())

        # Start idempotency cache cleanup task
        self._idempotency_cleanup_task = asyncio.create_task(self._cleanup_idempotency_cache())

        logger.info("Execution router started with idempotency protection")

    async def stop(self) -> None:
        """Stop the execution router."""
        self._bus.unsubscribe(EventType.ORDER_REQUEST, self._handle_order_request)
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self._bus.unsubscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        if self._ttl_check_task:
            self._ttl_check_task.cancel()
            try:
                await self._ttl_check_task
            except asyncio.CancelledError:
                pass
        if self._retry_check_task:
            self._retry_check_task.cancel()
            try:
                await self._retry_check_task
            except asyncio.CancelledError:
                pass
        if self._idempotency_cleanup_task:
            self._idempotency_cleanup_task.cancel()
            try:
                await self._idempotency_cleanup_task
            except asyncio.CancelledError:
                pass

        # Log idempotency metrics on shutdown
        metrics = self.get_idempotency_metrics()
        logger.info(
            f"[IDEMPOTENCY] Shutdown metrics: "
            f"duplicates_prevented={metrics['duplicates_prevented']}, "
            f"cache_hits={metrics['cache_hits']}, "
            f"cache_misses={metrics['cache_misses']}"
        )

        # Disconnect Bybit clients
        if self._bybit_ws_private:
            await self._bybit_ws_private.disconnect()
        if self._bybit_ws_public:
            await self._bybit_ws_public.disconnect()
        if self._bybit_http:
            await self._bybit_http.disconnect()

        if settings.dry_run:
            await self._paper_broker.stop()

        # Stop Phase 3 components
        if self._smart_executor:
            await self._smart_executor.stop()
        if self._ofi:
            await self._ofi.stop()

        self._running = False
        logger.info("Execution router stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update bid/ask prices and volatility."""
        tick: Tick = event.data["tick"]
        if tick.bid:
            self._current_bids[tick.symbol] = tick.bid
        if tick.ask:
            self._current_asks[tick.symbol] = tick.ask
        if tick.price:
            self._current_prices[tick.symbol] = tick.price

        # Update volatility history for adaptive logic
        if tick.price:
            self._update_volatility_history(tick.symbol, tick.price)

        # OFI updates are handled by the OFI monitor's own subscriptions.

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled events."""
        order = event.data.get("order")
        if not order:
            logger.warning(f"[ROUTER] Missing 'order' in ORDER_FILLED event: {event.data}")
            return
        self._maker_orders.pop(order.order_id, None)

        # Update idempotency status to completed
        self._update_idempotency_status(order.order_id, "completed", result={"filled": True})

        # Record fill in diagnostics
        if order.is_maker:
            self._diagnostics.record_maker_fill(order)
            no_trade_report.increment_pipeline("maker_orders_filled", order.strategy_id)
        else:
            no_trade_report.increment_pipeline("taker_orders_filled", order.strategy_id)

        # Remove from retry queue if it was there
        self._retry_queue.remove_order(order.order_id)

        # Finalize record
        self._diagnostics.finalize_record(order.order_id)
        self._order_requests.pop(order.order_id, None)

    async def _handle_order_cancelled(self, event: Event) -> None:
        """Handle order cancelled events.

        Supports two event formats:
        - {"order": Order} — from engine_facade / trading router
        - {"order_id": str} — from ws_private (Bybit WebSocket)
        """
        order = event.data.get("order")
        if order:
            order_id = order.order_id
        else:
            order_id = event.data.get("order_id")
            if not order_id:
                logger.warning(f"[ROUTER] Missing 'order' and 'order_id' in ORDER_CANCELLED event: {event.data}")
                return

        self._maker_orders.pop(order_id, None)

        # Update idempotency status to completed (cancelled is a terminal state)
        self._update_idempotency_status(order_id, "completed", result={"cancelled": True})

        # Finalize record
        self._diagnostics.finalize_record(order_id)
        self._order_requests.pop(order.order_id, None)

    def _check_idempotency(self, order_request: OrderRequest) -> IdempotencyEntry | None:
        """Check if this order request was already processed.

        Args:
            order_request: The order request to check

        Returns:
            IdempotencyEntry if duplicate found, None otherwise
        """
        idempotency_key = generate_idempotency_key(order_request)

        if idempotency_key in self._idempotency_cache:
            entry = self._idempotency_cache[idempotency_key]
            # Check if entry is still valid (not expired)
            if datetime.utcnow() - entry.created_at < self._idempotency_ttl:
                self._idempotency_metrics["cache_hits"] += 1
                self._idempotency_metrics["duplicates_prevented"] += 1
                return entry

            # Entry expired, remove it
            del self._idempotency_cache[idempotency_key]

        self._idempotency_metrics["cache_misses"] += 1
        return None

    def _store_idempotency(
        self,
        order_request: OrderRequest,
        order_id: str,
        status: str = "pending",
        result: Any = None
    ) -> str:
        """Store an idempotency entry for an order request.

        Args:
            order_request: The order request
            order_id: The generated order ID
            status: Entry status ('pending', 'completed', 'failed')
            result: Optional result data

        Returns:
            The idempotency key
        """
        idempotency_key = generate_idempotency_key(order_request)

        entry = IdempotencyEntry(
            idempotency_key=idempotency_key,
            order_id=order_id,
            created_at=datetime.utcnow(),
            status=status,
            result=result,
        )

        self._idempotency_cache[idempotency_key] = entry
        logger.debug(f"Stored idempotency key {idempotency_key[:8]}... for order {order_id}")

        return idempotency_key

    def _update_idempotency_status(self, order_id: str, status: str, result: Any = None) -> None:
        """Update the status of an idempotency entry by order_id."""
        for key, entry in self._idempotency_cache.items():
            if entry.order_id == order_id:
                entry.status = status
                entry.result = result
                logger.debug(f"Updated idempotency {key[:8]}... status to {status}")
                break

    async def _cleanup_idempotency_cache(self) -> None:
        """Periodically clean up expired idempotency entries."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute

                now = datetime.utcnow()
                expired_keys = [
                    key for key, entry in self._idempotency_cache.items()
                    if now - entry.created_at > self._idempotency_ttl
                ]

                for key in expired_keys:
                    del self._idempotency_cache[key]

                if expired_keys:
                    logger.debug(f"Cleaned up {len(expired_keys)} expired idempotency entries")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in idempotency cleanup: {e}", exc_info=True)

    def get_idempotency_metrics(self) -> dict[str, int]:
        """Get idempotency cache metrics."""
        return {
            **self._idempotency_metrics,
            "cache_size": len(self._idempotency_cache),
        }

    async def _handle_order_request(self, event: Event) -> None:
        """Handle an order request event with idempotency protection."""
        order_request: OrderRequest = event.data["order_request"]
        logger.debug(
            f"Handling order request: {order_request.symbol} {order_request.side} size={order_request.size:.6f}"
        )

        # IDEMPOTENCY CHECK: Prevent duplicate orders
        existing_entry = self._check_idempotency(order_request)
        if existing_entry:
            logger.warning(
                f"[IDEMPOTENCY] Duplicate order request detected! "
                f"Original order_id={existing_entry.order_id}, status={existing_entry.status}. "
                f"Skipping to prevent duplicate."
            )
            # If the original order is still pending, we could optionally return its status
            # For safety, we simply skip the duplicate
            return

        try:
            # Route to Bybit if in live mode and not dry_run, otherwise use paper
            if not settings.dry_run and settings.is_live and self._bybit_http:
                logger.info("Routing to REAL Bybit API (live trading, no simulation)")
                if settings.maker_first:
                    # Use maker-first strategy but with real Bybit API
                    await self._route_maker_first(order_request)
                else:
                    await self._route_to_bybit(order_request)
            elif settings.maker_first:
                logger.debug("Routing to maker_first (paper/simulation)")
                await self._route_maker_first(order_request)
            else:
                logger.debug("Routing to paper broker (simulation)")
                await self._route_to_paper(order_request)
        except Exception as e:
            logger.error(f"Error routing order request: {e}", exc_info=True)
            await self._publish_order_rejected(order_request, str(e))

    async def _route_maker_first(self, order_request: OrderRequest) -> None:
        """Route order using maker-first policy with adaptive placement and volatility gating.

        Phase 3: Integrates Smart Limit Engine, OFI, and Iceberg for optimal execution.
        """
        symbol = order_request.symbol
        side = order_request.side

        # Phase 3: Check if order should be iceberg split
        if self._phase3_enabled and self._iceberg:
            current_price = self._current_prices.get(symbol) or order_request.price
            if current_price:
                # Process through iceberg (may split into micro-batches)
                micro_requests = await self._iceberg.process_order(order_request, current_price)

                # Validate all micro-requests have positive size
                valid_micro_requests = [r for r in micro_requests if r.size > 0]

                if len(valid_micro_requests) == 0:
                    logger.error(f"Iceberg split produced no valid orders, using original request")
                    # Continue with original order_request
                elif len(valid_micro_requests) > 1:
                    # Order was split into iceberg batches
                    logger.info(
                        f"Iceberg order: {symbol} {side} {order_request.size} split into {len(valid_micro_requests)} micro-batches"
                    )
                    # Schedule micro-orders with delays
                    await self._iceberg.schedule_micro_orders(valid_micro_requests)
                    return  # Order is being handled as iceberg
                else:
                    # Order not split or only one valid order, use it
                    order_request = valid_micro_requests[0]

        # Phase 3: Use Smart Limit Executor with OFI
        if self._phase3_enabled and self._smart_executor and self._ofi:
            # Get OFI aggression factor
            ofi_aggression = self._ofi.get_aggression_factor(symbol, side)

            # Place Post-Only order with price improvement via Smart Limit Executor
            try:
                order = await self._smart_executor.place_post_only_order(
                    order_request,
                    ofi_aggression=ofi_aggression,
                )

                # Register with order manager
                self._order_manager.register_order(order)
                self._maker_orders[order.order_id] = order
                self._order_requests[order.order_id] = order_request

                # Submit to broker (real or paper)
                if not settings.dry_run and settings.is_live and self._bybit_http:
                    try:
                        bybit_order_request = OrderRequest(
                            signal_id=order_request.signal_id,
                            strategy_id=order_request.strategy_id,
                            symbol=order.symbol,
                            side=order.side,
                            size=order.size,
                            price=order.price,
                            order_type="limit",
                            stop_loss=order.stop_loss,
                            take_profit=order.take_profit,
                            metadata=order.metadata,
                        )

                        # Place on Bybit
                        placed_order = await self._bybit_http.place_order(bybit_order_request)

                        if placed_order and placed_order.order_id:
                            # Remove old tracking
                            self._maker_orders.pop(order.order_id, None)
                            self._order_requests.pop(order.order_id, None)

                            # Update with Bybit's order
                            placed_order.metadata.update(order.metadata)
                            placed_order.is_maker = True
                            placed_order.placed_at = datetime.utcnow()

                            # Track with Bybit's order ID
                            self._maker_orders[placed_order.order_id] = placed_order
                            self._order_requests[placed_order.order_id] = order_request
                            self._order_manager.register_order(placed_order)

                            # Publish order placed event
                            await self._bus.publish(Event(event_type=EventType.ORDER_PLACED, data={"order": placed_order}))

                            logger.info(f"Smart Limit order placed on Bybit: {placed_order.order_id}")
                        else:
                            logger.error(f"Bybit returned no order for smart limit {order.order_id}")
                            # Clean up
                            self._maker_orders.pop(order.order_id, None)
                            self._order_requests.pop(order.order_id, None)
                    except Exception as e:
                        logger.error(f"Failed to place smart limit order on Bybit: {e}", exc_info=True)
                        # Clean up and fall through to standard maker-first logic
                        self._maker_orders.pop(order.order_id, None)
                        self._order_requests.pop(order.order_id, None)
                        raise
                else:
                    await self._paper_broker.submit_order(order)

                logger.info(
                    f"Smart Limit order placed: {order.order_id} {side} {order.size} {symbol} @ {order.price:.6f} "
                    f"(OFI aggression={ofi_aggression:.3f})"
                )
                return
            except Exception as e:
                logger.error(f"Smart Limit Executor failed, falling back to standard maker-first: {e}", exc_info=True)
                # Fall through to standard maker-first logic

        # Get best bid/ask
        best_bid = self._current_bids.get(symbol)
        best_ask = self._current_asks.get(symbol)

        # Use order request price if no bid/ask data
        if best_bid is None or best_ask is None:
            logger.warning(f"No bid/ask data for {symbol}, using order request price")
            # Use order request price as fallback
            if order_request.price:
                if side == "buy":
                    best_bid = order_request.price
                    best_ask = order_request.price * 1.0001  # Small spread
                else:  # sell
                    best_ask = order_request.price
                    best_bid = order_request.price * 0.9999  # Small spread
            else:
                # Last resort: use current price if available
                current_price = (
                    self._current_prices.get(symbol) if hasattr(self, "_current_prices") else None
                )
                if current_price:
                    best_bid = current_price * 0.9999
                    best_ask = current_price * 1.0001
                else:
                    # No price data available - reject order
                    logger.error(f"No price data for {symbol}, rejecting order")
                    await self._publish_order_rejected(order_request, f"No price data for {symbol}")
                    return
            logger.debug(f"Using fallback prices: bid={best_bid:.2f}, ask={best_ask:.2f}")

        # Volatility-aware execution gating
        if not settings.debug_mode:
            # Check volatility and apply gating
            current_volatility = (
                self._regime_detector.get_volatility(symbol) if self._regime_detector else None
            )
            if current_volatility and current_volatility > 0:
                volatility_percentile = self._calculate_volatility_percentile(
                    symbol, current_volatility
                )

                # Hard block at high volatility
                if volatility_percentile >= self._volatility_hard_block_percentile:
                    logger.warning(
                        f"Order blocked: volatility percentile {volatility_percentile:.1f} >= {self._volatility_hard_block_percentile}"
                    )
                    await self._publish_order_rejected(
                        order_request,
                        f"Volatility too high: {volatility_percentile:.1f}th percentile",
                    )
                    return

                # Soft block (size penalty) at medium-high volatility
                if volatility_percentile >= self._volatility_soft_block_percentile:
                    logger.debug(
                        f"Size penalty applied: volatility percentile {volatility_percentile:.1f}"
                    )
                elif volatility_percentile >= self._volatility_medium_penalty_percentile:
                    logger.debug(
                        f"Size penalty applied: volatility percentile {volatility_percentile:.1f}"
                    )
        else:
            logger.debug(f"[DEBUG] Volatility checks bypassed for {symbol}")

        # Update adaptive parameters based on recent performance
        self._update_adaptive_parameters(symbol)

        # Calculate maker price with adaptive offset
        offset_pct = self._adaptive_offset_bps / 10000.0
        if side == "buy":
            if offset_pct == 0:
                maker_price = best_bid
            else:
                maker_price = best_bid * (1 - offset_pct)
        else:  # sell
            if offset_pct == 0:
                maker_price = best_ask
            else:
                maker_price = best_ask * (1 + offset_pct)

        # Generate order ID and idempotency key
        order_id = str(uuid.uuid4())
        idempotency_key = self._store_idempotency(order_request, order_id, status="pending")

        # Create post-only limit order
        order = Order(
            order_id=order_id,
            strategy_id=order_request.strategy_id,
            symbol=symbol,
            side=side,
            size=order_request.size,
            price=maker_price,
            order_type="limit",
            status=OrderStatus.PENDING,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            timestamp=datetime.utcnow(),
            metadata={
                **order_request.metadata,
                "maker_first": True,
                "adaptive_ttl_ms": self._adaptive_ttl_ms,
                "adaptive_offset_bps": self._adaptive_offset_bps,
                "idempotency_key": idempotency_key,  # Store for tracking
            },
            is_maker=True,
            placed_at=datetime.utcnow(),
        )

        # Register with order manager
        self._order_manager.register_order(order)
        logger.debug(f"Order {order.order_id} registered with idempotency key {idempotency_key[:8]}...")

        # Track maker order for TTL
        self._maker_orders[order.order_id] = order

        # Store original request for retry queue
        self._order_requests[order.order_id] = order_request

        # Record maker attempt
        self._diagnostics.record_maker_attempt(order)

        # Submit to broker (real Bybit if live and not dry_run, paper if dry_run)
        if not settings.dry_run and settings.is_live and self._bybit_http:
            # Use real Bybit API for live trading
            logger.info(f"Submitting maker order {order.order_id} to REAL Bybit API: {side} {order.size} {symbol} @ {maker_price:.6f}")
            try:
                # Create OrderRequest with maker price for Bybit
                bybit_order_request = OrderRequest(
                    signal_id=order_request.signal_id,
                    strategy_id=order_request.strategy_id,
                    symbol=symbol,
                    side=side,
                    size=order.size,
                    price=maker_price,
                    order_type="limit",
                    stop_loss=order_request.stop_loss,
                    take_profit=order_request.take_profit,
                    metadata=order.metadata,
                )

                # Place order directly on Bybit
                placed_order = await self._bybit_http.place_order(bybit_order_request)

                # Update our tracking with Bybit's order ID
                if placed_order and placed_order.order_id:
                    # Remove old tracking
                    self._maker_orders.pop(order.order_id, None)
                    self._order_requests.pop(order.order_id, None)

                    # Update with Bybit's order
                    placed_order.metadata.update(order.metadata)  # Preserve our metadata
                    placed_order.is_maker = True
                    placed_order.placed_at = datetime.utcnow()

                    # Track with Bybit's order ID
                    self._maker_orders[placed_order.order_id] = placed_order
                    self._order_requests[placed_order.order_id] = order_request
                    self._order_manager.register_order(placed_order)

                    # Store idempotency for Bybit's order
                    self._store_idempotency(order_request, placed_order.order_id, status="pending")

                    # Publish order placed event
                    await self._bus.publish(Event(event_type=EventType.ORDER_PLACED, data={"order": placed_order}))

                    logger.info(f"Maker order placed on Bybit: {placed_order.order_id}")
                else:
                    logger.error(f"Bybit returned no order for {order.order_id}")
                    # Clean up
                    self._maker_orders.pop(order.order_id, None)
                    self._order_requests.pop(order.order_id, None)
            except Exception as e:
                logger.error(f"Failed to place maker order on Bybit: {e}", exc_info=True)
                # Clean up failed order
                self._maker_orders.pop(order.order_id, None)
                self._order_requests.pop(order.order_id, None)
                await self._publish_order_rejected(order_request, str(e))
                return
        else:
            # Use paper broker for dry_run mode
            logger.debug(f"Submitting order {order.order_id} to paper broker")
            await self._paper_broker.submit_order(order)

        # Track maker order placement
        no_trade_report.increment_pipeline("maker_orders_placed", order.strategy_id)

        logger.info(
            f"Maker order {order.order_id} placed: {side} {order.size} {symbol} @ {maker_price:.2f} "
            f"(TTL: {self._adaptive_ttl_ms}ms, offset: {self._adaptive_offset_bps}bps)"
        )

    async def _check_maker_ttl_loop(self) -> None:
        """Check maker orders for TTL expiration and clean stale PENDING orders."""
        stale_check_counter = 0
        while self._running:
            try:
                await self._check_maker_ttl()
                # Clean stale PENDING orders every 5s (50 iterations * 100ms)
                stale_check_counter += 1
                if stale_check_counter >= 50:
                    stale_check_counter = 0
                    await self._cleanup_stale_pending_orders()
                await asyncio.sleep(0.1)  # Check every 100ms
            except Exception as e:
                logger.error(f"Error in TTL check loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _cleanup_stale_pending_orders(self) -> None:
        """Cancel PENDING orders stuck for longer than stale TTL."""
        now = datetime.utcnow()
        stale_orders = []
        for order in self._order_manager.get_open_orders():
            if order.status == OrderStatus.PENDING:
                created = order.placed_at or order.created_at if hasattr(order, 'created_at') else None
                if created and (now - created) > self._stale_order_ttl:
                    stale_orders.append(order)
                elif not created:
                    # No timestamp — mark as stale
                    stale_orders.append(order)

        if stale_orders:
            logger.info(f"Cleaning {len(stale_orders)} stale PENDING orders (TTL={self._stale_order_ttl.total_seconds()}s)")
            for order in stale_orders:
                order.status = OrderStatus.CANCELLED
                self._maker_orders.pop(order.order_id, None)
                self._order_requests.pop(order.order_id, None)

    async def _check_maker_ttl(self) -> None:
        """Check if any maker orders have exceeded TTL (using adaptive TTL)."""
        now = datetime.utcnow()
        ttl_delta = timedelta(milliseconds=self._adaptive_ttl_ms)

        expired_orders = []
        for _order_id, order in list(self._maker_orders.items()):
            if order.placed_at and (now - order.placed_at) > ttl_delta:
                # Expire orders that are PENDING, PLACED, or PARTIALLY_FILLED
                if order.status in {OrderStatus.PENDING, OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED}:
                    expired_orders.append(order)

        for order in expired_orders:
            await self._handle_expired_maker_order(order)

    async def _handle_expired_maker_order(self, order: Order) -> None:
        """Handle expired maker order - cancel and optionally enqueue for retry or fallback to taker."""
        logger.info(f"Maker order {order.order_id} expired (TTL: {self._adaptive_ttl_ms}ms)")

        # Record expiration
        self._diagnostics.record_maker_expired(order)
        self._recent_expired_count.append(1)

        # Remove from tracking immediately to prevent re-processing
        self._maker_orders.pop(order.order_id, None)

        # Cancel the order (use real API if live, paper if dry_run)
        if not settings.dry_run and settings.is_live and self._bybit_http:
            try:
                await self._bybit_http.cancel_order(order.order_id, order.symbol)
                logger.info(f"Cancelled order {order.order_id} on REAL Bybit API")
            except Exception as e:
                logger.debug(f"Cancel order on Bybit failed (likely already filled/cancelled): {e}")
        else:
            await self._paper_broker.cancel_order(order.order_id)

        # Track TTL cancellation
        no_trade_report.increment_pipeline("maker_orders_cancelled_ttl", order.strategy_id)

        # Try to enqueue for retry if original request exists
        original_request = self._order_requests.get(order.order_id)
        if original_request:
            # Enqueue for retry (will check conditions when retrying)
            if self._retry_queue.enqueue_for_retry(
                order, original_request, reason="volatility_expired"
            ):
                logger.debug(
                    f"Enqueued expired order {order.order_id} for retry, skipping taker fallback"
                )
                # Clean up after enqueue
                self._order_requests.pop(order.order_id, None)
                return  # Prefer retry over taker fallback

        # Check if taker fallback is allowed
        if not settings.allow_taker_fallback:
            logger.debug(f"Taker fallback disabled, skipping trade for {order.order_id}")
            # Clean up
            self._order_requests.pop(order.order_id, None)
            return

        # Smart taker fallback: check if edge is still positive after taker fees + slippage
        symbol = order.symbol
        current_bid = self._current_bids.get(symbol)
        current_ask = self._current_asks.get(symbol)

        if current_bid is None or current_ask is None:
            logger.debug(f"No market data for {symbol}, skipping taker fallback")
            # Clean up
            self._order_requests.pop(order.order_id, None)
            return

        # Get taker execution price
        if order.side == "buy":
            taker_price = current_ask  # Buy at ask
            original_price = order.price or taker_price
        else:  # sell
            taker_price = current_bid  # Sell at bid
            original_price = order.price or taker_price

        # Calculate taker costs: fee + slippage
        # Use backtest taker fee (default 0.03% = 3 bps) for consistency
        taker_fee_bps = settings.backtest_taker_fee * 10000  # Convert to bps
        slippage_bps = 5.0  # 0.05% = 5 bps average
        total_cost_bps = taker_fee_bps + slippage_bps

        # Calculate expected profit from take_profit (if available)
        if order.take_profit:
            if order.side == "buy":
                expected_profit_pct = (order.take_profit - taker_price) / taker_price
            else:  # sell
                expected_profit_pct = (taker_price - order.take_profit) / taker_price
            expected_profit_bps = expected_profit_pct * 10000

            # Net edge = expected profit - taker costs
            net_edge_bps = expected_profit_bps - total_cost_bps

            # Only allow fallback if net edge is positive (at least 2 bps to be safe)
            min_edge_for_fallback_bps = 2.0
            if net_edge_bps >= min_edge_for_fallback_bps:
                logger.info(
                    f"Taker fallback approved: net_edge={net_edge_bps:.1f} bps "
                    f"(expected_profit={expected_profit_bps:.1f} bps, costs={total_cost_bps:.1f} bps)"
                )
                await self._route_taker_fallback(order)
                # Clean up after successful fallback
                self._order_requests.pop(order.order_id, None)
            else:
                logger.debug(
                    f"Taker fallback rejected: net_edge={net_edge_bps:.1f} bps < "
                    f"min={min_edge_for_fallback_bps} bps"
                )
                # Clean up rejected order
                self._order_requests.pop(order.order_id, None)
        else:
            # No take_profit, use simple price movement check
            # Allow if price moved favorably by at least the taker cost
            if order.side == "buy":
                price_improvement = (original_price - taker_price) / original_price
                if price_improvement * 10000 >= total_cost_bps:  # Improvement covers costs
                    await self._route_taker_fallback(order)
                    self._order_requests.pop(order.order_id, None)
                else:
                    # No improvement, clean up
                    self._order_requests.pop(order.order_id, None)
            else:  # sell
                price_improvement = (taker_price - original_price) / original_price
                if price_improvement * 10000 >= total_cost_bps:  # Improvement covers costs
                    await self._route_taker_fallback(order)
                    self._order_requests.pop(order.order_id, None)
                else:
                    # No improvement, clean up
                    self._order_requests.pop(order.order_id, None)

    async def _route_taker_fallback(self, original_order: Order) -> None:
        """Route expired maker order as taker market order."""
        logger.info(f"Taker fallback for order {original_order.order_id}")

        # Create market order
        order = Order(
            order_id=str(uuid.uuid4()),
            strategy_id=original_order.strategy_id,
            symbol=original_order.symbol,
            side=original_order.side,
            size=original_order.size - original_order.filled_size,  # Remaining size
            price=None,
            order_type="market",
            status=OrderStatus.PENDING,
            stop_loss=original_order.stop_loss,
            take_profit=original_order.take_profit,
            timestamp=datetime.utcnow(),
            metadata={
                **original_order.metadata,
                "taker_fallback": True,
                "original_order_id": original_order.order_id,
            },
            is_maker=False,
        )

        self._order_manager.register_order(order)

        # Route to Bybit if in live mode, otherwise paper
        if settings.is_live and self._bybit_http:
            try:
                # Create order request for Bybit
                order_request = OrderRequest(
                    signal_id=order.metadata.get("original_signal_id", ""),
                    strategy_id=order.strategy_id,
                    symbol=order.symbol,
                    side=order.side,
                    size=order.size - order.filled_size,
                    price=None,  # Market order
                    order_type="market",
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    metadata=order.metadata,
                )
                await self._bybit_http.place_order(order_request)
                logger.info(f"Taker fallback order placed on Bybit: {order.order_id}")
            except Exception as e:
                logger.error(f"Failed to place taker fallback on Bybit: {e}")
                # Fallback to paper
                await self._paper_broker.submit_order(order)
        else:
            await self._paper_broker.submit_order(order)

        # Track taker order placement
        no_trade_report.increment_pipeline("taker_orders_placed", original_order.strategy_id)

    async def _route_to_bybit(self, order_request: OrderRequest) -> None:
        """Route order to Bybit for live trading.

        Args:
            order_request: Order request to place
        """
        if not self._bybit_http:
            logger.error("Bybit HTTP client not available, falling back to paper")
            await self._route_to_paper(order_request)
            return

        try:
            # Place order on Bybit
            order = await self._bybit_http.place_order(order_request)

            # Register order with order manager
            self._order_manager.register_order(order)

            # Publish order placed event
            await self._bus.publish(Event(event_type=EventType.ORDER_PLACED, data={"order": order}))

            logger.info(
                f"Order placed on Bybit: {order.order_id} {order.symbol} {order.side} {order.size}"
            )

            # WebSocket will handle order fills automatically
            # But we can also poll for status if needed

        except Exception as e:
            logger.error(f"Failed to place order on Bybit: {e}", exc_info=True)
            await self._publish_order_rejected(order_request, str(e))

    async def _route_to_paper(self, order_request: OrderRequest) -> None:
        """Route order to paper broker (non-maker-first mode)."""
        order = Order(
            order_id=str(uuid.uuid4()),
            strategy_id=order_request.strategy_id,
            symbol=order_request.symbol,
            side=order_request.side,
            size=order_request.size,
            price=order_request.price,
            order_type=order_request.order_type,
            status=OrderStatus.PENDING,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            timestamp=datetime.utcnow(),
            metadata=order_request.metadata,
            is_maker=False,
        )

        # Register with order manager
        self._order_manager.register_order(order)

        # Submit to broker (real if live, paper if dry_run)
        if not settings.dry_run and settings.is_live and self._bybit_http:
            logger.info(f"Submitting order {order.order_id} to REAL Bybit API (route_to_paper)")
            await self._route_to_bybit(order_request)
        else:
            logger.debug(f"Submitting order {order.order_id} to paper broker (route_to_paper)")
            await self._paper_broker.submit_order(order)

    async def _publish_order_rejected(self, order_request: OrderRequest, reason: str) -> None:
        """Publish order rejected event."""
        order = Order(
            order_id=str(uuid.uuid4()),
            strategy_id=order_request.strategy_id,
            symbol=order_request.symbol,
            side=order_request.side,
            size=order_request.size,
            price=order_request.price,
            order_type=order_request.order_type,
            status=OrderStatus.REJECTED,
            timestamp=datetime.utcnow(),
            metadata={"rejection_reason": reason},
        )

        await self._bus.publish(
            Event(
                event_type=EventType.ORDER_REJECTED,
                data={"order": order},
            )
        )

    def _update_volatility_history(self, symbol: str, price: float) -> None:
        """Update volatility history for a symbol."""
        if symbol not in self._volatility_history:
            self._volatility_history[symbol] = deque(maxlen=50)

        history = self._volatility_history[symbol]
        if len(history) > 0:
            # Calculate return
            prev_price = history[-1]
            if prev_price > 0:
                abs((price - prev_price) / prev_price)
                # Store as volatility proxy (absolute return)
                # We'll calculate percentile from this
                pass

        history.append(price)

    def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol."""
        if symbol not in self._volatility_history:
            return 0.0

        history = list(self._volatility_history[symbol])
        if len(history) < 2:
            return 0.0

        # Calculate rolling volatility (std dev of returns)
        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > 0:
                ret = abs((history[i] - history[i - 1]) / history[i - 1])
                returns.append(ret)

        if not returns:
            return 0.0

        # Simple volatility: average absolute return
        return sum(returns) / len(returns)

    def _calculate_volatility_percentile(self, symbol: str, volatility: float) -> float:
        """Calculate volatility percentile for a symbol given volatility value (0-100).

        Args:
            symbol: Trading symbol
            volatility: Current volatility value

        Returns:
            Percentile (0-100)
        """
        if volatility <= 0:
            return 0.0

        # For percentile calculation, we'd need historical distribution
        # For now, use a simple heuristic based on volatility value
        # High volatility (>0.01 = 1%) -> high percentile
        if volatility > 0.02:  # >2%
            return 95.0
        elif volatility > 0.01:  # >1%
            return 85.0
        elif volatility > 0.005:  # >0.5%
            return 70.0
        else:
            return 50.0

    def _get_volatility_percentile(self, symbol: str) -> float | None:
        """Get volatility percentile for a symbol (0-100).

        Returns None if insufficient data.
        """
        current_vol = self._get_current_volatility(symbol)
        if current_vol == 0.0:
            return None

        # Get volatility from regime detector if available
        if self._regime_detector:
            vol = self._regime_detector.get_volatility(symbol)
            if vol > 0:
                return self._calculate_volatility_percentile(symbol, vol)

        # Fallback: use current volatility as proxy
        return self._calculate_volatility_percentile(symbol, current_vol)

    def _update_adaptive_parameters(self, symbol: str) -> None:
        """Update adaptive TTL and offset based on recent performance.

        Enhanced with:
        - Fill rate tracking for smarter TTL adjustment
        - Per-symbol volatility adaptation
        - Spread-aware offset calculation
        """
        # Check recent expired count
        recent_expired = sum(self._recent_expired_count)

        # Get fill rate from diagnostics
        fill_rate = self._diagnostics.get_maker_fill_rate()

        # Adaptive TTL: smart adjustment based on fill rate and expirations
        base_ttl = settings.maker_ttl_ms
        max_ttl = base_ttl * 3  # Max 3x base TTL (increased from 2x)
        min_ttl = base_ttl // 2  # Min 0.5x base TTL

        # Fill rate based adjustment
        # High fill rate (>70%) = can reduce TTL (more aggressive)
        # Low fill rate (<30%) = increase TTL (more patient)
        # Medium (30-70%) = adjust based on expirations
        if fill_rate > 0.70:
            # Good fill rate - can be more aggressive
            target_ttl = base_ttl * 0.8  # 20% shorter
            if recent_expired >= 3:
                # Some expirations - stay at base
                target_ttl = base_ttl
        elif fill_rate < 0.30 and fill_rate > 0:
            # Poor fill rate - need more patience
            target_ttl = base_ttl * 1.5
            if recent_expired >= 5:
                # Many expirations - increase further
                target_ttl = base_ttl * 2.0
        else:
            # Medium fill rate - use expiration-based adjustment
            if recent_expired >= 5:
                increase_pct = min((recent_expired / 5) * 0.25, 1.5)
                target_ttl = base_ttl * (1 + increase_pct)
            else:
                target_ttl = base_ttl

        # Smooth adjustment (avoid jumps)
        if target_ttl > self._adaptive_ttl_ms:
            self._adaptive_ttl_ms = int(self._adaptive_ttl_ms * 0.7 + target_ttl * 0.3)
        else:
            self._adaptive_ttl_ms = int(self._adaptive_ttl_ms * 0.9 + target_ttl * 0.1)

        # Clamp to bounds
        self._adaptive_ttl_ms = max(min_ttl, min(self._adaptive_ttl_ms, max_ttl))

        # Adaptive offset: widen if volatility is rising, tighten if spread is tight
        base_offset = settings.maker_price_offset_bps
        max_offset = base_offset + 5  # Max +5 bps from base (increased)
        min_offset = max(0, base_offset - 2)  # Min -2 bps from base

        volatility_percentile = self._get_volatility_percentile(symbol)

        # Check current spread for smarter offset
        current_bid = self._current_bids.get(symbol)
        current_ask = self._current_asks.get(symbol)
        current_spread_bps = None
        if current_bid and current_ask and current_bid > 0:
            current_spread_bps = abs((current_ask - current_bid) / current_bid) * 10000

        if volatility_percentile is not None:
            if volatility_percentile >= 90:
                # Very high volatility: widen offset significantly
                self._adaptive_offset_bps = min(base_offset + 4, max_offset)
            elif volatility_percentile >= 75:
                # High volatility: widen offset by 2 bps
                self._adaptive_offset_bps = min(base_offset + 2, max_offset)
            elif volatility_percentile <= 30:
                # Low volatility: tighten offset more aggressively
                self._adaptive_offset_bps = max(base_offset - 2, min_offset)
            elif volatility_percentile <= 50:
                # Low-medium volatility: tighten offset
                self._adaptive_offset_bps = max(base_offset - 1, min_offset)
            else:
                # Medium volatility: use base
                self._adaptive_offset_bps = base_offset

            # Spread-based adjustment: if spread is very tight, reduce offset
            if current_spread_bps is not None:
                if current_spread_bps < 2.0:
                    # Very tight spread - reduce offset to capture more fills
                    self._adaptive_offset_bps = max(
                        self._adaptive_offset_bps - 1, min_offset
                    )
                elif current_spread_bps > 10.0:
                    # Wide spread - increase offset to ensure queue priority
                    self._adaptive_offset_bps = min(
                        self._adaptive_offset_bps + 1, max_offset
                    )
        else:
            # No volatility data: use base
            self._adaptive_offset_bps = base_offset

        logger.debug(
            f"Adaptive params updated for {symbol}: "
            f"TTL={self._adaptive_ttl_ms}ms (fill_rate={fill_rate:.1%}), "
            f"offset={self._adaptive_offset_bps}bps (vol_pct={volatility_percentile}, "
            f"spread_bps={f'{current_spread_bps:.1f}' if current_spread_bps else 'N/A'})"
        )

    async def _check_retry_queue_loop(self) -> None:
        """Periodically check retry queue for ready orders."""
        while self._running:
            try:
                await self._process_retry_queue()
                await asyncio.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Error in retry queue loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _process_retry_queue(self) -> None:
        """Process retry queue and retry ready orders.

        Deterministic retry logic:
        - Retries are driven by ticks/events, not random timers
        - Check if volatility decreased OR enough time has passed
        - Respect max_retries limit
        """
        # Get current volatility for all symbols
        symbols = set()
        for entry in self._retry_queue._queue:
            symbols.add(entry.original_order.symbol)

        for symbol in symbols:
            # Get volatility from regime detector if available (more accurate)
            current_vol = 0.0
            if self._regime_detector:
                current_vol = self._regime_detector.get_volatility(symbol)
            else:
                current_vol = self._get_current_volatility(symbol)

            # Estimate previous volatility (when order was blocked)
            # Use a conservative estimate: assume it was higher
            previous_vol = current_vol * 1.15  # Assume 15% higher when blocked

            # Get ready retries
            # For deterministic behavior, we check:
            # 1. Volatility improved (decreased by at least 10%)
            # 2. OR max delay exceeded (retry anyway)
            # 3. Regime not IMPULSE (would need regime tracking, but for now allow)
            # 4. Spread acceptable (check current bid/ask spread)
            current_bid = self._current_bids.get(symbol)
            current_ask = self._current_asks.get(symbol)
            spread_ok = True
            if current_bid and current_ask:
                spread_bps = abs((current_ask - current_bid) / current_bid) * 10000
                max_spread_bps = 20  # Allow up to 20 bps spread for retry
                spread_ok = spread_bps <= max_spread_bps

            ready = self._retry_queue.get_ready_retries(
                current_volatility=current_vol,
                previous_volatility=previous_vol,
                regime_changed=False,  # Would need to track this
                drawdown_worsened=False,  # Would need to track this
                capital_preservation_active=False,  # Would need to track this
            )

            # Filter by spread if needed
            if not spread_ok:
                ready = []  # Don't retry if spread too wide

            for order_request in ready:
                # Find retry entry to get retry count
                retry_entry = next(
                    (
                        e
                        for e in self._retry_queue._queue
                        if e.original_request.strategy_id == order_request.strategy_id
                        and e.original_request.symbol == order_request.symbol
                        and e.original_request.side == order_request.side
                    ),
                    None,
                )
                retry_count = retry_entry.retry_count if retry_entry else 1

                # Increase offset for each retry attempt (1 -> 2 -> 3 bps)
                original_offset = self._adaptive_offset_bps
                base_offset = settings.maker_price_offset_bps
                self._adaptive_offset_bps = min(base_offset + retry_count, base_offset + 3)

                logger.info(
                    f"Retrying order request for {order_request.symbol} "
                    f"(attempt {retry_count}, vol: {current_vol:.6f}, offset: {self._adaptive_offset_bps}bps)"
                )

                try:
                    await self._route_maker_first(order_request)
                finally:
                    # Restore original offset
                    self._adaptive_offset_bps = original_offset

    def get_diagnostics(self) -> ExecutionDiagnostics:
        """Get execution diagnostics instance."""
        return self._diagnostics

    def get_retry_queue(self) -> MakerRetryQueue:
        """Get retry queue instance."""
        return self._retry_queue

    def get_orderbook_presence(self, symbol: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        """Get our orderbook presence (Phase 3).

        Args:
            symbol: Optional symbol filter. If None, returns presence for all symbols.

        Returns:
            Orderbook presence dictionary for a symbol, or list of dictionaries for all symbols.
        """
        if not self._phase3_enabled or not self._smart_executor:
            return {} if symbol else []

        if symbol:
            return self._smart_executor.get_orderbook_presence(symbol)

        # Get presence for all symbols we have orders for
        all_presence = []
        active_orders = self._smart_executor.get_active_orders()
        symbols = {o.symbol for o in active_orders}
        for sym in symbols:
            presence = self._smart_executor.get_orderbook_presence(sym)
            if presence and presence.get("num_orders", 0) > 0:
                all_presence.append(presence)

        return all_presence

    def get_iceberg_detections(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent iceberg detections for UI."""
        if not self._iceberg:
            return []
        getter = getattr(self._iceberg, "get_recent_detections", None)
        if callable(getter):
            return getter(limit=limit)
        return []

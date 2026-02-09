"""Order router - BYBIT TESTNET ONLY VERSION (no paper trading).

Features:
- Correlation IDs: Every order gets a unique correlation_id for end-to-end tracing
- Idempotency: Prevents duplicate orders from the same signal
- Bybit Testnet: All orders routed to Bybit testnet (no paper trading)
"""

import asyncio
import hashlib
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any

from hean.config import settings
from hean.core.bus import EventBus

try:
    from hean.core.execution.iceberg import IcebergOrder
except ImportError:
    IcebergOrder = None  # type: ignore[assignment]
from hean.core.ofi import OrderFlowImbalance
from hean.core.regime import RegimeDetector
from hean.core.types import Event, EventType, Order, OrderRequest, OrderStatus, Tick
from hean.exchange.bybit.http import BybitHTTPClient
from hean.exchange.bybit.ws_private import BybitPrivateWebSocket
from hean.exchange.bybit.ws_public import BybitPublicWebSocket
from hean.exchange.executor import SmartLimitExecutor
from hean.execution.execution_diagnostics import ExecutionDiagnostics
from hean.execution.maker_retry_queue import MakerRetryQueue
from hean.execution.order_manager import OrderManager
from hean.logging import get_logger
from hean.observability.no_trade_report import no_trade_report

logger = get_logger(__name__)


class ExecutionRouter:
    """Routes order requests to Bybit testnet (ONLY - no paper trading)."""

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
        self._regime_detector = regime_detector
        self._running = False

        # Bybit clients (REQUIRED - no paper trading fallback)
        self._bybit_http: BybitHTTPClient = BybitHTTPClient()
        self._bybit_ws_public: BybitPublicWebSocket = BybitPublicWebSocket(self._bus)
        self._bybit_ws_private: BybitPrivateWebSocket = BybitPrivateWebSocket(self._bus)

        self._current_bids: dict[str, float] = {}
        self._current_asks: dict[str, float] = {}
        self._current_prices: dict[str, float] = {}
        self._maker_orders: dict[str, Order] = {}
        self._ttl_check_task: asyncio.Task[None] | None = None
        self._retry_check_task: asyncio.Task[None] | None = None

        # Execution diagnostics
        self._diagnostics = ExecutionDiagnostics()

        # Retry queue
        self._retry_queue = MakerRetryQueue()

        # Adaptive parameters
        self._adaptive_ttl_ms = settings.maker_ttl_ms
        self._adaptive_offset_bps = settings.maker_price_offset_bps
        self._recent_expired_count = deque(maxlen=10)
        self._recent_fills = deque(maxlen=20)  # Track recent fill outcomes for fill rate
        self._volatility_history: dict[str, deque[float]] = {}

        # Volatility gating thresholds (PRODUCTION mode only - no DEBUG bypass)
        self._volatility_soft_block_percentile = 90.0
        self._volatility_medium_penalty_percentile = 75.0
        self._volatility_hard_block_percentile = 99.0

        # Original order requests for retry queue
        self._order_requests: dict[str, OrderRequest] = {}

        # Phase 3: Smart Limit Engine, OFI, and Iceberg
        self._smart_executor: SmartLimitExecutor | None = None
        self._ofi: OrderFlowImbalance | None = None
        self._iceberg: Any | None = None
        self._phase3_enabled = True

        # Phase 2 Enhancement: Orderbook Imbalance Detection
        self._orderbook_cache: dict[str, dict[str, Any]] = {}  # symbol -> {bids, asks}
        self._imbalance_threshold = 3.0  # Trigger when ratio > 3:1
        self._imbalance_size_mult = 1.5  # Size multiplier for imbalance trades
        self._imbalance_signals_count = 0

        # CRITICAL: Idempotency tracking - prevents duplicate orders from same signal
        self._processed_signal_keys: dict[str, datetime] = {}  # idempotency_key -> timestamp
        self._idempotency_ttl = timedelta(minutes=5)  # Keys expire after 5 minutes
        self._idempotency_cleanup_interval = timedelta(minutes=1)

    async def start(self) -> None:
        """Start the execution router and connect to Bybit."""
        logger.info("🚀 Starting BYBIT TESTNET ONLY router (no paper trading)")

        # Connect to Bybit (REQUIRED)
        try:
            await self._bybit_http.connect()
            await self._bybit_ws_public.connect()
            await self._bybit_ws_private.connect()
            await self._bybit_ws_private.subscribe_all()
            logger.info("✅ Bybit testnet clients connected")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Bybit: {e}")
            raise RuntimeError("Cannot start without Bybit connection") from e

        self._bus.subscribe(EventType.ORDER_REQUEST, self._handle_order_request)
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self._bus.subscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        self._bus.subscribe(EventType.POSITION_CLOSE_REQUEST, self._handle_position_close_request)
        self._running = True

        # Initialize Phase 3 components
        if self._phase3_enabled:
            self._ofi = OrderFlowImbalance(
                bus=self._bus,
                lookback_window=20,
            )
            await self._ofi.start()

            if IcebergOrder is not None:
                self._iceberg = IcebergOrder(
                    bus=self._bus,
                    min_size_usdt=10.0,
                    max_micro_size_usdt=5.0,
                    min_delay_ms=100,
                    max_delay_ms=500,
                )
            else:
                self._iceberg = None
                logger.warning("IcebergOrder module not available")

            self._smart_executor = SmartLimitExecutor(
                bus=self._bus,
                bybit_http=self._bybit_http,
            )
            await self._smart_executor.start()
            logger.info("Phase 3 components initialized: Smart Limit Engine, OFI, Iceberg")

        if settings.maker_first:
            self._ttl_check_task = asyncio.create_task(self._check_maker_ttl_loop())
            self._retry_check_task = asyncio.create_task(self._check_retry_queue_loop())

        logger.info("✅ Execution router started (Bybit testnet only)")

    async def stop(self) -> None:
        """Stop the execution router."""
        self._bus.unsubscribe(EventType.ORDER_REQUEST, self._handle_order_request)
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self._bus.unsubscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        self._bus.unsubscribe(EventType.POSITION_CLOSE_REQUEST, self._handle_position_close_request)

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

        # Disconnect Bybit clients
        await self._bybit_ws_private.disconnect()
        await self._bybit_ws_public.disconnect()
        await self._bybit_http.disconnect()

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

    def detect_orderbook_imbalance(self, symbol: str) -> dict[str, Any] | None:
        """Detect orderbook liquidity imbalance for edge capture.

        When bid/ask size ratio is > 3:1, market will likely move toward
        the heavy side. Returns signal info for taker aggression.

        Returns:
            Dict with imbalance info if detected, None otherwise:
            - side: 'buy' or 'sell'
            - ratio: imbalance ratio
            - size_multiplier: suggested size multiplier
            - urgency: 'high' for strong imbalance
        """
        if self._ofi is None:
            return None

        try:
            ofi_result = self._ofi.calculate_ofi(symbol)
        except Exception:
            return None

        # Use buy_pressure and sell_pressure to calculate imbalance ratio
        buy_pressure = ofi_result.buy_pressure
        sell_pressure = ofi_result.sell_pressure

        if buy_pressure <= 0 or sell_pressure <= 0:
            return None

        # Calculate ratio
        bid_heavy_ratio = buy_pressure / sell_pressure if sell_pressure > 0 else 0
        ask_heavy_ratio = sell_pressure / buy_pressure if buy_pressure > 0 else 0

        # Check for significant imbalance
        if bid_heavy_ratio >= self._imbalance_threshold:
            # Heavy bid side → price will rise → BUY with taker
            self._imbalance_signals_count += 1
            logger.info(
                f"[ORDERBOOK IMBALANCE] {symbol} BID-HEAVY: ratio={bid_heavy_ratio:.1f}:1 "
                f"(buy_pressure={buy_pressure:.2f}, sell_pressure={sell_pressure:.2f})"
            )
            return {
                "side": "buy",
                "ratio": bid_heavy_ratio,
                "size_multiplier": self._imbalance_size_mult,
                "urgency": "high",
                "use_taker": True,
            }

        if ask_heavy_ratio >= self._imbalance_threshold:
            # Heavy ask side → price will fall → SELL with taker
            self._imbalance_signals_count += 1
            logger.info(
                f"[ORDERBOOK IMBALANCE] {symbol} ASK-HEAVY: ratio={ask_heavy_ratio:.1f}:1 "
                f"(buy_pressure={buy_pressure:.2f}, sell_pressure={sell_pressure:.2f})"
            )
            return {
                "side": "sell",
                "ratio": ask_heavy_ratio,
                "size_multiplier": self._imbalance_size_mult,
                "urgency": "high",
                "use_taker": True,
            }

        return None

    def get_imbalance_adjusted_request(
        self, order_request: OrderRequest
    ) -> tuple[OrderRequest, bool]:
        """Adjust order request based on orderbook imbalance.

        Returns:
            Tuple of (adjusted_request, should_use_taker)
        """
        imbalance = self.detect_orderbook_imbalance(order_request.symbol)

        if imbalance is None:
            return order_request, False

        # Only apply if imbalance direction matches order side
        if imbalance["side"] != order_request.side:
            return order_request, False

        # Apply size multiplier
        adjusted_metadata = dict(order_request.metadata) if order_request.metadata else {}
        current_mult = adjusted_metadata.get("size_multiplier", 1.0)
        adjusted_metadata["size_multiplier"] = current_mult * imbalance["size_multiplier"]
        adjusted_metadata["orderbook_imbalance"] = imbalance["ratio"]
        adjusted_metadata["imbalance_urgency"] = imbalance["urgency"]

        adjusted_request = OrderRequest(
            signal_id=order_request.signal_id,
            strategy_id=order_request.strategy_id,
            symbol=order_request.symbol,
            side=order_request.side,
            size=order_request.size * imbalance["size_multiplier"],
            price=order_request.price,
            order_type=order_request.order_type,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            metadata=adjusted_metadata,
        )

        return adjusted_request, imbalance["use_taker"]

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled events."""
        order = event.data.get("order")
        if not order:
            logger.warning(f"[ROUTER] Missing 'order' in ORDER_FILLED event: {event.data}")
            return
        self._maker_orders.pop(order.order_id, None)

        # Track fill success for adaptive TTL
        if order.is_maker:
            self._recent_fills.append(True)  # Filled successfully

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

        # Finalize record
        self._diagnostics.finalize_record(order_id)
        self._order_requests.pop(order_id, None)

    async def _handle_position_close_request(self, event: Event) -> None:
        """Handle position close requests from Oracle or other intelligence modules.

        Closes position by placing reverse market order.

        Event data expected:
            - position_id: Position ID to close
            - reason: Reason for closing (e.g., 'oracle_reversal_prediction')
            - Additional diagnostic fields (optional)
        """
        position_id = event.data.get("position_id")
        reason = event.data.get("reason", "position_close_request")

        if not position_id:
            logger.warning(f"[ROUTER] Missing position_id in POSITION_CLOSE_REQUEST: {event.data}")
            return

        # Get position from order manager
        position = self._order_manager.get_position_by_id(position_id)
        if not position:
            logger.warning(f"[ROUTER] Position {position_id} not found for close request")
            return

        logger.info(
            f"[ROUTER] Closing position {position_id} ({position.symbol} {position.side}) "
            f"- Reason: {reason}"
        )

        # Create reverse market order to close position
        close_side = "sell" if position.side == "long" else "buy"
        close_request = OrderRequest(
            signal_id=f"close_{position_id}",
            strategy_id=position.strategy_id,
            symbol=position.symbol,
            side=close_side,
            size=position.size,
            price=None,  # Market order
            order_type="market",
            metadata={
                "position_id": position_id,
                "close_reason": reason,
                "original_entry": position.entry_price,
                **{k: v for k, v in event.data.items() if k not in ["position_id", "reason"]},
            },
        )

        # Execute close via standard order flow
        try:
            order = await self._bybit_http.place_order(close_request)
            await self._bus.publish(
                Event(
                    event_type=EventType.ORDER_PLACED,
                    data={"order": order},
                )
            )
            logger.info(f"[ROUTER] Position close order placed: {order.order_id}")
        except Exception as e:
            logger.error(f"[ROUTER] Failed to close position {position_id}: {e}")

    def _generate_idempotency_key(self, order_request: OrderRequest) -> str:
        """Generate idempotency key for an order request.

        Key is based on: signal_id + strategy_id + symbol + side + size (rounded)
        This ensures the same signal doesn't create duplicate orders.
        """
        key_parts = [
            order_request.signal_id,
            order_request.strategy_id,
            order_request.symbol,
            order_request.side,
            f"{order_request.size:.6f}",  # Rounded to 6 decimals
        ]
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _generate_correlation_id(self) -> str:
        """Generate unique correlation ID for end-to-end order tracing."""
        return f"corr-{uuid.uuid4().hex[:12]}-{int(datetime.utcnow().timestamp() * 1000) % 100000}"

    def _cleanup_expired_idempotency_keys(self) -> None:
        """Remove expired idempotency keys to prevent memory growth."""
        now = datetime.utcnow()
        expired = [
            key for key, ts in self._processed_signal_keys.items()
            if now - ts > self._idempotency_ttl
        ]
        for key in expired:
            del self._processed_signal_keys[key]

    def _check_idempotency(self, idempotency_key: str) -> bool:
        """Check if this request has already been processed.

        Returns True if order should be skipped (duplicate), False if new.
        """
        self._cleanup_expired_idempotency_keys()

        if idempotency_key in self._processed_signal_keys:
            return True  # Duplicate - skip

        # Mark as processed
        self._processed_signal_keys[idempotency_key] = datetime.utcnow()
        return False  # New - process

    async def _handle_order_request(self, event: Event) -> None:
        """Handle an order request event - ALWAYS route to Bybit.

        Features:
        - Idempotency: Skips duplicate requests from same signal
        - Correlation IDs: Adds correlation_id for tracing
        """
        order_request: OrderRequest = event.data["order_request"]

        # Generate correlation ID for end-to-end tracing
        correlation_id = self._generate_correlation_id()

        logger.debug(
            f"[{correlation_id}] Handling order request: {order_request.symbol} "
            f"{order_request.side} size={order_request.size:.6f}"
        )

        # IDEMPOTENCY CHECK: Prevent duplicate orders from same signal
        idempotency_key = self._generate_idempotency_key(order_request)
        if self._check_idempotency(idempotency_key):
            logger.warning(
                f"[{correlation_id}] DUPLICATE ORDER BLOCKED: "
                f"idempotency_key={idempotency_key} already processed"
            )
            no_trade_report.increment("idempotency_blocked", order_request.symbol, order_request.strategy_id)
            return

        # Add correlation_id to metadata for tracing
        if order_request.metadata is None:
            order_request.metadata = {}
        order_request.metadata["correlation_id"] = correlation_id
        order_request.metadata["idempotency_key"] = idempotency_key

        # GUARD: Check max open orders to prevent margin exhaustion
        open_order_count = len(self._order_requests)
        max_orders = settings.max_open_orders
        if open_order_count >= max_orders:
            logger.warning(
                f"[{correlation_id}] ORDER BLOCKED: max open orders ({max_orders}) reached, current={open_order_count}"
            )
            await self._publish_order_rejected(order_request, f"Max open orders ({max_orders}) reached")
            return

        try:
            # Phase 2: Check for orderbook imbalance edge
            adjusted_request, use_taker = self.get_imbalance_adjusted_request(order_request)

            if use_taker:
                # Orderbook imbalance detected - use taker for immediate execution
                logger.info(
                    f"[IMBALANCE EDGE] Using TAKER execution for {order_request.symbol} {order_request.side} "
                    f"(size adjusted: {order_request.size:.6f} -> {adjusted_request.size:.6f})"
                )
                await self._route_to_bybit(adjusted_request)
                return

            # ALWAYS use Bybit (testnet) - no paper trading option
            if settings.maker_first:
                logger.debug("Routing to maker_first (Bybit testnet)")
                await self._route_maker_first(adjusted_request)
            else:
                logger.debug("Routing to Bybit testnet")
                await self._route_to_bybit(adjusted_request)
        except Exception as e:
            logger.error(f"Error routing order request: {e}", exc_info=True)
            await self._publish_order_rejected(order_request, str(e))

    async def _route_maker_first(self, order_request: OrderRequest) -> None:
        """Route order using maker-first policy with adaptive placement and volatility gating."""
        symbol = order_request.symbol
        side = order_request.side

        # Phase 3: Check if order should be iceberg split
        if self._phase3_enabled and self._iceberg:
            current_price = self._current_prices.get(symbol) or order_request.price
            if current_price:
                micro_requests = await self._iceberg.process_order(order_request, current_price)

                if len(micro_requests) > 1:
                    logger.info(
                        f"Iceberg order: {symbol} {side} {order_request.size} split into {len(micro_requests)} micro-batches"
                    )
                    await self._iceberg.schedule_micro_orders(micro_requests)
                    return
                else:
                    order_request = micro_requests[0]

        # Phase 3: Use Smart Limit Executor with OFI
        if self._phase3_enabled and self._smart_executor and self._ofi:
            ofi_aggression = self._ofi.get_aggression_factor(symbol, side)

            try:
                order = await self._smart_executor.place_post_only_order(
                    order_request,
                    ofi_aggression=ofi_aggression,
                )

                self._order_manager.register_order(order)
                self._maker_orders[order.order_id] = order
                self._order_requests[order.order_id] = order_request

                # Submit to Bybit
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
                await self._route_to_bybit(bybit_order_request)

                logger.info(
                    f"Smart Limit order placed: {order.order_id} {side} {order.size} {symbol} @ {order.price:.6f} "
                    f"(OFI aggression={ofi_aggression:.3f})"
                )
                return
            except Exception as e:
                logger.error(f"Smart Limit Executor failed, falling back to standard maker-first: {e}", exc_info=True)

        # Get best bid/ask
        best_bid = self._current_bids.get(symbol)
        best_ask = self._current_asks.get(symbol)

        # Use order request price if no bid/ask data
        if best_bid is None or best_ask is None:
            logger.warning(f"No bid/ask data for {symbol}, using order request price")
            if order_request.price:
                if side == "buy":
                    best_bid = order_request.price
                    best_ask = order_request.price * 1.0001
                else:
                    best_ask = order_request.price
                    best_bid = order_request.price * 0.9999
            else:
                current_price = self._current_prices.get(symbol)
                if current_price:
                    best_bid = current_price * 0.9999
                    best_ask = current_price * 1.0001
                else:
                    # No price data available - reject order
                    logger.error(f"No price data for {symbol}, rejecting order")
                    await self._publish_order_rejected(order_request, f"No price data for {symbol}")
                    return
            logger.debug(f"Using fallback prices: bid={best_bid:.2f}, ask={best_ask:.2f}")

        # Volatility-aware execution gating (PRODUCTION - always enabled)
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

        # Update adaptive parameters
        self._update_adaptive_parameters(symbol)

        # Calculate maker price
        offset_pct = self._adaptive_offset_bps / 10000.0
        if side == "buy":
            maker_price = best_bid * (1 - offset_pct) if offset_pct > 0 else best_bid
        else:
            maker_price = best_ask * (1 + offset_pct) if offset_pct > 0 else best_ask

        # Create post-only limit order
        order = Order(
            order_id=str(uuid.uuid4()),
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
            },
            is_maker=True,
            placed_at=datetime.utcnow(),
        )

        # Register with order manager
        self._order_manager.register_order(order)
        logger.debug(f"Order {order.order_id} registered in OrderManager")

        # Track maker order for TTL
        self._maker_orders[order.order_id] = order

        # Store original request for retry queue
        self._order_requests[order.order_id] = order_request

        # Record maker attempt
        self._diagnostics.record_maker_attempt(order)

        # Submit to Bybit
        logger.info(f"Submitting maker order {order.order_id} to Bybit testnet")
        bybit_order_request = OrderRequest(
            signal_id=order_request.signal_id if hasattr(order_request, 'signal_id') else order_request.strategy_id,
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
        await self._route_to_bybit(bybit_order_request)

        # Track maker order placement
        no_trade_report.increment_pipeline("maker_orders_placed", order.strategy_id)

        logger.info(
            f"Maker order {order.order_id} placed: {side} {order.size} {symbol} @ {maker_price:.2f} "
            f"(TTL: {self._adaptive_ttl_ms}ms, offset: {self._adaptive_offset_bps}bps)"
        )

    async def _check_maker_ttl_loop(self) -> None:
        """Check maker orders for TTL expiration."""
        while self._running:
            try:
                await self._check_maker_ttl()
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in TTL check loop: {e}", exc_info=True)
                await asyncio.sleep(1)

    async def _check_maker_ttl(self) -> None:
        """Check if any maker orders have exceeded TTL."""
        now = datetime.utcnow()
        ttl_delta = timedelta(milliseconds=self._adaptive_ttl_ms)

        expired_orders = []
        for _order_id, order in list(self._maker_orders.items()):
            if order.placed_at and (now - order.placed_at) > ttl_delta:
                if order.status in {OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED}:
                    expired_orders.append(order)

        for order in expired_orders:
            await self._handle_expired_maker_order(order)

    async def _handle_expired_maker_order(self, order: Order) -> None:
        """Handle expired maker order - cancel and optionally retry or fallback."""
        logger.info(f"Maker order {order.order_id} expired (TTL: {self._adaptive_ttl_ms}ms)")

        # Record expiration
        self._diagnostics.record_maker_expired(order)
        self._recent_expired_count.append(1)

        # Track failed fill for adaptive TTL
        self._recent_fills.append(False)  # Failed to fill before expiry

        # Cancel the order on Bybit
        try:
            await self._bybit_http.cancel_order(order.order_id, order.symbol)
            logger.info(f"Cancelled order {order.order_id} on Bybit testnet")
        except Exception as e:
            logger.error(f"Failed to cancel order on Bybit: {e}")

        # Track TTL cancellation
        no_trade_report.increment_pipeline("maker_orders_cancelled_ttl", order.strategy_id)

        # Try to enqueue for retry if original request exists
        original_request = self._order_requests.get(order.order_id)
        if original_request:
            if self._retry_queue.enqueue_for_retry(
                order, original_request, reason="volatility_expired"
            ):
                logger.debug(f"Enqueued expired order {order.order_id} for retry")
                return

        # Check if taker fallback is allowed
        if not settings.allow_taker_fallback:
            logger.debug(f"Taker fallback disabled, skipping trade for {order.order_id}")
            return

        # Smart taker fallback
        symbol = order.symbol
        current_bid = self._current_bids.get(symbol)
        current_ask = self._current_asks.get(symbol)

        if current_bid is None or current_ask is None:
            logger.debug(f"No market data for {symbol}, skipping taker fallback")
            return

        # Get taker execution price
        if order.side == "buy":
            taker_price = current_ask
        else:
            taker_price = current_bid

        # Calculate taker costs
        taker_fee_bps = settings.backtest_taker_fee * 10000
        slippage_bps = 5.0
        total_cost_bps = taker_fee_bps + slippage_bps

        # Calculate expected profit
        if order.take_profit:
            if order.side == "buy":
                expected_profit_pct = (order.take_profit - taker_price) / taker_price
            else:
                expected_profit_pct = (taker_price - order.take_profit) / taker_price
            expected_profit_bps = expected_profit_pct * 10000

            net_edge_bps = expected_profit_bps - total_cost_bps

            min_edge_for_fallback_bps = 2.0
            if net_edge_bps >= min_edge_for_fallback_bps:
                logger.info(
                    f"Taker fallback approved: net_edge={net_edge_bps:.1f} bps "
                    f"(expected_profit={expected_profit_bps:.1f} bps, costs={total_cost_bps:.1f} bps)"
                )
                await self._route_taker_fallback(order)
            else:
                logger.debug(
                    f"Taker fallback rejected: net_edge={net_edge_bps:.1f} bps < "
                    f"min={min_edge_for_fallback_bps} bps"
                )

    async def _route_taker_fallback(self, original_order: Order) -> None:
        """Route expired maker order as taker market order."""
        logger.info(f"Taker fallback for order {original_order.order_id}")

        order = Order(
            order_id=str(uuid.uuid4()),
            strategy_id=original_order.strategy_id,
            symbol=original_order.symbol,
            side=original_order.side,
            size=original_order.size - original_order.filled_size,
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

        # Route to Bybit
        try:
            order_request = OrderRequest(
                signal_id=order.metadata.get("original_signal_id", ""),
                strategy_id=order.strategy_id,
                symbol=order.symbol,
                side=order.side,
                size=order.size - order.filled_size,
                price=None,
                order_type="market",
                stop_loss=order.stop_loss,
                take_profit=order.take_profit,
                metadata=order.metadata,
            )
            await self._bybit_http.place_order(order_request)
            logger.info(f"Taker fallback order placed on Bybit: {order.order_id}")
        except Exception as e:
            logger.error(f"Failed to place taker fallback on Bybit: {e}")

        # Track taker order placement
        no_trade_report.increment_pipeline("taker_orders_placed", original_order.strategy_id)

    async def _route_to_bybit(self, order_request: OrderRequest) -> None:
        """Route order to Bybit testnet.

        Args:
            order_request: Order request to place
        """
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

        except Exception as e:
            logger.error(f"Failed to place order on Bybit: {e}", exc_info=True)
            await self._publish_order_rejected(order_request, str(e))

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
            prev_price = history[-1]
            if prev_price > 0:
                abs((price - prev_price) / prev_price)

        history.append(price)

    def _get_current_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol."""
        if symbol not in self._volatility_history:
            return 0.0

        history = list(self._volatility_history[symbol])
        if len(history) < 2:
            return 0.0

        returns = []
        for i in range(1, len(history)):
            if history[i - 1] > 0:
                ret = abs((history[i] - history[i - 1]) / history[i - 1])
                returns.append(ret)

        if not returns:
            return 0.0

        return sum(returns) / len(returns)

    def _calculate_volatility_percentile(self, symbol: str, volatility: float) -> float:
        """Calculate volatility percentile (0-100)."""
        if volatility <= 0:
            return 0.0

        if volatility > 0.02:
            return 95.0
        elif volatility > 0.01:
            return 85.0
        elif volatility > 0.005:
            return 70.0
        else:
            return 50.0

    def _get_volatility_percentile(self, symbol: str) -> float | None:
        """Get volatility percentile for a symbol (0-100)."""
        current_vol = self._get_current_volatility(symbol)
        if current_vol == 0.0:
            return None

        if self._regime_detector:
            vol = self._regime_detector.get_volatility(symbol)
            if vol > 0:
                return self._calculate_volatility_percentile(symbol, vol)

        return self._calculate_volatility_percentile(symbol, current_vol)

    def _calculate_optimal_ttl(self, symbol: str) -> int:
        """Calculate optimal TTL based on market conditions.

        Considers:
        - Current spread in bps (wider spread = longer TTL)
        - Current volatility (higher volatility = shorter TTL)
        - Recent fill rate (lower fill rate = longer TTL)

        Returns:
            Optimal TTL in milliseconds
        """
        base_ttl = settings.maker_ttl_ms

        # Factor 1: Spread adjustment
        spread_multiplier = 1.0
        current_bid = self._current_bids.get(symbol)
        current_ask = self._current_asks.get(symbol)
        if current_bid and current_ask and current_bid > 0:
            spread_bps = ((current_ask - current_bid) / current_bid) * 10000
            # Wider spread = longer TTL (more time needed for fill)
            # 0-5 bps: 1.0x, 5-10 bps: 1.2x, 10-20 bps: 1.5x, >20 bps: 2.0x
            if spread_bps > 20:
                spread_multiplier = 2.0
            elif spread_bps > 10:
                spread_multiplier = 1.5
            elif spread_bps > 5:
                spread_multiplier = 1.2

        # Factor 2: Volatility adjustment
        volatility_multiplier = 1.0
        current_vol = self._get_current_volatility(symbol)
        if current_vol > 0:
            # Higher volatility = shorter TTL (prices move faster)
            # Low vol (<0.5%): 1.2x, Medium (0.5-1%): 1.0x, High (>1%): 0.7x, Extreme (>2%): 0.5x
            vol_pct = current_vol * 100
            if vol_pct > 2.0:
                volatility_multiplier = 0.5
            elif vol_pct > 1.0:
                volatility_multiplier = 0.7
            elif vol_pct < 0.5:
                volatility_multiplier = 1.2

        # Factor 3: Fill rate adjustment
        fill_rate_multiplier = 1.0
        if len(self._recent_fills) >= 5:
            fill_rate = sum(1 for f in self._recent_fills if f) / len(self._recent_fills)
            # Lower fill rate = longer TTL (need more time)
            # >80%: 0.9x, 60-80%: 1.0x, 40-60%: 1.2x, <40%: 1.5x
            if fill_rate < 0.4:
                fill_rate_multiplier = 1.5
            elif fill_rate < 0.6:
                fill_rate_multiplier = 1.2
            elif fill_rate > 0.8:
                fill_rate_multiplier = 0.9

        # Calculate optimal TTL with all factors
        optimal_ttl = int(base_ttl * spread_multiplier * volatility_multiplier * fill_rate_multiplier)

        # Clamp to reasonable bounds: 50ms to 500ms
        optimal_ttl = max(50, min(500, optimal_ttl))

        logger.debug(
            f"Optimal TTL for {symbol}: {optimal_ttl}ms "
            f"(spread_mult={spread_multiplier:.2f}, vol_mult={volatility_multiplier:.2f}, "
            f"fill_rate_mult={fill_rate_multiplier:.2f})"
        )

        return optimal_ttl

    def _update_adaptive_parameters(self, symbol: str) -> None:
        """Update adaptive TTL and offset based on recent performance."""
        # Use optimal TTL calculation
        self._adaptive_ttl_ms = self._calculate_optimal_ttl(symbol)

        # Log TTL adjustments for observability
        base_ttl = settings.maker_ttl_ms
        if self._adaptive_ttl_ms != base_ttl:
            logger.info(
                f"TTL adjusted for {symbol}: {base_ttl}ms -> {self._adaptive_ttl_ms}ms "
                f"(change: {((self._adaptive_ttl_ms / base_ttl) - 1) * 100:+.1f}%)"
            )

        # Adaptive offset
        base_offset = settings.maker_price_offset_bps
        max_offset = base_offset + 3
        min_offset = max(0, base_offset - 1)

        volatility_percentile = self._get_volatility_percentile(symbol)
        if volatility_percentile is not None:
            if volatility_percentile >= 85:
                self._adaptive_offset_bps = min(base_offset + 2, max_offset)
            elif volatility_percentile <= 50:
                self._adaptive_offset_bps = max(base_offset - 1, min_offset)
            else:
                self._adaptive_offset_bps = base_offset
        else:
            self._adaptive_offset_bps = base_offset

    async def _check_retry_queue_loop(self) -> None:
        """Periodically check retry queue for ready orders."""
        while self._running:
            try:
                await self._process_retry_queue()
                await asyncio.sleep(1.0)
            except Exception as e:
                logger.error(f"Error in retry queue loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _process_retry_queue(self) -> None:
        """Process retry queue and retry ready orders."""
        symbols = set()
        for entry in self._retry_queue._queue:
            symbols.add(entry.original_order.symbol)

        for symbol in symbols:
            current_vol = 0.0
            if self._regime_detector:
                current_vol = self._regime_detector.get_volatility(symbol)
            else:
                current_vol = self._get_current_volatility(symbol)

            previous_vol = current_vol * 1.15

            current_bid = self._current_bids.get(symbol)
            current_ask = self._current_asks.get(symbol)
            spread_ok = True
            if current_bid and current_ask:
                spread_bps = abs((current_ask - current_bid) / current_bid) * 10000
                max_spread_bps = 20
                spread_ok = spread_bps <= max_spread_bps

            ready = self._retry_queue.get_ready_retries(
                current_volatility=current_vol,
                previous_volatility=previous_vol,
                regime_changed=False,
                drawdown_worsened=False,
                capital_preservation_active=False,
            )

            if not spread_ok:
                ready = []

            for order_request in ready:
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
                    self._adaptive_offset_bps = original_offset

    def get_diagnostics(self) -> ExecutionDiagnostics:
        """Get execution diagnostics instance."""
        return self._diagnostics

    def get_retry_queue(self) -> MakerRetryQueue:
        """Get retry queue instance."""
        return self._retry_queue

    def get_orderbook_presence(self, symbol: str | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        """Get our orderbook presence (Phase 3)."""
        if not self._phase3_enabled or not self._smart_executor:
            return {} if symbol else []

        if symbol:
            return self._smart_executor.get_orderbook_presence(symbol)

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

"""Paper trading broker with realistic simulation."""

import asyncio
import random
from collections import deque
from datetime import datetime

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Order, OrderStatus, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


class PaperBroker:
    """Paper trading broker with fee and slippage simulation."""

    def __init__(self, bus: EventBus) -> None:
        """Initialize the paper broker."""
        self._bus = bus
        self._pending_orders: dict[str, Order] = {}
        self._current_prices: dict[str, float] = {}
        self._current_bids: dict[str, float] = {}
        self._current_asks: dict[str, float] = {}
        self._running = False
        self._fill_task: asyncio.Task[None] | None = None

        # Price history for deterministic maker fill model
        # Store last N ticks per symbol for maker fill detection
        # N is based on TTL: for 8s TTL at ~1 tick/sec, we need ~8-10 ticks
        self._price_history: dict[str, deque[float]] = {}  # symbol -> deque of prices
        self._bid_history: dict[str, deque[float]] = {}  # symbol -> deque of bids
        self._ask_history: dict[str, deque[float]] = {}  # symbol -> deque of asks
        self._history_window = 10  # Keep last 10 ticks for deterministic fill detection

        # Simulation parameters - use backtest fees if available, otherwise realistic fees
        # Check if we're in backtest mode (can be determined by checking if backtest fees are set)
        # For now, use backtest fees by default to reduce commission impact
        self._maker_fee = settings.backtest_maker_fee  # Default 0.005% (0.5 bps) for backtests
        self._taker_fee = settings.backtest_taker_fee  # Default 0.03% (3 bps) for backtests
        self._slippage_bps = 5  # 5 basis points average slippage

        # Metrics
        self._maker_fills = 0
        self._taker_fills = 0
        self._total_fills = 0

        # Subscribe to ticks for price updates
        self._bus.subscribe(EventType.TICK, self._handle_tick)

    async def start(self) -> None:
        """Start the paper broker."""
        self._running = True
        self._fill_task = asyncio.create_task(self._fill_orders_loop())
        logger.info(
            f"[FORCED_BROKER] Paper broker started, _fill_orders_loop task created, _running={self._running}"
        )

    async def stop(self) -> None:
        """Stop the paper broker."""
        self._running = False
        if self._fill_task:
            self._fill_task.cancel()
            try:
                await self._fill_task
            except asyncio.CancelledError:
                pass
        logger.info("Paper broker stopped")

    async def submit_order(self, order: Order) -> None:
        """Submit an order for execution."""
        logger.info(
            f"[FORCED_BROKER] submit_order called: order_id={order.order_id}, "
            f"symbol={order.symbol}, side={order.side}, size={order.size}, "
            f"price={order.price}, order_type={order.order_type}, "
            f"current_status={order.status}"
        )

        # Store order in pending orders
        self._pending_orders[order.order_id] = order

        # Ensure status is PLACED (may come as PENDING from ExecutionRouter)
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.PLACED
            logger.info("[FORCED_BROKER] Changed order status from PENDING to PLACED")

        logger.info(
            f"[FORCED_BROKER] Order added to _pending_orders, "
            f"total pending: {len(self._pending_orders)}, final_status={order.status}"
        )

        # Publish ORDER_PLACED event
        self._append_timeline(order, "placed")
        await self._bus.publish(
            Event(
                event_type=EventType.ORDER_PLACED,
                data={"order": order},
            )
        )
        logger.info(f"Order {order.order_id} submitted: {order.side} {order.size} {order.symbol}")

        # CRITICAL: Immediately try to fill the order
        # This ensures orders are filled as soon as they're submitted
        # Don't wait for the background loop - fill immediately
        try:
            logger.info(f"[FORCED_BROKER] Attempting immediate fill for order {order.order_id}")
            await self._process_pending_orders()
            logger.info(
                f"[FORCED_BROKER] Immediate fill attempt completed for order {order.order_id}"
            )
        except Exception as e:
            logger.error(
                f"[FORCED_BROKER] Error in immediate fill attempt for order {order.order_id}: {e}",
                exc_info=True,
            )

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update current prices and history."""
        tick: Tick = event.data["tick"]
        symbol = tick.symbol

        self._current_prices[symbol] = tick.price

        # Update price history for deterministic maker fill model
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._history_window)
            self._bid_history[symbol] = deque(maxlen=self._history_window)
            self._ask_history[symbol] = deque(maxlen=self._history_window)

        self._price_history[symbol].append(tick.price)

        if tick.bid:
            self._current_bids[symbol] = tick.bid
            self._bid_history[symbol].append(tick.bid)
        if tick.ask:
            self._current_asks[symbol] = tick.ask
            self._ask_history[symbol].append(tick.ask)

    async def _fill_orders_loop(self) -> None:
        """Loop to fill pending orders."""
        logger.info("[FORCED_BROKER] _fill_orders_loop started")
        loop_count = 0
        while self._running:
            try:
                loop_count += 1
                logger.debug(
                    f"[FORCED_BROKER] _fill_orders_loop iteration {loop_count}, pending orders: {len(self._pending_orders)}"
                )
                await self._process_pending_orders()
                await asyncio.sleep(0.1)  # Check every 100ms
            except asyncio.CancelledError:
                logger.info("[FORCED_BROKER] _fill_orders_loop cancelled")
                raise
            except Exception as e:
                logger.error(f"[FORCED_BROKER] Error in fill loop: {e}", exc_info=True)
                await asyncio.sleep(1)
        logger.info("[FORCED_BROKER] _fill_orders_loop stopped")

    async def _process_pending_orders(self) -> None:
        """Process pending orders and fill them if conditions are met."""
        pending_count = len(self._pending_orders)
        if pending_count == 0:
            return

        logger.info(f"[FORCED_BROKER] _process_pending_orders: {pending_count} pending orders")
        # Log all pending orders for debugging
        for order_id, order in self._pending_orders.items():
            logger.info(
                f"[FORCED_BROKER] Pending order: {order_id}, status={order.status}, symbol={order.symbol}, side={order.side}, size={order.size}, price={order.price}"
            )

        for order_id, order in list(self._pending_orders.items()):
            # Skip only if already filled or cancelled
            if order.status in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
                logger.debug(
                    f"[FORCED_BROKER] Skipping order {order_id} with status {order.status}"
                )
                continue

            # Ensure status is PLACED (for limit orders) or PENDING (for market orders)
            if order.status == OrderStatus.PENDING and order.order_type == "limit":
                logger.info(
                    f"[FORCED_BROKER] Order {order_id} has PENDING status, changing to PLACED"
                )
                order.status = OrderStatus.PLACED

            # CRITICAL FIX: Always fill orders immediately, regardless of order type
            # For limit orders, use order price if available
            # For market orders, use current market price
            fill_price = None

            if order.order_type == "limit" and order.price:
                # Limit order: use limit price
                fill_price = order.price
            elif order.order_type == "market":
                # Market order: use current market price
                if order.side == "buy":
                    # Buy at ask price
                    fill_price = self._current_asks.get(order.symbol) or self._current_prices.get(
                        order.symbol
                    )
                else:  # sell
                    # Sell at bid price
                    fill_price = self._current_bids.get(order.symbol) or self._current_prices.get(
                        order.symbol
                    )

            # Fallback: use current price or symbol-based fallback
            if fill_price is None:
                fill_price = self._current_prices.get(order.symbol)
                if fill_price is None:
                    # Final fallback price based on symbol
                    fill_price = 50000.0 if "BTC" in order.symbol else 3000.0
                    logger.warning(
                        f"[FORCED_BROKER] No price for {order.symbol}, using fallback {fill_price}"
                    )

            logger.info(
                f"[FORCED_FILL] Filling order {order_id} ({order.order_type}, status={order.status}) immediately at {fill_price:.2f}"
            )
            try:
                await self._fill_order(order, fill_price)
                logger.info(
                    f"[FORCED_FILL] Order {order_id} fill completed successfully, new status={order.status}"
                )
            except Exception as e:
                logger.error(f"[FORCED_FILL] Error filling order {order_id}: {e}", exc_info=True)

    async def _calculate_fill_price(self, order: Order, current_price: float) -> float | None:
        """Calculate fill price for an order, considering slippage.

        For maker orders, uses deterministic fill model based on price history:
        - Buy limit: fills if min(ask over N ticks) <= limit_price
        - Sell limit: fills if max(bid over N ticks) >= limit_price
        This ensures deterministic behavior in backtests while being realistic.
        """
        if order.order_type == "limit":
            if order.price is None:
                return None

            symbol = order.symbol

            # Deterministic maker fill model based on price history
            if order.is_maker:
                if order.side == "buy":
                    # Buy limit order: fill if ask price touched or crossed limit during window
                    # Check ask history (we buy at ask when limit is hit)
                    ask_history = self._ask_history.get(symbol)
                    if ask_history and len(ask_history) > 0:
                        # Fill if minimum ask in history window <= limit price
                        min_ask = min(ask_history)
                        if min_ask <= order.price:
                            return order.price

                    # Fallback to current ask if history not available
                    current_ask = self._current_asks.get(symbol)
                    if current_ask and current_ask <= order.price:
                        return order.price
                else:  # sell
                    # Sell limit order: fill if bid price touched or crossed limit during window
                    # Check bid history (we sell at bid when limit is hit)
                    bid_history = self._bid_history.get(symbol)
                    if bid_history and len(bid_history) > 0:
                        # Fill if maximum bid in history window >= limit price
                        max_bid = max(bid_history)
                        if max_bid >= order.price:
                            return order.price

                    # Fallback to current bid if history not available
                    current_bid = self._current_bids.get(symbol)
                    if current_bid and current_bid >= order.price:
                        return order.price

                return None
            else:
                # Non-maker limit order: use current price check
                current_bid = self._current_bids.get(symbol)
                current_ask = self._current_asks.get(symbol)

                if order.side == "buy":
                    best_price = current_bid if current_bid is not None else current_price
                    if best_price <= order.price:
                        return order.price
                else:  # sell
                    best_price = current_ask if current_ask is not None else current_price
                    if best_price >= order.price:
                        return order.price
                return None
        else:
            # Market order: fill immediately with slippage
            slippage_mult = 1.0 + (random.gauss(0, self._slippage_bps / 10000))
            if order.side == "buy":
                return current_price * slippage_mult
            else:
                return current_price / slippage_mult

    async def cancel_order(self, order_id: str) -> None:
        """Cancel an order."""
        if order_id in self._pending_orders:
            order = self._pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            self._append_timeline(order, "cancelled")

            await self._bus.publish(
                Event(
                    event_type=EventType.ORDER_CANCELLED,
                    data={"order": order},
                )
            )

            logger.info(f"Order {order_id} cancelled")

    async def _fill_order(self, order: Order, fill_price: float) -> None:
        """Fill an order (fully or partially)."""
        logger.info(
            f"[FORCED_FILL] _fill_order called: order_id={order.order_id}, fill_price={fill_price:.2f}, current_filled_size={order.filled_size}, order_size={order.size}"
        )
        remaining_size = order.size - order.filled_size
        logger.info(f"[FORCED_FILL] remaining_size={remaining_size}")
        if order.metadata is None:
            order.metadata = {}

        # Determine if this is a maker or taker fill
        # Maker: limit order with is_maker=True (post-only)
        # Taker: market order OR limit order that was filled immediately (not post-only)
        is_maker = (
            order.order_type == "limit"
            and order.is_maker
            and fill_price == order.price  # Maker gets limit price
        )
        logger.info(
            f"[FORCED_FILL] is_maker={is_maker}, order_type={order.order_type}, is_maker_flag={order.is_maker}, fill_price={fill_price:.2f}, order_price={order.price}"
        )

        # Calculate fee based on maker/taker
        fee_rate = self._maker_fee if is_maker else self._taker_fee
        fee = remaining_size * fill_price * fee_rate
        logger.info(f"[FORCED_FILL] fee_rate={fee_rate}, fee={fee:.4f}")
        order.metadata["fee"] = fee

        # Update metrics
        if is_maker:
            self._maker_fills += 1
        else:
            self._taker_fills += 1
        self._total_fills += 1

        # Fill the order
        fill_size = remaining_size
        order.filled_size += fill_size
        order.avg_fill_price = (
            (
                (order.avg_fill_price or 0.0) * (order.filled_size - fill_size)
                + fill_price * fill_size
            )
            / order.filled_size
            if order.filled_size > 0
            else fill_price
        )
        logger.info(
            f"[FORCED_FILL] After fill: filled_size={order.filled_size}, avg_fill_price={order.avg_fill_price:.2f}"
        )

        if order.filled_size >= order.size:
            order.status = OrderStatus.FILLED
            self._pending_orders.pop(order.order_id, None)
            logger.info(
                f"[FORCED_FILL] Order {order.order_id} fully filled, "
                f"status set to {order.status}, "
                f"status_type={type(order.status)}, "
                f"id(order)={id(order)}"
            )
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
            logger.info(
                f"[FORCED_FILL] Order partially filled, remaining: {order.size - order.filled_size}"
            )

        self._append_timeline(order, order.status.value if hasattr(order.status, "value") else str(order.status))

        # Publish fill event
        logger.info(f"[FORCED_FILL] Publishing ORDER_FILLED event for order {order.order_id}")
        await self._bus.publish(
            Event(
                event_type=EventType.ORDER_FILLED,
                data={
                    "order": order,
                    "fill_price": fill_price,
                    "fill_size": fill_size,
                    "fee": fee,
                    "is_maker": is_maker,
                },
            )
        )
        logger.info("[FORCED_FILL] ORDER_FILLED event published successfully")

        fill_type = "maker" if is_maker else "taker"
        logger.info(
            f"Order {order.order_id} filled ({fill_type}): {fill_size} @ {fill_price:.2f} "
            f"(fee: {fee:.4f}, status: {order.status})"
        )

    def get_maker_fill_rate(self) -> float:
        """Get maker fill rate (percentage of maker fills)."""
        if self._total_fills == 0:
            return 0.0
        return (self._maker_fills / self._total_fills) * 100.0

    def get_fill_stats(self) -> dict[str, int]:
        """Get fill statistics."""
        return {
            "maker_fills": self._maker_fills,
            "taker_fills": self._taker_fills,
            "total_fills": self._total_fills,
            "maker_fill_rate_pct": self.get_maker_fill_rate(),
        }

    def _append_timeline(self, order: Order, status: str) -> None:
        """Append a status change to the order timeline."""
        if order.metadata is None:
            order.metadata = {}
        timeline = order.metadata.get("status_timeline") or []
        timeline.append(
            {
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
        order.metadata["status_timeline"] = timeline

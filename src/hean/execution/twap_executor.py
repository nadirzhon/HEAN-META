"""TWAP (Time-Weighted Average Price) Executor.

Splits large orders into smaller time-weighted slices to:
1. Reduce market impact
2. Avoid detection by other algos
3. Achieve better average fill price

Uses randomized timing to avoid predictable patterns.
"""

import asyncio
import random
from datetime import datetime

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Order, OrderRequest, OrderStatus
from hean.logging import get_logger

logger = get_logger(__name__)


class TWAPExecutor:
    """Execute large orders using TWAP algorithm."""

    def __init__(
        self,
        enabled: bool = False,
        threshold_usd: float = 500.0,
        duration_sec: int = 300,
        num_slices: int = 10,
        randomize: bool = True,
    ) -> None:
        """Initialize TWAP executor.

        Args:
            enabled: Whether TWAP is enabled
            threshold_usd: Minimum order size (USD) to use TWAP
            duration_sec: Total duration to spread order over (seconds)
            num_slices: Number of slices to split order into
            randomize: Whether to randomize slice timing (±20%)
        """
        self._enabled = enabled
        self._threshold_usd = threshold_usd
        self._duration_sec = duration_sec
        self._num_slices = num_slices
        self._randomize = randomize

        # Stats
        self._stats = {
            "twap_executed": 0,
            "total_slices": 0,
            "successful_slices": 0,
            "failed_slices": 0,
        }

    def should_use_twap(
        self,
        size: float,
        current_price: float,
        threshold_usd: float | None = None,
    ) -> bool:
        """Determine if order should use TWAP.

        Args:
            size: Order size in base currency
            current_price: Current price
            threshold_usd: Custom threshold (if None, uses default)

        Returns:
            True if order size exceeds threshold
        """
        if not self._enabled:
            return False

        threshold = threshold_usd if threshold_usd is not None else self._threshold_usd
        notional = size * current_price

        return notional >= threshold

    async def execute_twap(
        self,
        order_request: OrderRequest,
        bybit_http,
        bus: EventBus,
        duration_sec: int | None = None,
        num_slices: int | None = None,
        randomize: bool | None = None,
    ) -> list[Order]:
        """Execute order using TWAP algorithm.

        Args:
            order_request: Original order request
            bybit_http: BybitHTTPClient for placing orders
            bus: Event bus for publishing ORDER_PLACED events
            duration_sec: Override default duration
            num_slices: Override default number of slices
            randomize: Override default randomize setting

        Returns:
            List of filled orders
        """
        if not self._enabled:
            raise ValueError("TWAP is disabled (TWAP_ENABLED=false)")

        self._stats["twap_executed"] += 1

        duration = duration_sec if duration_sec is not None else self._duration_sec
        slices = num_slices if num_slices is not None else self._num_slices
        should_randomize = randomize if randomize is not None else self._randomize

        # Calculate slice parameters
        slice_size = order_request.size / slices
        base_interval = duration / slices

        logger.info(
            f"[TWAP] Executing {order_request.symbol} {order_request.side} "
            f"size={order_request.size:.4f} over {duration}s in {slices} slices "
            f"(slice_size={slice_size:.4f}, interval={base_interval:.1f}s)"
        )

        filled_orders: list[Order] = []

        for i in range(slices):
            # Calculate wait time with optional randomization
            if should_randomize and i > 0:
                # Randomize ±20%
                jitter = random.uniform(-0.2, 0.2)
                wait_time = base_interval * (1 + jitter)
                wait_time = max(1.0, wait_time)  # Minimum 1 second
                await asyncio.sleep(wait_time)
            elif i > 0:
                await asyncio.sleep(base_interval)

            # Create slice order request
            slice_request = OrderRequest(
                signal_id=f"{order_request.signal_id}_twap_{i}",
                strategy_id=order_request.strategy_id,
                symbol=order_request.symbol,
                side=order_request.side,
                size=slice_size,
                price=order_request.price,
                order_type=order_request.order_type,
                stop_loss=order_request.stop_loss,
                take_profit=order_request.take_profit,
                reduce_only=order_request.reduce_only,
                metadata={
                    **order_request.metadata,
                    "twap_slice": i + 1,
                    "twap_total_slices": slices,
                    "twap_parent_signal": order_request.signal_id,
                },
            )

            try:
                # Place slice order
                self._stats["total_slices"] += 1

                order = await self._place_slice(
                    slice_request, bybit_http, bus, i + 1, slices
                )

                if order and order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                    filled_orders.append(order)
                    self._stats["successful_slices"] += 1
                else:
                    self._stats["failed_slices"] += 1
                    logger.warning(
                        f"[TWAP] Slice {i+1}/{slices} failed or not filled: {order}"
                    )

            except Exception as e:
                self._stats["failed_slices"] += 1
                logger.error(
                    f"[TWAP] Error placing slice {i+1}/{slices}: {e}",
                    exc_info=True,
                )

        # Calculate VWAP
        if filled_orders:
            vwap = self._calculate_vwap(filled_orders)
            logger.info(
                f"[TWAP] Completed {order_request.symbol} {order_request.side}: "
                f"{len(filled_orders)}/{slices} slices filled, VWAP={vwap:.4f}"
            )
        else:
            logger.error(
                f"[TWAP] FAILED {order_request.symbol} {order_request.side}: "
                f"No slices filled"
            )

        return filled_orders

    async def _place_slice(
        self,
        slice_request: OrderRequest,
        bybit_http,
        bus: EventBus,
        slice_num: int,
        total_slices: int,
    ) -> Order | None:
        """Place a single TWAP slice order.

        Args:
            slice_request: Slice order request
            bybit_http: BybitHTTPClient instance
            bus: Event bus for publishing events
            slice_num: Current slice number (1-indexed)
            total_slices: Total number of slices

        Returns:
            Order object if successful, None otherwise
        """
        logger.debug(
            f"[TWAP] Placing slice {slice_num}/{total_slices}: "
            f"{slice_request.symbol} {slice_request.side} {slice_request.size:.4f}"
        )

        # Place order via HTTP client
        if slice_request.order_type == "market":
            result = await bybit_http.place_market_order(
                symbol=slice_request.symbol,
                side=slice_request.side,
                qty=slice_request.size,
                reduce_only=slice_request.reduce_only,
            )
        else:
            result = await bybit_http.place_limit_order(
                symbol=slice_request.symbol,
                side=slice_request.side,
                qty=slice_request.size,
                price=slice_request.price,
                reduce_only=slice_request.reduce_only,
            )

        if not result or "orderId" not in result:
            logger.error(f"[TWAP] Failed to place slice {slice_num}: {result}")
            return None

        # Create Order object
        order = Order(
            order_id=result["orderId"],
            strategy_id=slice_request.strategy_id,
            symbol=slice_request.symbol,
            side=slice_request.side,
            size=slice_request.size,
            filled_size=result.get("cumExecQty", 0),
            price=slice_request.price,
            avg_fill_price=result.get("avgPrice"),
            order_type=slice_request.order_type,
            status=OrderStatus.PLACED,
            stop_loss=slice_request.stop_loss,
            take_profit=slice_request.take_profit,
            timestamp=datetime.utcnow(),
            metadata={
                **slice_request.metadata,
                "twap_slice_num": slice_num,
                "twap_total_slices": total_slices,
            },
        )

        # Publish ORDER_PLACED event
        await bus.publish(
            Event(
                event_type=EventType.ORDER_PLACED,
                data={
                    "order": order,
                    "twap_slice": slice_num,
                    "twap_total": total_slices,
                },
            )
        )

        logger.info(
            f"[TWAP] Slice {slice_num}/{total_slices} placed: "
            f"order_id={order.order_id}"
        )

        return order

    def _calculate_vwap(self, orders: list[Order]) -> float:
        """Calculate volume-weighted average price.

        Args:
            orders: List of filled orders

        Returns:
            VWAP price
        """
        total_value = 0.0
        total_volume = 0.0

        for order in orders:
            if order.avg_fill_price and order.filled_size:
                total_value += order.avg_fill_price * order.filled_size
                total_volume += order.filled_size

        if total_volume == 0:
            return 0.0

        return total_value / total_volume

    def get_stats(self) -> dict:
        """Get TWAP execution statistics."""
        total_slices = self._stats["total_slices"]
        if total_slices == 0:
            fill_rate = 0.0
        else:
            fill_rate = self._stats["successful_slices"] / total_slices * 100

        return {
            **self._stats,
            "fill_rate_pct": fill_rate,
        }

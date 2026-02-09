"""Smart execution algorithms: TWAP, VWAP, Iceberg Detection.

This module provides advanced order execution strategies to minimize market impact:
- TWAP (Time-Weighted Average Price): Splits large orders across time intervals
- VWAP (Volume-Weighted Average Price): Executes orders tracking volume patterns
- Iceberg Detection: Identifies hidden large orders in the order book

All executions publish events to the EventBus for observability and integrate
with the existing ExecutionRouter and RiskGovernor systems.
"""

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, OrderRequest
from hean.logging import get_logger, log_exception

logger = get_logger(__name__)


@dataclass
class ExecutionSlice:
    """A single slice of a parent order for TWAP/VWAP execution."""

    slice_id: str
    parent_order_id: str
    symbol: str
    side: str
    size: float
    price: float | None  # None for market orders
    scheduled_time: datetime
    executed: bool = False
    order_id: str | None = None
    fill_price: float | None = None
    fill_time: datetime | None = None


@dataclass
class IcebergLevel:
    """Detected iceberg order level."""

    symbol: str
    side: str  # "buy" or "sell"
    price: float
    visible_size: float
    estimated_hidden_size: float
    confidence: float  # 0.0 to 1.0
    detected_at: datetime
    refresh_count: int  # Number of times level has been refreshed


class TWAPExecutor:
    """Time-Weighted Average Price executor for large orders.

    Splits large orders into equal time-sliced pieces to minimize market impact
    and avoid signaling intent to other market participants.
    """

    def __init__(self, bus: EventBus, execution_router: Any) -> None:
        """Initialize TWAP executor.

        Args:
            bus: Event bus for publishing execution events
            execution_router: ExecutionRouter instance for submitting slices
        """
        self._bus = bus
        self._router = execution_router
        self._active_orders: dict[str, list[ExecutionSlice]] = {}
        self._execution_tasks: dict[str, asyncio.Task[None]] = {}
        self._running = False

        logger.info("TWAP Executor initialized")

    async def start(self) -> None:
        """Start the TWAP executor."""
        self._running = True
        logger.info("TWAP Executor started")

    async def stop(self) -> None:
        """Stop the TWAP executor and cancel pending orders."""
        self._running = False

        # Cancel all active execution tasks
        for parent_order_id, task in list(self._execution_tasks.items()):
            logger.info(f"Cancelling TWAP execution for order {parent_order_id}")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._active_orders.clear()
        self._execution_tasks.clear()
        logger.info("TWAP Executor stopped")

    async def execute_twap(
        self,
        order_request: OrderRequest,
        num_slices: int = 5,
        interval_seconds: int = 60,
        use_limit_orders: bool = False,
    ) -> str:
        """Split order into time-sliced pieces and execute over time.

        Args:
            order_request: Original order request to split
            num_slices: Number of time slices (default: 5)
            interval_seconds: Seconds between each slice (default: 60)
            use_limit_orders: Use limit orders instead of market (default: False)

        Returns:
            Parent order ID for tracking

        Raises:
            ValueError: If num_slices < 1 or interval_seconds < 1
        """
        if num_slices < 1:
            raise ValueError(f"num_slices must be >= 1, got {num_slices}")
        if interval_seconds < 1:
            raise ValueError(f"interval_seconds must be >= 1, got {interval_seconds}")

        parent_order_id = str(uuid.uuid4())
        slice_size = order_request.size / num_slices
        now = datetime.utcnow()

        # Create execution slices
        slices: list[ExecutionSlice] = []
        for i in range(num_slices):
            scheduled_time = now + timedelta(seconds=i * interval_seconds)
            slice_id = f"{parent_order_id}_slice_{i}"

            execution_slice = ExecutionSlice(
                slice_id=slice_id,
                parent_order_id=parent_order_id,
                symbol=order_request.symbol,
                side=order_request.side,
                size=slice_size,
                price=order_request.price if use_limit_orders else None,
                scheduled_time=scheduled_time,
            )
            slices.append(execution_slice)

        self._active_orders[parent_order_id] = slices

        logger.info(
            f"TWAP order created: parent_id={parent_order_id}, symbol={order_request.symbol}, "
            f"total_size={order_request.size:.4f}, num_slices={num_slices}, "
            f"interval={interval_seconds}s, use_limit={use_limit_orders}"
        )

        # Publish TWAP start event
        await self._bus.publish(
            Event(
                event_type=EventType.ORDER_PLACED,
                data={
                    "parent_order_id": parent_order_id,
                    "execution_type": "TWAP",
                    "symbol": order_request.symbol,
                    "side": order_request.side,
                    "total_size": order_request.size,
                    "num_slices": num_slices,
                    "interval_seconds": interval_seconds,
                    "strategy_id": order_request.strategy_id,
                },
            )
        )

        # Start async execution task
        task = asyncio.create_task(self._execute_slices(parent_order_id, order_request))
        self._execution_tasks[parent_order_id] = task

        return parent_order_id

    async def _execute_slices(
        self, parent_order_id: str, original_request: OrderRequest
    ) -> None:
        """Execute slices according to schedule.

        Args:
            parent_order_id: Parent order ID
            original_request: Original order request for metadata
        """
        slices = self._active_orders.get(parent_order_id)
        if not slices:
            logger.error(f"No slices found for parent_order_id={parent_order_id}")
            return

        executed_count = 0
        failed_count = 0

        try:
            for slice_obj in slices:
                if not self._running:
                    logger.warning(
                        f"TWAP executor stopped, cancelling remaining slices "
                        f"for {parent_order_id}"
                    )
                    break

                # Wait until scheduled time
                now = datetime.utcnow()
                wait_seconds = (slice_obj.scheduled_time - now).total_seconds()
                if wait_seconds > 0:
                    await asyncio.sleep(wait_seconds)

                # Create order request for this slice
                slice_request = OrderRequest(
                    signal_id=f"{original_request.signal_id}_twap_{slice_obj.slice_id}",
                    strategy_id=original_request.strategy_id,
                    symbol=slice_obj.symbol,
                    side=slice_obj.side,
                    size=slice_obj.size,
                    price=slice_obj.price,
                    order_type="limit" if slice_obj.price else "market",
                    stop_loss=original_request.stop_loss,
                    take_profit=original_request.take_profit,
                    metadata={
                        **original_request.metadata,
                        "twap_parent_id": parent_order_id,
                        "twap_slice_id": slice_obj.slice_id,
                    },
                )

                # Submit to execution router
                try:
                    # Call the router's internal order submission method
                    # This assumes ExecutionRouter has a method to handle OrderRequest
                    logger.info(
                        f"Executing TWAP slice: {slice_obj.slice_id}, "
                        f"size={slice_obj.size:.4f}"
                    )

                    # Publish as ORDER_REQUEST event for the router to pick up
                    await self._bus.publish(
                        Event(
                            event_type=EventType.ORDER_REQUEST,
                            data=slice_request.model_dump(),
                        )
                    )

                    slice_obj.executed = True
                    slice_obj.fill_time = datetime.utcnow()
                    executed_count += 1

                except Exception as exc:
                    failed_count += 1
                    log_exception(
                        logger,
                        exc,
                        {
                            "parent_order_id": parent_order_id,
                            "slice_id": slice_obj.slice_id,
                            "symbol": slice_obj.symbol,
                        },
                    )

            # Publish completion event
            completion_status = "completed" if failed_count == 0 else "partial"
            await self._bus.publish(
                Event(
                    event_type=EventType.ORDER_FILLED if failed_count == 0 else EventType.ERROR,
                    data={
                        "parent_order_id": parent_order_id,
                        "execution_type": "TWAP",
                        "status": completion_status,
                        "executed_slices": executed_count,
                        "failed_slices": failed_count,
                        "total_slices": len(slices),
                    },
                )
            )

            logger.info(
                f"TWAP execution completed: parent_id={parent_order_id}, "
                f"executed={executed_count}/{len(slices)}, failed={failed_count}"
            )

        except asyncio.CancelledError:
            logger.warning(f"TWAP execution cancelled for {parent_order_id}")
            raise
        finally:
            # Cleanup
            self._execution_tasks.pop(parent_order_id, None)

    def get_order_status(self, parent_order_id: str) -> dict[str, Any]:
        """Get status of a TWAP order.

        Args:
            parent_order_id: Parent order ID

        Returns:
            Status dictionary with execution progress
        """
        slices = self._active_orders.get(parent_order_id)
        if not slices:
            return {"error": "Order not found", "parent_order_id": parent_order_id}

        executed = sum(1 for s in slices if s.executed)
        pending = len(slices) - executed

        return {
            "parent_order_id": parent_order_id,
            "total_slices": len(slices),
            "executed_slices": executed,
            "pending_slices": pending,
            "progress_pct": (executed / len(slices) * 100) if slices else 0.0,
            "slices": [
                {
                    "slice_id": s.slice_id,
                    "size": s.size,
                    "scheduled_time": s.scheduled_time.isoformat(),
                    "executed": s.executed,
                    "fill_price": s.fill_price,
                    "fill_time": s.fill_time.isoformat() if s.fill_time else None,
                }
                for s in slices
            ],
        }


class VWAPExecutor:
    """Volume-Weighted Average Price executor.

    Executes orders by tracking market volume and maintaining a target
    participation rate to minimize market impact while ensuring timely fills.
    """

    def __init__(self, bus: EventBus, execution_router: Any) -> None:
        """Initialize VWAP executor.

        Args:
            bus: Event bus for publishing execution events
            execution_router: ExecutionRouter instance for submitting orders
        """
        self._bus = bus
        self._router = execution_router
        self._volume_history: dict[str, deque[tuple[datetime, float]]] = {}
        self._active_orders: dict[str, dict[str, Any]] = {}
        self._running = False

        # VWAP parameters
        self._volume_window_seconds = 300  # 5 minutes for volume calculation
        self._min_participation_rate = 0.01  # 1% minimum
        self._max_participation_rate = 0.25  # 25% maximum

        logger.info("VWAP Executor initialized")

    async def start(self) -> None:
        """Start the VWAP executor."""
        self._running = True
        logger.info("VWAP Executor started")

    async def stop(self) -> None:
        """Stop the VWAP executor."""
        self._running = False
        self._active_orders.clear()
        self._volume_history.clear()
        logger.info("VWAP Executor stopped")

    def update_volume(self, symbol: str, volume: float) -> None:
        """Update volume data for a symbol.

        Args:
            symbol: Trading symbol
            volume: Volume amount
        """
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=1000)

        self._volume_history[symbol].append((datetime.utcnow(), volume))

    def _get_recent_volume_rate(self, symbol: str) -> float:
        """Calculate recent volume rate (volume per second).

        Args:
            symbol: Trading symbol

        Returns:
            Volume rate in size per second
        """
        if symbol not in self._volume_history:
            return 0.0

        now = datetime.utcnow()
        cutoff = now - timedelta(seconds=self._volume_window_seconds)

        recent_volumes = [
            vol for ts, vol in self._volume_history[symbol] if ts >= cutoff
        ]

        if not recent_volumes:
            return 0.0

        total_volume = sum(recent_volumes)
        return total_volume / self._volume_window_seconds

    async def execute_vwap(
        self,
        order_request: OrderRequest,
        target_participation: float = 0.1,
        max_duration_seconds: int = 600,
    ) -> str:
        """Execute order tracking VWAP with participation rate.

        Args:
            order_request: Original order request
            target_participation: Target participation rate (0.0 to 1.0, default: 0.1 = 10%)
            max_duration_seconds: Maximum execution duration (default: 600s = 10min)

        Returns:
            Parent order ID for tracking

        Raises:
            ValueError: If participation rate is invalid
        """
        if not (0.0 < target_participation <= 1.0):
            raise ValueError(f"target_participation must be in (0, 1], got {target_participation}")

        # Clamp participation to safe bounds
        clamped_participation = max(
            self._min_participation_rate,
            min(target_participation, self._max_participation_rate),
        )

        if clamped_participation != target_participation:
            logger.warning(
                f"Clamped participation rate from {target_participation:.2%} "
                f"to {clamped_participation:.2%}"
            )

        parent_order_id = str(uuid.uuid4())

        self._active_orders[parent_order_id] = {
            "order_request": order_request,
            "target_participation": clamped_participation,
            "max_duration_seconds": max_duration_seconds,
            "remaining_size": order_request.size,
            "executed_size": 0.0,
            "start_time": datetime.utcnow(),
        }

        logger.info(
            f"VWAP order created: parent_id={parent_order_id}, symbol={order_request.symbol}, "
            f"size={order_request.size:.4f}, participation={clamped_participation:.2%}, "
            f"max_duration={max_duration_seconds}s"
        )

        # Publish VWAP start event
        await self._bus.publish(
            Event(
                event_type=EventType.ORDER_PLACED,
                data={
                    "parent_order_id": parent_order_id,
                    "execution_type": "VWAP",
                    "symbol": order_request.symbol,
                    "side": order_request.side,
                    "total_size": order_request.size,
                    "target_participation": clamped_participation,
                    "max_duration_seconds": max_duration_seconds,
                    "strategy_id": order_request.strategy_id,
                },
            )
        )

        # Start execution loop (simplified for production)
        # In production, this would be a background task checking volume periodically
        # For now, we execute immediately with volume-adjusted sizing
        await self._execute_vwap_order(parent_order_id)

        return parent_order_id

    async def _execute_vwap_order(self, parent_order_id: str) -> None:
        """Execute VWAP order based on market volume.

        Args:
            parent_order_id: Parent order ID
        """
        order_data = self._active_orders.get(parent_order_id)
        if not order_data:
            logger.error(f"No order data found for parent_order_id={parent_order_id}")
            return

        original_request: OrderRequest = order_data["order_request"]
        target_participation: float = order_data["target_participation"]

        # Calculate volume-adjusted order size
        volume_rate = self._get_recent_volume_rate(original_request.symbol)

        if volume_rate > 0:
            # Adjust size based on participation rate
            adjusted_size = min(
                original_request.size,
                volume_rate * target_participation * 60,  # 1 minute worth
            )
        else:
            # No volume data, use small fraction
            adjusted_size = original_request.size * 0.2

        # Create adjusted order request
        vwap_request = OrderRequest(
            signal_id=f"{original_request.signal_id}_vwap",
            strategy_id=original_request.strategy_id,
            symbol=original_request.symbol,
            side=original_request.side,
            size=adjusted_size,
            price=original_request.price,
            order_type=original_request.order_type,
            stop_loss=original_request.stop_loss,
            take_profit=original_request.take_profit,
            metadata={
                **original_request.metadata,
                "vwap_parent_id": parent_order_id,
                "volume_rate": volume_rate,
                "participation_rate": target_participation,
            },
        )

        logger.info(
            f"Executing VWAP order: parent_id={parent_order_id}, "
            f"adjusted_size={adjusted_size:.4f}, volume_rate={volume_rate:.2f}"
        )

        # Publish to event bus for execution router
        await self._bus.publish(
            Event(
                event_type=EventType.ORDER_REQUEST,
                data=vwap_request.model_dump(),
            )
        )

        # Update tracking
        order_data["executed_size"] += adjusted_size
        order_data["remaining_size"] -= adjusted_size

        # Publish completion event
        await self._bus.publish(
            Event(
                event_type=EventType.ORDER_FILLED,
                data={
                    "parent_order_id": parent_order_id,
                    "execution_type": "VWAP",
                    "executed_size": order_data["executed_size"],
                    "remaining_size": order_data["remaining_size"],
                },
            )
        )

    def get_order_status(self, parent_order_id: str) -> dict[str, Any]:
        """Get status of a VWAP order.

        Args:
            parent_order_id: Parent order ID

        Returns:
            Status dictionary with execution progress
        """
        order_data = self._active_orders.get(parent_order_id)
        if not order_data:
            return {"error": "Order not found", "parent_order_id": parent_order_id}

        original_request = order_data["order_request"]
        elapsed = (datetime.utcnow() - order_data["start_time"]).total_seconds()

        return {
            "parent_order_id": parent_order_id,
            "symbol": original_request.symbol,
            "total_size": original_request.size,
            "executed_size": order_data["executed_size"],
            "remaining_size": order_data["remaining_size"],
            "progress_pct": (order_data["executed_size"] / original_request.size * 100)
            if original_request.size > 0
            else 0.0,
            "elapsed_seconds": elapsed,
            "target_participation": order_data["target_participation"],
        }


class IcebergDetector:
    """Detect hidden iceberg orders in orderbook.

    Iceberg orders are large orders that only show a small portion in the
    order book, refreshing when filled. Detection helps identify major
    support/resistance levels and institutional activity.
    """

    def __init__(self) -> None:
        """Initialize iceberg detector."""
        self._detected_levels: dict[str, dict[float, IcebergLevel]] = {}
        self._refresh_threshold = 3  # Require 3+ refreshes to confirm iceberg
        self._confidence_decay_seconds = 300  # 5 minutes

        logger.info("Iceberg Detector initialized")

    def update_orderbook(
        self, symbol: str, bids: list[tuple[float, float]], asks: list[tuple[float, float]]
    ) -> None:
        """Update orderbook data and detect icebergs.

        Args:
            symbol: Trading symbol
            bids: List of (price, size) for bid levels
            asks: List of (price, size) for ask levels
        """
        if symbol not in self._detected_levels:
            self._detected_levels[symbol] = {}

        # Check for level refreshes (indicating iceberg)
        for price, size in bids + asks:
            side = "buy" if (price, size) in bids else "sell"
            self._check_level_refresh(symbol, side, price, size)

    def _check_level_refresh(
        self, symbol: str, side: str, price: float, size: float
    ) -> None:
        """Check if a price level has been refreshed (iceberg indicator).

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            price: Price level
            size: Current visible size
        """
        level_key = price

        if level_key in self._detected_levels[symbol]:
            level = self._detected_levels[symbol][level_key]
            # Check if this is a refresh (same price, similar size after time gap)
            time_since_last = (datetime.utcnow() - level.detected_at).total_seconds()

            if time_since_last > 5:  # At least 5 seconds between refreshes
                level.refresh_count += 1
                level.detected_at = datetime.utcnow()

                # Update confidence based on refresh count
                level.confidence = min(
                    1.0, level.refresh_count / (self._refresh_threshold * 2)
                )

                # Estimate hidden size (heuristic: visible * refresh_count)
                level.estimated_hidden_size = size * level.refresh_count

                if level.refresh_count >= self._refresh_threshold:
                    logger.info(
                        f"Iceberg detected: {symbol} {side} @ {price:.2f}, "
                        f"refreshes={level.refresh_count}, confidence={level.confidence:.2f}, "
                        f"estimated_hidden={level.estimated_hidden_size:.4f}"
                    )
        else:
            # First detection of this level
            self._detected_levels[symbol][level_key] = IcebergLevel(
                symbol=symbol,
                side=side,
                price=price,
                visible_size=size,
                estimated_hidden_size=size,
                confidence=0.1,  # Low initial confidence
                detected_at=datetime.utcnow(),
                refresh_count=1,
            )

    def detect_iceberg(self, symbol: str, side: str) -> list[dict[str, Any]]:
        """Return detected iceberg levels with confidence.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell" (filter by side)

        Returns:
            List of iceberg level dictionaries sorted by confidence
        """
        if symbol not in self._detected_levels:
            return []

        now = datetime.utcnow()
        icebergs = []

        for level in self._detected_levels[symbol].values():
            # Filter by side
            if level.side != side:
                continue

            # Decay confidence over time
            age_seconds = (now - level.detected_at).total_seconds()
            if age_seconds > self._confidence_decay_seconds:
                continue  # Too old, skip

            # Only return confirmed icebergs
            if level.refresh_count >= self._refresh_threshold:
                icebergs.append(
                    {
                        "price": level.price,
                        "visible_size": level.visible_size,
                        "estimated_hidden_size": level.estimated_hidden_size,
                        "confidence": level.confidence,
                        "refresh_count": level.refresh_count,
                        "age_seconds": age_seconds,
                    }
                )

        # Sort by confidence descending
        icebergs.sort(key=lambda x: x["confidence"], reverse=True)
        return icebergs

    def get_all_icebergs(self, min_confidence: float = 0.5) -> dict[str, list[dict[str, Any]]]:
        """Get all detected icebergs across all symbols.

        Args:
            min_confidence: Minimum confidence threshold (default: 0.5)

        Returns:
            Dictionary mapping symbol to list of iceberg levels
        """
        result: dict[str, list[dict[str, Any]]] = {}

        for symbol in self._detected_levels:
            buy_icebergs = [
                iceberg
                for iceberg in self.detect_iceberg(symbol, "buy")
                if iceberg["confidence"] >= min_confidence
            ]
            sell_icebergs = [
                iceberg
                for iceberg in self.detect_iceberg(symbol, "sell")
                if iceberg["confidence"] >= min_confidence
            ]

            if buy_icebergs or sell_icebergs:
                result[symbol] = {
                    "buy": buy_icebergs,
                    "sell": sell_icebergs,
                }

        return result

    def clear_old_levels(self, max_age_seconds: int = 600) -> int:
        """Clear old iceberg detections.

        Args:
            max_age_seconds: Maximum age in seconds (default: 600 = 10 minutes)

        Returns:
            Number of levels cleared
        """
        now = datetime.utcnow()
        cleared = 0

        for symbol in list(self._detected_levels.keys()):
            levels = self._detected_levels[symbol]
            for price in list(levels.keys()):
                level = levels[price]
                age_seconds = (now - level.detected_at).total_seconds()
                if age_seconds > max_age_seconds:
                    del levels[price]
                    cleared += 1

            # Clean up empty symbol entries
            if not levels:
                del self._detected_levels[symbol]

        if cleared > 0:
            logger.debug(f"Cleared {cleared} old iceberg levels")

        return cleared

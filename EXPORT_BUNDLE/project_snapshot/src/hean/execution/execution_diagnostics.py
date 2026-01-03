"""Execution diagnostics - tracks order execution metrics."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from hean.core.types import Order
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OrderExecutionRecord:
    """Record of a single order's execution lifecycle."""

    order_id: str
    strategy_id: str
    symbol: str
    side: str
    maker_attempted: bool = False
    maker_filled: bool = False
    maker_expired: bool = False
    rejected_by_volatility_soft: bool = False
    rejected_by_volatility_hard: bool = False
    time_to_fill_ms: int | None = None
    placed_at: datetime | None = None
    filled_at: datetime | None = None
    expired_at: datetime | None = None


class ExecutionDiagnostics:
    """Tracks execution metrics for orders."""

    def __init__(self, max_history: int = 1000) -> None:
        """Initialize execution diagnostics.

        Args:
            max_history: Maximum number of order records to keep in history
        """
        self._records: dict[str, OrderExecutionRecord] = {}
        self._history: deque[OrderExecutionRecord] = deque(maxlen=max_history)
        self._maker_attempted = 0
        self._maker_filled = 0
        self._maker_expired = 0
        self._rejected_soft = 0
        self._rejected_hard = 0
        self._total_fill_time_ms = 0
        self._fill_count = 0

    def record_maker_attempt(self, order: Order) -> None:
        """Record that a maker order was attempted."""
        record = self._get_or_create_record(order)
        record.maker_attempted = True
        record.placed_at = order.placed_at or datetime.utcnow()
        self._maker_attempted += 1
        logger.debug(f"Recorded maker attempt for order {order.order_id}")

    def record_maker_fill(self, order: Order) -> None:
        """Record that a maker order was filled."""
        record = self._get_or_create_record(order)
        record.maker_filled = True
        record.filled_at = datetime.utcnow()

        if record.placed_at:
            time_to_fill = (record.filled_at - record.placed_at).total_seconds() * 1000
            record.time_to_fill_ms = int(time_to_fill)
            self._total_fill_time_ms += record.time_to_fill_ms
            self._fill_count += 1

        self._maker_filled += 1
        logger.debug(f"Recorded maker fill for order {order.order_id}")

    def record_maker_expired(self, order: Order) -> None:
        """Record that a maker order expired."""
        record = self._get_or_create_record(order)
        record.maker_expired = True
        record.expired_at = datetime.utcnow()
        self._maker_expired += 1
        logger.debug(f"Recorded maker expiration for order {order.order_id}")

    def record_volatility_rejection_soft(self, order: Order) -> None:
        """Record a soft volatility rejection (retry later)."""
        record = self._get_or_create_record(order)
        record.rejected_by_volatility_soft = True
        self._rejected_soft += 1
        logger.debug(f"Recorded soft volatility rejection for order {order.order_id}")

    def record_volatility_rejection_hard(self, order: Order) -> None:
        """Record a hard volatility rejection (skip entirely)."""
        record = self._get_or_create_record(order)
        record.rejected_by_volatility_hard = True
        self._rejected_hard += 1
        logger.debug(f"Recorded hard volatility rejection for order {order.order_id}")

    def _get_or_create_record(self, order: Order) -> OrderExecutionRecord:
        """Get existing record or create a new one."""
        if order.order_id not in self._records:
            record = OrderExecutionRecord(
                order_id=order.order_id,
                strategy_id=order.strategy_id,
                symbol=order.symbol,
                side=order.side,
            )
            self._records[order.order_id] = record
        return self._records[order.order_id]

    def finalize_record(self, order_id: str) -> None:
        """Move record to history when order lifecycle is complete."""
        if order_id in self._records:
            record = self._records.pop(order_id)
            self._history.append(record)

    def get_maker_fill_rate(self) -> float:
        """Get maker fill rate as percentage."""
        if self._maker_attempted == 0:
            return 0.0
        return (self._maker_filled / self._maker_attempted) * 100.0

    def get_avg_time_to_fill_ms(self) -> float:
        """Get average time to fill in milliseconds."""
        if self._fill_count == 0:
            return 0.0
        return self._total_fill_time_ms / self._fill_count

    def get_volatility_rejection_rate(self) -> float:
        """Get volatility rejection rate as percentage of attempts."""
        total_rejections = self._rejected_soft + self._rejected_hard
        total_attempts = self._maker_attempted + total_rejections
        if total_attempts == 0:
            return 0.0
        return (total_rejections / total_attempts) * 100.0

    def get_recent_expired_count(self, lookback_seconds: int = 60) -> int:
        """Get count of expired orders in recent time window."""
        cutoff = datetime.utcnow() - timedelta(seconds=lookback_seconds)
        count = 0
        for record in self._history:
            if record.maker_expired and record.expired_at and record.expired_at >= cutoff:
                count += 1
        # Also check active records
        for record in self._records.values():
            if record.maker_expired and record.expired_at and record.expired_at >= cutoff:
                count += 1
        return count

    def snapshot(self) -> dict[str, float]:
        """Get a snapshot of current execution metrics.

        Returns:
            Dictionary with execution metrics:
            - maker_fill_rate: Percentage of maker orders that filled
            - avg_time_to_fill_ms: Average time to fill in milliseconds
            - volatility_rejection_rate: Percentage of orders rejected by volatility
            - maker_attempted: Total maker orders attempted
            - maker_filled: Total maker orders filled
            - maker_expired: Total maker orders expired
            - volatility_rejections_soft: Soft volatility rejections
            - volatility_rejections_hard: Hard volatility rejections
        """
        return {
            "maker_fill_rate": self.get_maker_fill_rate(),
            "avg_time_to_fill_ms": self.get_avg_time_to_fill_ms(),
            "volatility_rejection_rate": self.get_volatility_rejection_rate(),
            "maker_attempted": float(self._maker_attempted),
            "maker_filled": float(self._maker_filled),
            "maker_expired": float(self._maker_expired),
            "volatility_rejections_soft": float(self._rejected_soft),
            "volatility_rejections_hard": float(self._rejected_hard),
        }

    def reset(self) -> None:
        """Reset all metrics (primarily for tests)."""
        self._records.clear()
        self._history.clear()
        self._maker_attempted = 0
        self._maker_filled = 0
        self._maker_expired = 0
        self._rejected_soft = 0
        self._rejected_hard = 0
        self._total_fill_time_ms = 0
        self._fill_count = 0

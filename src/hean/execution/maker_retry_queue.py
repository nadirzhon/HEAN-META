"""Maker retry queue - retries expired maker orders when volatility improves."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from hean.core.types import Order, OrderRequest
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class RetryEntry:
    """Entry in the retry queue."""

    original_order: Order
    original_request: OrderRequest
    retry_count: int
    first_attempt_at: datetime
    last_attempt_at: datetime
    reason: str  # Why it was queued (e.g., "volatility_expired")


class MakerRetryQueue:
    """Manages retry queue for expired maker orders."""

    def __init__(
        self,
        max_retries: int = 2,
        min_retry_delay_seconds: int = 5,
        max_retry_delay_seconds: int = 60,
    ) -> None:
        """Initialize the retry queue.

        Args:
            max_retries: Maximum number of retries per order
            min_retry_delay_seconds: Minimum delay before retry
            max_retry_delay_seconds: Maximum delay before retry
        """
        self._queue: list[RetryEntry] = []
        self._max_retries = max_retries
        self._min_retry_delay = timedelta(seconds=min_retry_delay_seconds)
        self._max_retry_delay = timedelta(seconds=max_retry_delay_seconds)
        self._successful_retries = 0
        self._failed_retries = 0

    def enqueue_for_retry(
        self,
        order: Order,
        original_request: OrderRequest,
        reason: str = "volatility_expired",
    ) -> bool:
        """Enqueue an order for retry.

        Args:
            order: The expired order
            original_request: Original order request
            reason: Reason for retry

        Returns:
            True if enqueued, False if max retries exceeded
        """
        # Check if already in queue
        existing = next(
            (e for e in self._queue if e.original_order.order_id == order.order_id),
            None,
        )
        if existing:
            if existing.retry_count >= self._max_retries:
                logger.debug(f"Order {order.order_id} already at max retries ({self._max_retries})")
                return False
            existing.retry_count += 1
            existing.last_attempt_at = datetime.utcnow()
            logger.debug(f"Updated retry count for order {order.order_id}: {existing.retry_count}")
            return True

        # Create new entry
        entry = RetryEntry(
            original_order=order,
            original_request=original_request,
            retry_count=1,
            first_attempt_at=datetime.utcnow(),
            last_attempt_at=datetime.utcnow(),
            reason=reason,
        )
        self._queue.append(entry)
        logger.info(f"Enqueued order {order.order_id} for retry (reason: {reason}, attempt 1)")
        return True

    def get_ready_retries(
        self,
        current_volatility: float,
        previous_volatility: float,
        regime_changed: bool = False,
        drawdown_worsened: bool = False,
        capital_preservation_active: bool = False,
    ) -> list[OrderRequest]:
        """Get orders ready for retry.

        Args:
            current_volatility: Current volatility value
            previous_volatility: Previous volatility value (when order expired)
            regime_changed: Whether market regime has changed
            drawdown_worsened: Whether drawdown has worsened
            capital_preservation_active: Whether capital preservation mode is active

        Returns:
            List of OrderRequests ready to retry
        """
        if capital_preservation_active:
            logger.debug("Capital preservation active, skipping retries")
            return []

        if regime_changed:
            logger.debug("Regime changed, clearing retry queue")
            self._queue.clear()
            return []

        if drawdown_worsened:
            logger.debug("Drawdown worsened, skipping retries")
            return []

        ready: list[OrderRequest] = []
        now = datetime.utcnow()
        to_remove: list[RetryEntry] = []

        for entry in self._queue:
            # Check if max retries exceeded (remove immediately, don't wait for ready)
            if entry.retry_count > self._max_retries:
                to_remove.append(entry)
                self._failed_retries += 1
                logger.debug(
                    f"Order {entry.original_order.order_id} exceeded max retries ({entry.retry_count} > {self._max_retries})"
                )
                continue

            # Check if enough time has passed
            time_since_last = now - entry.last_attempt_at
            if time_since_last < self._min_retry_delay:
                continue

            # Check if volatility has improved
            # Volatility improved if current is lower than previous
            volatility_improved = current_volatility < previous_volatility * 0.9  # 10% improvement

            if not volatility_improved:
                # Check if max delay exceeded (retry anyway if volatility hasn't improved much)
                if time_since_last >= self._max_retry_delay:
                    logger.debug(f"Retrying order {entry.original_order.order_id} after max delay")
                else:
                    continue

            # Ready for retry
            ready.append(entry.original_request)
            to_remove.append(entry)
            self._successful_retries += 1
            logger.info(
                f"Order {entry.original_order.order_id} ready for retry "
                f"(attempt {entry.retry_count}, volatility improved: {volatility_improved})"
            )

        # Remove processed entries
        for entry in to_remove:
            self._queue.remove(entry)

        return ready

    def remove_order(self, order_id: str) -> None:
        """Remove an order from the retry queue (e.g., if it was filled)."""
        self._queue = [e for e in self._queue if e.original_order.order_id != order_id]

    def get_retry_success_rate(self) -> float:
        """Get retry success rate as percentage."""
        total = self._successful_retries + self._failed_retries
        if total == 0:
            return 0.0
        return (self._successful_retries / total) * 100.0

    def get_queue_size(self) -> int:
        """Get current queue size."""
        return len(self._queue)

    def clear(self) -> None:
        """Clear the retry queue."""
        self._queue.clear()

    def reset_metrics(self) -> None:
        """Reset metrics (primarily for tests)."""
        self._successful_retries = 0
        self._failed_retries = 0

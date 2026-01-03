"""Profit target tracking - tracks progress to $100/day goal."""

from datetime import datetime

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)


class ProfitTargetTracker:
    """Tracks progress toward daily profit target.

    Monitors profit accumulation throughout the day and provides
    progress metrics toward the goal (default $100/day).
    """

    def __init__(self) -> None:
        """Initialize profit target tracker."""
        self._daily_target = settings.profit_target_daily_usd
        self._daily_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        self._daily_profit = 0.0
        self._daily_trades = 0
        logger.info(f"Profit target tracker initialized: ${self._daily_target:.2f}/day")

    def record_profit(self, profit: float) -> None:
        """Record profit from closed position.

        Args:
            profit: Realized profit from closed position (can be negative)
        """
        self._check_daily_reset()
        self._daily_profit += profit
        self._daily_trades += 1
        logger.debug(
            f"Profit recorded: ${profit:.2f}, daily total: ${self._daily_profit:.2f}, "
            f"trades: {self._daily_trades}"
        )

    def get_progress(self) -> dict[str, float]:
        """Get current progress toward daily target.

        Returns:
            Dictionary with progress metrics:
            - target: Daily profit target
            - current: Current profit for today
            - remaining: Remaining to reach target
            - progress_pct: Percentage of target achieved (0-100+)
            - trades: Number of trades today
        """
        self._check_daily_reset()
        progress_pct = (
            (self._daily_profit / self._daily_target) * 100 if self._daily_target > 0 else 0.0
        )
        remaining = max(0.0, self._daily_target - self._daily_profit)

        return {
            "target": self._daily_target,
            "current": self._daily_profit,
            "remaining": remaining,
            "progress_pct": progress_pct,
            "trades": float(self._daily_trades),
        }

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day has started."""
        now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        if now > self._daily_start:
            logger.info(
                f"Daily reset: Previous day profit=${self._daily_profit:.2f}, "
                f"trades={self._daily_trades}, target was ${self._daily_target:.2f}"
            )
            self._daily_start = now
            self._daily_profit = 0.0
            self._daily_trades = 0

    def should_increase_aggressiveness(self) -> bool:
        """Check if system should increase aggressiveness.

        Determines if current progress is behind schedule and needs
        more aggressive trading to reach daily target.

        Returns:
            True if should increase aggressiveness (behind schedule)
        """
        self._check_daily_reset()
        hours_passed = (datetime.utcnow() - self._daily_start).total_seconds() / 3600

        # Too early to make decisions (less than 1 hour)
        if hours_passed < 1:
            return False

        # Calculate expected progress based on time elapsed
        # If we're 12 hours into the day, we should have ~50% of target
        expected_progress = (hours_passed / 24) * self._daily_target

        # If less than 50% of expected progress, increase aggressiveness
        threshold = expected_progress * 0.5
        should_increase = self._daily_profit < threshold

        if should_increase:
            logger.warning(
                f"Behind schedule: ${self._daily_profit:.2f} / ${expected_progress:.2f} expected "
                f"after {hours_passed:.1f} hours. Consider increasing aggressiveness."
            )

        return should_increase

    def get_daily_start(self) -> datetime:
        """Get daily start timestamp."""
        return self._daily_start

    @property
    def daily_target(self) -> float:
        """Get daily profit target."""
        return self._daily_target

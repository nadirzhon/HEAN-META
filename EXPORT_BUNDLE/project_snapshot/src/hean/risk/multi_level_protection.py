"""Multi-level protection system with circuit breakers.

This module implements 5 levels of capital protection:
Level 1: Per-trade limit (1% risk per trade) - ALWAYS!
Level 2: Per-strategy limit (max $15-20 loss per strategy = 5-7% of $300)
Level 3: Hourly limit (max $30-45 loss per hour = 10-15% of $300)
Level 4: Daily limit (max $45-60 loss per day = 15-20% of $300 = killswitch)
Level 5: Consecutive losses (5 losses in a row = 1 hour pause)
"""

from collections import defaultdict
from datetime import datetime, timedelta

from hean.logging import get_logger

logger = get_logger(__name__)


class MultiLevelProtection:
    """Multi-level protection system with circuit breakers.

    Implements 5 levels of protection to safeguard capital:
    1. Per-trade limit (checked in PositionSizer)
    2. Per-strategy limit
    3. Hourly limit
    4. Daily limit (killswitch)
    5. Consecutive losses limit
    """

    def __init__(
        self,
        initial_capital: float,
        max_strategy_loss_pct: float = 7.0,
        max_hourly_loss_pct: float = 15.0,
        max_daily_drawdown_pct: float = 20.0,
        consecutive_losses_limit: int = 5,
        consecutive_losses_cooldown_hours: int = 1,
    ) -> None:
        """Initialize multi-level protection.

        Args:
            initial_capital: Initial capital for percentage calculations
            max_strategy_loss_pct: Maximum loss per strategy as % of initial capital (default 7%)
            max_hourly_loss_pct: Maximum loss per hour as % of initial capital (default 15%)
            max_daily_drawdown_pct: Maximum daily drawdown as % (default 20%)
            consecutive_losses_limit: Number of consecutive losses before pause (default 5)
            consecutive_losses_cooldown_hours: Hours to pause after consecutive losses (default 1)
        """
        self._initial_capital = initial_capital
        self._max_strategy_loss = initial_capital * (max_strategy_loss_pct / 100.0)
        self._max_hourly_loss = initial_capital * (max_hourly_loss_pct / 100.0)
        self._max_daily_drawdown_pct = max_daily_drawdown_pct
        self._consecutive_losses_limit = consecutive_losses_limit
        self._consecutive_losses_cooldown_hours = consecutive_losses_cooldown_hours

        # Per-strategy tracking
        self._strategy_losses: dict[str, float] = defaultdict(float)
        self._strategy_consecutive_losses: dict[str, int] = defaultdict(int)
        self._strategy_last_loss_time: dict[str, datetime | None] = defaultdict(lambda: None)
        self._strategy_cooldown_until: dict[str, datetime | None] = defaultdict(lambda: None)

        # Hourly tracking
        self._hourly_losses: dict[datetime, float] = {}
        self._current_hour_start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

        # Daily tracking
        self._daily_start_equity: float | None = None
        self._daily_start_date = datetime.utcnow().date()

    def _reset_daily_if_needed(self) -> None:
        """Reset daily tracking if new day."""
        now = datetime.utcnow()
        if now.date() > self._daily_start_date:
            self._daily_start_equity = None
            self._daily_start_date = now.date()
            # Reset hourly tracking
            self._hourly_losses.clear()
            self._current_hour_start = now.replace(minute=0, second=0, microsecond=0)
            logger.info("MultiLevelProtection: Daily reset")

    def _reset_hourly_if_needed(self) -> None:
        """Reset hourly tracking if new hour."""
        now = datetime.utcnow()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        if current_hour > self._current_hour_start:
            # New hour - clear old hourly data
            cutoff = current_hour - timedelta(hours=1)
            self._hourly_losses = {
                hour: loss for hour, loss in self._hourly_losses.items() if hour >= cutoff
            }
            self._current_hour_start = current_hour

    def _get_hourly_loss(self) -> float:
        """Get total loss for current hour."""
        self._reset_hourly_if_needed()
        now = datetime.utcnow()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        return self._hourly_losses.get(current_hour, 0.0)

    def record_loss(self, strategy_id: str, loss_amount: float) -> None:
        """Record a loss for tracking.

        Args:
            strategy_id: Strategy identifier
            loss_amount: Loss amount (positive value)
        """
        if loss_amount <= 0:
            return

        self._reset_daily_if_needed()
        self._reset_hourly_if_needed()

        # Per-strategy tracking
        self._strategy_losses[strategy_id] += loss_amount
        self._strategy_consecutive_losses[strategy_id] += 1
        self._strategy_last_loss_time[strategy_id] = datetime.utcnow()

        # Hourly tracking
        now = datetime.utcnow()
        current_hour = now.replace(minute=0, second=0, microsecond=0)
        self._hourly_losses[current_hour] = self._hourly_losses.get(current_hour, 0.0) + loss_amount

        logger.debug(
            f"MultiLevelProtection: Recorded loss ${loss_amount:.2f} for {strategy_id}, "
            f"strategy_total=${self._strategy_losses[strategy_id]:.2f}, "
            f"consecutive={self._strategy_consecutive_losses[strategy_id]}"
        )

    def record_win(self, strategy_id: str) -> None:
        """Record a win to reset consecutive losses.

        Args:
            strategy_id: Strategy identifier
        """
        self._strategy_consecutive_losses[strategy_id] = 0
        logger.debug(
            f"MultiLevelProtection: Win recorded for {strategy_id}, consecutive losses reset"
        )

    def check_all_protections(
        self, strategy_id: str, equity: float, initial_capital: float
    ) -> tuple[bool, str]:
        """Check all protection levels.

        Args:
            strategy_id: Strategy identifier
            equity: Current equity
            initial_capital: Initial capital

        Returns:
            (allowed, reason_if_blocked) tuple
        """
        self._reset_daily_if_needed()
        self._reset_hourly_if_needed()

        # Level 1: Per-trade limit (checked in PositionSizer, not here)

        # Level 2: Per-strategy limit
        strategy_loss = self._strategy_losses.get(strategy_id, 0.0)
        if strategy_loss >= self._max_strategy_loss:
            return False, (
                f"Strategy {strategy_id} reached loss limit ${self._max_strategy_loss:.2f} "
                f"(current loss: ${strategy_loss:.2f})"
            )

        # Level 3: Hourly limit
        hourly_loss = self._get_hourly_loss()
        if hourly_loss >= self._max_hourly_loss:
            return False, (
                f"Hourly loss limit ${self._max_hourly_loss:.2f} reached "
                f"(current hourly loss: ${hourly_loss:.2f})"
            )

        # Level 4: Daily limit (drawdown-based)
        if self._daily_start_equity is None:
            self._daily_start_equity = equity

        daily_drawdown = ((self._daily_start_equity - equity) / self._daily_start_equity) * 100
        if daily_drawdown > self._max_daily_drawdown_pct:
            return False, (
                f"Daily drawdown {daily_drawdown:.1f}% exceeds limit "
                f"{self._max_daily_drawdown_pct:.1f}%"
            )

        # Level 5: Consecutive losses
        consecutive = self._strategy_consecutive_losses.get(strategy_id, 0)
        if consecutive >= self._consecutive_losses_limit:
            # Check if cooldown period has passed
            cooldown_until = self._strategy_cooldown_until.get(strategy_id)
            if cooldown_until is None:
                # Start cooldown
                cooldown_until = datetime.utcnow() + timedelta(
                    hours=self._consecutive_losses_cooldown_hours
                )
                self._strategy_cooldown_until[strategy_id] = cooldown_until
                logger.warning(
                    f"MultiLevelProtection: {consecutive} consecutive losses for {strategy_id}, "
                    f"cooldown until {cooldown_until}"
                )

            if datetime.utcnow() < cooldown_until:
                return False, (
                    f"{consecutive} consecutive losses for {strategy_id}, "
                    f"cooldown until {cooldown_until}"
                )
            else:
                # Cooldown expired, reset consecutive losses
                self._strategy_consecutive_losses[strategy_id] = 0
                self._strategy_cooldown_until[strategy_id] = None
                logger.info(f"MultiLevelProtection: Cooldown expired for {strategy_id}")

        return True, ""

    def get_strategy_loss(self, strategy_id: str) -> float:
        """Get total loss for a strategy."""
        return self._strategy_losses.get(strategy_id, 0.0)

    def get_hourly_loss(self) -> float:
        """Get total loss for current hour."""
        return self._get_hourly_loss()

    def reset_strategy(self, strategy_id: str) -> None:
        """Reset tracking for a strategy (e.g., after manual intervention)."""
        self._strategy_losses.pop(strategy_id, None)
        self._strategy_consecutive_losses.pop(strategy_id, None)
        self._strategy_last_loss_time.pop(strategy_id, None)
        self._strategy_cooldown_until.pop(strategy_id, None)
        logger.info(f"MultiLevelProtection: Reset tracking for {strategy_id}")

    def reset_all(self) -> None:
        """Reset all tracking (for testing)."""
        self._strategy_losses.clear()
        self._strategy_consecutive_losses.clear()
        self._strategy_last_loss_time.clear()
        self._strategy_cooldown_until.clear()
        self._hourly_losses.clear()
        self._daily_start_equity = None
        self._daily_start_date = datetime.utcnow().date()
        self._current_hour_start = datetime.utcnow().replace(minute=0, second=0, microsecond=0)

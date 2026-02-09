"""Killswitch for emergency trading halt."""

from datetime import datetime, timedelta

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class KillSwitch:
    """Adaptive killswitch that triggers STOP_TRADING event on various conditions.

    Enhanced with:
    - Adaptive drawdown limits based on performance
    - Rolling profit factor monitoring
    - Consecutive loss tracking
    - Equity drop from initial capital monitoring
    """

    def __init__(self, bus: EventBus) -> None:
        """Initialize the adaptive killswitch."""
        self._bus = bus
        self._triggered = False
        self._trigger_reason = ""
        self._triggered_at: datetime | None = None
        self._reasons: list[str] = []  # List of all trigger reasons
        self._daily_high_equity: float | None = None
        self._daily_reset_time = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._error_count = 0
        self._error_window_start = datetime.utcnow()
        self._max_errors_per_hour = 10
        self._initial_capital: float | None = None
        self._rolling_pf_history: list[float] = []
        self._max_pf_history = 20  # Track last 20 PF calculations

        # Auto-reset safety controls
        self._auto_reset_enabled = False  # Disabled by default - require explicit enable
        self._min_cooldown_hours = 4  # Minimum hours before auto-reset allowed
        self._required_recovery_pct = 50  # Must recover 50% of drawdown before auto-reset
        self._equity_at_trigger: float | None = None  # Track equity when triggered
        self._auto_reset_count_today = 0  # Track auto-resets per day
        self._max_auto_resets_per_day = 2  # Maximum auto-resets per day

    def set_initial_capital(self, initial_capital: float) -> None:
        """Set initial capital for equity drop monitoring."""
        self._initial_capital = initial_capital

    def get_adaptive_drawdown_limit(self, current_equity: float) -> float:
        """Get adaptive drawdown limit based on current equity.

        Adaptive limits:
        - Capital < $500: 15% limit (stricter for small capital)
        - Capital $500-1000: 18% limit
        - Capital > $1000: 20% limit (can be more lenient)

        Args:
            current_equity: Current portfolio equity

        Returns:
            Adaptive drawdown limit percentage
        """
        # Use current equity or initial capital for determination
        reference_capital = (
            current_equity if self._initial_capital is None else self._initial_capital
        )

        if reference_capital < 500:
            return 15.0
        elif reference_capital < 1000:
            return 18.0
        else:
            return 20.0

    def update_rolling_pf(self, rolling_pf: float) -> None:
        """Update rolling profit factor history."""
        self._rolling_pf_history.append(rolling_pf)
        if len(self._rolling_pf_history) > self._max_pf_history:
            self._rolling_pf_history.pop(0)

    async def check_drawdown(
        self, equity: float, peak_equity: float, regime=None, rolling_pf: float | None = None
    ) -> bool:
        """Check if drawdown exceeds limit and trigger if needed.

        Enhanced with:
        - Adaptive limits based on rolling PF
        - Equity drop from initial capital check
        - Stricter limits when PF is poor
        """
        if self._triggered:
            # Check if auto-reset is possible
            if await self._check_auto_reset_conditions(equity):
                return False  # Killswitch was reset, continue trading
            return True

        # Reset daily high if new day (but NOT the triggered state - that requires explicit reset)
        now = datetime.utcnow()
        if now.date() > self._daily_reset_time.date():
            self._daily_high_equity = None
            self._daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self._auto_reset_count_today = 0  # Reset daily auto-reset counter
            logger.info("Killswitch daily stats reset (triggered state preserved)")

        # Update rolling PF history
        if rolling_pf is not None:
            self.update_rolling_pf(rolling_pf)

        # Check equity drop from initial capital (more critical)
        if self._initial_capital is not None and self._initial_capital > 0:
            equity_drop_from_initial = (
                (self._initial_capital - equity) / self._initial_capital
            ) * 100
            # Critical: If equity drops >= killswitch_drawdown_pct from initial, trigger immediately
            if equity_drop_from_initial >= settings.killswitch_drawdown_pct:
                await self._trigger(
                    f"CRITICAL: Equity drop {equity_drop_from_initial:.2f}% from initial capital "
                    f"(${self._initial_capital:.2f} -> ${equity:.2f})",
                    equity=equity,
                )
                return True

        # Update daily high (use peak_equity if provided, otherwise use equity)
        daily_high = peak_equity if peak_equity is not None else equity
        if self._daily_high_equity is None or daily_high > self._daily_high_equity:
            self._daily_high_equity = daily_high

        # Calculate drawdown from daily high
        if self._daily_high_equity is not None and self._daily_high_equity > 0:
            drawdown_pct = ((self._daily_high_equity - equity) / self._daily_high_equity) * 100

            # CRITICAL FIX: Don't block trading on drawdown from daily high if:
            # 1. Equity is still above initial capital (we're in profit)
            # 2. The drawdown is just normal market fluctuations after profit
            # This prevents "freezing" after reaching profit goals
            profit_above_initial = False
            if self._initial_capital is not None and self._initial_capital > 0:
                profit_pct = ((equity - self._initial_capital) / self._initial_capital) * 100
                profit_above_initial = equity > self._initial_capital * 1.05  # At least 5% above initial

                # If we have significant profit (>50%), allow larger drawdown from daily high
                if profit_pct > 50.0:
                    logger.debug(
                        f"Equity ${equity:.2f} is {profit_pct:.1f}% above initial ${self._initial_capital:.2f}, "
                        f"allowing drawdown from daily high ${self._daily_high_equity:.2f}"
                    )
                    # Skip daily high drawdown check if we're in significant profit
                    # Only check drawdown from initial capital (already checked above)
                    return self._triggered

            # Get adaptive drawdown limit based on capital size
            adaptive_limit = self.get_adaptive_drawdown_limit(equity)

            # Use stricter of adaptive limit or configured limit
            max_dd_limit = min(adaptive_limit, settings.max_daily_drawdown_pct)

            # Tighter limit if rolling PF is poor
            if rolling_pf is not None and rolling_pf < 0.8:
                max_dd_limit *= 0.7  # 30% tighter if PF < 0.8
                logger.debug(f"Tighter killswitch limit due to poor PF: {rolling_pf:.2f}")
            elif rolling_pf is not None and rolling_pf < 1.0:
                max_dd_limit *= 0.85  # 15% tighter if PF < 1.0

            # Tighter drawdown limit in IMPULSE regime
            if regime and hasattr(regime, "value") and regime.value == "impulse":
                max_dd_limit = max_dd_limit * 0.8  # 20% tighter

            # If we have profit above initial, be more lenient with daily high drawdown
            if profit_above_initial:
                # Allow 2x the normal drawdown limit when in profit
                max_dd_limit = max_dd_limit * 2.0
                logger.debug(
                    f"Relaxed drawdown limit to {max_dd_limit:.2f}% due to profit above initial capital"
                )

            if drawdown_pct >= max_dd_limit:
                pf_str = f"{rolling_pf:.2f}" if rolling_pf is not None else "N/A"
                await self._trigger(
                    f"Daily drawdown {drawdown_pct:.2f}% exceeds adaptive limit {max_dd_limit:.2f}% "
                    f"(PF={pf_str})",
                    equity=equity,
                )
                return True

        return self._triggered

    async def record_error(self) -> None:
        """Record an error and trigger if too many errors."""
        if self._triggered:
            return

        now = datetime.utcnow()
        # Reset error count if more than an hour has passed
        if (now - self._error_window_start) > timedelta(hours=1):
            self._error_count = 0
            self._error_window_start = now

        self._error_count += 1

        if self._error_count >= self._max_errors_per_hour:
            await self._trigger(f"Too many errors: {self._error_count} in the last hour")

    async def check_volatility(self, spread_pct: float) -> None:
        """Check volatility/spread and trigger if extreme."""
        if self._triggered:
            return

        # Trigger if spread exceeds 1% (extreme volatility)
        if spread_pct > 1.0:
            await self._trigger(f"Extreme spread: {spread_pct:.2f}%")

    async def _check_auto_reset_conditions(self, current_equity: float) -> bool:
        """Check if conditions are met for auto-reset.

        Auto-reset requires:
        1. auto_reset_enabled = True
        2. Minimum cooldown period has passed
        3. Equity has recovered sufficiently
        4. Not exceeded max auto-resets per day

        Returns:
            True if killswitch was reset, False otherwise
        """
        if not self._auto_reset_enabled:
            return False

        if self._triggered_at is None:
            return False

        now = datetime.utcnow()

        # Check cooldown period
        time_since_trigger = now - self._triggered_at
        if time_since_trigger < timedelta(hours=self._min_cooldown_hours):
            logger.debug(
                f"Auto-reset blocked: cooldown not met "
                f"({time_since_trigger.total_seconds() / 3600:.1f}h < {self._min_cooldown_hours}h)"
            )
            return False

        # Check max auto-resets per day
        if self._auto_reset_count_today >= self._max_auto_resets_per_day:
            logger.warning(
                f"Auto-reset blocked: max daily auto-resets reached "
                f"({self._auto_reset_count_today}/{self._max_auto_resets_per_day})"
            )
            return False

        # Check equity recovery
        if self._equity_at_trigger is not None and self._initial_capital is not None:
            loss_at_trigger = self._initial_capital - self._equity_at_trigger
            current_loss = self._initial_capital - current_equity

            if loss_at_trigger > 0:
                recovery_pct = ((loss_at_trigger - current_loss) / loss_at_trigger) * 100

                if recovery_pct < self._required_recovery_pct:
                    logger.debug(
                        f"Auto-reset blocked: insufficient recovery "
                        f"({recovery_pct:.1f}% < {self._required_recovery_pct}%)"
                    )
                    return False

        # All conditions met - perform auto-reset
        logger.warning(
            f"KILLSWITCH AUTO-RESET: Conditions met after {time_since_trigger.total_seconds() / 3600:.1f}h. "
            f"Previous reason: {self._trigger_reason}"
        )

        self._triggered = False
        self._trigger_reason = ""
        self._triggered_at = None
        self._reasons = []
        self._equity_at_trigger = None
        self._auto_reset_count_today += 1

        # Publish reset event
        await self._bus.publish(
            Event(
                event_type=EventType.KILLSWITCH_RESET,
                data={
                    "reset_type": "auto",
                    "cooldown_hours": time_since_trigger.total_seconds() / 3600,
                    "auto_reset_count_today": self._auto_reset_count_today,
                },
            )
        )

        return True

    async def _trigger(self, reason: str, equity: float | None = None) -> None:
        """Trigger the killswitch."""
        if self._triggered:
            # Add additional reason if already triggered
            if reason not in self._reasons:
                self._reasons.append(reason)
            return

        self._triggered = True
        self._trigger_reason = reason
        self._triggered_at = datetime.utcnow()
        self._reasons = [reason]  # Initialize with first reason

        # Store equity at trigger for recovery calculation
        if equity is not None:
            self._equity_at_trigger = equity

        logger.critical(f"KILLSWITCH TRIGGERED: {reason}")

        await self._bus.publish(
            Event(
                event_type=EventType.KILLSWITCH_TRIGGERED,
                data={"reason": reason, "triggered_at": self._triggered_at.isoformat()},
            )
        )

        await self._bus.publish(
            Event(
                event_type=EventType.STOP_TRADING,
                data={"reason": reason},
            )
        )

    def is_triggered(self) -> bool:
        """Check if killswitch is triggered."""
        return self._triggered

    def get_reason(self) -> str:
        """Get the reason for killswitch trigger."""
        return self._trigger_reason

    def reset(self, force: bool = False) -> bool:
        """Reset the killswitch manually.

        Args:
            force: If True, reset regardless of conditions (requires explicit user action)

        Returns:
            True if reset was successful, False otherwise
        """
        if not self._triggered:
            logger.info("Killswitch reset called but not triggered")
            return True

        if not force:
            # Require minimum cooldown even for manual reset
            if self._triggered_at is not None:
                time_since = datetime.utcnow() - self._triggered_at
                min_manual_cooldown = timedelta(minutes=30)  # 30 min minimum for manual reset
                if time_since < min_manual_cooldown:
                    logger.warning(
                        f"Manual killswitch reset blocked: minimum cooldown not met "
                        f"({time_since.total_seconds() / 60:.1f}m < 30m)"
                    )
                    return False

        logger.warning(
            f"KILLSWITCH MANUAL RESET (force={force}). Previous reason: {self._trigger_reason}"
        )

        self._triggered = False
        self._trigger_reason = ""
        self._triggered_at = None
        self._reasons = []
        self._error_count = 0
        self._equity_at_trigger = None
        return True

    def enable_auto_reset(self, enable: bool = True) -> None:
        """Enable or disable automatic reset after conditions are met.

        WARNING: Auto-reset should only be enabled after careful consideration.
        It allows the system to resume trading automatically after a killswitch event.

        Args:
            enable: True to enable auto-reset, False to disable
        """
        self._auto_reset_enabled = enable
        logger.info(f"Killswitch auto-reset {'ENABLED' if enable else 'DISABLED'}")

    def configure_auto_reset(
        self,
        min_cooldown_hours: int = 4,
        required_recovery_pct: int = 50,
        max_auto_resets_per_day: int = 2,
    ) -> None:
        """Configure auto-reset parameters.

        Args:
            min_cooldown_hours: Minimum hours before auto-reset can occur
            required_recovery_pct: Required equity recovery percentage (0-100)
            max_auto_resets_per_day: Maximum auto-resets allowed per day
        """
        self._min_cooldown_hours = max(1, min_cooldown_hours)
        self._required_recovery_pct = max(0, min(100, required_recovery_pct))
        self._max_auto_resets_per_day = max(1, max_auto_resets_per_day)
        logger.info(
            f"Killswitch auto-reset configured: cooldown={self._min_cooldown_hours}h, "
            f"recovery={self._required_recovery_pct}%, max_resets={self._max_auto_resets_per_day}/day"
        )

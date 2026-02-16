"""Killswitch for emergency trading halt.

Acts as a graduated safety net, not a hard kill:
- Soft brake: reduce position sizes (via risk state escalation)
- Hard stop: only on catastrophic drawdown from initial capital
- Daily drawdown: pauses trading but auto-resets after cooldown
- Never penalizes early-stage trading when PF is naturally low
"""

from datetime import datetime, timedelta

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class KillSwitch:
    """Adaptive killswitch that triggers STOP_TRADING event on various conditions.

    Design philosophy: Safety net, not project killer.
    - Only hard-kill on catastrophic loss (>= killswitch_drawdown_pct from initial)
    - Daily drawdown triggers a pause with fast auto-reset (15min cooldown)
    - PF-based throttling is informational only (logged, not enforced)
    - Auto-reset is enabled by default for testnet
    """

    def __init__(self, bus: EventBus) -> None:
        """Initialize the adaptive killswitch."""
        self._bus = bus
        self._triggered = False
        self._trigger_reason = ""
        self._trigger_type = ""  # "HARD_STOP" or "DAILY_PAUSE"
        self._triggered_at: datetime | None = None
        self._reasons: list[str] = []
        self._daily_high_equity: float | None = None
        self._daily_reset_time = datetime.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._error_count = 0
        self._error_window_start = datetime.utcnow()
        self._max_errors_per_hour = 10
        self._initial_capital: float | None = None
        self._rolling_pf_history: list[float] = []
        self._max_pf_history = 20

        # Auto-reset safety controls — lenient defaults for testnet
        self._auto_reset_enabled = True  # Enabled by default
        self._min_cooldown_minutes = 5  # 5 minutes cooldown
        self._required_recovery_pct = 10  # Must recover 10% of drawdown (was 25%)
        self._equity_at_trigger: float | None = None
        self._auto_reset_count_today = 0
        self._max_auto_resets_per_day = 10  # Generous for testnet (was 5)
        self._total_triggers = 0  # Lifetime trigger count for observability

    def set_initial_capital(self, initial_capital: float) -> None:
        """Set initial capital for equity drop monitoring."""
        self._initial_capital = initial_capital

    def get_adaptive_drawdown_limit(self, current_equity: float) -> float:
        """Get adaptive drawdown limit based on current equity.

        Returns the configured max_daily_drawdown_pct without capital-based
        reduction. The config value is the single source of truth.
        """
        return settings.max_daily_drawdown_pct

    def update_rolling_pf(self, rolling_pf: float) -> None:
        """Update rolling profit factor history (informational only)."""
        self._rolling_pf_history.append(rolling_pf)
        if len(self._rolling_pf_history) > self._max_pf_history:
            self._rolling_pf_history.pop(0)

    def _get_avg_rolling_pf(self) -> float | None:
        """Get average rolling PF from history."""
        if not self._rolling_pf_history:
            return None
        return sum(self._rolling_pf_history) / len(self._rolling_pf_history)

    async def check_drawdown(
        self, equity: float, peak_equity: float, regime=None, rolling_pf: float | None = None
    ) -> bool:
        """Check if drawdown exceeds limit and trigger if needed.

        Two-tier system:
        1. HARD_STOP: equity drops >= killswitch_drawdown_pct from initial capital
           → Requires manual reset or new day
        2. DAILY_PAUSE: daily drawdown from peak >= max_daily_drawdown_pct
           → Auto-resets after 15-min cooldown
        """
        if self._triggered:
            # Check if auto-reset is possible
            if await self._check_auto_reset_conditions(equity):
                return False  # Killswitch was reset, continue trading
            return True

        # Reset daily high if new day
        now = datetime.utcnow()
        if now.date() > self._daily_reset_time.date():
            self._daily_high_equity = None
            self._daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self._auto_reset_count_today = 0
            logger.info("Killswitch daily stats reset (triggered state preserved)")

        # Update rolling PF history (informational — NOT used for penalties)
        if rolling_pf is not None:
            self.update_rolling_pf(rolling_pf)

        # ────────────────────────────────────────────────────
        # TIER 1: HARD STOP — catastrophic loss from initial capital
        # ────────────────────────────────────────────────────
        if self._initial_capital is not None and self._initial_capital > 0:
            equity_drop_from_initial = (
                (self._initial_capital - equity) / self._initial_capital
            ) * 100

            if equity_drop_from_initial >= settings.killswitch_drawdown_pct:
                await self._trigger(
                    f"HARD_STOP: Equity drop {equity_drop_from_initial:.2f}% from initial capital "
                    f"(${self._initial_capital:.2f} -> ${equity:.2f})",
                    equity=equity,
                    trigger_type="HARD_STOP",
                )
                return True

        # ────────────────────────────────────────────────────
        # TIER 2: DAILY PAUSE — drawdown from daily high
        # ────────────────────────────────────────────────────
        daily_high = peak_equity if peak_equity is not None else equity
        if self._daily_high_equity is None or daily_high > self._daily_high_equity:
            self._daily_high_equity = daily_high

        if self._daily_high_equity is not None and self._daily_high_equity > 0:
            drawdown_pct = ((self._daily_high_equity - equity) / self._daily_high_equity) * 100

            # If we're still above initial capital, be lenient with daily drawdown
            if self._initial_capital is not None and self._initial_capital > 0:
                profit_pct = ((equity - self._initial_capital) / self._initial_capital) * 100

                # Skip daily drawdown check entirely if we have >20% profit above initial
                if profit_pct > 20.0:
                    logger.debug(
                        f"Equity ${equity:.2f} is {profit_pct:.1f}% above initial, "
                        f"skipping daily drawdown check"
                    )
                    return False

                # If we're above initial but within 20% profit, use relaxed limit (2x)
                if profit_pct > 0:
                    max_dd_limit = settings.max_daily_drawdown_pct * 2.0
                else:
                    max_dd_limit = settings.max_daily_drawdown_pct
            else:
                max_dd_limit = settings.max_daily_drawdown_pct

            # Log PF for observability but do NOT tighten limits based on it
            if rolling_pf is not None and rolling_pf < 1.0:
                logger.debug(
                    f"Rolling PF is low ({rolling_pf:.2f}) — monitoring but not penalizing"
                )

            if drawdown_pct >= max_dd_limit:
                pf_str = f"{rolling_pf:.2f}" if rolling_pf is not None else "N/A"
                await self._trigger(
                    f"DAILY_PAUSE: Drawdown {drawdown_pct:.2f}% from daily high "
                    f"${self._daily_high_equity:.2f} exceeds limit {max_dd_limit:.2f}% "
                    f"(PF={pf_str})",
                    equity=equity,
                    trigger_type="DAILY_PAUSE",
                )
                return True

        return False

    async def record_error(self) -> None:
        """Record an error and trigger if too many errors."""
        if self._triggered:
            return

        now = datetime.utcnow()
        if (now - self._error_window_start) > timedelta(hours=1):
            self._error_count = 0
            self._error_window_start = now

        self._error_count += 1

        if self._error_count >= self._max_errors_per_hour:
            await self._trigger(
                f"ERROR_PAUSE: {self._error_count} errors in the last hour",
                trigger_type="DAILY_PAUSE",  # Errors are pausable, not hard stops
            )

    async def check_volatility(self, spread_pct: float) -> None:
        """Check volatility/spread and trigger if extreme."""
        if self._triggered:
            return

        # Only trigger on truly extreme spreads (>2% instead of 1%)
        if spread_pct > 2.0:
            await self._trigger(
                f"VOLATILITY_PAUSE: Extreme spread {spread_pct:.2f}%",
                trigger_type="DAILY_PAUSE",
            )

    async def _check_auto_reset_conditions(self, current_equity: float) -> bool:
        """Check if conditions are met for auto-reset.

        HARD_STOP requires manual reset or new day.
        DAILY_PAUSE auto-resets after short cooldown with minimal recovery.
        """
        if not self._auto_reset_enabled:
            return False

        if self._triggered_at is None:
            return False

        # HARD_STOP does NOT auto-reset (requires manual intervention)
        if self._trigger_type == "HARD_STOP":
            now = datetime.utcnow()
            # Exception: reset on new day
            if now.date() > self._triggered_at.date():
                logger.warning(
                    "KILLSWITCH HARD_STOP auto-reset: new trading day started. "
                    f"Previous reason: {self._trigger_reason}"
                )
                await self._do_reset("new_day")
                return True
            return False

        now = datetime.utcnow()

        # Check cooldown period for DAILY_PAUSE
        time_since_trigger = now - self._triggered_at
        cooldown = timedelta(minutes=self._min_cooldown_minutes)
        if time_since_trigger < cooldown:
            remaining = (cooldown - time_since_trigger).total_seconds()
            logger.debug(
                f"Auto-reset blocked: {remaining:.0f}s remaining in cooldown"
            )
            return False

        # Check max auto-resets per day
        if self._auto_reset_count_today >= self._max_auto_resets_per_day:
            logger.warning(
                f"Auto-reset blocked: max daily auto-resets reached "
                f"({self._auto_reset_count_today}/{self._max_auto_resets_per_day})"
            )
            return False

        # Check equity recovery (lenient: only 10% recovery needed)
        if self._equity_at_trigger is not None and self._initial_capital is not None:
            loss_at_trigger = self._initial_capital - self._equity_at_trigger
            current_loss = self._initial_capital - current_equity

            if loss_at_trigger > 0:
                recovery_pct = ((loss_at_trigger - current_loss) / loss_at_trigger) * 100
                if recovery_pct < self._required_recovery_pct:
                    logger.debug(
                        f"Auto-reset blocked: recovery {recovery_pct:.1f}% "
                        f"< required {self._required_recovery_pct}%"
                    )
                    return False

        # All conditions met — auto-reset
        elapsed_min = time_since_trigger.total_seconds() / 60
        logger.info(
            f"KILLSWITCH AUTO-RESET after {elapsed_min:.0f}min pause. "
            f"Reason was: {self._trigger_reason}"
        )
        await self._do_reset("auto")
        return True

    async def _do_reset(self, reset_type: str) -> None:
        """Perform the actual reset and notify system via KILLSWITCH_RESET event."""
        prev_reason = self._trigger_reason
        prev_type = self._trigger_type
        self._triggered = False
        self._trigger_reason = ""
        self._trigger_type = ""
        self._triggered_at = None
        self._reasons = []
        self._equity_at_trigger = None
        self._auto_reset_count_today += 1

        # Notify system that killswitch has been reset — trading can resume
        await self._bus.publish(
            Event(
                event_type=EventType.KILLSWITCH_RESET,
                data={
                    "reset_type": reset_type,
                    "previous_reason": prev_reason,
                    "previous_trigger_type": prev_type,
                    "auto_resets_today": self._auto_reset_count_today,
                },
            )
        )
        logger.info(
            f"KILLSWITCH RESET ({reset_type}): trading can resume. "
            f"Previous: {prev_type} — {prev_reason}"
        )

    async def _trigger(
        self, reason: str, equity: float | None = None, trigger_type: str = "DAILY_PAUSE"
    ) -> None:
        """Trigger the killswitch."""
        if self._triggered:
            if reason not in self._reasons:
                self._reasons.append(reason)
            return

        self._triggered = True
        self._trigger_reason = reason
        self._trigger_type = trigger_type
        self._triggered_at = datetime.utcnow()
        self._reasons = [reason]
        self._total_triggers += 1

        if equity is not None:
            self._equity_at_trigger = equity

        if trigger_type == "HARD_STOP":
            logger.critical(f"KILLSWITCH HARD_STOP: {reason}")
        else:
            logger.warning(f"KILLSWITCH DAILY_PAUSE: {reason}")

        await self._bus.publish(
            Event(
                event_type=EventType.KILLSWITCH_TRIGGERED,
                data={
                    "reason": reason,
                    "trigger_type": trigger_type,
                    "triggered_at": self._triggered_at.isoformat(),
                    "total_triggers": self._total_triggers,
                },
            )
        )

        await self._bus.publish(
            Event(
                event_type=EventType.STOP_TRADING,
                data={"reason": reason, "trigger_type": trigger_type},
            )
        )

    def is_triggered(self) -> bool:
        """Check if killswitch is triggered."""
        return self._triggered

    def get_reason(self) -> str:
        """Get the reason for killswitch trigger."""
        return self._trigger_reason

    def get_state(self) -> dict:
        """Get full killswitch state for observability."""
        return {
            "triggered": self._triggered,
            "trigger_type": self._trigger_type,
            "trigger_reason": self._trigger_reason,
            "triggered_at": self._triggered_at.isoformat() if self._triggered_at else None,
            "initial_capital": self._initial_capital,
            "daily_high_equity": self._daily_high_equity,
            "auto_reset_enabled": self._auto_reset_enabled,
            "auto_resets_today": self._auto_reset_count_today,
            "max_auto_resets_per_day": self._max_auto_resets_per_day,
            "cooldown_minutes": self._min_cooldown_minutes,
            "total_lifetime_triggers": self._total_triggers,
            "avg_rolling_pf": self._get_avg_rolling_pf(),
        }

    async def reset(self, force: bool = False) -> bool:
        """Reset the killswitch manually.

        Args:
            force: If True, reset regardless of trigger type

        Returns:
            True if reset was successful
        """
        if not self._triggered:
            logger.info("Killswitch reset called but not triggered")
            return True

        logger.warning(
            f"KILLSWITCH MANUAL RESET (force={force}). "
            f"Type: {self._trigger_type}. Reason: {self._trigger_reason}"
        )

        await self._do_reset("manual")
        self._error_count = 0
        return True

    def enable_auto_reset(self, enable: bool = True) -> None:
        """Enable or disable automatic reset after conditions are met."""
        self._auto_reset_enabled = enable
        logger.info(f"Killswitch auto-reset {'ENABLED' if enable else 'DISABLED'}")

    def configure_auto_reset(
        self,
        min_cooldown_hours: int = 1,
        required_recovery_pct: int = 25,
        max_auto_resets_per_day: int = 5,
    ) -> None:
        """Configure auto-reset parameters.

        Note: min_cooldown_hours is converted to minutes internally.
        For backward compatibility, accepts hours but stores as minutes.
        """
        # Convert hours to minutes but enforce minimum of 10 minutes
        self._min_cooldown_minutes = max(10, min_cooldown_hours * 60)
        self._required_recovery_pct = max(0, min(100, required_recovery_pct))
        self._max_auto_resets_per_day = max(1, max_auto_resets_per_day)
        logger.info(
            f"Killswitch auto-reset configured: cooldown={self._min_cooldown_minutes}min, "
            f"recovery={self._required_recovery_pct}%, max_resets={self._max_auto_resets_per_day}/day"
        )

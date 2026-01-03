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
            return True

        # Reset daily high if new day
        now = datetime.utcnow()
        if now.date() > self._daily_reset_time.date():
            self._daily_high_equity = None
            self._daily_reset_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
            self._triggered = False  # Reset on new day
            logger.info("Killswitch daily reset")

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
                    f"(${self._initial_capital:.2f} -> ${equity:.2f})"
                )
                return True

        # Update daily high (use peak_equity if provided, otherwise use equity)
        daily_high = peak_equity if peak_equity is not None else equity
        if self._daily_high_equity is None or daily_high > self._daily_high_equity:
            self._daily_high_equity = daily_high

        # Calculate drawdown from daily high
        if self._daily_high_equity is not None and self._daily_high_equity > 0:
            drawdown_pct = ((self._daily_high_equity - equity) / self._daily_high_equity) * 100

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

            if drawdown_pct >= max_dd_limit:
                await self._trigger(
                    f"Daily drawdown {drawdown_pct:.2f}% exceeds adaptive limit {max_dd_limit:.2f}% "
                    f"(PF={rolling_pf:.2f if rolling_pf else 'N/A'})"
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

    async def _trigger(self, reason: str) -> None:
        """Trigger the killswitch."""
        if self._triggered:
            return

        self._triggered = True
        self._trigger_reason = reason

        logger.critical(f"KILLSWITCH TRIGGERED: {reason}")

        await self._bus.publish(
            Event(
                event_type=EventType.KILLSWITCH_TRIGGERED,
                data={"reason": reason},
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

    def reset(self) -> None:
        """Reset the killswitch (only on new day)."""
        self._triggered = False
        self._trigger_reason = ""
        self._error_count = 0

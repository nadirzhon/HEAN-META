"""Dynamic risk scaling based on performance and market conditions."""

from collections import deque
from datetime import datetime

from hean.core.regime import Regime
from hean.logging import get_logger

logger = get_logger(__name__)


class DynamicRiskManager:
    """Manages dynamic risk scaling based on performance metrics.

    Adjusts position sizing risk multiplier (0.5x - 1.5x) based on:
    - Current market regime
    - Rolling profit factor
    - Recent drawdown
    - Volatility percentile

    Safeguards:
    - Never increases risk after drawdown spike
    - Caps max risk in IMPULSE regime
    """

    def __init__(self) -> None:
        """Initialize the dynamic risk manager."""
        # Risk multiplier bounds
        self._min_multiplier = 0.5
        self._max_multiplier = 1.5
        self._max_impulse_multiplier = 1.2  # Cap for IMPULSE regime

        # Drawdown spike detection
        self._recent_drawdowns: deque[float] = deque(maxlen=20)  # Track last 20 drawdowns
        self._drawdown_spike_threshold = 2.0  # % change in drawdown to trigger spike
        self._last_drawdown: float = 0.0
        self._drawdown_spike_detected: bool = False
        self._spike_cooldown_hours = 24  # Hours before allowing risk increase after spike

        # Volatility tracking for percentile calculation
        self._volatility_history: deque[float] = deque(maxlen=100)  # Track last 100 volatilities
        self._volatility_window = 100

        # Metrics tracking
        self._risk_multiplier_history: deque[float] = deque(maxlen=1000)
        self._risk_reductions_triggered: int = 0
        self._last_spike_time: datetime | None = None

    def calculate_risk_multiplier(
        self,
        current_regime: Regime,
        rolling_pf: float,
        recent_drawdown: float,
        volatility_percentile: float,
    ) -> float:
        """Calculate dynamic risk multiplier.

        Args:
            current_regime: Current market regime (RANGE, NORMAL, IMPULSE)
            rolling_pf: Rolling profit factor (wins/losses)
            recent_drawdown: Recent drawdown percentage
            volatility_percentile: Current volatility percentile (0-100)

        Returns:
            Risk multiplier between 0.5x and 1.5x (capped at 1.2x for IMPULSE)
        """
        # Update drawdown tracking
        self._update_drawdown_tracking(recent_drawdown)

        # Base multiplier starts at 1.0
        multiplier = 1.0

        # Factor 1: Profit Factor adjustment
        # PF > 1.5: increase risk (+0.2)
        # PF > 1.2: slight increase (+0.1)
        # PF < 0.8: decrease risk (-0.2)
        # PF < 1.0: slight decrease (-0.1)
        if rolling_pf > 1.5:
            multiplier += 0.2
        elif rolling_pf > 1.2:
            multiplier += 0.1
        elif rolling_pf < 0.8:
            multiplier -= 0.2
        elif rolling_pf < 1.0:
            multiplier -= 0.1

        # Factor 2: Drawdown adjustment
        # High drawdown (>5%): reduce risk
        # Moderate drawdown (2-5%): slight reduction
        if recent_drawdown > 5.0:
            multiplier -= 0.3
        elif recent_drawdown > 2.0:
            multiplier -= 0.15

        # Factor 3: Volatility percentile adjustment
        # High volatility (>80th percentile): reduce risk
        # Low volatility (<20th percentile): slight increase
        if volatility_percentile > 80.0:
            multiplier -= 0.2
        elif volatility_percentile < 20.0:
            multiplier += 0.1

        # Factor 4: Regime adjustment
        # RANGE: slight reduction (already conservative)
        # IMPULSE: slight increase but capped
        if current_regime == Regime.RANGE:
            multiplier -= 0.1
        elif current_regime == Regime.IMPULSE:
            multiplier += 0.1

        # Apply bounds - CRITICAL: Respect configured minimum
        multiplier = max(self._min_multiplier, min(self._max_multiplier, multiplier))

        # Safeguard 1: Never increase risk after drawdown spike
        if self._drawdown_spike_detected:
            # Check if cooldown period has passed
            if self._last_spike_time:
                hours_since_spike = (
                    datetime.utcnow() - self._last_spike_time
                ).total_seconds() / 3600
                if hours_since_spike < self._spike_cooldown_hours:
                    # Force reduction if trying to increase
                    if multiplier > 1.0:
                        multiplier = 1.0
                        self._risk_reductions_triggered += 1
                        logger.warning(
                            f"Risk increase blocked due to recent drawdown spike "
                            f"({hours_since_spike:.1f}h ago)"
                        )
                else:
                    # Cooldown passed, reset spike flag
                    self._drawdown_spike_detected = False
                    self._last_spike_time = None
                    logger.info("Drawdown spike cooldown expired, risk scaling enabled")
            else:
                # Spike detected but no timestamp (shouldn't happen)
                if multiplier > 1.0:
                    multiplier = 1.0

        # Safeguard 2: Cap max risk in IMPULSE regime
        if current_regime == Regime.IMPULSE:
            multiplier = min(multiplier, self._max_impulse_multiplier)

        # Track metrics
        self._risk_multiplier_history.append(multiplier)

        logger.debug(
            f"Dynamic risk: regime={current_regime.value}, PF={rolling_pf:.2f}, "
            f"DD={recent_drawdown:.2f}%, vol_pct={volatility_percentile:.1f}, "
            f"multiplier={multiplier:.2f}x"
        )

        return multiplier

    def _update_drawdown_tracking(self, current_drawdown: float) -> None:
        """Update drawdown tracking and detect spikes."""
        # Detect drawdown spike (sudden increase in drawdown)
        if self._last_drawdown > 0:
            drawdown_change = current_drawdown - self._last_drawdown
            if drawdown_change > self._drawdown_spike_threshold:
                # Spike detected
                self._drawdown_spike_detected = True
                self._last_spike_time = datetime.utcnow()
                self._risk_reductions_triggered += 1
                logger.warning(
                    f"Drawdown spike detected: {self._last_drawdown:.2f}% -> "
                    f"{current_drawdown:.2f}% (change: {drawdown_change:.2f}%)"
                )

        self._recent_drawdowns.append(current_drawdown)
        self._last_drawdown = current_drawdown

    def update_volatility(self, volatility: float) -> None:
        """Update volatility history for percentile calculation."""
        self._volatility_history.append(volatility)

    def calculate_volatility_percentile(self, current_volatility: float) -> float:
        """Calculate percentile of current volatility in historical distribution.

        Args:
            current_volatility: Current volatility value

        Returns:
            Percentile (0-100) where current volatility ranks in history
        """
        if len(self._volatility_history) < 10:
            # Not enough history, return neutral (50th percentile)
            return 50.0

        # Count how many historical values are below current
        below_count = sum(1 for v in self._volatility_history if v < current_volatility)
        percentile = (below_count / len(self._volatility_history)) * 100.0

        return percentile

    def get_metrics(self) -> dict[str, float]:
        """Get risk scaling metrics.

        Returns:
            Dictionary with avg_risk_multiplier and risk_reductions_triggered
        """
        avg_multiplier = 1.0
        if self._risk_multiplier_history:
            avg_multiplier = sum(self._risk_multiplier_history) / len(self._risk_multiplier_history)

        return {
            "avg_risk_multiplier": avg_multiplier,
            "risk_reductions_triggered": float(self._risk_reductions_triggered),
            "current_multiplier": (
                self._risk_multiplier_history[-1] if self._risk_multiplier_history else 1.0
            ),
            "drawdown_spike_active": 1.0 if self._drawdown_spike_detected else 0.0,
        }

    def reset(self) -> None:
        """Reset risk manager state (for testing)."""
        self._recent_drawdowns.clear()
        self._volatility_history.clear()
        self._risk_multiplier_history.clear()
        self._drawdown_spike_detected = False
        self._last_drawdown = 0.0
        self._last_spike_time = None
        self._risk_reductions_triggered = 0


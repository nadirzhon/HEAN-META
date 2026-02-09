"""Signal Decay Model.

Models how signal confidence degrades over time. Signals that aren't
executed quickly lose their edge due to changing market conditions.

Integrates with OrderTimingOptimizer to penalize delayed execution.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


class DecayCurve(str, Enum):
    """Types of decay curves."""

    LINEAR = "linear"              # Constant decay rate
    EXPONENTIAL = "exponential"    # Accelerating decay
    LOGARITHMIC = "logarithmic"    # Decelerating decay
    STEP = "step"                  # Step function (sudden drops)


@dataclass
class DecayParameters:
    """Parameters for signal decay calculation."""

    curve_type: DecayCurve = DecayCurve.EXPONENTIAL
    half_life_seconds: float = 300.0  # Time for confidence to halve (5 min default)
    min_confidence: float = 0.2       # Minimum confidence floor
    volatility_multiplier: float = 1.0  # Faster decay in high volatility
    regime_multiplier: float = 1.0     # Faster decay in impulse regime


@dataclass
class SignalDecayState:
    """Tracks decay state for a specific signal."""

    signal_id: str
    initial_confidence: float
    current_confidence: float
    creation_time: datetime
    last_update: datetime
    decay_params: DecayParameters
    age_seconds: float = 0.0


class SignalDecayModel:
    """Models signal confidence decay over time.

    Different signal types decay at different rates:
    - Momentum signals decay quickly (market moves fast)
    - Mean reversion signals decay slower (ranges persist)
    - Breakout signals decay medium (confirmation window)
    """

    # Default decay parameters by signal type
    DEFAULT_DECAY_PARAMS = {
        "momentum": DecayParameters(
            curve_type=DecayCurve.EXPONENTIAL,
            half_life_seconds=180.0,  # 3 minutes
            min_confidence=0.15,
        ),
        "mean_reversion": DecayParameters(
            curve_type=DecayCurve.LOGARITHMIC,
            half_life_seconds=600.0,  # 10 minutes
            min_confidence=0.25,
        ),
        "breakout": DecayParameters(
            curve_type=DecayCurve.EXPONENTIAL,
            half_life_seconds=300.0,  # 5 minutes
            min_confidence=0.20,
        ),
        "arbitrage": DecayParameters(
            curve_type=DecayCurve.EXPONENTIAL,
            half_life_seconds=30.0,   # 30 seconds (arb decays FAST)
            min_confidence=0.10,
        ),
        "default": DecayParameters(
            curve_type=DecayCurve.EXPONENTIAL,
            half_life_seconds=300.0,  # 5 minutes
            min_confidence=0.20,
        ),
    }

    def __init__(self):
        """Initialize signal decay model."""
        # Track active signals
        self._active_signals: dict[str, SignalDecayState] = {}

        # Decay statistics
        self._total_decayed = 0
        self._total_expired = 0  # Signals that decayed to min confidence

        logger.info("SignalDecayModel initialized")

    def register_signal(
        self,
        signal_id: str,
        initial_confidence: float,
        signal_type: str = "default",
        custom_params: DecayParameters | None = None,
    ) -> None:
        """Register a new signal for decay tracking.

        Args:
            signal_id: Unique signal identifier
            initial_confidence: Initial confidence (0.0 to 1.0)
            signal_type: Signal type (momentum/mean_reversion/breakout/arbitrage)
            custom_params: Optional custom decay parameters
        """
        # Get decay parameters
        if custom_params:
            params = custom_params
        elif signal_type in self.DEFAULT_DECAY_PARAMS:
            params = self.DEFAULT_DECAY_PARAMS[signal_type]
        else:
            params = self.DEFAULT_DECAY_PARAMS["default"]
            logger.warning(f"Unknown signal type '{signal_type}', using default decay")

        # Create decay state
        now = datetime.utcnow()
        state = SignalDecayState(
            signal_id=signal_id,
            initial_confidence=initial_confidence,
            current_confidence=initial_confidence,
            creation_time=now,
            last_update=now,
            decay_params=params,
        )

        self._active_signals[signal_id] = state

        logger.debug(
            f"Registered signal {signal_id}: type={signal_type}, "
            f"confidence={initial_confidence:.3f}, "
            f"half_life={params.half_life_seconds:.0f}s"
        )

    def get_current_confidence(
        self,
        signal_id: str,
        now: datetime | None = None,
    ) -> float:
        """Get current decayed confidence for a signal.

        Args:
            signal_id: Signal identifier
            now: Current time (uses UTC now if None)

        Returns:
            Current confidence (0.0 to 1.0)
        """
        if now is None:
            now = datetime.utcnow()

        if signal_id not in self._active_signals:
            logger.warning(f"Signal {signal_id} not registered")
            return 0.0

        state = self._active_signals[signal_id]

        # Calculate age
        age_seconds = (now - state.creation_time).total_seconds()
        state.age_seconds = age_seconds

        # Calculate decay
        decayed_confidence = self._calculate_decay(
            state.initial_confidence,
            age_seconds,
            state.decay_params,
        )

        # Update state
        state.current_confidence = decayed_confidence
        state.last_update = now

        # Check if expired
        if decayed_confidence <= state.decay_params.min_confidence:
            if signal_id in self._active_signals:
                self._total_expired += 1
                logger.debug(
                    f"Signal {signal_id} expired: "
                    f"confidence={decayed_confidence:.3f} <= {state.decay_params.min_confidence}"
                )

        return decayed_confidence

    def adjust_for_market_conditions(
        self,
        signal_id: str,
        volatility_percentile: float,
        regime: str,
    ) -> None:
        """Adjust decay parameters based on market conditions.

        High volatility and IMPULSE regime accelerate decay.

        Args:
            signal_id: Signal identifier
            volatility_percentile: Current volatility percentile (0-100)
            regime: Current regime (NORMAL/IMPULSE/RANGE)
        """
        if signal_id not in self._active_signals:
            return

        state = self._active_signals[signal_id]
        params = state.decay_params

        # Volatility adjustment
        # High volatility (>75th percentile) = faster decay
        # Low volatility (<25th percentile) = slower decay
        if volatility_percentile > 75:
            params.volatility_multiplier = 1.5  # 50% faster decay
        elif volatility_percentile > 50:
            params.volatility_multiplier = 1.2  # 20% faster
        elif volatility_percentile < 25:
            params.volatility_multiplier = 0.8  # 20% slower
        else:
            params.volatility_multiplier = 1.0  # Normal

        # Regime adjustment
        if regime == "IMPULSE":
            params.regime_multiplier = 1.3  # 30% faster decay in impulse
        elif regime == "RANGE":
            params.regime_multiplier = 0.9  # 10% slower in range
        else:
            params.regime_multiplier = 1.0  # Normal

        logger.debug(
            f"Market conditions adjusted for {signal_id}: "
            f"vol_mult={params.volatility_multiplier:.2f}, "
            f"regime_mult={params.regime_multiplier:.2f}"
        )

    def _calculate_decay(
        self,
        initial_confidence: float,
        age_seconds: float,
        params: DecayParameters,
    ) -> float:
        """Calculate decayed confidence.

        Args:
            initial_confidence: Initial confidence
            age_seconds: Signal age in seconds
            params: Decay parameters

        Returns:
            Decayed confidence
        """
        # Apply market condition multipliers to effective half-life
        effective_half_life = params.half_life_seconds / (
            params.volatility_multiplier * params.regime_multiplier
        )

        if params.curve_type == DecayCurve.EXPONENTIAL:
            # Exponential decay: C(t) = C0 * (0.5)^(t/half_life)
            decay_factor = 0.5 ** (age_seconds / effective_half_life)
            confidence = initial_confidence * decay_factor

        elif params.curve_type == DecayCurve.LINEAR:
            # Linear decay: C(t) = C0 - (C0 - min) * (t / (2 * half_life))
            max_age = 2 * effective_half_life  # Reaches min at 2x half-life
            decay_rate = (initial_confidence - params.min_confidence) / max_age
            confidence = initial_confidence - (decay_rate * age_seconds)

        elif params.curve_type == DecayCurve.LOGARITHMIC:
            # Logarithmic decay: C(t) = C0 - (C0 - min) * log(1 + t) / log(1 + half_life)
            if age_seconds == 0:
                confidence = initial_confidence
            else:
                decay_factor = np.log(1 + age_seconds) / np.log(1 + effective_half_life * 2)
                confidence = initial_confidence - (
                    (initial_confidence - params.min_confidence) * decay_factor
                )

        elif params.curve_type == DecayCurve.STEP:
            # Step decay: confidence drops at specific time intervals
            step_interval = effective_half_life
            steps_elapsed = int(age_seconds / step_interval)
            step_decay = 0.2  # 20% drop per step
            confidence = initial_confidence * ((1 - step_decay) ** steps_elapsed)

        else:
            # Default to exponential
            decay_factor = 0.5 ** (age_seconds / effective_half_life)
            confidence = initial_confidence * decay_factor

        # Clamp to min confidence
        confidence = max(params.min_confidence, confidence)

        self._total_decayed += 1

        return confidence

    def remove_signal(self, signal_id: str) -> None:
        """Remove signal from tracking (e.g., when executed or cancelled).

        Args:
            signal_id: Signal identifier
        """
        if signal_id in self._active_signals:
            del self._active_signals[signal_id]
            logger.debug(f"Removed signal {signal_id} from decay tracking")

    def get_signal_state(self, signal_id: str) -> SignalDecayState | None:
        """Get decay state for a signal.

        Args:
            signal_id: Signal identifier

        Returns:
            SignalDecayState or None if not found
        """
        return self._active_signals.get(signal_id)

    def get_statistics(self) -> dict[str, Any]:
        """Get decay model statistics.

        Returns:
            Dictionary of statistics
        """
        active_count = len(self._active_signals)

        # Calculate average age and confidence of active signals
        if active_count > 0:
            avg_age_seconds = sum(s.age_seconds for s in self._active_signals.values()) / active_count
            avg_confidence = sum(s.current_confidence for s in self._active_signals.values()) / active_count
        else:
            avg_age_seconds = 0.0
            avg_confidence = 0.0

        return {
            "active_signals": active_count,
            "total_decayed": self._total_decayed,
            "total_expired": self._total_expired,
            "avg_age_seconds": avg_age_seconds,
            "avg_current_confidence": avg_confidence,
        }

    def cleanup_expired_signals(self, max_age_minutes: int = 60) -> int:
        """Remove signals older than max age.

        Args:
            max_age_minutes: Maximum age in minutes

        Returns:
            Number of signals removed
        """
        now = datetime.utcnow()
        max_age = timedelta(minutes=max_age_minutes)

        expired_ids = [
            signal_id
            for signal_id, state in self._active_signals.items()
            if (now - state.creation_time) > max_age
        ]

        for signal_id in expired_ids:
            del self._active_signals[signal_id]

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired signals (age > {max_age_minutes}min)")

        return len(expired_ids)


class DecayAwareOrderTiming:
    """Integrates signal decay with order timing optimization.

    Combines OrderTimingOptimizer with SignalDecayModel to make
    decay-aware timing decisions.
    """

    def __init__(
        self,
        timing_optimizer: Any,  # OrderTimingOptimizer instance
        decay_model: SignalDecayModel,
        decay_threshold: float = 0.4,  # Execute immediately if below this
    ):
        """Initialize decay-aware order timing.

        Args:
            timing_optimizer: OrderTimingOptimizer instance
            decay_model: SignalDecayModel instance
            decay_threshold: Confidence threshold for immediate execution
        """
        self._timing_optimizer = timing_optimizer
        self._decay_model = decay_model
        self._decay_threshold = decay_threshold

        logger.info(
            f"DecayAwareOrderTiming initialized with decay_threshold={decay_threshold}"
        )

    def should_execute_now(
        self,
        signal_id: str,
        symbol: str,
        side: str,
    ) -> tuple[bool, str]:
        """Determine if signal should execute now considering decay.

        Args:
            signal_id: Signal identifier
            symbol: Trading symbol
            side: Order side

        Returns:
            Tuple of (should_execute, reason)
        """
        # Get current decayed confidence
        current_confidence = self._decay_model.get_current_confidence(signal_id)

        # If confidence is below threshold, execute immediately to capture remaining edge
        if current_confidence < self._decay_threshold:
            return True, f"decay_urgent_confidence={current_confidence:.3f}"

        # Otherwise, consult timing optimizer
        timing_rec = self._timing_optimizer.get_timing_recommendation(
            symbol=symbol,
            side=side,
            is_urgent=False,
        )

        if timing_rec.urgency == "optimal":
            return True, "timing_optimal"
        elif timing_rec.urgency == "wait":
            # Check if waiting will cause too much decay
            state = self._decay_model.get_signal_state(signal_id)
            if state:
                # Estimate confidence after waiting
                future_age = state.age_seconds + (timing_rec.wait_minutes * 60)
                future_confidence = self._decay_model._calculate_decay(
                    state.initial_confidence,
                    future_age,
                    state.decay_params,
                )

                # If future confidence will drop below threshold, execute now
                if future_confidence < self._decay_threshold:
                    return True, f"decay_will_expire_confidence={future_confidence:.3f}"

            return False, f"timing_wait_{timing_rec.wait_minutes}min"
        else:
            return True, "timing_immediate"

    def get_adjusted_confidence(
        self,
        signal_id: str,
        base_confidence: float,
    ) -> float:
        """Get confidence adjusted for signal decay.

        Args:
            signal_id: Signal identifier
            base_confidence: Base confidence from confirmation

        Returns:
            Decay-adjusted confidence
        """
        # Get decay multiplier
        current_confidence = self._decay_model.get_current_confidence(signal_id)
        state = self._decay_model.get_signal_state(signal_id)

        if not state:
            return base_confidence

        # Decay factor (0.0 to 1.0)
        decay_factor = current_confidence / state.initial_confidence if state.initial_confidence > 0 else 1.0

        # Apply decay to base confidence
        adjusted_confidence = base_confidence * decay_factor

        logger.debug(
            f"Adjusted confidence: base={base_confidence:.3f}, "
            f"decay_factor={decay_factor:.3f}, adjusted={adjusted_confidence:.3f}"
        )

        return adjusted_confidence

"""Dynamic Oracle Weighting: Regime-Aware Signal Fusion.

Adapts Oracle signal weights based on current market regime (from Physics component):
- Volatile/unpredictable markets → increase TCN weight (price reversal prediction)
- Strong news-driven trends → increase sentiment weights (FinBERT, Ollama)
- Calm/stable markets → balanced weights

Production-grade with full observability and audit trail.
"""

import time
from dataclasses import dataclass

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WeightConfig:
    """Weight configuration for Oracle signal sources."""
    tcn_weight: float  # TCN price reversal predictor
    finbert_weight: float  # FinBERT text sentiment
    ollama_weight: float  # Ollama local LLM sentiment
    brain_weight: float  # Claude Brain analysis
    timestamp: float
    regime: str  # Market regime that produced these weights


class DynamicOracleWeightManager:
    """Manages dynamic weighting of Oracle signal sources based on market regime.

    Default weights (baseline):
    - TCN: 40%
    - FinBERT: 20%
    - Ollama: 20%
    - Brain: 20%

    Regime adjustments:
    - VAPOR (high volatility/entropy): TCN 60%, sentiment 40% (price action > sentiment)
    - ICE (low volatility): TCN 30%, sentiment 70% (patience, wait for catalyst)
    - WATER (trending): Balanced 40/20/20/20
    - Laplace (SSD resonance): TCN 55%, Brain 25% (deterministic regime, trust models)
    - Silent (SSD noise): Disable all sources (return None)
    """

    # Base weight (normal market conditions - WATER phase)
    BASE_WEIGHTS = {
        "tcn": 0.40,
        "finbert": 0.20,
        "ollama": 0.20,
        "brain": 0.20,
    }

    # Regime-specific weight adjustments
    REGIME_WEIGHTS = {
        "vapor": {  # High volatility, chaos
            "tcn": 0.60,
            "finbert": 0.15,
            "ollama": 0.15,
            "brain": 0.10,
        },
        "ice": {  # Low volatility, sideways
            "tcn": 0.30,
            "finbert": 0.25,
            "ollama": 0.25,
            "brain": 0.20,
        },
        "water": {  # Normal trending
            "tcn": 0.40,
            "finbert": 0.20,
            "ollama": 0.20,
            "brain": 0.20,
        },
        "accumulation": {  # Bottom formation
            "tcn": 0.35,
            "finbert": 0.25,
            "ollama": 0.20,
            "brain": 0.20,
        },
        "markup": {  # Trending up
            "tcn": 0.45,
            "finbert": 0.20,
            "ollama": 0.15,
            "brain": 0.20,
        },
        "distribution": {  # Top formation
            "tcn": 0.35,
            "finbert": 0.25,
            "ollama": 0.20,
            "brain": 0.20,
        },
        "markdown": {  # Trending down
            "tcn": 0.45,
            "finbert": 0.20,
            "ollama": 0.15,
            "brain": 0.20,
        },
        "unknown": {  # No clear regime
            "tcn": 0.40,
            "finbert": 0.20,
            "ollama": 0.20,
            "brain": 0.20,
        },
    }

    # SSD mode adjustments (applied on top of regime weights)
    SSD_ADJUSTMENTS = {
        "laplace": {  # Deterministic regime
            "tcn": 1.3,  # Boost TCN (models work well)
            "finbert": 1.0,
            "ollama": 0.8,
            "brain": 1.2,  # Boost Brain (AI analysis valuable)
        },
        "normal": {  # Standard operation
            "tcn": 1.0,
            "finbert": 1.0,
            "ollama": 1.0,
            "brain": 1.0,
        },
        "silent": {  # Entropy diverging, noise
            # Return None in get_weights() - no trading
            "tcn": 0.0,
            "finbert": 0.0,
            "ollama": 0.0,
            "brain": 0.0,
        },
    }

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._current_weights: dict[str, WeightConfig] = {}  # symbol -> WeightConfig
        self._physics_states: dict[str, dict] = {}  # symbol -> physics state
        self._last_update: dict[str, float] = {}

    async def start(self) -> None:
        """Start dynamic weight manager."""
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        logger.info("DynamicOracleWeightManager started")

    async def stop(self) -> None:
        """Stop dynamic weight manager."""
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        logger.info("DynamicOracleWeightManager stopped")

    async def _handle_physics_update(self, event: Event) -> None:
        """Handle physics state updates."""
        data = event.data
        symbol = data.get("symbol")
        if not symbol:
            return

        physics = data.get("physics", {})
        self._physics_states[symbol] = physics
        self._last_update[symbol] = time.time()

        # Recalculate weights for this symbol
        new_weights = self._calculate_weights(symbol, physics)
        if new_weights:
            old_weights = self._current_weights.get(symbol)
            self._current_weights[symbol] = new_weights

            # Log significant changes
            if old_weights and self._weights_changed_significantly(old_weights, new_weights):
                logger.info(
                    f"Oracle weights updated for {symbol}: "
                    f"TCN {old_weights.tcn_weight:.2%}→{new_weights.tcn_weight:.2%}, "
                    f"FinBERT {old_weights.finbert_weight:.2%}→{new_weights.finbert_weight:.2%}, "
                    f"Ollama {old_weights.ollama_weight:.2%}→{new_weights.ollama_weight:.2%}, "
                    f"Brain {old_weights.brain_weight:.2%}→{new_weights.brain_weight:.2%} "
                    f"(regime: {new_weights.regime})"
                )

    def _calculate_weights(self, symbol: str, physics: dict) -> WeightConfig | None:
        """Calculate Oracle weights based on physics state.

        Returns:
            WeightConfig with adjusted weights, or None if trading should be blocked (Silent mode)
        """
        phase = physics.get("phase", "unknown")
        ssd_mode = physics.get("ssd_mode", "normal")
        entropy = physics.get("entropy", 0.5)
        temperature = physics.get("temperature", 500.0)

        # BLOCK in Silent mode (entropy diverging)
        if ssd_mode == "silent":
            logger.warning(f"Oracle weights disabled for {symbol}: SSD SILENT mode")
            return None

        # Get base weights for current regime
        regime_weights = self.REGIME_WEIGHTS.get(phase, self.BASE_WEIGHTS)

        # Get SSD adjustments
        ssd_adjustments = self.SSD_ADJUSTMENTS.get(ssd_mode, self.SSD_ADJUSTMENTS["normal"])

        # Apply SSD adjustments
        tcn = regime_weights["tcn"] * ssd_adjustments["tcn"]
        finbert = regime_weights["finbert"] * ssd_adjustments["finbert"]
        ollama = regime_weights["ollama"] * ssd_adjustments["ollama"]
        brain = regime_weights["brain"] * ssd_adjustments["brain"]

        # Additional entropy/temperature adjustments
        # High entropy + high temp → boost TCN even more (chaos, trust price action)
        if entropy > 0.7 and temperature > 800:
            tcn *= 1.15
            finbert *= 0.9
            ollama *= 0.9

        # Low entropy + low temp → boost sentiment (stable, wait for catalyst)
        if entropy < 0.3 and temperature < 400:
            tcn *= 0.85
            finbert *= 1.15
            ollama *= 1.15

        # Normalize to sum to 1.0
        total = tcn + finbert + ollama + brain
        if total <= 0:
            logger.error(f"Invalid weight sum for {symbol}: {total}")
            return None

        tcn /= total
        finbert /= total
        ollama /= total
        brain /= total

        return WeightConfig(
            tcn_weight=tcn,
            finbert_weight=finbert,
            ollama_weight=ollama,
            brain_weight=brain,
            timestamp=time.time(),
            regime=f"{phase}/{ssd_mode}",
        )

    def _weights_changed_significantly(
        self, old: WeightConfig, new: WeightConfig, threshold: float = 0.05
    ) -> bool:
        """Check if weights changed by more than threshold (5% by default)."""
        return (
            abs(old.tcn_weight - new.tcn_weight) > threshold
            or abs(old.finbert_weight - new.finbert_weight) > threshold
            or abs(old.ollama_weight - new.ollama_weight) > threshold
            or abs(old.brain_weight - new.brain_weight) > threshold
        )

    def get_weights(self, symbol: str) -> WeightConfig | None:
        """Get current Oracle weights for a symbol.

        Returns:
            WeightConfig with current weights, or None if trading blocked (Silent mode)
        """
        weights = self._current_weights.get(symbol)

        if not weights:
            # No physics data yet, return base weights
            logger.debug(f"No physics-based weights for {symbol}, using base weights")
            return WeightConfig(
                tcn_weight=self.BASE_WEIGHTS["tcn"],
                finbert_weight=self.BASE_WEIGHTS["finbert"],
                ollama_weight=self.BASE_WEIGHTS["ollama"],
                brain_weight=self.BASE_WEIGHTS["brain"],
                timestamp=time.time(),
                regime="default",
            )

        # Check staleness (weights older than 60s)
        age = time.time() - weights.timestamp
        if age > 60.0:
            logger.warning(
                f"Stale Oracle weights for {symbol} (age={age:.1f}s), using base weights"
            )
            return WeightConfig(
                tcn_weight=self.BASE_WEIGHTS["tcn"],
                finbert_weight=self.BASE_WEIGHTS["finbert"],
                ollama_weight=self.BASE_WEIGHTS["ollama"],
                brain_weight=self.BASE_WEIGHTS["brain"],
                timestamp=time.time(),
                regime="stale/default",
            )

        return weights

    def get_all_weights(self) -> dict[str, WeightConfig]:
        """Get all current Oracle weights."""
        return self._current_weights.copy()

    def force_weights(self, symbol: str, weights: WeightConfig) -> None:
        """Manually override weights for a symbol (for testing/manual intervention)."""
        self._current_weights[symbol] = weights
        logger.warning(
            f"Oracle weights manually overridden for {symbol}: "
            f"TCN={weights.tcn_weight:.2%}, FinBERT={weights.finbert_weight:.2%}, "
            f"Ollama={weights.ollama_weight:.2%}, Brain={weights.brain_weight:.2%}"
        )

    def reset_to_base(self, symbol: str) -> None:
        """Reset symbol to base weights."""
        self._current_weights[symbol] = WeightConfig(
            tcn_weight=self.BASE_WEIGHTS["tcn"],
            finbert_weight=self.BASE_WEIGHTS["finbert"],
            ollama_weight=self.BASE_WEIGHTS["ollama"],
            brain_weight=self.BASE_WEIGHTS["brain"],
            timestamp=time.time(),
            regime="manual_reset",
        )
        logger.info(f"Oracle weights reset to base for {symbol}")

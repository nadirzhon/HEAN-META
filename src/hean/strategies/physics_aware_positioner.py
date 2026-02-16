"""Physics-Aware Position Sizing and Confidence Filtering.

Integrates Physics component (temperature, entropy, phase) into trading decisions:
- Market phase → position size multiplier
- Temperature/entropy → confidence threshold adjustment
- SSD resonance → boost high-confidence signals

Production-grade with full observability and error handling.
"""

import time
from dataclasses import dataclass

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PhysicsContext:
    """Current physics state for decision making."""
    symbol: str
    temperature: float
    entropy: float
    phase: str
    phase_confidence: float
    ssd_mode: str
    resonance_strength: float
    is_resonant: bool
    causal_weight: float
    timestamp: float


class PhysicsAwarePositioner:
    """Adjusts position sizing and signal confidence based on Physics component state.

    Key principles:
    1. Phase alignment: Larger positions when signal matches detected phase
    2. Temperature/entropy filtering: Adjust confidence thresholds based on market state
    3. SSD resonance boost: Increase sizing when market is deterministic (Laplace mode)
    4. Silent mode protection: Block trades during high-entropy divergence
    """

    def __init__(self, bus: EventBus):
        self._bus = bus
        self._physics_states: dict[str, PhysicsContext] = {}
        self._last_update: dict[str, float] = {}
        self._state_staleness_threshold = 30.0  # seconds

        # Phase-specific size multipliers
        self._phase_multipliers = {
            "accumulation": 1.2,  # Good for mean reversion strategies
            "markup": 1.5,        # Strong for momentum/impulse
            "distribution": 0.8,  # Risky, reduce size
            "markdown": 1.3,      # Good for shorts
            "ice": 1.0,           # Neutral
            "water": 1.2,         # Trending
            "vapor": 0.5,         # Chaos, reduce size
            "unknown": 0.7,       # Conservative default
        }

        # SSD mode multipliers
        self._ssd_multipliers = {
            "silent": 0.0,    # No trades during noise regime
            "normal": 1.0,    # Standard operation
            "laplace": 1.5,   # Boost sizing in deterministic regime
        }

    async def start(self) -> None:
        """Start physics-aware positioner."""
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        logger.info("PhysicsAwarePositioner started")

    async def stop(self) -> None:
        """Stop physics-aware positioner."""
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        logger.info("PhysicsAwarePositioner stopped")

    async def _handle_physics_update(self, event: Event) -> None:
        """Handle physics state updates from PhysicsEngine."""
        data = event.data
        symbol = data.get("symbol")
        if not symbol:
            return

        physics = data.get("physics", {})

        context = PhysicsContext(
            symbol=symbol,
            temperature=physics.get("temperature", 0.5),
            entropy=physics.get("entropy", 0.5),
            phase=physics.get("phase", "unknown"),
            phase_confidence=physics.get("phase_confidence", 0.0),
            ssd_mode=physics.get("ssd_mode", "normal"),
            resonance_strength=physics.get("resonance_strength", 0.0),
            is_resonant=physics.get("is_resonant", False),
            causal_weight=physics.get("causal_weight", 1.0),
            timestamp=time.time(),
        )

        self._physics_states[symbol] = context
        self._last_update[symbol] = context.timestamp

    def get_physics_adjusted_signal(self, signal: Signal) -> Signal | None:
        """Adjust signal based on physics state, or None if should be blocked.

        Returns:
            Adjusted signal with physics-aware sizing, or None if trade should be blocked
        """
        symbol = signal.symbol
        physics = self._physics_states.get(symbol)

        if not physics:
            logger.debug(f"No physics state for {symbol}, allowing signal with 0.7x default multiplier")
            return self._apply_size_multiplier(signal, 0.7)

        # Check staleness
        age = time.time() - physics.timestamp
        if age > self._state_staleness_threshold:
            logger.warning(
                f"Stale physics state for {symbol} (age={age:.1f}s), using conservative 0.7x multiplier"
            )
            return self._apply_size_multiplier(signal, 0.7)

        # BLOCK in SSD Silent mode (entropy diverging, noise regime)
        if physics.ssd_mode == "silent":
            logger.warning(
                f"PHYSICS BLOCK: {symbol} in SSD SILENT mode (entropy={physics.entropy:.3f}), "
                f"signal rejected"
            )
            return None

        # Calculate total size multiplier
        phase_mult = self._phase_multipliers.get(physics.phase, 0.7)
        ssd_mult = self._ssd_multipliers.get(physics.ssd_mode, 1.0)

        # Resonance boost (only in Laplace mode)
        resonance_boost = 1.0
        if physics.ssd_mode == "laplace" and physics.is_resonant:
            resonance_boost = 1.0 + (physics.resonance_strength * 0.3)  # Up to +30%

        # Temperature confidence adjustment
        temp_confidence = self._get_temperature_confidence(physics.temperature)

        # Entropy confidence penalty (high entropy = lower confidence)
        entropy_penalty = max(0.5, 1.0 - (physics.entropy * 0.3))

        # Combined multiplier
        total_mult = phase_mult * ssd_mult * resonance_boost * temp_confidence * entropy_penalty

        # Apply phase confidence filter (low confidence → reduce size)
        if physics.phase_confidence < 0.3:
            total_mult *= 0.7
            logger.debug(
                f"Low phase confidence ({physics.phase_confidence:.2f}) for {symbol}, reducing multiplier"
            )

        logger.info(
            f"Physics Adjustment {symbol}: phase={physics.phase}({phase_mult:.2f}x) "
            f"ssd={physics.ssd_mode}({ssd_mult:.2f}x) resonance={resonance_boost:.2f}x "
            f"temp_conf={temp_confidence:.2f}x entropy_penalty={entropy_penalty:.2f}x "
            f"→ TOTAL={total_mult:.2f}x"
        )

        adjusted_signal = self._apply_size_multiplier(signal, total_mult)

        # Add physics metadata for auditing
        if adjusted_signal.metadata is None:
            adjusted_signal.metadata = {}
        adjusted_signal.metadata.update({
            "physics_phase": physics.phase,
            "physics_ssd_mode": physics.ssd_mode,
            "physics_resonance": physics.resonance_strength,
            "physics_size_mult": total_mult,
            "physics_temperature": physics.temperature,
            "physics_entropy": physics.entropy,
        })

        return adjusted_signal

    def _apply_size_multiplier(self, signal: Signal, multiplier: float) -> Signal:
        """Apply size multiplier to signal (creates new signal, does not mutate)."""
        # NOTE: Signal is a dataclass, we need to create a new instance
        # with modified metadata
        new_metadata = dict(signal.metadata) if signal.metadata else {}
        current_mult = new_metadata.get("size_multiplier", 1.0)
        new_metadata["size_multiplier"] = current_mult * multiplier

        # Return new signal with updated metadata
        return Signal(
            strategy_id=signal.strategy_id,
            symbol=signal.symbol,
            side=signal.side,
            entry_price=signal.entry_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            metadata=new_metadata,
        )

    def _get_temperature_confidence(self, temperature: float) -> float:
        """Convert temperature to confidence multiplier.

        - Cold markets (T < 400): High confidence, steady trends → 1.1x
        - Warm markets (400 <= T < 800): Medium confidence → 1.0x
        - Hot markets (T >= 800): Low confidence, volatile → 0.7x
        """
        if temperature < 400:
            return 1.1  # Cold, stable
        elif temperature < 800:
            return 1.0  # Warm, normal
        else:
            # Very hot, reduce confidence
            heat_factor = min(temperature / 1000.0, 2.0)
            return max(0.5, 1.0 - (heat_factor - 0.8) * 0.5)

    def should_increase_confidence_threshold(self, symbol: str) -> tuple[bool, float]:
        """Check if confidence threshold should be raised due to physics state.

        Returns:
            (should_increase, suggested_threshold_adjustment)
        """
        physics = self._physics_states.get(symbol)
        if not physics:
            return False, 0.0

        # High entropy + high temperature = chaos, increase threshold
        if physics.entropy > 0.7 and physics.temperature > 700:
            adjustment = 0.15  # Increase threshold by 15%
            logger.debug(
                f"High entropy+temp for {symbol}, suggest threshold +{adjustment:.2%}"
            )
            return True, adjustment

        # Low temp + low entropy = stable, can lower threshold
        if physics.entropy < 0.3 and physics.temperature < 400:
            adjustment = -0.10  # Decrease threshold by 10%
            logger.debug(
                f"Low entropy+temp for {symbol}, suggest threshold {adjustment:.2%}"
            )
            return True, adjustment

        return False, 0.0

    def get_physics_state(self, symbol: str) -> PhysicsContext | None:
        """Get current physics context for a symbol."""
        return self._physics_states.get(symbol)

    def get_all_physics_states(self) -> dict[str, PhysicsContext]:
        """Get all current physics states."""
        return self._physics_states.copy()

"""Physics Engine - Integrates all physics components with EventBus.

Subscribes to TICK events, calculates T/S/phase, and publishes
PHYSICS_UPDATE events for downstream consumers.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger
from hean.physics.entropy import MarketEntropy
from hean.physics.phase_detector import PhaseDetector, ResonanceState
from hean.physics.szilard import SzilardEngine
from hean.physics.temperature import MarketTemperature

logger = get_logger(__name__)


@dataclass
class PhysicsState:
    """Current physics state for a symbol."""

    symbol: str
    temperature: float = 0.0
    temperature_regime: str = "COLD"
    entropy: float = 0.0
    entropy_state: str = "COMPRESSED"
    phase: str = "unknown"
    phase_confidence: float = 0.0
    szilard_profit: float = 0.0
    should_trade: bool = False
    trade_reason: str = ""
    size_multiplier: float = 0.5
    timestamp: float = field(default_factory=time.time)
    # SSD: Singular Spectral Determinism fields
    entropy_flow: float = 0.0            # dH/dt — entropy rate of change
    entropy_flow_smooth: float = 0.0     # EMA-smoothed entropy flow
    ssd_mode: str = "normal"             # "silent" | "normal" | "laplace"
    resonance_strength: float = 0.0      # 0-1 vector alignment score
    is_resonant: bool = False            # True when market vectors align
    causal_weight: float = 1.0           # Data collapse factor (0.05 in Laplace, 1.0 normal)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "temperature": self.temperature,
            "temperature_regime": self.temperature_regime,
            "entropy": self.entropy,
            "entropy_state": self.entropy_state,
            "phase": self.phase,
            "phase_confidence": self.phase_confidence,
            "szilard_profit": self.szilard_profit,
            "should_trade": self.should_trade,
            "trade_reason": self.trade_reason,
            "size_multiplier": self.size_multiplier,
            "timestamp": self.timestamp,
            # SSD fields
            "entropy_flow": self.entropy_flow,
            "entropy_flow_smooth": self.entropy_flow_smooth,
            "ssd_mode": self.ssd_mode,
            "resonance_strength": self.resonance_strength,
            "is_resonant": self.is_resonant,
            "causal_weight": self.causal_weight,
        }


class PhysicsEngine:
    """Main physics engine integrating all components."""

    def __init__(
        self,
        bus: EventBus,
        window_size: int = 100,
    ) -> None:
        self._bus = bus
        self._temperature = MarketTemperature(window_size=window_size)
        self._entropy = MarketEntropy(window_size=window_size)
        self._phase_detector = PhaseDetector()
        self._szilard = SzilardEngine()

        # Rolling price/volume buffers per symbol
        self._prices: dict[str, deque[float]] = {}
        self._volumes: dict[str, deque[float]] = {}
        self._buffer_size = window_size

        # Current state cache
        self._states: dict[str, PhysicsState] = {}

        self._running = False

    async def start(self) -> None:
        """Start the physics engine."""
        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        logger.info("PhysicsEngine started")

    async def stop(self) -> None:
        """Stop the physics engine."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("PhysicsEngine stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle incoming tick event."""
        if not self._running:
            return

        data = event.data
        tick = data.get("tick")
        if tick is None:
            return

        symbol = tick.symbol
        price = tick.price
        volume = tick.volume

        # Initialize buffers
        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=self._buffer_size)
            self._volumes[symbol] = deque(maxlen=self._buffer_size)

        self._prices[symbol].append(price)
        self._volumes[symbol].append(volume)

        # Need minimum samples
        if len(self._prices[symbol]) < 10:
            return

        prices = list(self._prices[symbol])
        volumes = list(self._volumes[symbol])

        # Calculate temperature
        temp_reading = self._temperature.calculate(prices, volumes, symbol)

        # Calculate entropy (now includes SSD entropy flow)
        entropy_reading = self._entropy.calculate(volumes, symbol)

        # SSD: Feed price/volume vectors for resonance calculation
        self._phase_detector.update_market_vectors(
            symbol, price, volume, entropy_reading.entropy_flow_smooth
        )

        # Detect phase (now includes SSD resonance and mode)
        phase = self._phase_detector.detect(
            temp_reading.value, entropy_reading.value, symbol,
            entropy_flow=entropy_reading.entropy_flow_smooth,
        )

        # Get phase reading for confidence + SSD state
        phase_reading = self._phase_detector.get_current(symbol)
        phase_confidence = phase_reading.confidence if phase_reading else 0.0
        ssd_mode = phase_reading.ssd_mode.value if phase_reading else "normal"
        resonance = phase_reading.resonance if phase_reading else ResonanceState()

        # SSD: Data Collapse — in Laplace mode, causal_weight = 0.05 (keep only 5% of inputs)
        causal_weight = 1.0
        if ssd_mode == "laplace":
            causal_weight = 0.05  # 95% collapse
        elif ssd_mode == "silent":
            causal_weight = 0.0  # Full collapse — ignore everything

        # Calculate Szilard profit (enhanced with SSD resonance)
        szilard_probability = 0.5
        if ssd_mode == "laplace":
            szilard_probability = min(0.95, 0.5 + resonance.strength * 0.45)
        szilard = self._szilard.calculate_max_profit(
            temp_reading.value, szilard_probability
        )

        # Should we trade? (SSD-aware)
        if ssd_mode == "silent":
            should_trade = False
            trade_reason = "SSD SILENT — entropy diverging, noise regime"
        elif ssd_mode == "laplace":
            should_trade = True
            trade_reason = (
                f"SSD LAPLACE — resonance={resonance.strength:.3f}, "
                f"entropy_flow={entropy_reading.entropy_flow_smooth:.4f}"
            )
        else:
            should_trade, trade_reason = self._szilard.should_trade(
                temp_reading.value, entropy_reading.value, phase.value
            )

        # Optimal size (boosted in Laplace mode)
        size_mult = self._szilard.calculate_optimal_size_multiplier(
            temp_reading.value, entropy_reading.value, phase.value
        )
        if ssd_mode == "laplace":
            size_mult = min(2.0, size_mult * (1.0 + resonance.strength))
        elif ssd_mode == "silent":
            size_mult = 0.0

        # Build state
        state = PhysicsState(
            symbol=symbol,
            temperature=temp_reading.value,
            temperature_regime=temp_reading.regime,
            entropy=entropy_reading.value,
            entropy_state=entropy_reading.state,
            phase=phase.value,
            phase_confidence=phase_confidence,
            szilard_profit=szilard.max_profit,
            should_trade=should_trade,
            trade_reason=trade_reason,
            size_multiplier=size_mult,
            entropy_flow=entropy_reading.entropy_flow,
            entropy_flow_smooth=entropy_reading.entropy_flow_smooth,
            ssd_mode=ssd_mode,
            resonance_strength=resonance.strength,
            is_resonant=resonance.is_resonant,
            causal_weight=causal_weight,
        )

        self._states[symbol] = state

        # Publish physics update
        await self._bus.publish(
            Event(
                event_type=EventType.PHYSICS_UPDATE,
                data={
                    "symbol": symbol,
                    "physics": state.to_dict(),
                },
            )
        )

    def get_state(self, symbol: str) -> PhysicsState | None:
        return self._states.get(symbol)

    def get_all_states(self) -> dict[str, PhysicsState]:
        return self._states.copy()

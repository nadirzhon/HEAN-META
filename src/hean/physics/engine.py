"""Physics Engine - Integrates all physics components with EventBus.

Subscribes to TICK events, calculates T/S/phase, and publishes
PHYSICS_UPDATE events for downstream consumers.
"""

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger
from hean.physics.entropy import MarketEntropy
from hean.physics.phase_detector import MarketPhase, PhaseDetector
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

        # Calculate entropy
        entropy_reading = self._entropy.calculate(volumes, symbol)

        # Detect phase
        phase = self._phase_detector.detect(
            temp_reading.value, entropy_reading.value, symbol
        )

        # Get phase reading for confidence
        phase_reading = self._phase_detector.get_current(symbol)
        phase_confidence = phase_reading.confidence if phase_reading else 0.0

        # Calculate Szilard profit
        szilard = self._szilard.calculate_max_profit(
            temp_reading.value, 0.5  # Default probability
        )

        # Should we trade?
        should_trade, trade_reason = self._szilard.should_trade(
            temp_reading.value, entropy_reading.value, phase.value
        )

        # Optimal size
        size_mult = self._szilard.calculate_optimal_size_multiplier(
            temp_reading.value, entropy_reading.value, phase.value
        )

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

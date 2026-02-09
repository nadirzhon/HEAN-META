"""Market Phase Transition Detector.

Detects phase transitions using temperature and entropy:
- ICE: Low T, low S (sideways, consolidation)
- WATER: Medium T, medium S (trend, directional)
- VAPOR: High T, high S (cascade, crash, chaos)

Key insight: ICE -> WATER transitions offer maximum edge with minimum risk.
"""

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum

from hean.logging import get_logger

logger = get_logger(__name__)


class MarketPhase(str, Enum):
    ICE = "ice"
    WATER = "water"
    VAPOR = "vapor"
    UNKNOWN = "unknown"


@dataclass
class PhaseReading:
    phase: MarketPhase
    temperature: float
    entropy: float
    confidence: float
    timestamp: float
    symbol: str


@dataclass
class PhaseTransition:
    from_phase: MarketPhase
    to_phase: MarketPhase
    temperature: float
    entropy: float
    timestamp: float
    symbol: str
    is_favorable: bool


class PhaseDetector:
    """Detect market phase transitions using temperature and entropy."""

    ICE_TEMP_MAX = 400.0
    WATER_TEMP_MAX = 800.0
    ICE_ENTROPY_MAX = 2.5
    WATER_ENTROPY_MAX = 3.5
    HYSTERESIS_FACTOR = 0.9

    def __init__(self, history_size: int = 100) -> None:
        self._history: dict[str, deque[PhaseReading]] = {}
        self._history_size = history_size
        self._transitions: dict[str, deque[PhaseTransition]] = {}
        self._transition_history_size = 50

    def detect(self, temperature: float, entropy: float, symbol: str = "UNKNOWN") -> MarketPhase:
        timestamp = time.time()
        prev_reading = self.get_current(symbol)
        prev_phase = prev_reading.phase if prev_reading else MarketPhase.UNKNOWN

        phase = self._detect_with_hysteresis(temperature, entropy, prev_phase)
        confidence = self._calculate_confidence(temperature, entropy, phase)

        reading = PhaseReading(
            phase=phase,
            temperature=temperature,
            entropy=entropy,
            confidence=confidence,
            timestamp=timestamp,
            symbol=symbol,
        )

        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._history_size)
        self._history[symbol].append(reading)

        if prev_reading and prev_reading.phase != phase:
            transition = PhaseTransition(
                from_phase=prev_reading.phase,
                to_phase=phase,
                temperature=temperature,
                entropy=entropy,
                timestamp=timestamp,
                symbol=symbol,
                is_favorable=self._is_favorable_transition(prev_reading.phase, phase),
            )
            if symbol not in self._transitions:
                self._transitions[symbol] = deque(maxlen=self._transition_history_size)
            self._transitions[symbol].append(transition)

            logger.info(
                f"[Phase] TRANSITION {symbol}: {prev_reading.phase.value.upper()} -> "
                f"{phase.value.upper()} (T={temperature:.1f}, S={entropy:.2f}, "
                f"favorable={transition.is_favorable})"
            )

        return phase

    def _detect_with_hysteresis(
        self, temperature: float, entropy: float, prev_phase: MarketPhase
    ) -> MarketPhase:
        if prev_phase != MarketPhase.UNKNOWN:
            ice_temp_max = self.ICE_TEMP_MAX * self.HYSTERESIS_FACTOR
            water_temp_max = self.WATER_TEMP_MAX * self.HYSTERESIS_FACTOR
            ice_entropy_max = self.ICE_ENTROPY_MAX * self.HYSTERESIS_FACTOR
            water_entropy_max = self.WATER_ENTROPY_MAX * self.HYSTERESIS_FACTOR
        else:
            ice_temp_max = self.ICE_TEMP_MAX
            water_temp_max = self.WATER_TEMP_MAX
            ice_entropy_max = self.ICE_ENTROPY_MAX
            water_entropy_max = self.WATER_ENTROPY_MAX

        if temperature < ice_temp_max and entropy < ice_entropy_max:
            return MarketPhase.ICE
        elif temperature >= water_temp_max and entropy >= water_entropy_max:
            return MarketPhase.VAPOR
        else:
            return MarketPhase.WATER

    def _calculate_confidence(
        self, temperature: float, entropy: float, phase: MarketPhase
    ) -> float:
        if phase == MarketPhase.ICE:
            t_dist = max(0, self.ICE_TEMP_MAX - temperature) / self.ICE_TEMP_MAX
            s_dist = max(0, self.ICE_ENTROPY_MAX - entropy) / self.ICE_ENTROPY_MAX
            return min(1.0, (t_dist + s_dist) / 2)
        elif phase == MarketPhase.VAPOR:
            t_dist = max(0, temperature - self.WATER_TEMP_MAX) / self.WATER_TEMP_MAX
            s_dist = max(0, entropy - self.WATER_ENTROPY_MAX) / self.WATER_ENTROPY_MAX
            return min(1.0, (t_dist + s_dist) / 2)
        elif phase == MarketPhase.WATER:
            t_ok = 1.0 if self.ICE_TEMP_MAX <= temperature < self.WATER_TEMP_MAX else 0.5
            s_ok = 1.0 if self.ICE_ENTROPY_MAX <= entropy < self.WATER_ENTROPY_MAX else 0.5
            return (t_ok + s_ok) / 2
        return 0.0

    def _is_favorable_transition(self, from_phase: MarketPhase, to_phase: MarketPhase) -> bool:
        # ICE -> WATER is the golden transition
        if from_phase == MarketPhase.ICE and to_phase == MarketPhase.WATER:
            return True
        if from_phase == MarketPhase.WATER and to_phase == MarketPhase.ICE:
            return True
        if to_phase == MarketPhase.VAPOR:
            return False
        if from_phase == MarketPhase.VAPOR:
            return False
        return False

    def get_current(self, symbol: str) -> PhaseReading | None:
        if symbol not in self._history or not self._history[symbol]:
            return None
        return self._history[symbol][-1]

    def get_history(self, symbol: str, limit: int | None = None) -> list[PhaseReading]:
        if symbol not in self._history:
            return []
        history = list(self._history[symbol])
        if limit is not None:
            history = history[-limit:]
        return history

    def get_transitions(self, symbol: str, limit: int | None = None) -> list[PhaseTransition]:
        if symbol not in self._transitions:
            return []
        transitions = list(self._transitions[symbol])
        if limit is not None:
            transitions = transitions[-limit:]
        return transitions

    def get_last_favorable_transition(self, symbol: str) -> PhaseTransition | None:
        if symbol not in self._transitions:
            return None
        for transition in reversed(self._transitions[symbol]):
            if transition.is_favorable:
                return transition
        return None

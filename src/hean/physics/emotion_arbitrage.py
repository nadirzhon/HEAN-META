"""Emotion Arbitrage - News Impact Wave Exploitation.

Tracks 4 news impact waves and detects overreactions to fade:
- Wave 1 (0-2s): Algos react
- Wave 2 (2-30s): First traders
- Wave 3 (30s-5m): Main wave
- Wave 4 (5-30m): Late arrivals
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


class ImpactWave(Enum):
    ALGO = "algo"           # 0-2 seconds
    FIRST_TRADERS = "first" # 2-30 seconds
    MAIN_WAVE = "main"      # 30s-5min
    LATE_ARRIVALS = "late"  # 5-30 min


@dataclass
class NewsImpact:
    id: str
    timestamp: float
    symbol: str
    initial_price: float
    max_deviation_pct: float = 0.0
    current_wave: ImpactWave = ImpactWave.ALGO
    waves: dict[str, float] = field(default_factory=dict)  # wave -> price_change_pct
    fade_opportunity: bool = False
    fade_confidence: float = 0.0
    resolved: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "initial_price": self.initial_price,
            "max_deviation_pct": self.max_deviation_pct,
            "current_wave": self.current_wave.value,
            "waves": self.waves,
            "fade_opportunity": self.fade_opportunity,
            "fade_confidence": self.fade_confidence,
        }


class EmotionArbitrage:
    """Detect and exploit news-driven emotional overreactions."""

    # Wave boundaries (seconds)
    WAVE_BOUNDARIES = {
        ImpactWave.ALGO: (0, 2),
        ImpactWave.FIRST_TRADERS: (2, 30),
        ImpactWave.MAIN_WAVE: (30, 300),
        ImpactWave.LATE_ARRIVALS: (300, 1800),
    }

    # Overreaction threshold (price deviation from expected)
    OVERREACTION_THRESHOLD = 0.02  # 2%
    FADE_THRESHOLD = 0.015  # 1.5% fade expected

    def __init__(self) -> None:
        self._impacts: deque[NewsImpact] = deque(maxlen=100)
        self._active_impacts: list[NewsImpact] = []
        self._counter = 0

        # Historical fade success rate
        self._fade_successes: int = 0
        self._fade_attempts: int = 0

    def register_event(self, symbol: str, price: float) -> NewsImpact:
        """Register a new news/event impact."""
        self._counter += 1
        impact = NewsImpact(
            id=f"news_{self._counter}",
            timestamp=time.time(),
            symbol=symbol,
            initial_price=price,
        )
        self._active_impacts.append(impact)
        self._impacts.append(impact)

        logger.info(f"[Emotion] News event registered on {symbol} at ${price:.2f}")
        return impact

    def update(self, symbol: str, price: float) -> list[NewsImpact]:
        """Update active impacts with new price and return those with fade opportunities."""
        now = time.time()
        opportunities = []
        resolved = []

        for i, impact in enumerate(self._active_impacts):
            if impact.symbol != symbol:
                continue

            elapsed = now - impact.timestamp

            # Determine current wave
            wave = self._get_wave(elapsed)
            if wave is None:
                impact.resolved = True
                resolved.append(i)
                continue

            impact.current_wave = wave

            # Calculate deviation
            deviation_pct = (price - impact.initial_price) / impact.initial_price
            impact.max_deviation_pct = max(
                impact.max_deviation_pct, abs(deviation_pct), key=abs
            )

            # Record wave data
            impact.waves[wave.value] = deviation_pct

            # Check for fade opportunity (overreaction in wave 2-3)
            if wave in (ImpactWave.FIRST_TRADERS, ImpactWave.MAIN_WAVE):
                if abs(deviation_pct) > self.OVERREACTION_THRESHOLD:
                    impact.fade_opportunity = True
                    impact.fade_confidence = min(
                        0.95, abs(deviation_pct) / self.OVERREACTION_THRESHOLD * 0.5
                    )
                    opportunities.append(impact)

                    logger.info(
                        f"[Emotion] FADE opportunity on {symbol}: "
                        f"deviation={deviation_pct*100:.2f}%, "
                        f"wave={wave.value}, confidence={impact.fade_confidence:.2f}"
                    )

        # Remove resolved
        for i in sorted(resolved, reverse=True):
            self._active_impacts.pop(i)

        return opportunities

    def _get_wave(self, elapsed_seconds: float) -> ImpactWave | None:
        for wave, (start, end) in self.WAVE_BOUNDARIES.items():
            if start <= elapsed_seconds < end:
                return wave
        return None

    def get_active_impacts(self) -> list[NewsImpact]:
        return [i for i in self._active_impacts if not i.resolved]

    def get_recent_impacts(self, limit: int = 20) -> list[NewsImpact]:
        return list(self._impacts)[-limit:]

    def get_fade_success_rate(self) -> float:
        if self._fade_attempts == 0:
            return 0.0
        return self._fade_successes / self._fade_attempts

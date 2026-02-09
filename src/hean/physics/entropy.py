"""Market Entropy Calculation.

Entropy measures the disorder of volume distribution:
    S = -Sum p_i * log(p_i)
    where p_i = V_i / Sum(V) (normalized volume share)

Low entropy = compressed (coiled spring, breakout imminent)
High entropy = equilibrium (dispersed, no edge)
"""

import math
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EntropyReading:
    """Single entropy reading."""

    value: float
    state: str  # COMPRESSED, NORMAL, EQUILIBRIUM
    n_samples: int
    timestamp: float
    symbol: str
    is_compression: bool = False  # True when entropy drops significantly


class MarketEntropy:
    """Calculate market entropy from volume distribution."""

    # State thresholds
    COMPRESSED_THRESHOLD = 2.0
    EQUILIBRIUM_THRESHOLD = 3.5

    # Compression detection
    COMPRESSION_DROP_RATE = 0.3  # 30% drop from rolling avg

    def __init__(self, window_size: int = 50, history_size: int = 500) -> None:
        self._window_size = window_size
        self._history: dict[str, deque[EntropyReading]] = {}
        self._history_size = history_size
        self._rolling_avg: dict[str, float] = {}

    def calculate(self, volumes: list[float], symbol: str = "UNKNOWN") -> EntropyReading:
        """Calculate market entropy from volume distribution.

        Args:
            volumes: Recent trade volumes
            symbol: Trading symbol

        Returns:
            EntropyReading with value and state
        """
        if not volumes:
            return EntropyReading(
                value=0.0,
                state="COMPRESSED",
                n_samples=0,
                timestamp=time.time(),
                symbol=symbol,
            )

        vol_arr = np.array(volumes, dtype=np.float64)
        vol_arr = vol_arr[vol_arr > 0]  # Filter zero volumes

        if len(vol_arr) == 0:
            return EntropyReading(
                value=0.0,
                state="COMPRESSED",
                n_samples=0,
                timestamp=time.time(),
                symbol=symbol,
            )

        total_volume = np.sum(vol_arr)
        if total_volume == 0:
            return EntropyReading(
                value=0.0,
                state="COMPRESSED",
                n_samples=len(vol_arr),
                timestamp=time.time(),
                symbol=symbol,
            )

        # Calculate entropy: S = -Sum p_i * log(p_i)
        probabilities = vol_arr / total_volume
        entropy = -float(np.sum(probabilities * np.log(probabilities)))

        # Detect state
        if entropy < self.COMPRESSED_THRESHOLD:
            state = "COMPRESSED"
        elif entropy >= self.EQUILIBRIUM_THRESHOLD:
            state = "EQUILIBRIUM"
        else:
            state = "NORMAL"

        # Detect compression (entropy dropping)
        rolling_avg = self._rolling_avg.get(symbol, entropy)
        is_compression = (
            entropy < rolling_avg * (1 - self.COMPRESSION_DROP_RATE)
            and rolling_avg > 0
        )

        # Update rolling average (EMA)
        alpha = 0.1
        self._rolling_avg[symbol] = alpha * entropy + (1 - alpha) * rolling_avg

        reading = EntropyReading(
            value=entropy,
            state=state,
            n_samples=len(vol_arr),
            timestamp=time.time(),
            symbol=symbol,
            is_compression=is_compression,
        )

        # Store history
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._history_size)
        self._history[symbol].append(reading)

        if is_compression:
            logger.info(
                f"[Entropy] COMPRESSION detected {symbol}: S={entropy:.2f} "
                f"(avg={rolling_avg:.2f}, drop={(1-entropy/rolling_avg)*100:.1f}%)"
            )

        logger.debug(
            f"[Entropy] {symbol}: S={entropy:.2f} ({state}), N={len(vol_arr)}"
        )

        return reading

    def get_current(self, symbol: str) -> EntropyReading | None:
        if symbol not in self._history or not self._history[symbol]:
            return None
        return self._history[symbol][-1]

    def get_history(self, symbol: str, limit: int | None = None) -> list[EntropyReading]:
        if symbol not in self._history:
            return []
        history = list(self._history[symbol])
        if limit is not None:
            history = history[-limit:]
        return history

    def reset(self, symbol: str | None = None) -> None:
        if symbol is None:
            self._history.clear()
            self._rolling_avg.clear()
        else:
            self._history.pop(symbol, None)
            self._rolling_avg.pop(symbol, None)

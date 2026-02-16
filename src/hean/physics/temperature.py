"""Market Temperature Calculation.

Temperature measures the kinetic energy of the market:
    T = KE / N
    where KE = Sum (deltaP_i * V_i)^2

Higher temperature = more energy = more volatile.
"""

import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemperatureReading:
    """Single temperature reading."""

    value: float
    regime: str  # HOT, WARM, COLD
    kinetic_energy: float
    n_samples: int
    timestamp: float
    symbol: str
    is_spike: bool = False


class MarketTemperature:
    """Calculate market temperature from price and volume data."""

    # Regime thresholds
    HOT_THRESHOLD = 800.0
    WARM_THRESHOLD = 400.0

    # Spike detection
    SPIKE_MULTIPLIER = 3.0  # 3x rolling average = spike

    def __init__(self, window_size: int = 100, history_size: int = 500) -> None:
        self._window_size = window_size
        self._prices: dict[str, deque[float]] = {}
        self._volumes: dict[str, deque[float]] = {}
        self._history: dict[str, deque[TemperatureReading]] = {}
        self._history_size = history_size
        self._rolling_avg: dict[str, float] = {}

    def calculate(
        self, prices: list[float], volumes: list[float], symbol: str = "UNKNOWN"
    ) -> TemperatureReading:
        """Calculate market temperature.

        Args:
            prices: Recent price values
            volumes: Corresponding volume values
            symbol: Trading symbol

        Returns:
            TemperatureReading with value and regime
        """
        if len(prices) < 2 or len(prices) != len(volumes):
            return TemperatureReading(
                value=0.0,
                regime="COLD",
                kinetic_energy=0.0,
                n_samples=0,
                timestamp=time.time(),
                symbol=symbol,
            )

        # Calculate kinetic energy: KE = Sum (deltaP * V)^2
        price_arr = np.array(prices)
        vol_arr = np.array(volumes)

        delta_p = np.diff(price_arr)
        vol_matched = vol_arr[1:]

        kinetic_energy = float(np.sum((delta_p * vol_matched) ** 2))
        n = len(prices)
        temperature = kinetic_energy / n

        # Detect regime
        if temperature >= self.HOT_THRESHOLD:
            regime = "HOT"
        elif temperature >= self.WARM_THRESHOLD:
            regime = "WARM"
        else:
            regime = "COLD"

        # Detect spikes
        rolling_avg = self._rolling_avg.get(symbol, temperature)
        is_spike = temperature > rolling_avg * self.SPIKE_MULTIPLIER and rolling_avg > 0

        # Update rolling average (EMA)
        alpha = 0.1
        self._rolling_avg[symbol] = alpha * temperature + (1 - alpha) * rolling_avg

        reading = TemperatureReading(
            value=temperature,
            regime=regime,
            kinetic_energy=kinetic_energy,
            n_samples=n,
            timestamp=time.time(),
            symbol=symbol,
            is_spike=is_spike,
        )

        # Store history
        if symbol not in self._history:
            self._history[symbol] = deque(maxlen=self._history_size)
        self._history[symbol].append(reading)

        if is_spike:
            logger.warning(
                f"[Temperature] SPIKE detected {symbol}: T={temperature:.1f} "
                f"(avg={rolling_avg:.1f}, {temperature/rolling_avg:.1f}x)"
            )

        logger.debug(
            f"[Temperature] {symbol}: T={temperature:.1f} ({regime}), "
            f"KE={kinetic_energy:.1f}, N={n}"
        )

        return reading

    def get_current(self, symbol: str) -> TemperatureReading | None:
        if symbol not in self._history or not self._history[symbol]:
            return None
        return self._history[symbol][-1]

    def get_history(self, symbol: str, limit: int | None = None) -> list[TemperatureReading]:
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

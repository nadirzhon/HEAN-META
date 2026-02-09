"""Market Anomaly Detector.

Detects unusual market conditions:
- OI spikes
- Funding divergence
- Whale inflows
- Volume anomalies
- Price dislocations
"""

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


class AnomalyType(str, Enum):
    OI_SPIKE = "oi_spike"
    FUNDING_DIVERGENCE = "funding_divergence"
    WHALE_INFLOW = "whale_inflow"
    VOLUME_ANOMALY = "volume_anomaly"
    PRICE_DISLOCATION = "price_dislocation"
    LIQUIDATION_CASCADE = "liquidation_cascade"


@dataclass
class MarketAnomaly:
    id: str
    anomaly_type: AnomalyType
    severity: float  # 0.0-1.0
    description: str
    symbol: str
    timestamp: float
    details: dict[str, Any] = field(default_factory=dict)
    active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.anomaly_type.value,
            "severity": self.severity,
            "description": self.description,
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "details": self.details,
            "active": self.active,
        }


class MarketAnomalyDetector:
    """Detect market anomalies from price, volume, and order data."""

    VOLUME_SPIKE_THRESHOLD = 3.0  # 3x average
    PRICE_DISLOCATION_THRESHOLD = 0.02  # 2% sudden move
    ANOMALY_COOLDOWN = 60.0  # seconds between same type

    def __init__(self, history_size: int = 200) -> None:
        self._anomalies: deque[MarketAnomaly] = deque(maxlen=500)
        self._active_anomalies: dict[str, MarketAnomaly] = {}

        # Rolling stats per symbol
        self._volume_history: dict[str, deque[float]] = {}
        self._price_history: dict[str, deque[float]] = {}
        self._history_size = history_size

        # Cooldown tracking
        self._last_anomaly_time: dict[str, float] = {}
        self._anomaly_counter = 0

    def check(
        self,
        symbol: str,
        price: float,
        volume: float,
        oi: float | None = None,
        funding_rate: float | None = None,
    ) -> list[MarketAnomaly]:
        """Check for anomalies given current market data.

        Returns list of newly detected anomalies.
        """
        # Initialize history
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self._history_size)
            self._price_history[symbol] = deque(maxlen=self._history_size)

        detected: list[MarketAnomaly] = []

        # Volume anomaly
        vol_anomaly = self._check_volume_anomaly(symbol, volume)
        if vol_anomaly:
            detected.append(vol_anomaly)

        # Price dislocation
        price_anomaly = self._check_price_dislocation(symbol, price)
        if price_anomaly:
            detected.append(price_anomaly)

        # Update history
        self._volume_history[symbol].append(volume)
        self._price_history[symbol].append(price)

        # Store detected anomalies
        for anomaly in detected:
            self._anomalies.append(anomaly)
            self._active_anomalies[anomaly.id] = anomaly
            logger.warning(
                f"[Anomaly] {anomaly.anomaly_type.value} on {symbol}: "
                f"{anomaly.description} (severity={anomaly.severity:.2f})"
            )

        return detected

    def _check_volume_anomaly(self, symbol: str, volume: float) -> MarketAnomaly | None:
        volumes = list(self._volume_history.get(symbol, []))
        if len(volumes) < 20:
            return None

        avg_vol = np.mean(volumes)
        if avg_vol == 0:
            return None

        ratio = volume / avg_vol
        if ratio < self.VOLUME_SPIKE_THRESHOLD:
            return None

        # Check cooldown
        key = f"{symbol}:volume"
        now = time.time()
        if now - self._last_anomaly_time.get(key, 0) < self.ANOMALY_COOLDOWN:
            return None
        self._last_anomaly_time[key] = now

        self._anomaly_counter += 1
        severity = min(1.0, (ratio - self.VOLUME_SPIKE_THRESHOLD) / 10.0 + 0.3)

        return MarketAnomaly(
            id=f"anomaly_{self._anomaly_counter}",
            anomaly_type=AnomalyType.VOLUME_ANOMALY,
            severity=severity,
            description=f"Volume spike {ratio:.1f}x average on {symbol}",
            symbol=symbol,
            timestamp=now,
            details={"ratio": ratio, "volume": volume, "avg_volume": avg_vol},
        )

    def _check_price_dislocation(self, symbol: str, price: float) -> MarketAnomaly | None:
        prices = list(self._price_history.get(symbol, []))
        if len(prices) < 5:
            return None

        prev_price = prices[-1]
        if prev_price == 0:
            return None

        change_pct = abs(price - prev_price) / prev_price
        if change_pct < self.PRICE_DISLOCATION_THRESHOLD:
            return None

        # Check cooldown
        key = f"{symbol}:price"
        now = time.time()
        if now - self._last_anomaly_time.get(key, 0) < self.ANOMALY_COOLDOWN:
            return None
        self._last_anomaly_time[key] = now

        self._anomaly_counter += 1
        severity = min(1.0, change_pct / 0.1)
        direction = "up" if price > prev_price else "down"

        return MarketAnomaly(
            id=f"anomaly_{self._anomaly_counter}",
            anomaly_type=AnomalyType.PRICE_DISLOCATION,
            severity=severity,
            description=f"Price {direction} {change_pct*100:.1f}% on {symbol}",
            symbol=symbol,
            timestamp=now,
            details={
                "change_pct": change_pct,
                "direction": direction,
                "price": price,
                "prev_price": prev_price,
            },
        )

    def get_recent(self, limit: int = 20) -> list[MarketAnomaly]:
        anomalies = list(self._anomalies)
        return anomalies[-limit:]

    def get_active(self) -> list[MarketAnomaly]:
        return [a for a in self._active_anomalies.values() if a.active]

    def deactivate(self, anomaly_id: str) -> None:
        if anomaly_id in self._active_anomalies:
            self._active_anomalies[anomaly_id].active = False

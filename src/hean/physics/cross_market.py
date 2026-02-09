"""Cross-Market Impulse Detection.

Detects BTC -> ETH -> alts propagation patterns.
Learns propagation delays in real-time.
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ImpulseEvent:
    source_symbol: str
    source_price_change_pct: float
    timestamp: float
    propagated_to: dict[str, float] = field(default_factory=dict)  # symbol -> delay_ms
    predicted_targets: list[str] = field(default_factory=list)


@dataclass
class PropagationStats:
    source: str
    target: str
    avg_delay_ms: float
    correlation: float
    sample_count: int


class CrossMarketImpulse:
    """Detect and predict cross-market impulse propagation."""

    SIGNIFICANT_MOVE_PCT = 0.003  # 0.3% = significant move
    MAX_PROPAGATION_DELAY_MS = 5000  # 5 seconds max
    MIN_SAMPLES = 10

    def __init__(
        self,
        leader_symbols: list[str] | None = None,
        follower_symbols: list[str] | None = None,
    ):
        self.leaders = leader_symbols or ["BTCUSDT"]
        self.followers = follower_symbols or ["ETHUSDT", "SOLUSDT"]

        # Price history per symbol: (timestamp, price)
        self._prices: dict[str, deque] = {}
        for s in self.leaders + self.followers:
            self._prices[s] = deque(maxlen=5000)

        # Propagation delay history: (source, target) -> delays in ms
        self._delays: dict[tuple[str, str], deque[float]] = {}
        for leader in self.leaders:
            for follower in self.followers:
                self._delays[(leader, follower)] = deque(maxlen=200)

        # Recent impulse events
        self._impulses: deque[ImpulseEvent] = deque(maxlen=100)

        # Pending impulses waiting for propagation
        self._pending: list[ImpulseEvent] = []

    def update(self, symbol: str, price: float) -> ImpulseEvent | None:
        """Update with new price data. Returns ImpulseEvent if significant move detected."""
        now = time.time()

        if symbol not in self._prices:
            self._prices[symbol] = deque(maxlen=5000)

        self._prices[symbol].append((now, price))

        # Check for propagation of pending impulses
        self._check_propagations(symbol, price, now)

        # Check if this is a significant move on a leader
        if symbol in self.leaders:
            impulse = self._check_leader_impulse(symbol, price, now)
            if impulse:
                self._pending.append(impulse)
                self._impulses.append(impulse)
                logger.info(
                    f"[Impulse] {symbol} moved {impulse.source_price_change_pct*100:.2f}% "
                    f"- watching for propagation to {self.followers}"
                )
                return impulse

        return None

    def _check_leader_impulse(
        self, symbol: str, price: float, now: float
    ) -> ImpulseEvent | None:
        prices = self._prices[symbol]
        if len(prices) < 5:
            return None

        # Compare to price 1 second ago
        lookback_prices = [(t, p) for t, p in prices if now - t < 1.0]
        if not lookback_prices:
            return None

        old_price = lookback_prices[0][1]
        if old_price == 0:
            return None

        change_pct = (price - old_price) / old_price

        if abs(change_pct) < self.SIGNIFICANT_MOVE_PCT:
            return None

        return ImpulseEvent(
            source_symbol=symbol,
            source_price_change_pct=change_pct,
            timestamp=now,
            predicted_targets=list(self.followers),
        )

    def _check_propagations(self, symbol: str, price: float, now: float) -> None:
        """Check if pending impulses have propagated to this symbol."""
        if symbol not in self.followers:
            return

        resolved = []
        for i, impulse in enumerate(self._pending):
            delay_ms = (now - impulse.timestamp) * 1000

            if delay_ms > self.MAX_PROPAGATION_DELAY_MS:
                resolved.append(i)
                continue

            # Check if this symbol shows correlated move
            follower_prices = self._prices[symbol]
            if len(follower_prices) < 3:
                continue

            ref_time = impulse.timestamp
            ref_prices = [(t, p) for t, p in follower_prices if t <= ref_time]
            if not ref_prices:
                continue

            ref_price = ref_prices[-1][1]
            if ref_price == 0:
                continue

            follower_change = (price - ref_price) / ref_price

            # Same direction and significant
            if (
                abs(follower_change) >= self.SIGNIFICANT_MOVE_PCT * 0.5
                and np.sign(follower_change) == np.sign(impulse.source_price_change_pct)
            ):
                impulse.propagated_to[symbol] = delay_ms
                key = (impulse.source_symbol, symbol)
                self._delays[key].append(delay_ms)

                logger.info(
                    f"[Impulse] Propagation: {impulse.source_symbol} -> {symbol} "
                    f"in {delay_ms:.0f}ms (change={follower_change*100:.2f}%)"
                )

        # Remove resolved
        for i in sorted(resolved, reverse=True):
            self._pending.pop(i)

    def get_propagation_stats(self) -> list[PropagationStats]:
        """Get propagation statistics for all leader->follower pairs."""
        stats = []
        for (source, target), delays in self._delays.items():
            delay_list = list(delays)
            if len(delay_list) < self.MIN_SAMPLES:
                continue
            stats.append(PropagationStats(
                source=source,
                target=target,
                avg_delay_ms=float(np.mean(delay_list)),
                correlation=0.7,  # Simplified
                sample_count=len(delay_list),
            ))
        return stats

    def get_recent_impulses(self, limit: int = 20) -> list[ImpulseEvent]:
        return list(self._impulses)[-limit:]

    def get_optimal_entry_delay(self, source: str, target: str) -> float | None:
        """Get optimal entry delay in ms for a given pair."""
        key = (source, target)
        delays = list(self._delays.get(key, []))
        if len(delays) < self.MIN_SAMPLES:
            return None
        return float(np.percentile(delays, 25))  # Enter at 25th percentile

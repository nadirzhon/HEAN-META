"""Enhanced Adaptive TTL for Maker Orders.

Improvements over Phase 1:
1. Spread-based TTL adjustment (wider spread = longer TTL)
2. Time-of-day pattern learning (different TTL for different hours)
3. Volatility-aware scaling
4. Historical fill pattern learning
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FillPatternStats:
    """Statistics for fill patterns at a specific hour."""

    hour: int
    fills: int = 0
    expirations: int = 0
    total_attempts: int = 0
    avg_fill_time_ms: float = 0.0
    fill_rate: float = 0.0

    # Running stats for fill times
    fill_times: deque = field(default_factory=lambda: deque(maxlen=50))

    def update_fill(self, fill_time_ms: float) -> None:
        """Update with a successful fill."""
        self.fills += 1
        self.total_attempts += 1
        self.fill_times.append(fill_time_ms)
        self._recalculate()

    def update_expiration(self) -> None:
        """Update with an expiration."""
        self.expirations += 1
        self.total_attempts += 1
        self._recalculate()

    def _recalculate(self) -> None:
        """Recalculate derived stats."""
        if self.total_attempts > 0:
            self.fill_rate = self.fills / self.total_attempts

        if len(self.fill_times) > 0:
            self.avg_fill_time_ms = sum(self.fill_times) / len(self.fill_times)


@dataclass
class SpreadBucket:
    """Statistics for fills at different spread levels."""

    spread_bps_min: float
    spread_bps_max: float
    fills: int = 0
    expirations: int = 0
    optimal_ttl_ms: float = 0.0

    def update_fill(self, ttl_ms: float) -> None:
        """Update with a successful fill."""
        self.fills += 1
        # Update optimal TTL (exponential moving average)
        if self.optimal_ttl_ms == 0:
            self.optimal_ttl_ms = ttl_ms
        else:
            self.optimal_ttl_ms = 0.7 * self.optimal_ttl_ms + 0.3 * ttl_ms

    def update_expiration(self) -> None:
        """Update with an expiration."""
        self.expirations += 1

    def get_fill_rate(self) -> float:
        """Get fill rate for this spread bucket."""
        total = self.fills + self.expirations
        return self.fills / total if total > 0 else 0.0


class EnhancedAdaptiveTTL:
    """Enhanced adaptive TTL calculator with multi-factor learning.

    Learns optimal TTL based on:
    - Time of day (hourly patterns)
    - Spread levels (tight vs wide spread)
    - Volatility regime
    - Recent fill patterns
    """

    # Spread buckets (in basis points)
    SPREAD_BUCKETS = [
        (0.0, 2.0),    # Very tight
        (2.0, 5.0),    # Tight
        (5.0, 10.0),   # Normal
        (10.0, 20.0),  # Wide
        (20.0, 100.0), # Very wide
    ]

    def __init__(
        self,
        base_ttl_ms: float = 500.0,
        min_ttl_ms: float = 200.0,
        max_ttl_ms: float = 3000.0,
        learning_rate: float = 0.1,
    ):
        """Initialize enhanced adaptive TTL.

        Args:
            base_ttl_ms: Base TTL in milliseconds
            min_ttl_ms: Minimum TTL
            max_ttl_ms: Maximum TTL
            learning_rate: How fast to adapt (0.0 to 1.0)
        """
        self._base_ttl_ms = base_ttl_ms
        self._min_ttl_ms = min_ttl_ms
        self._max_ttl_ms = max_ttl_ms
        self._learning_rate = max(0.0, min(1.0, learning_rate))

        # Time-of-day patterns (24 hours)
        self._hourly_stats: dict[int, FillPatternStats] = {
            hour: FillPatternStats(hour=hour) for hour in range(24)
        }

        # Spread-based patterns
        self._spread_buckets: list[SpreadBucket] = [
            SpreadBucket(spread_bps_min=min_bps, spread_bps_max=max_bps)
            for min_bps, max_bps in self.SPREAD_BUCKETS
        ]

        # Volatility-based adjustments
        self._volatility_multipliers: dict[str, float] = {
            "low": 0.8,      # Reduce TTL in low volatility (faster market)
            "medium": 1.0,   # Normal
            "high": 1.3,     # Increase TTL in high volatility (wait for better fills)
            "extreme": 1.5,  # Further increase in extreme volatility
        }

        # Current adaptive TTL
        self._current_ttl_ms = base_ttl_ms

        # Recent performance tracking
        self._recent_fills: deque = deque(maxlen=100)
        self._recent_expirations: deque = deque(maxlen=100)

        logger.info(
            f"EnhancedAdaptiveTTL initialized: base={base_ttl_ms}ms, "
            f"range=[{min_ttl_ms}, {max_ttl_ms}]ms, learning_rate={learning_rate}"
        )

    def calculate_ttl(
        self,
        symbol: str,
        spread_bps: float,
        volatility_regime: str = "medium",
        current_hour: int | None = None,
    ) -> float:
        """Calculate optimal TTL for current conditions.

        Args:
            symbol: Trading symbol
            spread_bps: Current bid-ask spread in basis points
            volatility_regime: Volatility regime (low/medium/high/extreme)
            current_hour: Current hour UTC (0-23), uses now if None

        Returns:
            Optimal TTL in milliseconds
        """
        if current_hour is None:
            current_hour = datetime.utcnow().hour

        # Start with base TTL
        ttl_ms = self._base_ttl_ms

        # Adjust for time of day
        hourly_adjustment = self._get_hourly_adjustment(current_hour)
        ttl_ms *= hourly_adjustment

        # Adjust for spread
        spread_adjustment = self._get_spread_adjustment(spread_bps)
        ttl_ms *= spread_adjustment

        # Adjust for volatility
        vol_multiplier = self._volatility_multipliers.get(volatility_regime, 1.0)
        ttl_ms *= vol_multiplier

        # Clamp to bounds
        ttl_ms = max(self._min_ttl_ms, min(self._max_ttl_ms, ttl_ms))

        self._current_ttl_ms = ttl_ms

        logger.debug(
            f"Calculated TTL: {ttl_ms:.0f}ms "
            f"(hour_adj={hourly_adjustment:.2f}, spread_adj={spread_adjustment:.2f}, "
            f"vol_mult={vol_multiplier:.2f})"
        )

        return ttl_ms

    def record_fill(
        self,
        symbol: str,
        spread_bps: float,
        fill_time_ms: float,
        hour: int | None = None,
    ) -> None:
        """Record a successful maker fill for learning.

        Args:
            symbol: Trading symbol
            spread_bps: Spread at fill time
            fill_time_ms: Time to fill in milliseconds
            hour: Hour of fill (uses now if None)
        """
        if hour is None:
            hour = datetime.utcnow().hour

        # Update hourly stats
        self._hourly_stats[hour].update_fill(fill_time_ms)

        # Update spread bucket stats
        bucket = self._get_spread_bucket(spread_bps)
        if bucket:
            bucket.update_fill(fill_time_ms)

        # Track recent fills
        self._recent_fills.append({
            "symbol": symbol,
            "spread_bps": spread_bps,
            "fill_time_ms": fill_time_ms,
            "hour": hour,
            "timestamp": datetime.utcnow(),
        })

        logger.debug(
            f"Recorded fill: symbol={symbol}, spread={spread_bps:.2f}bps, "
            f"fill_time={fill_time_ms:.0f}ms, hour={hour}"
        )

    def record_expiration(
        self,
        symbol: str,
        spread_bps: float,
        hour: int | None = None,
    ) -> None:
        """Record a maker order expiration for learning.

        Args:
            symbol: Trading symbol
            spread_bps: Spread at expiration
            hour: Hour of expiration (uses now if None)
        """
        if hour is None:
            hour = datetime.utcnow().hour

        # Update hourly stats
        self._hourly_stats[hour].update_expiration()

        # Update spread bucket stats
        bucket = self._get_spread_bucket(spread_bps)
        if bucket:
            bucket.update_expiration()

        # Track recent expirations
        self._recent_expirations.append({
            "symbol": symbol,
            "spread_bps": spread_bps,
            "hour": hour,
            "timestamp": datetime.utcnow(),
        })

        logger.debug(
            f"Recorded expiration: symbol={symbol}, spread={spread_bps:.2f}bps, hour={hour}"
        )

    def _get_hourly_adjustment(self, hour: int) -> float:
        """Get TTL adjustment for specific hour.

        Args:
            hour: Hour of day (0-23)

        Returns:
            TTL multiplier (0.5 to 2.0)
        """
        stats = self._hourly_stats[hour]

        if stats.total_attempts < 5:
            # Insufficient data - use baseline
            return 1.0

        # Adjust based on fill rate
        # High fill rate (>70%) = can reduce TTL (more aggressive)
        # Low fill rate (<30%) = increase TTL (more patient)
        fill_rate = stats.fill_rate

        if fill_rate > 0.70:
            # High fill rate - reduce TTL
            adjustment = 0.8
        elif fill_rate < 0.30:
            # Low fill rate - increase TTL
            adjustment = 1.3
        else:
            # Medium fill rate - maintain or slight increase
            # Interpolate between 0.9 and 1.1
            adjustment = 0.9 + (fill_rate - 0.3) * 0.5

        # Also consider average fill time
        if stats.avg_fill_time_ms > 0:
            # If average fill time is much less than base TTL, can reduce
            fill_time_ratio = stats.avg_fill_time_ms / self._base_ttl_ms
            if fill_time_ratio < 0.5:
                # Fills happening very quickly - reduce TTL
                adjustment *= 0.9
            elif fill_time_ratio > 0.9:
                # Fills taking almost full TTL - increase TTL
                adjustment *= 1.1

        return max(0.5, min(2.0, adjustment))

    def _get_spread_adjustment(self, spread_bps: float) -> float:
        """Get TTL adjustment based on spread.

        Wider spreads = longer TTL (more time to get filled).
        Tighter spreads = shorter TTL (fills happen faster).

        Args:
            spread_bps: Bid-ask spread in basis points

        Returns:
            TTL multiplier (0.7 to 1.5)
        """
        bucket = self._get_spread_bucket(spread_bps)
        if not bucket:
            return 1.0

        # Use learned optimal TTL if available
        if bucket.fills > 5 and bucket.optimal_ttl_ms > 0:
            optimal_ratio = bucket.optimal_ttl_ms / self._base_ttl_ms
            return max(0.7, min(1.5, optimal_ratio))

        # Fallback: heuristic based on spread
        if spread_bps < 2.0:
            # Very tight spread - reduce TTL
            return 0.8
        elif spread_bps < 5.0:
            # Tight spread - slightly reduce
            return 0.9
        elif spread_bps < 10.0:
            # Normal spread - baseline
            return 1.0
        elif spread_bps < 20.0:
            # Wide spread - increase TTL
            return 1.2
        else:
            # Very wide spread - significant increase
            return 1.4

    def _get_spread_bucket(self, spread_bps: float) -> SpreadBucket | None:
        """Get the appropriate spread bucket for a given spread.

        Args:
            spread_bps: Bid-ask spread in basis points

        Returns:
            SpreadBucket instance or None
        """
        for bucket in self._spread_buckets:
            if bucket.spread_bps_min <= spread_bps < bucket.spread_bps_max:
                return bucket
        return None

    def get_current_ttl(self) -> float:
        """Get current adaptive TTL."""
        return self._current_ttl_ms

    def get_statistics(self) -> dict[str, Any]:
        """Get adaptive TTL statistics.

        Returns:
            Dictionary of statistics
        """
        # Calculate overall fill rate
        total_fills = sum(1 for f in self._recent_fills)
        total_expirations = len(self._recent_expirations)
        total_attempts = total_fills + total_expirations
        overall_fill_rate = total_fills / total_attempts if total_attempts > 0 else 0.0

        # Calculate average fill time
        if self._recent_fills:
            avg_fill_time = sum(f["fill_time_ms"] for f in self._recent_fills) / len(self._recent_fills)
        else:
            avg_fill_time = 0.0

        # Hourly stats summary
        hourly_fill_rates = {
            hour: stats.fill_rate
            for hour, stats in self._hourly_stats.items()
            if stats.total_attempts > 0
        }

        # Best and worst hours
        if hourly_fill_rates:
            best_hour = max(hourly_fill_rates.items(), key=lambda x: x[1])[0]
            worst_hour = min(hourly_fill_rates.items(), key=lambda x: x[1])[0]
        else:
            best_hour = None
            worst_hour = None

        # Spread bucket stats
        spread_stats = [
            {
                "range_bps": f"{b.spread_bps_min:.1f}-{b.spread_bps_max:.1f}",
                "fill_rate": b.get_fill_rate(),
                "fills": b.fills,
                "optimal_ttl_ms": b.optimal_ttl_ms,
            }
            for b in self._spread_buckets
            if b.fills + b.expirations > 0
        ]

        return {
            "current_ttl_ms": self._current_ttl_ms,
            "base_ttl_ms": self._base_ttl_ms,
            "overall_fill_rate": overall_fill_rate,
            "avg_fill_time_ms": avg_fill_time,
            "total_fills": total_fills,
            "total_expirations": total_expirations,
            "best_hour": best_hour,
            "worst_hour": worst_hour,
            "hourly_fill_rates": hourly_fill_rates,
            "spread_bucket_stats": spread_stats,
        }

    def reset_learning(self) -> None:
        """Reset all learned patterns (use with caution)."""
        for stats in self._hourly_stats.values():
            stats.fills = 0
            stats.expirations = 0
            stats.total_attempts = 0
            stats.fill_times.clear()

        for bucket in self._spread_buckets:
            bucket.fills = 0
            bucket.expirations = 0
            bucket.optimal_ttl_ms = 0.0

        self._recent_fills.clear()
        self._recent_expirations.clear()

        logger.warning("Adaptive TTL learning data reset")

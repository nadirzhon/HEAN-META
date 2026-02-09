"""Order Timing Optimization.

Optimizes order placement timing based on:
- Market microstructure patterns
- High-liquidity windows
- Funding rate timing (8-hour cycles)
- Volume profile analysis

This module helps maximize fill rates and minimize slippage
by timing orders during optimal liquidity conditions.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from hean.logging import get_logger

logger = get_logger(__name__)


class LiquidityPhase(str, Enum):
    """Market liquidity phases."""
    HIGH = "HIGH"        # Asian/London/NY overlap - best execution
    MEDIUM = "MEDIUM"    # Single session active
    LOW = "LOW"          # Between sessions
    FUNDING = "FUNDING"  # Near funding rate settlement (high activity)


@dataclass
class TimingRecommendation:
    """Order timing recommendation."""
    phase: LiquidityPhase
    size_multiplier: float  # Recommended size adjustment
    urgency: str  # "immediate", "wait", "optimal"
    wait_minutes: int  # How long to wait if urgency is "wait"
    reason: str
    next_optimal_time: datetime | None = None


class OrderTimingOptimizer:
    """Optimizes order timing for best execution.

    Analyzes market microstructure and timing patterns to recommend
    optimal order placement times.
    """

    # Funding settlement times (UTC) - every 8 hours
    FUNDING_TIMES = [0, 8, 16]  # 00:00, 08:00, 16:00 UTC

    # High liquidity windows (UTC)
    # These correspond to major market overlaps
    HIGH_LIQUIDITY_WINDOWS = [
        (7, 10),   # London open
        (13, 16),  # London/NY overlap - BEST
        (20, 23),  # Asia open
    ]

    # Medium liquidity windows
    MEDIUM_LIQUIDITY_WINDOWS = [
        (0, 7),    # Asia session
        (10, 13),  # London morning
        (16, 20),  # NY afternoon
    ]

    # Funding window: 15 minutes before/after funding time
    FUNDING_WINDOW_MINUTES = 15

    def __init__(self):
        """Initialize order timing optimizer."""
        # Track volume patterns
        self._volume_by_hour: dict[int, deque] = {h: deque(maxlen=100) for h in range(24)}
        self._recent_fills: deque = deque(maxlen=100)

        # Track fill quality by hour
        self._fill_quality_by_hour: dict[int, deque] = {h: deque(maxlen=50) for h in range(24)}

        logger.info("OrderTimingOptimizer initialized")

    def get_current_phase(self, now: datetime | None = None) -> LiquidityPhase:
        """Get current liquidity phase.

        Args:
            now: Current time (uses UTC now if None)

        Returns:
            Current liquidity phase
        """
        if now is None:
            now = datetime.utcnow()

        hour = now.hour
        _ = hour  # Used for reference only

        # Check if near funding time
        for funding_hour in self.FUNDING_TIMES:
            # Check if within 15 minutes of funding
            funding_time = now.replace(hour=funding_hour, minute=0, second=0)

            # Handle day boundary
            diff_minutes = abs((now - funding_time).total_seconds() / 60)
            if diff_minutes <= self.FUNDING_WINDOW_MINUTES:
                return LiquidityPhase.FUNDING

            # Also check previous day's late funding
            yesterday_funding = funding_time - timedelta(days=1)
            diff_minutes_prev = abs((now - yesterday_funding).total_seconds() / 60)
            if diff_minutes_prev <= self.FUNDING_WINDOW_MINUTES:
                return LiquidityPhase.FUNDING

        # Check high liquidity windows
        for start, end in self.HIGH_LIQUIDITY_WINDOWS:
            if start <= hour < end:
                return LiquidityPhase.HIGH

        # Check medium liquidity windows
        for start, end in self.MEDIUM_LIQUIDITY_WINDOWS:
            if start <= hour < end:
                return LiquidityPhase.MEDIUM

        return LiquidityPhase.LOW

    def get_timing_recommendation(
        self,
        symbol: str,
        side: str,
        is_urgent: bool = False,
        now: datetime | None = None,
    ) -> TimingRecommendation:
        """Get timing recommendation for an order.

        Args:
            symbol: Trading symbol
            side: Order side ("buy" or "sell")
            is_urgent: Whether order is time-sensitive
            now: Current time (uses UTC now if None)

        Returns:
            TimingRecommendation with optimal timing advice
        """
        if now is None:
            now = datetime.utcnow()

        phase = self.get_current_phase(now)

        # Urgent orders: always execute immediately
        if is_urgent:
            return TimingRecommendation(
                phase=phase,
                size_multiplier=self._get_phase_size_multiplier(phase),
                urgency="immediate",
                wait_minutes=0,
                reason="urgent_order",
            )

        # Phase-specific recommendations
        if phase == LiquidityPhase.HIGH:
            return TimingRecommendation(
                phase=phase,
                size_multiplier=1.2,  # Can size up in high liquidity
                urgency="optimal",
                wait_minutes=0,
                reason="high_liquidity_window",
            )

        elif phase == LiquidityPhase.FUNDING:
            # Near funding: can be volatile but high activity
            # Recommend smaller size or waiting
            next_safe = self._next_non_funding_time(now)
            wait_minutes = int((next_safe - now).total_seconds() / 60) if next_safe else 20

            return TimingRecommendation(
                phase=phase,
                size_multiplier=0.7,  # Reduce size near funding
                urgency="wait" if not is_urgent else "immediate",
                wait_minutes=wait_minutes,
                reason="near_funding_settlement",
                next_optimal_time=next_safe,
            )

        elif phase == LiquidityPhase.MEDIUM:
            return TimingRecommendation(
                phase=phase,
                size_multiplier=1.0,  # Normal size
                urgency="optimal",
                wait_minutes=0,
                reason="medium_liquidity",
            )

        else:  # LOW
            # Low liquidity: wait for better conditions if possible
            next_high = self._next_high_liquidity_time(now)
            wait_minutes = int((next_high - now).total_seconds() / 60) if next_high else 60

            # If wait is too long, execute anyway with reduced size
            if wait_minutes > 60:
                return TimingRecommendation(
                    phase=phase,
                    size_multiplier=0.8,
                    urgency="immediate",
                    wait_minutes=0,
                    reason="low_liquidity_but_long_wait",
                )

            return TimingRecommendation(
                phase=phase,
                size_multiplier=0.8,  # Reduced size in low liquidity
                urgency="wait",
                wait_minutes=min(wait_minutes, 30),  # Max 30 min wait
                reason="low_liquidity",
                next_optimal_time=next_high,
            )

    def _get_phase_size_multiplier(self, phase: LiquidityPhase) -> float:
        """Get size multiplier for a liquidity phase."""
        multipliers = {
            LiquidityPhase.HIGH: 1.2,
            LiquidityPhase.MEDIUM: 1.0,
            LiquidityPhase.LOW: 0.8,
            LiquidityPhase.FUNDING: 0.7,
        }
        return multipliers.get(phase, 1.0)

    def _next_high_liquidity_time(self, now: datetime) -> datetime | None:
        """Find next high liquidity window start."""
        for hours_ahead in range(1, 25):
            check_time = now + timedelta(hours=hours_ahead)
            check_hour = check_time.hour

            for start, _end in self.HIGH_LIQUIDITY_WINDOWS:
                if check_hour == start:
                    return check_time.replace(minute=0, second=0, microsecond=0)

        return None

    def _next_non_funding_time(self, now: datetime) -> datetime | None:
        """Find next time that's outside funding window."""
        for minutes_ahead in range(1, 60):
            check_time = now + timedelta(minutes=minutes_ahead)

            is_funding = False
            for funding_hour in self.FUNDING_TIMES:
                funding_time = check_time.replace(
                    hour=funding_hour, minute=0, second=0
                )
                diff = abs((check_time - funding_time).total_seconds() / 60)
                if diff <= self.FUNDING_WINDOW_MINUTES:
                    is_funding = True
                    break

            if not is_funding:
                return check_time

        return now + timedelta(minutes=self.FUNDING_WINDOW_MINUTES + 1)

    def record_fill(
        self,
        symbol: str,
        side: str,
        fill_quality: float,  # 0.0 = bad slippage, 1.0 = good fill
        now: datetime | None = None,
    ) -> None:
        """Record a fill for learning optimal timing.

        Args:
            symbol: Trading symbol
            side: Order side
            fill_quality: Fill quality score (0.0 to 1.0)
            now: Fill time (uses UTC now if None)
        """
        if now is None:
            now = datetime.utcnow()

        hour = now.hour

        # Record fill quality for this hour
        self._fill_quality_by_hour[hour].append(fill_quality)

        # Record fill info
        self._recent_fills.append({
            "symbol": symbol,
            "side": side,
            "quality": fill_quality,
            "hour": hour,
            "phase": self.get_current_phase(now).value,
            "time": now,
        })

    def get_statistics(self) -> dict:
        """Get timing statistics."""
        # Calculate average fill quality by hour
        quality_by_hour = {}
        for hour in range(24):
            fills = list(self._fill_quality_by_hour[hour])
            if fills:
                quality_by_hour[hour] = sum(fills) / len(fills)
            else:
                quality_by_hour[hour] = None

        # Find best and worst hours
        valid_hours = [(h, q) for h, q in quality_by_hour.items() if q is not None]
        best_hours = sorted(valid_hours, key=lambda x: x[1], reverse=True)[:3]
        worst_hours = sorted(valid_hours, key=lambda x: x[1])[:3]

        return {
            "current_phase": self.get_current_phase().value,
            "total_fills_tracked": len(self._recent_fills),
            "quality_by_hour": quality_by_hour,
            "best_hours": [h for h, q in best_hours],
            "worst_hours": [h for h, q in worst_hours],
            "funding_times_utc": self.FUNDING_TIMES,
        }

    def should_delay_order(
        self,
        symbol: str,
        side: str,
        max_delay_minutes: int = 30,
    ) -> tuple[bool, int]:
        """Check if order should be delayed for better execution.

        Args:
            symbol: Trading symbol
            side: Order side
            max_delay_minutes: Maximum acceptable delay

        Returns:
            Tuple of (should_delay, delay_minutes)
        """
        rec = self.get_timing_recommendation(symbol, side)

        if rec.urgency == "wait" and rec.wait_minutes <= max_delay_minutes:
            return True, rec.wait_minutes

        return False, 0

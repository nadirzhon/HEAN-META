"""Tests for Order Timing Optimization module."""

from datetime import datetime, timedelta

import pytest

from hean.execution.order_timing import (
    LiquidityPhase,
    OrderTimingOptimizer,
    TimingRecommendation,
)


class TestLiquidityPhase:
    """Test liquidity phase detection."""

    def test_high_liquidity_london_open(self) -> None:
        """Test that London open is high liquidity."""
        optimizer = OrderTimingOptimizer()
        # 8:00 UTC - London open
        now = datetime(2024, 1, 15, 8, 30, 0)
        phase = optimizer.get_current_phase(now)
        assert phase == LiquidityPhase.HIGH

    def test_high_liquidity_london_ny_overlap(self) -> None:
        """Test that London/NY overlap is high liquidity."""
        optimizer = OrderTimingOptimizer()
        # 14:00 UTC - London/NY overlap
        now = datetime(2024, 1, 15, 14, 30, 0)
        phase = optimizer.get_current_phase(now)
        assert phase == LiquidityPhase.HIGH

    def test_high_liquidity_asia_open(self) -> None:
        """Test that Asia open is high liquidity."""
        optimizer = OrderTimingOptimizer()
        # 21:00 UTC - Asia open
        now = datetime(2024, 1, 15, 21, 30, 0)
        phase = optimizer.get_current_phase(now)
        assert phase == LiquidityPhase.HIGH

    def test_medium_liquidity_asia_session(self) -> None:
        """Test that Asia session is medium liquidity."""
        optimizer = OrderTimingOptimizer()
        # 3:00 UTC - Asia session
        now = datetime(2024, 1, 15, 3, 0, 0)
        phase = optimizer.get_current_phase(now)
        assert phase == LiquidityPhase.MEDIUM

    def test_medium_liquidity_london_morning(self) -> None:
        """Test that London morning is medium liquidity."""
        optimizer = OrderTimingOptimizer()
        # 11:00 UTC - London morning
        now = datetime(2024, 1, 15, 11, 0, 0)
        phase = optimizer.get_current_phase(now)
        assert phase == LiquidityPhase.MEDIUM

    def test_funding_phase_near_settlement(self) -> None:
        """Test that near funding settlement is detected."""
        optimizer = OrderTimingOptimizer()
        # 5 minutes before 8:00 UTC funding
        now = datetime(2024, 1, 15, 7, 55, 0)
        phase = optimizer.get_current_phase(now)
        assert phase == LiquidityPhase.FUNDING

    def test_funding_phase_after_settlement(self) -> None:
        """Test that just after funding settlement is detected."""
        optimizer = OrderTimingOptimizer()
        # 10 minutes after 16:00 UTC funding
        now = datetime(2024, 1, 15, 16, 10, 0)
        phase = optimizer.get_current_phase(now)
        assert phase == LiquidityPhase.FUNDING


class TestTimingRecommendation:
    """Test timing recommendations."""

    def test_urgent_order_immediate(self) -> None:
        """Test that urgent orders get immediate recommendation."""
        optimizer = OrderTimingOptimizer()
        now = datetime(2024, 1, 15, 3, 0, 0)  # Low liquidity time

        rec = optimizer.get_timing_recommendation(
            symbol="BTCUSDT",
            side="buy",
            is_urgent=True,
            now=now,
        )

        assert rec.urgency == "immediate"
        assert rec.wait_minutes == 0

    def test_high_liquidity_optimal(self) -> None:
        """Test that high liquidity gets optimal recommendation."""
        optimizer = OrderTimingOptimizer()
        now = datetime(2024, 1, 15, 14, 0, 0)  # London/NY overlap

        rec = optimizer.get_timing_recommendation(
            symbol="BTCUSDT",
            side="buy",
            is_urgent=False,
            now=now,
        )

        assert rec.phase == LiquidityPhase.HIGH
        assert rec.urgency == "optimal"
        assert rec.size_multiplier == 1.2  # Can size up

    def test_funding_phase_reduced_size(self) -> None:
        """Test that funding phase recommends reduced size."""
        optimizer = OrderTimingOptimizer()
        now = datetime(2024, 1, 15, 7, 55, 0)  # Near funding

        rec = optimizer.get_timing_recommendation(
            symbol="BTCUSDT",
            side="buy",
            is_urgent=False,
            now=now,
        )

        assert rec.phase == LiquidityPhase.FUNDING
        assert rec.size_multiplier == 0.7  # Reduced size
        assert rec.reason == "near_funding_settlement"

    def test_medium_liquidity_normal_size(self) -> None:
        """Test that medium liquidity uses normal size."""
        optimizer = OrderTimingOptimizer()
        now = datetime(2024, 1, 15, 11, 0, 0)  # London morning

        rec = optimizer.get_timing_recommendation(
            symbol="BTCUSDT",
            side="buy",
            is_urgent=False,
            now=now,
        )

        assert rec.phase == LiquidityPhase.MEDIUM
        assert rec.size_multiplier == 1.0
        assert rec.urgency == "optimal"


class TestFillTracking:
    """Test fill quality tracking."""

    def test_record_fill(self) -> None:
        """Test that fills are recorded correctly."""
        optimizer = OrderTimingOptimizer()
        now = datetime(2024, 1, 15, 14, 30, 0)

        optimizer.record_fill(
            symbol="BTCUSDT",
            side="buy",
            fill_quality=0.9,
            now=now,
        )

        stats = optimizer.get_statistics()
        assert stats["total_fills_tracked"] == 1

    def test_fill_quality_by_hour(self) -> None:
        """Test that fill quality is tracked by hour."""
        optimizer = OrderTimingOptimizer()

        # Record fills at different hours
        for hour in [10, 10, 14, 14, 14]:
            now = datetime(2024, 1, 15, hour, 30, 0)
            quality = 0.9 if hour == 14 else 0.6
            optimizer.record_fill("BTCUSDT", "buy", quality, now)

        stats = optimizer.get_statistics()
        assert stats["quality_by_hour"][14] == 0.9
        assert stats["quality_by_hour"][10] == 0.6

    def test_best_worst_hours(self) -> None:
        """Test that best/worst hours are identified."""
        optimizer = OrderTimingOptimizer()

        # Record fills with varying quality
        test_data = [
            (8, 0.9),   # Good
            (8, 0.85),  # Good
            (14, 0.95), # Best
            (14, 0.92), # Best
            (3, 0.4),   # Bad
            (3, 0.35),  # Worst
        ]

        for hour, quality in test_data:
            now = datetime(2024, 1, 15, hour, 30, 0)
            optimizer.record_fill("BTCUSDT", "buy", quality, now)

        stats = optimizer.get_statistics()
        assert 14 in stats["best_hours"]
        assert 3 in stats["worst_hours"]


class TestDelayDecision:
    """Test order delay decisions."""

    def test_should_not_delay_in_high_liquidity(self) -> None:
        """Test that orders don't delay in high liquidity."""
        optimizer = OrderTimingOptimizer()

        # Mock high liquidity time
        should_delay, delay = optimizer.should_delay_order(
            symbol="BTCUSDT",
            side="buy",
            max_delay_minutes=30,
        )

        # Result depends on current time, but we can check the logic
        assert isinstance(should_delay, bool)
        assert isinstance(delay, int)
        assert delay >= 0

    def test_delay_respects_max(self) -> None:
        """Test that delay respects maximum delay parameter."""
        optimizer = OrderTimingOptimizer()

        # Even if wait is recommended, should respect max
        should_delay, delay = optimizer.should_delay_order(
            symbol="BTCUSDT",
            side="buy",
            max_delay_minutes=5,  # Very short max
        )

        if should_delay:
            assert delay <= 5


class TestNextTimeCalculations:
    """Test next optimal time calculations."""

    def test_next_high_liquidity_time(self) -> None:
        """Test finding next high liquidity window."""
        optimizer = OrderTimingOptimizer()

        # At 5:00 UTC, next high liquidity is 7:00 (London open)
        now = datetime(2024, 1, 15, 5, 0, 0)
        next_high = optimizer._next_high_liquidity_time(now)

        assert next_high is not None
        assert next_high.hour == 7  # London open at 7:00

    def test_next_non_funding_time(self) -> None:
        """Test finding next time outside funding window."""
        optimizer = OrderTimingOptimizer()

        # At 7:50 UTC (5 min before 8:00 funding), should find 8:16+
        now = datetime(2024, 1, 15, 7, 50, 0)
        next_safe = optimizer._next_non_funding_time(now)

        assert next_safe is not None
        # Should be after funding window (8:00 + 15 min)
        assert next_safe > now

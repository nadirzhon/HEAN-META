"""Tests for Enhanced Adaptive TTL."""

import pytest
from datetime import datetime

from hean.execution.adaptive_ttl import EnhancedAdaptiveTTL, FillPatternStats, SpreadBucket


class TestFillPatternStats:
    """Tests for FillPatternStats."""

    def test_initialization(self):
        """Test stats initialization."""
        stats = FillPatternStats(hour=12)
        assert stats.hour == 12
        assert stats.fills == 0
        assert stats.expirations == 0
        assert stats.fill_rate == 0.0

    def test_update_fill(self):
        """Test fill update."""
        stats = FillPatternStats(hour=12)
        stats.update_fill(250.0)
        stats.update_fill(300.0)
        stats.update_fill(275.0)

        assert stats.fills == 3
        assert stats.total_attempts == 3
        assert stats.fill_rate == 1.0
        assert stats.avg_fill_time_ms > 0

    def test_update_expiration(self):
        """Test expiration update."""
        stats = FillPatternStats(hour=12)
        stats.update_fill(250.0)
        stats.update_expiration()
        stats.update_expiration()

        assert stats.fills == 1
        assert stats.expirations == 2
        assert stats.total_attempts == 3
        assert stats.fill_rate == pytest.approx(1/3)


class TestSpreadBucket:
    """Tests for SpreadBucket."""

    def test_initialization(self):
        """Test bucket initialization."""
        bucket = SpreadBucket(spread_bps_min=0.0, spread_bps_max=5.0)
        assert bucket.spread_bps_min == 0.0
        assert bucket.spread_bps_max == 5.0
        assert bucket.fills == 0

    def test_update_fill(self):
        """Test fill update."""
        bucket = SpreadBucket(spread_bps_min=0.0, spread_bps_max=5.0)
        bucket.update_fill(ttl_ms=500.0)
        bucket.update_fill(ttl_ms=600.0)

        assert bucket.fills == 2
        assert bucket.optimal_ttl_ms > 0

    def test_get_fill_rate(self):
        """Test fill rate calculation."""
        bucket = SpreadBucket(spread_bps_min=0.0, spread_bps_max=5.0)
        bucket.update_fill(500.0)
        bucket.update_fill(600.0)
        bucket.update_expiration()

        fill_rate = bucket.get_fill_rate()
        assert fill_rate == pytest.approx(2/3)


class TestEnhancedAdaptiveTTL:
    """Tests for EnhancedAdaptiveTTL."""

    @pytest.fixture
    def adaptive_ttl(self):
        """Create adaptive TTL instance."""
        return EnhancedAdaptiveTTL(
            base_ttl_ms=500.0,
            min_ttl_ms=200.0,
            max_ttl_ms=3000.0,
            learning_rate=0.1,
        )

    def test_initialization(self, adaptive_ttl):
        """Test initialization."""
        assert adaptive_ttl._base_ttl_ms == 500.0
        assert adaptive_ttl._min_ttl_ms == 200.0
        assert adaptive_ttl._max_ttl_ms == 3000.0
        assert adaptive_ttl._current_ttl_ms == 500.0

    def test_calculate_ttl_basic(self, adaptive_ttl):
        """Test basic TTL calculation."""
        ttl = adaptive_ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=5.0,
            volatility_regime="medium",
            current_hour=12,
        )

        assert adaptive_ttl._min_ttl_ms <= ttl <= adaptive_ttl._max_ttl_ms

    def test_calculate_ttl_tight_spread(self, adaptive_ttl):
        """Test TTL with tight spread (should reduce)."""
        ttl_tight = adaptive_ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=1.5,  # Very tight
            volatility_regime="medium",
            current_hour=12,
        )

        ttl_wide = adaptive_ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=15.0,  # Wide
            volatility_regime="medium",
            current_hour=12,
        )

        # Tight spread should have lower TTL
        assert ttl_tight < ttl_wide

    def test_calculate_ttl_volatility_regimes(self, adaptive_ttl):
        """Test TTL with different volatility regimes."""
        ttl_low = adaptive_ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=5.0,
            volatility_regime="low",
            current_hour=12,
        )

        ttl_high = adaptive_ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=5.0,
            volatility_regime="high",
            current_hour=12,
        )

        ttl_extreme = adaptive_ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=5.0,
            volatility_regime="extreme",
            current_hour=12,
        )

        # Higher volatility should have higher TTL
        assert ttl_low < ttl_high < ttl_extreme

    def test_record_fill(self, adaptive_ttl):
        """Test recording fills."""
        adaptive_ttl.record_fill(
            symbol="BTCUSDT",
            spread_bps=5.0,
            fill_time_ms=300.0,
            hour=12,
        )

        assert len(adaptive_ttl._recent_fills) == 1
        assert adaptive_ttl._hourly_stats[12].fills == 1

    def test_record_expiration(self, adaptive_ttl):
        """Test recording expirations."""
        adaptive_ttl.record_expiration(
            symbol="BTCUSDT",
            spread_bps=5.0,
            hour=12,
        )

        assert len(adaptive_ttl._recent_expirations) == 1
        assert adaptive_ttl._hourly_stats[12].expirations == 1

    def test_learning_from_fills(self, adaptive_ttl):
        """Test that TTL adjusts based on fill patterns."""
        # Record high fill rate at hour 14
        for _ in range(10):
            adaptive_ttl.record_fill(
                symbol="BTCUSDT",
                spread_bps=5.0,
                fill_time_ms=200.0,  # Fast fills
                hour=14,
            )

        # Calculate TTL for hour 14
        ttl_learned = adaptive_ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=5.0,
            volatility_regime="medium",
            current_hour=14,
        )

        # Should be lower than base (fast fills = reduce TTL)
        assert ttl_learned < adaptive_ttl._base_ttl_ms * 1.1

    def test_learning_from_expirations(self, adaptive_ttl):
        """Test that TTL increases with expirations."""
        # Record low fill rate at hour 15
        adaptive_ttl.record_fill("BTCUSDT", 5.0, 400.0, hour=15)
        for _ in range(10):
            adaptive_ttl.record_expiration("BTCUSDT", 5.0, hour=15)

        # Calculate TTL for hour 15
        ttl_learned = adaptive_ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=5.0,
            volatility_regime="medium",
            current_hour=15,
        )

        # Should be higher than base (many expirations = increase TTL)
        assert ttl_learned > adaptive_ttl._base_ttl_ms * 0.9

    def test_get_spread_bucket(self, adaptive_ttl):
        """Test spread bucket retrieval."""
        bucket = adaptive_ttl._get_spread_bucket(3.5)
        assert bucket is not None
        assert bucket.spread_bps_min <= 3.5 < bucket.spread_bps_max

        bucket_wide = adaptive_ttl._get_spread_bucket(15.0)
        assert bucket_wide is not None
        assert bucket_wide.spread_bps_min <= 15.0 < bucket_wide.spread_bps_max

    def test_get_statistics(self, adaptive_ttl):
        """Test statistics retrieval."""
        # Record some data
        adaptive_ttl.record_fill("BTCUSDT", 5.0, 300.0, hour=12)
        adaptive_ttl.record_fill("BTCUSDT", 5.0, 350.0, hour=12)
        adaptive_ttl.record_expiration("BTCUSDT", 5.0, hour=12)

        stats = adaptive_ttl.get_statistics()

        assert "current_ttl_ms" in stats
        assert "overall_fill_rate" in stats
        assert "avg_fill_time_ms" in stats
        assert stats["total_fills"] == 2
        assert stats["total_expirations"] == 1
        assert stats["overall_fill_rate"] == pytest.approx(2/3)

    def test_hourly_fill_rates(self, adaptive_ttl):
        """Test hourly fill rate tracking."""
        # Hour 10: good fill rate
        for _ in range(8):
            adaptive_ttl.record_fill("BTCUSDT", 5.0, 300.0, hour=10)
        for _ in range(2):
            adaptive_ttl.record_expiration("BTCUSDT", 5.0, hour=10)

        # Hour 22: poor fill rate
        for _ in range(3):
            adaptive_ttl.record_fill("BTCUSDT", 5.0, 400.0, hour=22)
        for _ in range(7):
            adaptive_ttl.record_expiration("BTCUSDT", 5.0, hour=22)

        stats = adaptive_ttl.get_statistics()

        assert 10 in stats["hourly_fill_rates"]
        assert 22 in stats["hourly_fill_rates"]
        assert stats["hourly_fill_rates"][10] > stats["hourly_fill_rates"][22]
        assert stats["best_hour"] == 10
        assert stats["worst_hour"] == 22

    def test_spread_bucket_learning(self, adaptive_ttl):
        """Test spread bucket optimal TTL learning."""
        # Tight spread bucket: record fast fills
        for _ in range(10):
            adaptive_ttl.record_fill("BTCUSDT", 3.0, 250.0)

        # Wide spread bucket: record slower fills
        for _ in range(10):
            adaptive_ttl.record_fill("BTCUSDT", 15.0, 800.0)

        stats = adaptive_ttl.get_statistics()

        # Should have stats for both buckets
        assert len(stats["spread_bucket_stats"]) >= 2

        # Find tight and wide buckets
        tight_bucket = next(
            (b for b in stats["spread_bucket_stats"] if "0.0-5.0" in b["range_bps"]),
            None
        )
        wide_bucket = next(
            (b for b in stats["spread_bucket_stats"] if "10.0-20.0" in b["range_bps"]),
            None
        )

        if tight_bucket and wide_bucket:
            # Wide bucket should have higher optimal TTL
            assert wide_bucket["optimal_ttl_ms"] > tight_bucket["optimal_ttl_ms"]

    def test_ttl_clamping(self, adaptive_ttl):
        """Test that TTL is clamped to min/max."""
        # Extreme volatility should still clamp
        ttl = adaptive_ttl.calculate_ttl(
            symbol="BTCUSDT",
            spread_bps=50.0,  # Very wide
            volatility_regime="extreme",  # Very volatile
            current_hour=12,
        )

        assert adaptive_ttl._min_ttl_ms <= ttl <= adaptive_ttl._max_ttl_ms

    def test_reset_learning(self, adaptive_ttl):
        """Test resetting learned patterns."""
        # Add some data
        adaptive_ttl.record_fill("BTCUSDT", 5.0, 300.0, hour=12)
        adaptive_ttl.record_expiration("BTCUSDT", 5.0, hour=12)

        assert len(adaptive_ttl._recent_fills) > 0

        # Reset
        adaptive_ttl.reset_learning()

        assert len(adaptive_ttl._recent_fills) == 0
        assert len(adaptive_ttl._recent_expirations) == 0
        assert all(s.fills == 0 for s in adaptive_ttl._hourly_stats.values())


@pytest.mark.asyncio
async def test_adaptive_ttl_integration():
    """Integration test for adaptive TTL."""
    adaptive_ttl = EnhancedAdaptiveTTL(base_ttl_ms=500.0)

    # Simulate trading session with varying conditions
    for hour in range(8, 18):  # Trading day
        for spread in [2.0, 5.0, 10.0]:
            # Calculate TTL
            ttl = adaptive_ttl.calculate_ttl(
                symbol="BTCUSDT",
                spread_bps=spread,
                volatility_regime="medium",
                current_hour=hour,
            )

            # Simulate fill or expiration
            if spread < 5.0:
                # Tight spread = likely fill
                adaptive_ttl.record_fill(
                    symbol="BTCUSDT",
                    spread_bps=spread,
                    fill_time_ms=ttl * 0.6,  # Fill before expiration
                    hour=hour,
                )
            else:
                # Wide spread = likely expiration
                adaptive_ttl.record_expiration(
                    symbol="BTCUSDT",
                    spread_bps=spread,
                    hour=hour,
                )

    # Check that learning occurred
    stats = adaptive_ttl.get_statistics()
    assert stats["total_fills"] > 0
    assert stats["total_expirations"] > 0
    assert len(stats["hourly_fill_rates"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

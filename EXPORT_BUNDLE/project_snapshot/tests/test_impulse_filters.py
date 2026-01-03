"""Tests for impulse engine micro-filters."""

from datetime import datetime

import pytest

from hean.config import settings
from hean.core.regime import Regime
from hean.core.types import Tick
from hean.strategies.impulse_filters import (
    ImpulseFilterPipeline,
    SpreadFilter,
    TimeWindowFilter,
    VolatilityExpansionFilter,
)


class TestSpreadFilter:
    """Tests for SpreadFilter."""

    def test_allows_trade_when_spread_is_acceptable(self) -> None:
        """Test that filter allows trade when spread is below threshold."""
        filter_obj = SpreadFilter()
        
        # Set max spread to 10 bps
        original_value = settings.impulse_max_spread_bps
        settings.impulse_max_spread_bps = 10
        
        try:
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime.utcnow(),
                bid=49995.0,  # 5 bps spread
                ask=50005.0,
            )
            
            result = filter_obj.allow(tick)
            assert result is True
        finally:
            settings.impulse_max_spread_bps = original_value

    def test_blocks_trade_when_spread_exceeds_threshold(self) -> None:
        """Test that filter blocks trade when spread exceeds threshold."""
        filter_obj = SpreadFilter()
        
        # Set max spread to 10 bps
        original_value = settings.impulse_max_spread_bps
        settings.impulse_max_spread_bps = 10
        
        try:
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime.utcnow(),
                bid=49950.0,  # 50 bps spread (exceeds 10 bps)
                ask=50050.0,
            )
            
            result = filter_obj.allow(tick)
            assert result is False
        finally:
            settings.impulse_max_spread_bps = original_value

    def test_allows_trade_when_no_bid_ask(self) -> None:
        """Test that filter allows trade when bid/ask are not available."""
        filter_obj = SpreadFilter()
        
        tick = Tick(
            symbol="BTCUSDT",
            price=50000.0,
            timestamp=datetime.utcnow(),
            bid=None,
            ask=None,
        )
        
        result = filter_obj.allow(tick)
        assert result is True


class TestVolatilityExpansionFilter:
    """Tests for VolatilityExpansionFilter."""

    def test_allows_trade_when_volatility_expands(self) -> None:
        """Test that filter allows trade when short-term vol > long-term vol * ratio."""
        filter_obj = VolatilityExpansionFilter()
        
        # Set required ratio to 1.15 (short-term must be >= 1.15 * long-term)
        original_value = settings.impulse_vol_expansion_ratio
        settings.impulse_vol_expansion_ratio = 1.15
        
        try:
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime.utcnow(),
            )
            
            # Short-term vol = 0.02, long-term vol = 0.01 (ratio = 2.0 > 1.15)
            context = {
                "vol_short": 0.02,
                "vol_long": 0.01,
            }
            
            result = filter_obj.allow(tick, context)
            assert result is True
        finally:
            settings.impulse_vol_expansion_ratio = original_value

    def test_blocks_trade_when_volatility_contracts(self) -> None:
        """Test that filter blocks trade when short-term vol <= long-term vol * ratio."""
        filter_obj = VolatilityExpansionFilter()
        
        # Set required ratio to 1.15
        original_value = settings.impulse_vol_expansion_ratio
        settings.impulse_vol_expansion_ratio = 1.15
        
        try:
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime.utcnow(),
            )
            
            # Short-term vol = 0.01, long-term vol = 0.01 (ratio = 1.0 < 1.15)
            context = {
                "vol_short": 0.01,
                "vol_long": 0.01,
            }
            
            result = filter_obj.allow(tick, context)
            assert result is False
        finally:
            settings.impulse_vol_expansion_ratio = original_value

    def test_allows_trade_with_insufficient_data(self) -> None:
        """Test that filter allows trade when there's insufficient volatility data."""
        filter_obj = VolatilityExpansionFilter()
        
        tick = Tick(
            symbol="BTCUSDT",
            price=50000.0,
            timestamp=datetime.utcnow(),
        )
        
        # No volatility data in context
        context = {}
        
        result = filter_obj.allow(tick, context)
        assert result is True

    def test_allows_trade_when_long_term_vol_is_zero(self) -> None:
        """Test that filter allows trade when long-term volatility is zero."""
        filter_obj = VolatilityExpansionFilter()
        
        tick = Tick(
            symbol="BTCUSDT",
            price=50000.0,
            timestamp=datetime.utcnow(),
        )
        
        # Long-term vol is zero
        context = {
            "vol_short": 0.01,
            "vol_long": 0.0,
        }
        
        result = filter_obj.allow(tick, context)
        assert result is True


class TestTimeWindowFilter:
    """Tests for TimeWindowFilter."""

    def test_allows_trade_when_in_allowed_hours(self) -> None:
        """Test that filter allows trade when time is in allowed range."""
        filter_obj = TimeWindowFilter()
        
        original_value = settings.impulse_allowed_hours
        settings.impulse_allowed_hours = ["09:00-17:00"]
        
        try:
            # 12:00 UTC (within 09:00-17:00)
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
            )
            
            result = filter_obj.allow(tick)
            assert result is True
        finally:
            settings.impulse_allowed_hours = original_value

    def test_blocks_trade_when_outside_allowed_hours(self) -> None:
        """Test that filter blocks trade when time is outside allowed range."""
        filter_obj = TimeWindowFilter()
        
        original_value = settings.impulse_allowed_hours
        settings.impulse_allowed_hours = ["09:00-17:00"]
        
        try:
            # 20:00 UTC (outside 09:00-17:00)
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 1, 20, 0, 0),
            )
            
            result = filter_obj.allow(tick)
            assert result is False
        finally:
            settings.impulse_allowed_hours = original_value

    def test_allows_trade_when_no_restrictions(self) -> None:
        """Test that filter allows trade when no hours are configured."""
        filter_obj = TimeWindowFilter()
        
        original_value = settings.impulse_allowed_hours
        settings.impulse_allowed_hours = None
        
        try:
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime.utcnow(),
            )
            
            result = filter_obj.allow(tick)
            assert result is True
        finally:
            settings.impulse_allowed_hours = original_value

    def test_allows_trade_when_crossing_midnight(self) -> None:
        """Test that filter handles time ranges that cross midnight."""
        filter_obj = TimeWindowFilter()
        
        original_value = settings.impulse_allowed_hours
        settings.impulse_allowed_hours = ["22:00-06:00"]  # 22:00 to 06:00 next day
        
        try:
            # 23:00 UTC (within 22:00-06:00 range)
            tick1 = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 1, 23, 0, 0),
            )
            result1 = filter_obj.allow(tick1)
            assert result1 is True
            
            # 03:00 UTC (within 22:00-06:00 range)
            tick2 = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 2, 3, 0, 0),
            )
            result2 = filter_obj.allow(tick2)
            assert result2 is True
            
            # 12:00 UTC (outside 22:00-06:00 range)
            tick3 = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
            )
            result3 = filter_obj.allow(tick3)
            assert result3 is False
        finally:
            settings.impulse_allowed_hours = original_value

    def test_allows_trade_with_multiple_ranges(self) -> None:
        """Test that filter works with multiple time ranges."""
        filter_obj = TimeWindowFilter()
        
        original_value = settings.impulse_allowed_hours
        settings.impulse_allowed_hours = ["09:00-12:00", "14:00-17:00"]
        
        try:
            # 10:00 UTC (in first range)
            tick1 = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 1, 10, 0, 0),
            )
            result1 = filter_obj.allow(tick1)
            assert result1 is True
            
            # 15:00 UTC (in second range)
            tick2 = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 1, 15, 0, 0),
            )
            result2 = filter_obj.allow(tick2)
            assert result2 is True
            
            # 13:00 UTC (outside both ranges)
            tick3 = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 1, 13, 0, 0),
            )
            result3 = filter_obj.allow(tick3)
            assert result3 is False
        finally:
            settings.impulse_allowed_hours = original_value


class TestImpulseFilterPipeline:
    """Tests for ImpulseFilterPipeline composition."""

    def test_all_filters_pass(self) -> None:
        """Test that pipeline passes when all filters pass."""
        # Configure all filters to pass
        original_spread = settings.impulse_max_spread_bps
        original_vol = settings.impulse_vol_expansion_ratio
        original_hours = settings.impulse_allowed_hours
        
        settings.impulse_max_spread_bps = 10
        settings.impulse_vol_expansion_ratio = 1.15
        settings.impulse_allowed_hours = None  # No time restriction
        
        try:
            pipeline = ImpulseFilterPipeline([
                SpreadFilter(),
                VolatilityExpansionFilter(),
                TimeWindowFilter(),
            ])
            
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                bid=49995.0,  # 5 bps spread (acceptable)
                ask=50005.0,
            )
            
            context = {
                "regime": Regime.IMPULSE,
                "spread_bps": 5.0,
                "vol_short": 0.02,  # Expanding volatility
                "vol_long": 0.01,
                "timestamp": tick.timestamp,
            }
            
            result = pipeline.allow(tick, context)
            assert result is True
        finally:
            settings.impulse_max_spread_bps = original_spread
            settings.impulse_vol_expansion_ratio = original_vol
            settings.impulse_allowed_hours = original_hours

    def test_spread_filter_blocks(self) -> None:
        """Test that pipeline blocks when spread filter fails."""
        original_spread = settings.impulse_max_spread_bps
        original_vol = settings.impulse_vol_expansion_ratio
        original_hours = settings.impulse_allowed_hours
        
        settings.impulse_max_spread_bps = 10
        settings.impulse_vol_expansion_ratio = 1.15
        settings.impulse_allowed_hours = None
        
        try:
            pipeline = ImpulseFilterPipeline([
                SpreadFilter(),
                VolatilityExpansionFilter(),
                TimeWindowFilter(),
            ])
            
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime.utcnow(),
                bid=49950.0,  # 50 bps spread (exceeds 10 bps)
                ask=50050.0,
            )
            
            context = {
                "regime": Regime.IMPULSE,
                "spread_bps": 50.0,
                "vol_short": 0.02,
                "vol_long": 0.01,
                "timestamp": tick.timestamp,
            }
            
            result = pipeline.allow(tick, context)
            assert result is False
        finally:
            settings.impulse_max_spread_bps = original_spread
            settings.impulse_vol_expansion_ratio = original_vol
            settings.impulse_allowed_hours = original_hours

    def test_volatility_filter_blocks(self) -> None:
        """Test that pipeline blocks when volatility filter fails."""
        original_spread = settings.impulse_max_spread_bps
        original_vol = settings.impulse_vol_expansion_ratio
        original_hours = settings.impulse_allowed_hours
        
        settings.impulse_max_spread_bps = 10
        settings.impulse_vol_expansion_ratio = 1.15
        settings.impulse_allowed_hours = None
        
        try:
            pipeline = ImpulseFilterPipeline([
                SpreadFilter(),
                VolatilityExpansionFilter(),
                TimeWindowFilter(),
            ])
            
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime.utcnow(),
                bid=49995.0,  # 5 bps spread (acceptable)
                ask=50005.0,
            )
            
            # Contracting volatility (short <= long * ratio)
            context = {
                "regime": Regime.IMPULSE,
                "spread_bps": 5.0,
                "vol_short": 0.01,  # Ratio = 1.0 < 1.15
                "vol_long": 0.01,
                "timestamp": tick.timestamp,
            }
            
            result = pipeline.allow(tick, context)
            assert result is False
        finally:
            settings.impulse_max_spread_bps = original_spread
            settings.impulse_vol_expansion_ratio = original_vol
            settings.impulse_allowed_hours = original_hours

    def test_time_filter_blocks(self) -> None:
        """Test that pipeline blocks when time filter fails."""
        original_spread = settings.impulse_max_spread_bps
        original_vol = settings.impulse_vol_expansion_ratio
        original_hours = settings.impulse_allowed_hours
        
        settings.impulse_max_spread_bps = 10
        settings.impulse_vol_expansion_ratio = 1.15
        settings.impulse_allowed_hours = ["09:00-17:00"]
        
        try:
            pipeline = ImpulseFilterPipeline([
                SpreadFilter(),
                VolatilityExpansionFilter(),
                TimeWindowFilter(),
            ])
            
            tick = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime(2024, 1, 1, 20, 0, 0),  # Outside allowed hours
                bid=49995.0,  # 5 bps spread (acceptable)
                ask=50005.0,
            )
            
            context = {
                "regime": Regime.IMPULSE,
                "spread_bps": 5.0,
                "vol_short": 0.02,
                "vol_long": 0.01,
                "timestamp": tick.timestamp,
            }
            
            result = pipeline.allow(tick, context)
            assert result is False
        finally:
            settings.impulse_max_spread_bps = original_spread
            settings.impulse_vol_expansion_ratio = original_vol
            settings.impulse_allowed_hours = original_hours

    def test_pass_rate_calculation(self) -> None:
        """Test that pass rate is calculated correctly."""
        original_spread = settings.impulse_max_spread_bps
        original_vol = settings.impulse_vol_expansion_ratio
        original_hours = settings.impulse_allowed_hours
        
        settings.impulse_max_spread_bps = 10
        settings.impulse_vol_expansion_ratio = 1.15
        settings.impulse_allowed_hours = None
        
        try:
            pipeline = ImpulseFilterPipeline([
                SpreadFilter(),
                VolatilityExpansionFilter(),
                TimeWindowFilter(),
            ])
            
            tick_good = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime.utcnow(),
                bid=49995.0,
                ask=50005.0,
            )
            
            tick_bad = Tick(
                symbol="BTCUSDT",
                price=50000.0,
                timestamp=datetime.utcnow(),
                bid=49950.0,  # High spread
                ask=50050.0,
            )
            
            context_good = {
                "regime": Regime.IMPULSE,
                "spread_bps": 5.0,
                "vol_short": 0.02,
                "vol_long": 0.01,
                "timestamp": tick_good.timestamp,
            }
            
            context_bad = {
                "regime": Regime.IMPULSE,
                "spread_bps": 50.0,
                "vol_short": 0.02,
                "vol_long": 0.01,
                "timestamp": tick_bad.timestamp,
            }
            
            # Check 3 good, 2 bad
            pipeline.allow(tick_good, context_good)
            pipeline.allow(tick_good, context_good)
            pipeline.allow(tick_good, context_good)
            pipeline.allow(tick_bad, context_bad)
            pipeline.allow(tick_bad, context_bad)
            
            pass_rate_pct = pipeline.get_pass_rate_pct()
            assert pass_rate_pct == 60.0  # 3 out of 5 passed = 60%
            
            blocked_count = pipeline.get_blocked_count()
            assert blocked_count == 2
        finally:
            settings.impulse_max_spread_bps = original_spread
            settings.impulse_vol_expansion_ratio = original_vol
            settings.impulse_allowed_hours = original_hours






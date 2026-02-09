"""Tests for trade density control."""

from datetime import datetime, timedelta

import pytest

from hean.core.trade_density import TradeDensityTracker, trade_density


class TestTradeDensityTracker:
    """Test trade density tracking and relaxation."""

    def test_initial_state(self) -> None:
        """Test initial state with no trades."""
        tracker = TradeDensityTracker()

        # No trades recorded
        idle_days = tracker.get_idle_days("test_strategy")
        assert idle_days == 30.0  # Default for new strategies

        relaxation_level = tracker.get_relaxation_level("test_strategy")
        assert relaxation_level == 2  # Should be at max relaxation (30 days > 7)

        trades_last_7 = tracker.get_trades_last_N_days("test_strategy", 7)
        assert trades_last_7 == 0

    def test_record_trade(self) -> None:
        """Test recording a trade."""
        tracker = TradeDensityTracker()
        now = datetime.utcnow()

        tracker.record_trade("test_strategy", now)

        idle_days = tracker.get_idle_days("test_strategy", now)
        assert idle_days == 0.0

        relaxation_level = tracker.get_relaxation_level("test_strategy", now)
        assert relaxation_level == 0

        trades_last_7 = tracker.get_trades_last_N_days("test_strategy", 7, now)
        assert trades_last_7 == 1

    def test_idle_days_calculation(self) -> None:
        """Test idle days calculation."""
        tracker = TradeDensityTracker()
        base_time = datetime.utcnow()

        # Record trade 3 days ago
        trade_time = base_time - timedelta(days=3)
        tracker.record_trade("test_strategy", trade_time)

        idle_days = tracker.get_idle_days("test_strategy", base_time)
        assert idle_days == pytest.approx(3.0, abs=0.1)

    def test_relaxation_levels(self) -> None:
        """Test relaxation level thresholds."""
        tracker = TradeDensityTracker()
        base_time = datetime.utcnow()

        # No trades - should be at level 2
        level = tracker.get_relaxation_level("test_strategy", base_time)
        assert level == 2

        # Trade 4 days ago - should be at level 0 (below threshold 1)
        trade_time = base_time - timedelta(days=4)
        tracker.record_trade("test_strategy", trade_time)
        level = tracker.get_relaxation_level("test_strategy", base_time)
        assert level == 0

        # Trade 6 days ago - should be at level 1 (above threshold 1, below threshold 2)
        trade_time = base_time - timedelta(days=6)
        tracker.record_trade("test_strategy", trade_time)
        level = tracker.get_relaxation_level("test_strategy", base_time)
        assert level == 1

        # Trade 8 days ago - should be at level 2 (above threshold 2)
        trade_time = base_time - timedelta(days=8)
        tracker.record_trade("test_strategy", trade_time)
        level = tracker.get_relaxation_level("test_strategy", base_time)
        assert level == 2

    def test_volatility_relaxation_factor(self) -> None:
        """Test volatility relaxation factor."""
        tracker = TradeDensityTracker()
        base_time = datetime.utcnow()

        # No relaxation (level 0)
        trade_time = base_time - timedelta(days=2)
        tracker.record_trade("test_strategy", trade_time)
        factor = tracker.get_volatility_relaxation_factor("test_strategy", base_time)
        assert factor == 1.0

        # Level 1 relaxation (5 days)
        trade_time = base_time - timedelta(days=6)
        tracker.record_trade("test_strategy", trade_time)
        factor = tracker.get_volatility_relaxation_factor("test_strategy", base_time)
        assert factor == pytest.approx(0.9, abs=0.01)  # 10% reduction

        # Level 2 relaxation (7+ days) - same as level 1
        trade_time = base_time - timedelta(days=8)
        tracker.record_trade("test_strategy", trade_time)
        factor = tracker.get_volatility_relaxation_factor("test_strategy", base_time)
        assert factor == pytest.approx(0.9, abs=0.01)

    def test_time_window_expansion(self) -> None:
        """Test time window expansion."""
        tracker = TradeDensityTracker()
        base_time = datetime.utcnow()

        # No expansion (level 0 or 1)
        trade_time = base_time - timedelta(days=6)
        tracker.record_trade("test_strategy", trade_time)
        expansion = tracker.get_time_window_expansion_hours("test_strategy", base_time)
        assert expansion == 0

        # Expansion at level 2 (7+ days)
        trade_time = base_time - timedelta(days=8)
        tracker.record_trade("test_strategy", trade_time)
        expansion = tracker.get_time_window_expansion_hours("test_strategy", base_time)
        assert expansion == 2

    def test_trade_resets_state(self) -> None:
        """Test that trade resets density state."""
        tracker = TradeDensityTracker()
        base_time = datetime.utcnow()

        # Record trade 8 days ago (level 2)
        old_trade = base_time - timedelta(days=8)
        tracker.record_trade("test_strategy", old_trade)

        level_before = tracker.get_relaxation_level("test_strategy", base_time)
        assert level_before == 2

        # Record new trade now
        tracker.record_trade("test_strategy", base_time)

        level_after = tracker.get_relaxation_level("test_strategy", base_time)
        assert level_after == 0

        idle_days = tracker.get_idle_days("test_strategy", base_time)
        assert idle_days == 0.0

    def test_trades_last_N_days(self) -> None:
        """Test counting trades in last N days."""
        tracker = TradeDensityTracker()
        base_time = datetime.utcnow()

        # Record trades at different times
        tracker.record_trade("test_strategy", base_time - timedelta(days=1))
        tracker.record_trade("test_strategy", base_time - timedelta(days=3))
        tracker.record_trade("test_strategy", base_time - timedelta(days=5))
        tracker.record_trade("test_strategy", base_time - timedelta(days=10))  # Outside window

        # Should count 3 trades in last 7 days
        count = tracker.get_trades_last_N_days("test_strategy", 7, base_time)
        assert count == 3

        # Should count 2 trades in last 4 days
        count = tracker.get_trades_last_N_days("test_strategy", 4, base_time)
        assert count == 2

    def test_density_state(self) -> None:
        """Test complete density state dictionary."""
        tracker = TradeDensityTracker()
        base_time = datetime.utcnow()

        # Record trade 6 days ago (level 1)
        trade_time = base_time - timedelta(days=6)
        tracker.record_trade("test_strategy", trade_time)

        state = tracker.get_density_state("test_strategy", base_time)

        assert "idle_days" in state
        assert state["idle_days"] == pytest.approx(6.0, abs=0.1)

        assert "trades_last_7_days" in state
        assert state["trades_last_7_days"] == 1

        assert "density_relaxation_level" in state
        assert state["density_relaxation_level"] == 1

        assert "volatility_relaxation_factor" in state
        assert state["volatility_relaxation_factor"] == pytest.approx(0.9, abs=0.01)

        assert "time_window_expansion_hours" in state
        assert state["time_window_expansion_hours"] == 0

    def test_multiple_strategies(self) -> None:
        """Test tracking multiple strategies independently."""
        tracker = TradeDensityTracker()
        base_time = datetime.utcnow()

        # Strategy 1: trade 2 days ago
        tracker.record_trade("strategy1", base_time - timedelta(days=2))

        # Strategy 2: trade 8 days ago
        tracker.record_trade("strategy2", base_time - timedelta(days=8))

        level1 = tracker.get_relaxation_level("strategy1", base_time)
        level2 = tracker.get_relaxation_level("strategy2", base_time)

        assert level1 == 0
        assert level2 == 2

        idle1 = tracker.get_idle_days("strategy1", base_time)
        idle2 = tracker.get_idle_days("strategy2", base_time)

        assert idle1 == pytest.approx(2.0, abs=0.1)
        assert idle2 == pytest.approx(8.0, abs=0.1)

    def test_trade_history_limit(self) -> None:
        """Test that trade history is limited to maxlen."""
        tracker = TradeDensityTracker()
        base_time = datetime.utcnow()

        # Record more than 30 trades
        for i in range(35):
            tracker.record_trade("test_strategy", base_time - timedelta(days=i))

        # Should only track last 30 trades
        count = tracker.get_trades_last_N_days("test_strategy", 35, base_time)
        assert count == 30


class TestGlobalTradeDensity:
    """Test global trade density instance."""

    def test_global_instance(self) -> None:
        """Test that global instance works."""
        now = datetime.utcnow()

        trade_density.record_trade("test_strategy", now)

        idle_days = trade_density.get_idle_days("test_strategy", now)
        assert idle_days == 0.0

        state = trade_density.get_density_state("test_strategy", now)
        assert state["idle_days"] == 0.0
        assert state["density_relaxation_level"] == 0







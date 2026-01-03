"""Tests for DensityController (trade density anti-starvation)."""

from datetime import datetime, timedelta

from hean.core.density import DensityController


class TestDensityController:
    """Tests for per-strategy DensityController."""

    def test_relaxation_increases_with_idle_days(self) -> None:
        """Relaxation level should increase as idle_days grows."""
        strategy_id = "density_test_strategy"
        controller = DensityController(strategy_id)

        base_time = datetime.utcnow()

        # No trades recorded: trade_density defaults to large idle_days (30.0),
        # which should map to max relaxation level 3.
        level_initial = controller.get_relaxation_level(base_time)
        assert level_initial == 3

        # Record a trade 1 day ago -> idle_days ~= 1 -> level 0
        trade_time_recent = base_time - timedelta(days=1)
        controller.record_trade(trade_time_recent)
        level_recent = controller.get_relaxation_level(base_time)
        assert level_recent == 0

        # Record a trade 4 days ago -> idle_days ~= 4 -> level 1
        trade_time_level1 = base_time - timedelta(days=4)
        controller.record_trade(trade_time_level1)
        level1 = controller.get_relaxation_level(base_time)
        assert level1 == 1

        # Record a trade 8 days ago -> idle_days ~= 8 -> level 2
        trade_time_level2 = base_time - timedelta(days=8)
        controller.record_trade(trade_time_level2)
        level2 = controller.get_relaxation_level(base_time)
        assert level2 == 2

        # Record a trade 15 days ago -> idle_days ~= 15 -> level 3
        trade_time_level3 = base_time - timedelta(days=15)
        controller.record_trade(trade_time_level3)
        level3 = controller.get_relaxation_level(base_time)
        assert level3 == 3

    def test_relaxation_resets_after_trade(self) -> None:
        """Relaxation level should reset to 0 immediately after a fresh trade."""
        strategy_id = "density_reset_strategy"
        controller = DensityController(strategy_id)

        base_time = datetime.utcnow()

        # Start with an old trade so we are at high relaxation.
        old_trade_time = base_time - timedelta(days=20)
        controller.record_trade(old_trade_time)
        level_before = controller.get_relaxation_level(base_time)
        assert level_before == 3

        # Now record a new trade "now" and ensure we fully reset.
        controller.record_trade(base_time)
        level_after = controller.get_relaxation_level(base_time)
        idle_days_after = controller.get_idle_days(base_time)

        assert level_after == 0
        assert idle_days_after == 0.0








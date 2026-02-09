"""Tests for enhanced Kelly Criterion with confidence scaling and adaptive fractions."""

from unittest.mock import MagicMock

import pytest

from hean.portfolio.accounting import PortfolioAccounting
from hean.risk.kelly_criterion import KellyCriterion, StrategyPerformanceTracker


@pytest.fixture
def mock_accounting() -> MagicMock:
    """Create a mock PortfolioAccounting instance."""
    accounting = MagicMock(spec=PortfolioAccounting)
    accounting.get_strategy_metrics.return_value = {}
    return accounting


@pytest.fixture
def kelly(mock_accounting: MagicMock) -> KellyCriterion:
    """Create a KellyCriterion instance with mock accounting."""
    return KellyCriterion(mock_accounting)


class TestKellyCriterionBasics:
    """Test basic Kelly Criterion calculations."""

    def test_positive_edge_returns_positive_kelly(self, kelly: KellyCriterion) -> None:
        """Test that positive edge returns positive Kelly fraction."""
        metrics = {
            "test_strategy": {
                "wins": 60,
                "losses": 40,
                "avg_win": 2.0,
                "avg_loss": 1.0,
            }
        }

        fraction = kelly.calculate_kelly_fraction("test_strategy", metrics)
        assert fraction > 0, "Positive edge should return positive Kelly fraction"

    def test_negative_edge_returns_zero(self, kelly: KellyCriterion) -> None:
        """Test that negative edge returns zero Kelly fraction."""
        metrics = {
            "test_strategy": {
                "wins": 30,
                "losses": 70,
                "avg_win": 1.0,
                "avg_loss": 1.0,
            }
        }

        fraction = kelly.calculate_kelly_fraction("test_strategy", metrics)
        assert fraction == 0, "Negative edge should return zero"

    def test_fractional_kelly_applied(self, kelly: KellyCriterion) -> None:
        """Test that fractional Kelly is applied (should be < full Kelly)."""
        # High edge scenario
        metrics = {
            "test_strategy": {
                "wins": 70,
                "losses": 30,
                "avg_win": 3.0,
                "avg_loss": 1.0,
            }
        }

        fraction = kelly.calculate_kelly_fraction("test_strategy", metrics)
        # Full Kelly would be quite high, fractional should cap it
        assert fraction <= kelly.MAX_FRACTIONAL_KELLY


class TestConfidenceBasedKelly:
    """Test confidence-based Kelly scaling."""

    def test_high_confidence_higher_kelly(self, kelly: KellyCriterion) -> None:
        """Test that high confidence returns higher Kelly fraction."""
        metrics = {
            "test_strategy": {
                "wins": 55,
                "losses": 45,
                "avg_win": 1.5,
                "avg_loss": 1.0,
            }
        }

        low_conf = kelly.calculate_kelly_with_confidence(
            "test_strategy", 0.3, metrics
        )
        high_conf = kelly.calculate_kelly_with_confidence(
            "test_strategy", 0.9, metrics
        )

        assert high_conf >= low_conf, "Higher confidence should return >= Kelly"

    def test_zero_confidence_returns_minimum(self, kelly: KellyCriterion) -> None:
        """Test that zero confidence returns minimum Kelly."""
        metrics = {
            "test_strategy": {
                "wins": 60,
                "losses": 40,
                "avg_win": 2.0,
                "avg_loss": 1.0,
            }
        }

        fraction = kelly.calculate_kelly_with_confidence(
            "test_strategy", 0.0, metrics
        )

        # Should be very conservative with zero confidence
        assert fraction <= kelly.MIN_FRACTIONAL_KELLY

    def test_confidence_scales_linearly(self, kelly: KellyCriterion) -> None:
        """Test that confidence scaling is monotonic."""
        metrics = {
            "test_strategy": {
                "wins": 60,
                "losses": 40,
                "avg_win": 1.5,
                "avg_loss": 1.0,
            }
        }

        confidences = [0.2, 0.4, 0.6, 0.8]
        fractions = [
            kelly.calculate_kelly_with_confidence("test_strategy", c, metrics)
            for c in confidences
        ]

        # Each should be >= previous
        for i in range(1, len(fractions)):
            assert fractions[i] >= fractions[i-1], "Kelly should increase with confidence"


class TestStreakTracking:
    """Test win/loss streak tracking."""

    def test_winning_streak_tracked(self, kelly: KellyCriterion) -> None:
        """Test that winning streaks are tracked."""
        # Record 5 wins
        for _ in range(5):
            kelly.record_trade_result("test_strategy", is_win=True, pnl_pct=2.0)

        summary = kelly.get_strategy_performance_summary("test_strategy")
        assert summary is not None
        assert summary["current_streak"] == 5
        assert summary["max_win_streak"] == 5

    def test_losing_streak_tracked(self, kelly: KellyCriterion) -> None:
        """Test that losing streaks are tracked as negative."""
        # Record 3 losses
        for _ in range(3):
            kelly.record_trade_result("test_strategy", is_win=False, pnl_pct=-1.0)

        summary = kelly.get_strategy_performance_summary("test_strategy")
        assert summary is not None
        assert summary["current_streak"] == -3
        assert summary["max_loss_streak"] == 3  # max_loss_streak is stored as positive

    def test_streak_reset_on_direction_change(self, kelly: KellyCriterion) -> None:
        """Test that streak resets when direction changes."""
        # Win, win, loss
        kelly.record_trade_result("test_strategy", is_win=True, pnl_pct=2.0)
        kelly.record_trade_result("test_strategy", is_win=True, pnl_pct=1.5)
        kelly.record_trade_result("test_strategy", is_win=False, pnl_pct=-1.0)

        summary = kelly.get_strategy_performance_summary("test_strategy")
        assert summary["current_streak"] == -1
        assert summary["max_win_streak"] == 2


class TestAdaptiveFraction:
    """Test adaptive Kelly fraction."""

    def test_fraction_increases_on_wins(self, kelly: KellyCriterion) -> None:
        """Test that adaptive fraction increases on wins."""
        initial_summary = kelly.get_strategy_performance_summary("test_strategy")
        initial_fraction = initial_summary.get("adaptive_kelly", 0.25)

        # Record several wins
        for _ in range(10):
            kelly.record_trade_result("test_strategy", is_win=True, pnl_pct=2.0)

        summary = kelly.get_strategy_performance_summary("test_strategy")
        assert summary["adaptive_kelly"] >= initial_fraction

    def test_fraction_decreases_on_losses(self, kelly: KellyCriterion) -> None:
        """Test that adaptive fraction decreases on losses."""
        # Start with some wins to establish a baseline
        for _ in range(5):
            kelly.record_trade_result("test_strategy", is_win=True, pnl_pct=2.0)

        summary_before = kelly.get_strategy_performance_summary("test_strategy")
        fraction_before = summary_before["adaptive_kelly"]

        # Record several losses
        for _ in range(10):
            kelly.record_trade_result("test_strategy", is_win=False, pnl_pct=-1.5)

        summary_after = kelly.get_strategy_performance_summary("test_strategy")
        assert summary_after["adaptive_kelly"] <= fraction_before

    def test_fraction_bounded(self, kelly: KellyCriterion) -> None:
        """Test that adaptive fraction stays within bounds."""
        # Many wins - should not exceed max
        for _ in range(50):
            kelly.record_trade_result("test_strategy", is_win=True, pnl_pct=5.0)

        summary = kelly.get_strategy_performance_summary("test_strategy")
        assert summary["adaptive_kelly"] <= kelly.MAX_FRACTIONAL_KELLY
        assert summary["adaptive_kelly"] >= kelly.MIN_FRACTIONAL_KELLY


class TestBayesianWinRate:
    """Test Bayesian win rate estimation."""

    def test_bayesian_with_few_trades(self, kelly: KellyCriterion) -> None:
        """Test that Bayesian estimate is conservative with few trades."""
        # Only 5 trades, all wins - shouldn't assume 100% win rate
        metrics = {
            "test_strategy": {
                "wins": 5,
                "losses": 0,
                "avg_win": 2.0,
                "avg_loss": 1.0,
            }
        }

        win_rate = kelly.calculate_bayesian_win_rate("test_strategy", metrics)
        # Should be less than 100% due to prior
        assert win_rate < 1.0
        assert win_rate > 0.5  # But should still be positive

    def test_bayesian_with_many_trades(self, kelly: KellyCriterion) -> None:
        """Test that Bayesian converges to actual rate with many trades."""
        # 100 trades, 60% win rate
        metrics = {
            "test_strategy": {
                "wins": 60,
                "losses": 40,
                "avg_win": 2.0,
                "avg_loss": 1.0,
            }
        }

        win_rate = kelly.calculate_bayesian_win_rate("test_strategy", metrics)
        # Should be close to 60%
        assert 0.55 <= win_rate <= 0.65


class TestStreakMultiplier:
    """Test streak-based position scaling."""

    def test_losing_streak_penalty(self, kelly: KellyCriterion) -> None:
        """Test that losing streaks apply penalty."""
        # Record losing streak exceeding threshold
        for _ in range(kelly.STREAK_PENALTY_THRESHOLD + 2):
            kelly.record_trade_result("test_strategy", is_win=False, pnl_pct=-1.0)

        # Get tracker and check multiplier
        tracker = kelly._get_or_create_tracker("test_strategy")
        multiplier = kelly._calculate_streak_multiplier(tracker)
        assert multiplier < 1.0, "Losing streak should reduce multiplier"

    def test_winning_streak_boost(self, kelly: KellyCriterion) -> None:
        """Test that winning streaks apply boost."""
        # Record winning streak exceeding threshold
        for _ in range(kelly.STREAK_BOOST_THRESHOLD + 2):
            kelly.record_trade_result("test_strategy", is_win=True, pnl_pct=2.0)

        # Get tracker and check multiplier
        tracker = kelly._get_or_create_tracker("test_strategy")
        multiplier = kelly._calculate_streak_multiplier(tracker)
        assert multiplier >= 1.0, "Winning streak should not reduce multiplier"


class TestGlobalPerformance:
    """Test global performance aggregation."""

    def test_global_performance_aggregates(self, kelly: KellyCriterion) -> None:
        """Test that global performance aggregates across strategies."""
        # Record trades for multiple strategies
        for _ in range(5):
            kelly.record_trade_result("strategy_a", is_win=True, pnl_pct=2.0)
            kelly.record_trade_result("strategy_b", is_win=False, pnl_pct=-1.0)

        global_perf = kelly.get_global_performance()

        assert global_perf is not None
        assert "strategies_tracked" in global_perf
        assert global_perf["strategies_tracked"] == 2
        assert global_perf["total_trades"] == 10
        assert global_perf["total_wins"] == 5

    def test_missing_strategy_returns_default(self, kelly: KellyCriterion) -> None:
        """Test that missing strategy returns default summary."""
        summary = kelly.get_strategy_performance_summary("nonexistent")
        # Should return default values for nonexistent strategy
        assert summary is not None
        assert summary["trades"] == 0
        assert summary["current_streak"] == 0

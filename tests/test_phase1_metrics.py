"""Tests for Phase 1 metrics tracking."""

import pytest

from hean.observability.phase1_metrics import Phase1Metrics


def test_kelly_calculation_tracking() -> None:
    """Test Kelly fraction calculation tracking."""
    metrics = Phase1Metrics()

    # Record initial calculation
    metrics.record_kelly_calculation("strategy_a", 0.25)
    assert metrics.kelly_fractions["strategy_a"] == 0.25
    assert metrics.kelly_adjustments == 0  # No adjustment yet

    # Record significant change (triggers adjustment)
    metrics.record_kelly_calculation("strategy_a", 0.30)
    assert metrics.kelly_fractions["strategy_a"] == 0.30
    assert metrics.kelly_adjustments == 1

    # Record small change (no adjustment)
    metrics.record_kelly_calculation("strategy_a", 0.305)
    assert metrics.kelly_adjustments == 1  # Still 1


def test_confidence_scaling_tracking() -> None:
    """Test confidence scaling event tracking."""
    metrics = Phase1Metrics()

    metrics.record_confidence_scaling(boosted=True)
    assert metrics.confidence_boosts == 1

    metrics.record_confidence_scaling(boosted=True)
    assert metrics.confidence_boosts == 2

    metrics.record_confidence_scaling(boosted=False)
    assert metrics.confidence_boosts == 2  # No change


def test_streak_penalty_tracking() -> None:
    """Test streak penalty tracking."""
    metrics = Phase1Metrics()

    metrics.record_streak_penalty()
    assert metrics.streak_penalties == 1

    metrics.record_streak_penalty()
    assert metrics.streak_penalties == 2


def test_ttl_adjustment_tracking() -> None:
    """Test adaptive TTL adjustment tracking."""
    metrics = Phase1Metrics()

    # Set initial TTL
    metrics.record_ttl_adjustment(100.0)
    assert metrics.adaptive_ttl_ms == 100.0
    assert metrics.ttl_adjustments == 0  # No adjustment yet

    # Significant change (>10ms)
    metrics.record_ttl_adjustment(120.0)
    assert metrics.adaptive_ttl_ms == 120.0
    assert metrics.ttl_adjustments == 1

    # Small change (<=10ms)
    metrics.record_ttl_adjustment(125.0)
    assert metrics.ttl_adjustments == 1  # Still 1


def test_offset_adjustment_tracking() -> None:
    """Test adaptive offset tracking."""
    metrics = Phase1Metrics()

    metrics.record_offset_adjustment(2.5)
    assert metrics.adaptive_offset_bps == 2.5

    metrics.record_offset_adjustment(3.0)
    assert metrics.adaptive_offset_bps == 3.0


def test_maker_fill_tracking() -> None:
    """Test maker fill tracking."""
    metrics = Phase1Metrics()

    metrics.record_maker_fill()
    assert metrics.maker_fills == 1
    assert len(metrics.fill_rate_window) == 1
    assert metrics.fill_rate_window[0] is True


def test_maker_expiration_tracking() -> None:
    """Test maker expiration tracking."""
    metrics = Phase1Metrics()

    metrics.record_maker_expiration()
    assert metrics.maker_expirations == 1
    assert len(metrics.fill_rate_window) == 1
    assert metrics.fill_rate_window[0] is False


def test_fill_rate_calculation() -> None:
    """Test fill rate percentage calculation."""
    metrics = Phase1Metrics()

    # Empty
    assert metrics.get_fill_rate_pct() == 0.0

    # 100% fill rate
    metrics.record_maker_fill()
    metrics.record_maker_fill()
    assert metrics.get_fill_rate_pct() == 100.0

    # 50% fill rate
    metrics.record_maker_expiration()
    metrics.record_maker_expiration()
    assert metrics.get_fill_rate_pct() == 50.0

    # 75% fill rate
    metrics.record_maker_fill()
    metrics.record_maker_fill()
    assert metrics.get_fill_rate_pct() == pytest.approx(66.67, rel=0.01)


def test_imbalance_signal_tracking() -> None:
    """Test orderbook imbalance signal tracking."""
    metrics = Phase1Metrics()

    metrics.record_imbalance_signal(edge_bps=5.0)
    assert metrics.imbalance_signals == 1
    assert metrics.imbalance_edge_bps_total == 5.0

    metrics.record_imbalance_signal(edge_bps=7.0)
    assert metrics.imbalance_signals == 2
    assert metrics.imbalance_edge_bps_total == 12.0

    assert metrics.get_average_imbalance_edge_bps() == 6.0


def test_regime_sizing_tracking() -> None:
    """Test regime-aware sizing tracking."""
    metrics = Phase1Metrics()

    # Boost
    metrics.record_regime_sizing("IMPULSE", 1.15, is_boost=True)
    assert metrics.current_regime == "IMPULSE"
    assert metrics.current_size_multiplier == 1.15
    assert metrics.regime_boosts == 1
    assert metrics.regime_reductions == 0

    # Reduction
    metrics.record_regime_sizing("RANGE", 0.7, is_boost=False)
    assert metrics.current_regime == "RANGE"
    assert metrics.current_size_multiplier == 0.7
    assert metrics.regime_boosts == 1
    assert metrics.regime_reductions == 1


def test_summary_export() -> None:
    """Test metrics summary export."""
    metrics = Phase1Metrics()

    # Populate metrics
    metrics.record_kelly_calculation("strategy_a", 0.25)
    metrics.record_kelly_calculation("strategy_b", 0.30)
    metrics.record_ttl_adjustment(100.0)
    metrics.record_maker_fill()
    metrics.record_maker_fill()
    metrics.record_maker_expiration()
    metrics.record_imbalance_signal(5.0)
    metrics.record_regime_sizing("IMPULSE", 1.15, is_boost=True)

    summary = metrics.get_summary()

    # Verify structure
    assert "kelly_strategies_tracked" in summary
    assert "kelly_avg_fraction" in summary
    assert "adaptive_ttl_ms" in summary
    assert "maker_fill_rate_pct" in summary
    assert "imbalance_avg_edge_bps" in summary
    assert "current_regime" in summary

    # Verify values
    assert summary["kelly_strategies_tracked"] == 2
    assert summary["kelly_avg_fraction"] == pytest.approx(0.275)
    assert summary["adaptive_ttl_ms"] == 100.0
    assert summary["maker_fill_rate_pct"] == pytest.approx(66.67, rel=0.01)
    assert summary["imbalance_avg_edge_bps"] == 5.0
    assert summary["current_regime"] == "IMPULSE"
    assert summary["current_size_multiplier"] == 1.15


def test_reset() -> None:
    """Test metrics reset."""
    metrics = Phase1Metrics()

    # Populate
    metrics.record_kelly_calculation("strategy_a", 0.25)
    metrics.record_ttl_adjustment(100.0)
    metrics.record_maker_fill()
    metrics.record_imbalance_signal(5.0)
    metrics.record_regime_sizing("IMPULSE", 1.15, is_boost=True)

    # Reset
    metrics.reset()

    # Verify all cleared
    assert len(metrics.kelly_fractions) == 0
    assert metrics.kelly_adjustments == 0
    assert metrics.adaptive_ttl_ms == 0.0
    assert metrics.maker_fills == 0
    assert len(metrics.fill_rate_window) == 0
    assert metrics.imbalance_signals == 0
    assert metrics.current_regime is None
    assert metrics.update_count == 0

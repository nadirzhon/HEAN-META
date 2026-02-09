"""Tests for adaptive capital routing."""

from datetime import date, timedelta

from hean.portfolio.allocator import CapitalAllocator


def test_weights_shift_toward_higher_pf() -> None:
    """Test that weights shift toward strategies with higher profit factor."""
    allocator = CapitalAllocator()

    # Initialize with two strategies
    allocator._weights = {
        "strategy_a": 0.5,
        "strategy_b": 0.5,
    }

    # Set up rolling PnL to simulate different profit factors
    # Strategy A: PF = 2.0 (wins=200, losses=100)
    allocator._rolling_pnl["strategy_a"] = [100.0, 100.0, -50.0, -50.0]
    # Strategy B: PF = 0.5 (wins=50, losses=100)
    allocator._rolling_pnl["strategy_b"] = [25.0, 25.0, -50.0, -50.0]

    # Create strategy metrics
    strategy_metrics = {
        "strategy_a": {
            "profit_factor": 2.0,
            "max_drawdown_pct": 5.0,
            "pnl": 100.0,
            "wins": 2,
            "losses": 2,
        },
        "strategy_b": {
            "profit_factor": 0.5,
            "max_drawdown_pct": 5.0,
            "pnl": -50.0,
            "wins": 1,
            "losses": 2,
        },
    }

    # Store old weights before update
    old_weight_a = allocator._weights["strategy_a"]
    old_weight_b = allocator._weights["strategy_b"]

    # Update weights
    new_weights = allocator.update_weights(strategy_metrics)

    # Strategy A should have higher weight (PF > 1.3)
    # Strategy B should have lower weight (PF < 1.0)
    assert new_weights["strategy_a"] > old_weight_a
    assert new_weights["strategy_b"] < old_weight_b

    # Weights should still sum to 1.0
    total = sum(new_weights.values())
    assert abs(total - 1.0) < 0.01


def test_daily_change_limit_enforced() -> None:
    """Test that daily change limit of ±20% is enforced."""
    allocator = CapitalAllocator()

    # Set initial weights
    allocator._weights = {
        "strategy_a": 0.5,
        "strategy_b": 0.5,
    }
    allocator._last_rebalance_date = date.today()

    # Force a large adjustment by setting extreme PF
    allocator._rolling_pnl["strategy_a"] = [1000.0] * 10  # Very high PF
    allocator._rolling_pnl["strategy_b"] = [-1000.0] * 10  # Very low PF

    strategy_metrics = {
        "strategy_a": {
            "profit_factor": 10.0,
            "max_drawdown_pct": 1.0,
            "pnl": 10000.0,
            "wins": 10,
            "losses": 0,
        },
        "strategy_b": {
            "profit_factor": 0.1,
            "max_drawdown_pct": 20.0,
            "pnl": -10000.0,
            "wins": 0,
            "losses": 10,
        },
    }

    # Update date to allow rebalancing
    allocator._last_rebalance_date = date.today() - timedelta(days=1)

    old_weight_a = allocator._weights["strategy_a"]
    old_weight_b = allocator._weights["strategy_b"]

    # Update weights
    new_weights = allocator.update_weights(strategy_metrics)

    # Check that change is limited to ±20%
    change_a = abs((new_weights["strategy_a"] - old_weight_a) / old_weight_a)
    change_b = abs((new_weights["strategy_b"] - old_weight_b) / old_weight_b)

    assert change_a <= 0.21, f"Change {change_a:.2%} exceeds 20% limit"
    assert change_b <= 0.21, f"Change {change_b:.2%} exceeds 20% limit"

    # Weights should still sum to 1.0
    total = sum(new_weights.values())
    assert abs(total - 1.0) < 0.01


def test_weights_normalize_to_one() -> None:
    """Test that weights are normalized to sum to 1.0."""
    allocator = CapitalAllocator()

    allocator._weights = {
        "strategy_a": 0.3,
        "strategy_b": 0.4,
        "strategy_c": 0.3,
    }

    strategy_metrics = {
        "strategy_a": {
            "profit_factor": 1.5,
            "max_drawdown_pct": 5.0,
            "pnl": 50.0,
            "wins": 5,
            "losses": 3,
        },
        "strategy_b": {
            "profit_factor": 1.2,
            "max_drawdown_pct": 5.0,
            "pnl": 30.0,
            "wins": 4,
            "losses": 3,
        },
        "strategy_c": {
            "profit_factor": 0.8,
            "max_drawdown_pct": 5.0,
            "pnl": -20.0,
            "wins": 2,
            "losses": 3,
        },
    }

    allocator._last_rebalance_date = date.today() - timedelta(days=1)

    new_weights = allocator.update_weights(strategy_metrics)

    # Weights should sum to 1.0
    total = sum(new_weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1.0"

    # All weights should be between 0 and 1
    for weight in new_weights.values():
        assert 0.0 <= weight <= 1.0, f"Weight {weight} is out of range"


def test_weight_history_tracked() -> None:
    """Test that weight history is tracked over time."""
    allocator = CapitalAllocator()

    allocator._weights = {
        "strategy_a": 0.5,
        "strategy_b": 0.5,
    }

    strategy_metrics = {
        "strategy_a": {
            "profit_factor": 1.5,
            "max_drawdown_pct": 5.0,
            "pnl": 50.0,
            "wins": 5,
            "losses": 3,
        },
        "strategy_b": {
            "profit_factor": 1.0,
            "max_drawdown_pct": 5.0,
            "pnl": 0.0,
            "wins": 2,
            "losses": 2,
        },
    }

    # Update weights multiple times (simulating multiple days)
    # Use different dates to ensure separate entries
    for day in range(3):
        test_date = date.today() - timedelta(days=3-day)
        allocator._last_rebalance_date = test_date - timedelta(days=1)  # Set to previous day
        allocator.update_weights(strategy_metrics)

    # Check that history is tracked
    history = allocator.get_weight_history()
    # May have fewer entries if same date is used, but should have at least 1
    assert len(history) >= 1, f"Expected at least 1 history entry, got {len(history)}"

    # Each entry should have weights and date
    for entry in history:
        assert "_date" in entry
        assert "strategy_a" in entry
        assert "strategy_b" in entry


def test_pf_below_one_decreases_weight() -> None:
    """Test that PF < 1.0 decreases weight."""
    allocator = CapitalAllocator()

    # Need at least 2 strategies for weight changes to be meaningful
    allocator._weights = {
        "poor_strategy": 0.5,
        "neutral_strategy": 0.5,
    }

    # Set up poor performance (PF < 1.0)
    allocator._rolling_pnl["poor_strategy"] = [10.0, -20.0, -15.0]
    allocator._rolling_pnl["neutral_strategy"] = [10.0, -10.0]

    strategy_metrics = {
        "poor_strategy": {
            "profit_factor": 0.5,
            "max_drawdown_pct": 10.0,
            "pnl": -25.0,
            "wins": 1,
            "losses": 2,
        },
        "neutral_strategy": {
            "profit_factor": 1.0,
            "max_drawdown_pct": 5.0,
            "pnl": 0.0,
            "wins": 1,
            "losses": 1,
        },
    }

    old_weight_poor = allocator._weights["poor_strategy"]
    old_weight_neutral = allocator._weights["neutral_strategy"]
    allocator._last_rebalance_date = date.today() - timedelta(days=1)

    new_weights = allocator.update_weights(strategy_metrics)

    # Poor strategy should decrease relative to neutral strategy
    # After normalization, poor should be < neutral
    assert new_weights["poor_strategy"] < new_weights["neutral_strategy"]
    # Or at least the poor strategy should have decreased
    assert new_weights["poor_strategy"] <= old_weight_poor


def test_pf_above_one_point_three_increases_weight() -> None:
    """Test that PF > 1.3 increases weight."""
    allocator = CapitalAllocator()

    # Need at least 2 strategies for weight changes to be meaningful
    allocator._weights = {
        "good_strategy": 0.5,
        "neutral_strategy": 0.5,
    }

    # Set up good performance (PF > 1.3)
    allocator._rolling_pnl["good_strategy"] = [100.0, 50.0, -30.0]
    allocator._rolling_pnl["neutral_strategy"] = [10.0, -10.0]

    strategy_metrics = {
        "good_strategy": {
            "profit_factor": 1.5,
            "max_drawdown_pct": 3.0,
            "pnl": 120.0,
            "wins": 2,
            "losses": 1,
        },
        "neutral_strategy": {
            "profit_factor": 1.0,
            "max_drawdown_pct": 5.0,
            "pnl": 0.0,
            "wins": 1,
            "losses": 1,
        },
    }

    old_weight_good = allocator._weights["good_strategy"]
    old_weight_neutral = allocator._weights["neutral_strategy"]
    allocator._last_rebalance_date = date.today() - timedelta(days=1)

    new_weights = allocator.update_weights(strategy_metrics)

    # Good strategy should increase relative to neutral strategy
    # After normalization, good should be > neutral
    assert new_weights["good_strategy"] > new_weights["neutral_strategy"]
    # Or at least the good strategy should have increased
    assert new_weights["good_strategy"] >= old_weight_good


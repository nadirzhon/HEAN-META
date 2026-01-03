"""Tests for strategy memory and penalty system."""

from datetime import date, datetime, timedelta

import pytest

from hean.core.regime import Regime
from hean.portfolio.allocator import CapitalAllocator
from hean.portfolio.strategy_memory import StrategyMemory


def test_weight_reduced_after_poor_pf() -> None:
    """Test that weight is reduced after poor profit factor."""
    allocator = CapitalAllocator()
    memory = allocator.get_strategy_memory()
    
    # Initialize with two strategies
    allocator._weights = {
        "strategy_a": 0.5,
        "strategy_b": 0.5,
    }
    
    # Record multiple losing trades for strategy_a to get PF < 1.0
    for _ in range(10):
        memory.record_trade("strategy_a", -10.0, Regime.NORMAL)
    
    # Record some winning trades but not enough
    for _ in range(5):
        memory.record_trade("strategy_a", 5.0, Regime.NORMAL)
    
    # Strategy_a should have PF < 1.0 (50 wins / 100 losses = 0.5)
    pf = memory.get_rolling_profit_factor("strategy_a")
    assert pf < 1.0, f"Expected PF < 1.0, got {pf}"
    
    # Create strategy metrics
    strategy_metrics = {
        "strategy_a": {
            "profit_factor": pf,
            "max_drawdown_pct": 5.0,
            "pnl": -75.0,  # Net loss
            "wins": 5,
            "losses": 10,
        },
        "strategy_b": {
            "profit_factor": 1.5,
            "max_drawdown_pct": 5.0,
            "pnl": 50.0,
            "wins": 10,
            "losses": 5,
        },
    }
    
    old_weight_a = allocator._weights["strategy_a"]
    allocator._last_rebalance_date = date.today() - timedelta(days=1)
    
    # Update weights
    new_weights = allocator.update_weights(strategy_metrics)
    
    # Strategy_a should have reduced weight due to poor PF
    assert new_weights["strategy_a"] < old_weight_a, \
        f"Expected reduced weight, got {new_weights['strategy_a']} >= {old_weight_a}"
    
    # Weights should still sum to 1.0
    total = sum(new_weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1.0"


def test_regime_specific_penalty_works() -> None:
    """Test that regime-specific penalties work."""
    memory = StrategyMemory()
    
    # Record 3 consecutive losses in IMPULSE regime for strategy_a
    for _ in range(3):
        memory.record_trade("strategy_a", -10.0, Regime.IMPULSE)
    
    # Record some winning trades in NORMAL regime to keep PF good for NORMAL
    for _ in range(5):
        memory.record_trade("strategy_a", 15.0, Regime.NORMAL)
    
    # Regime should be blacklisted
    assert memory.is_regime_blacklisted("strategy_a", Regime.IMPULSE), \
        "Expected IMPULSE regime to be blacklisted"
    
    # Other regimes should not be blacklisted
    assert not memory.is_regime_blacklisted("strategy_a", Regime.NORMAL), \
        "NORMAL regime should not be blacklisted"
    assert not memory.is_regime_blacklisted("strategy_a", Regime.RANGE), \
        "RANGE regime should not be blacklisted"
    
    # should_reduce_weight should return True for blacklisted regime
    assert memory.should_reduce_weight("strategy_a", Regime.IMPULSE), \
        "Should reduce weight for blacklisted regime"
    
    # should_reduce_weight should return False for non-blacklisted regime (PF is good)
    assert not memory.should_reduce_weight("strategy_a", Regime.NORMAL), \
        "Should not reduce weight for non-blacklisted regime with good PF"
    
    # Penalty multiplier should be < 1.0 for blacklisted regime
    multiplier = memory.get_penalty_multiplier("strategy_a", Regime.IMPULSE)
    assert multiplier < 1.0, f"Expected penalty multiplier < 1.0, got {multiplier}"
    
    # Penalty multiplier should be 1.0 for non-blacklisted regime (PF is good)
    multiplier_normal = memory.get_penalty_multiplier("strategy_a", Regime.NORMAL)
    assert multiplier_normal == 1.0, \
        f"Expected no penalty for NORMAL regime with good PF, got {multiplier_normal}"


def test_memory_recovers_after_cooldown() -> None:
    """Test that memory recovers after cooldown period expires."""
    memory = StrategyMemory()
    
    # Set up high drawdown to trigger cooldown
    memory.update_equity("strategy_a", 10000.0)  # Initial equity
    memory.update_equity("strategy_a", 8000.0)  # 20% drawdown (above 15% threshold)
    
    # Record some winning trades to keep PF good
    for _ in range(5):
        memory.record_trade("strategy_a", 15.0, Regime.NORMAL)
    
    # Strategy should be in cooldown
    assert memory.is_in_cooldown("strategy_a"), \
        "Expected strategy to be in cooldown after high drawdown"
    
    assert memory.should_reduce_weight("strategy_a", Regime.NORMAL), \
        "Should reduce weight during cooldown"
    
    # Simulate time passing by manually expiring the cooldown
    memory._cooldown_until["strategy_a"] = date.today() - timedelta(days=1)
    
    # Also reduce drawdown below threshold
    memory.update_equity("strategy_a", 9500.0)  # Drawdown now ~5% (below 15% threshold)
    
    # Cooldown should be expired
    assert not memory.is_in_cooldown("strategy_a"), \
        "Expected cooldown to be expired"
    
    # Should not reduce weight after cooldown expires (PF is good, drawdown is low)
    assert not memory.should_reduce_weight("strategy_a", Regime.NORMAL), \
        "Should not reduce weight after cooldown expires and conditions improve"
    
    # Penalty multiplier should be 1.0 after cooldown expires and conditions improve
    multiplier = memory.get_penalty_multiplier("strategy_a", Regime.NORMAL)
    assert multiplier == 1.0, \
        f"Expected no penalty after cooldown expires and conditions improve, got {multiplier}"


def test_regime_blacklist_expires() -> None:
    """Test that regime blacklist expires after duration."""
    memory = StrategyMemory()
    
    # Record 3 consecutive losses to trigger blacklist
    for _ in range(3):
        memory.record_trade("strategy_a", -10.0, Regime.IMPULSE)
    
    # Record winning trades in IMPULSE to improve PF for that regime
    for _ in range(5):
        memory.record_trade("strategy_a", 15.0, Regime.IMPULSE)
    
    # Regime should be blacklisted
    assert memory.is_regime_blacklisted("strategy_a", Regime.IMPULSE)
    
    # Manually expire the blacklist
    memory._regime_blacklist[("strategy_a", "impulse")] = date.today() - timedelta(days=1)
    
    # Blacklist should be expired
    assert not memory.is_regime_blacklisted("strategy_a", Regime.IMPULSE), \
        "Expected blacklist to be expired"
    
    # should_reduce_weight should return False (blacklist expired, PF is good)
    assert not memory.should_reduce_weight("strategy_a", Regime.IMPULSE), \
        "Should not reduce weight after blacklist expires and PF improves"


def test_rolling_pf_calculation() -> None:
    """Test rolling profit factor calculation."""
    memory = StrategyMemory()
    
    # Record trades: 5 wins of $10, 3 losses of $5
    # PF = (5 * 10) / (3 * 5) = 50 / 15 = 3.33
    for _ in range(5):
        memory.record_trade("strategy_a", 10.0, Regime.NORMAL)
    for _ in range(3):
        memory.record_trade("strategy_a", -5.0, Regime.NORMAL)
    
    pf = memory.get_rolling_profit_factor("strategy_a")
    expected_pf = (5 * 10.0) / (3 * 5.0)  # 50 / 15 = 3.33...
    assert abs(pf - expected_pf) < 0.01, \
        f"Expected PF ≈ {expected_pf}, got {pf}"


def test_drawdown_tracking() -> None:
    """Test drawdown tracking."""
    memory = StrategyMemory()
    
    # Set initial equity
    memory.update_equity("strategy_a", 10000.0)
    assert memory.get_rolling_drawdown("strategy_a") == 0.0
    
    # Increase equity (new peak)
    memory.update_equity("strategy_a", 12000.0)
    assert memory.get_rolling_drawdown("strategy_a") == 0.0
    
    # Decrease equity (drawdown)
    memory.update_equity("strategy_a", 11000.0)
    drawdown = memory.get_rolling_drawdown("strategy_a")
    expected_dd = ((12000.0 - 11000.0) / 12000.0) * 100.0  # 8.33%
    assert abs(drawdown - expected_dd) < 0.01, \
        f"Expected drawdown ≈ {expected_dd}%, got {drawdown}%"


def test_penalty_applied_before_normalization() -> None:
    """Test that penalties are applied before weight normalization."""
    allocator = CapitalAllocator()
    memory = allocator.get_strategy_memory()
    
    # Initialize with two strategies
    allocator._weights = {
        "strategy_a": 0.5,
        "strategy_b": 0.5,
    }
    
    # Make strategy_a have poor PF (will trigger PF penalty and regime blacklist)
    for _ in range(10):
        memory.record_trade("strategy_a", -10.0, Regime.NORMAL)
    for _ in range(3):
        memory.record_trade("strategy_a", 5.0, Regime.NORMAL)
    
    strategy_metrics = {
        "strategy_a": {
            "profit_factor": 0.15,  # Very poor (15 wins / 100 losses)
            "max_drawdown_pct": 5.0,
            "pnl": -85.0,
            "wins": 3,
            "losses": 10,
        },
        "strategy_b": {
            "profit_factor": 1.5,
            "max_drawdown_pct": 5.0,
            "pnl": 50.0,
            "wins": 10,
            "losses": 5,
        },
    }
    
    allocator._last_rebalance_date = date.today() - timedelta(days=1)
    
    # Update weights
    new_weights = allocator.update_weights(strategy_metrics)
    
    # Strategy_a should have lower weight due to penalties (PF < 1.0 and regime blacklist)
    # With penalties applied, weight should be reduced
    assert new_weights["strategy_a"] < 0.5, \
        f"Expected strategy_a weight < 0.5 after penalties, got {new_weights['strategy_a']}"
    
    # Weights should still sum to 1.0 (normalization preserved)
    total = sum(new_weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1.0"


def test_min_max_weight_bounds_respected() -> None:
    """Test that min/max weight bounds are respected after penalties."""
    allocator = CapitalAllocator()
    memory = allocator.get_strategy_memory()
    
    # Initialize with three strategies
    allocator._weights = {
        "strategy_a": 0.33,
        "strategy_b": 0.33,
        "strategy_c": 0.34,
    }
    
    # Make strategy_a have very poor performance (multiple penalties)
    for _ in range(10):
        memory.record_trade("strategy_a", -10.0, Regime.IMPULSE)  # Will blacklist regime
    
    # Set high drawdown
    memory.update_equity("strategy_a", 10000.0)
    memory.update_equity("strategy_a", 8000.0)  # 20% drawdown
    
    strategy_metrics = {
        "strategy_a": {
            "profit_factor": 0.1,
            "max_drawdown_pct": 20.0,
            "pnl": -2000.0,
            "wins": 0,
            "losses": 10,
        },
        "strategy_b": {
            "profit_factor": 1.5,
            "max_drawdown_pct": 5.0,
            "pnl": 50.0,
            "wins": 10,
            "losses": 5,
        },
        "strategy_c": {
            "profit_factor": 1.2,
            "max_drawdown_pct": 5.0,
            "pnl": 30.0,
            "wins": 8,
            "losses": 5,
        },
    }
    
    allocator._last_rebalance_date = date.today() - timedelta(days=1)
    
    # Update weights
    new_weights = allocator.update_weights(strategy_metrics)
    
    # Strategy_a should still have minimum weight (5%)
    assert new_weights["strategy_a"] >= 0.05, \
        f"Expected minimum weight 0.05, got {new_weights['strategy_a']}"
    
    # All weights should be between 0 and 1
    for strategy_id, weight in new_weights.items():
        assert 0.0 <= weight <= 1.0, \
            f"Weight for {strategy_id} is {weight}, out of bounds"
    
    # Weights should sum to 1.0
    total = sum(new_weights.values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, not 1.0"






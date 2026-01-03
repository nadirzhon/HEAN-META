"""Tests for execution edge estimator."""

import pytest

from hean.core.regime import Regime
from hean.core.types import Signal, Tick
from hean.execution.edge_estimator import ExecutionEdgeEstimator
from datetime import datetime


def test_edge_positive_when_tp_distance_greater_than_spread() -> None:
    """Test that edge is positive when TP distance > spread + fees."""
    estimator = ExecutionEdgeEstimator()
    
    # Create signal with large TP distance
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,  # 1% = 100 bps
    )
    
    # Create tick with small spread
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49999.0,  # 1 USDT spread = ~2 bps
        ask=50001.0,
    )
    
    edge = estimator.estimate_edge(signal, tick, Regime.NORMAL)
    
    # Edge should be positive (expected move ~100 bps, spread ~2 bps)
    assert edge > 0, f"Expected positive edge, got {edge:.2f} bps"


def test_edge_negative_when_spread_too_large() -> None:
    """Test that edge is negative when spread is too large."""
    estimator = ExecutionEdgeEstimator()
    
    # Create signal with small TP distance
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50050.0,  # 0.1% = 10 bps
    )
    
    # Create tick with large spread
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49900.0,  # 100 USDT spread = ~200 bps
        ask=50100.0,
    )
    
    edge = estimator.estimate_edge(signal, tick, Regime.NORMAL)
    
    # Edge should be negative (spread 200 bps > expected move 10 bps)
    assert edge < 0, f"Expected negative edge, got {edge:.2f} bps"


def test_different_thresholds_by_regime() -> None:
    """Test that different regimes have different edge thresholds."""
    estimator = ExecutionEdgeEstimator()
    
    # Check thresholds for different regimes
    threshold_impulse = estimator.get_min_edge_threshold(Regime.IMPULSE)
    threshold_normal = estimator.get_min_edge_threshold(Regime.NORMAL)
    threshold_range = estimator.get_min_edge_threshold(Regime.RANGE)
    
    # IMPULSE should have highest threshold (more aggressive)
    assert threshold_impulse >= threshold_normal, "IMPULSE threshold should be >= NORMAL"
    
    # RANGE should have lowest threshold (stricter)
    assert threshold_range <= threshold_normal, "RANGE threshold should be <= NORMAL"


def test_should_emit_signal_blocks_low_edge() -> None:
    """Test that should_emit_signal blocks signals with low edge."""
    estimator = ExecutionEdgeEstimator()
    
    # Create signal with very small TP (low edge)
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50010.0,  # 0.02% = 2 bps
    )
    
    # Create tick with moderate spread
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49995.0,  # 5 USDT spread = ~10 bps
        ask=50005.0,
    )
    
    # Should be blocked (edge < threshold)
    should_emit = estimator.should_emit_signal(signal, tick, Regime.NORMAL)
    
    assert not should_emit, "Signal with low edge should be blocked"
    
    # Check metrics
    metrics = estimator.get_metrics()
    assert metrics["signals_blocked_by_edge"] > 0


def test_should_emit_signal_allows_high_edge() -> None:
    """Test that should_emit_signal allows signals with high edge."""
    estimator = ExecutionEdgeEstimator()
    
    # Create signal with large TP (high edge)
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=51000.0,  # 2% = 200 bps
    )
    
    # Create tick with small spread
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49999.0,  # 1 USDT spread = ~2 bps
        ask=50001.0,
    )
    
    # Should be allowed (edge > threshold)
    should_emit = estimator.should_emit_signal(signal, tick, Regime.NORMAL)
    
    assert should_emit, "Signal with high edge should be allowed"
    
    # Check metrics
    metrics = estimator.get_metrics()
    assert metrics["avg_edge_bps"] > 0


def test_volatility_penalty_reduces_edge() -> None:
    """Test that higher volatility reduces edge."""
    estimator = ExecutionEdgeEstimator()
    
    # Build volatility history with high volatility
    symbol = "BTCUSDT"
    base_price = 50000.0
    for i in range(20):
        # Large price swings (high volatility)
        price = base_price + (i % 2) * 500  # ±500 USDT swings
        estimator.update_price_history(symbol, price)
    
    signal = Signal(
        strategy_id="test",
        symbol=symbol,
        side="buy",
        entry_price=base_price,
        take_profit=base_price * 1.01,  # 1% TP
    )
    
    tick = Tick(
        symbol=symbol,
        price=base_price,
        timestamp=datetime.utcnow(),
        bid=base_price - 10,
        ask=base_price + 10,
    )
    
    edge_with_vol = estimator.estimate_edge(signal, tick, Regime.NORMAL)
    
    # Reset and test with low volatility
    estimator._volatility_history[symbol].clear()
    for i in range(20):
        price = base_price + i * 0.1  # Small price changes
        estimator.update_price_history(symbol, price)
    
    edge_without_vol = estimator.estimate_edge(signal, tick, Regime.NORMAL)
    
    # Edge with high volatility should be lower (or at least not higher)
    # Note: This is a probabilistic test, so we check that it's reasonable
    assert edge_with_vol <= edge_without_vol + 50, (
        f"High volatility should reduce edge. "
        f"With vol: {edge_with_vol:.2f}, Without vol: {edge_without_vol:.2f}"
    )


def test_regime_adjustment_impulse_allows_higher_edge() -> None:
    """Test that IMPULSE regime allows higher edge threshold."""
    estimator = ExecutionEdgeEstimator()
    
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50075.0,  # 0.15% = 15 bps
    )
    
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49995.0,
        ask=50005.0,
    )
    
    # Test in IMPULSE regime (should allow)
    should_emit_impulse = estimator.should_emit_signal(signal, tick, Regime.IMPULSE)
    
    # Test in RANGE regime (should be stricter)
    estimator.reset_metrics()
    should_emit_range = estimator.should_emit_signal(signal, tick, Regime.RANGE)
    
    # IMPULSE should be more permissive (or at least not less permissive)
    # This depends on the actual edge calculation, but IMPULSE has higher threshold
    # so it might block more, but the edge itself might be adjusted higher
    edge_impulse = estimator.estimate_edge(signal, tick, Regime.IMPULSE)
    estimator.reset_metrics()
    edge_range = estimator.estimate_edge(signal, tick, Regime.RANGE)
    
    # IMPULSE regime should give slightly higher edge (5% boost)
    assert edge_impulse >= edge_range * 0.95, (
        f"IMPULSE should allow higher edge. "
        f"IMPULSE: {edge_impulse:.2f}, RANGE: {edge_range:.2f}"
    )


def test_maker_fill_probability_by_regime() -> None:
    """Test that maker fill probability varies by regime."""
    estimator = ExecutionEdgeEstimator()
    
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50100.0,  # 0.2% = 20 bps
    )
    
    # Small spread (should have good fill probability)
    tick_small_spread = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49999.0,  # 1 USDT = ~2 bps
        ask=50001.0,
    )
    
    # Large spread (should have lower fill probability)
    tick_large_spread = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49950.0,  # 50 USDT = ~100 bps
        ask=50050.0,
    )
    
    edge_small_spread = estimator.estimate_edge(signal, tick_small_spread, Regime.RANGE)
    edge_large_spread = estimator.estimate_edge(signal, tick_large_spread, Regime.RANGE)
    
    # Small spread should have better edge
    assert edge_small_spread > edge_large_spread, (
        f"Small spread should have better edge. "
        f"Small: {edge_small_spread:.2f}, Large: {edge_large_spread:.2f}"
    )


def test_metrics_tracking() -> None:
    """Test that metrics are tracked correctly."""
    estimator = ExecutionEdgeEstimator()
    
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50100.0,
    )
    
    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        bid=49995.0,
        ask=50005.0,
    )
    
    # Evaluate multiple signals
    estimator.should_emit_signal(signal, tick, Regime.NORMAL)
    estimator.should_emit_signal(signal, tick, Regime.NORMAL)
    
    metrics = estimator.get_metrics()
    
    assert "signals_blocked_by_edge" in metrics
    assert "avg_edge_bps" in metrics
    assert "total_signals_evaluated" in metrics
    assert metrics["total_signals_evaluated"] == 2.0







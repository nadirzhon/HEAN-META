"""Tests for ML-based edge estimator functionality."""

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from hean.core.regime import Regime
from hean.core.types import Signal, Tick
from hean.execution.edge_estimator import ExecutionEdgeEstimator


@pytest.fixture
def estimator() -> ExecutionEdgeEstimator:
    """Create edge estimator for testing."""
    return ExecutionEdgeEstimator()


def test_ml_components_initialization(estimator: ExecutionEdgeEstimator) -> None:
    """Test that ML components are properly initialized."""
    assert hasattr(estimator, "_ml_enabled")
    assert hasattr(estimator, "_ml_model")
    assert hasattr(estimator, "_training_data")
    assert hasattr(estimator, "_ofi_history")
    assert estimator._ml_enabled is False
    assert estimator._ml_model is None


def test_extract_features(estimator: ExecutionEdgeEstimator) -> None:
    """Test feature extraction for ML model."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    features = estimator._extract_features(signal, tick, Regime.NORMAL)

    # Should have 8 features
    assert features.shape == (8,)
    assert isinstance(features, np.ndarray)

    # Check feature types
    spread_bps = features[0]
    assert spread_bps > 0  # Should be positive

    volatility = features[1]
    assert 0 <= volatility <= 1  # Should be normalized

    time_of_day = features[3]
    assert 0 <= time_of_day <= 1  # Normalized hour

    expected_move_bps = features[6]
    assert expected_move_bps > 0  # TP is above entry for buy


def test_ofi_calculation(estimator: ExecutionEdgeEstimator) -> None:
    """Test Order Flow Imbalance calculation."""
    # Add several ticks with buy pressure to build history
    for _ in range(5):
        tick_buy_pressure = Tick(
            symbol="BTCUSDT",
            price=50004.0,  # Closer to ask
            timestamp=datetime.utcnow(),
            volume=1.0,
            bid=49995.0,
            ask=50005.0,
        )
        ofi_buy = estimator._get_ofi(tick_buy_pressure)

    # After several ticks, OFI should be positive
    assert ofi_buy > 0, "OFI should be positive when price closer to ask"

    # Now add sell pressure ticks (more than buy to override the average)
    for _ in range(20):
        tick_sell_pressure = Tick(
            symbol="BTCUSDT",
            price=49996.0,  # Closer to bid
            timestamp=datetime.utcnow(),
            volume=1.0,
            bid=49995.0,
            ask=50005.0,
        )
        ofi_sell = estimator._get_ofi(tick_sell_pressure)

    # After several ticks, OFI should become negative
    assert ofi_sell < 0, f"OFI should be negative when price closer to bid, got {ofi_sell}"


def test_ofi_smoothing(estimator: ExecutionEdgeEstimator) -> None:
    """Test that OFI is smoothed over time."""
    # Add multiple ticks
    for i in range(20):
        tick = Tick(
            symbol="BTCUSDT",
            price=50000.0 + i,
            timestamp=datetime.utcnow(),
            volume=1.0,
            bid=49995.0 + i,
            ask=50005.0 + i,
        )
        estimator._get_ofi(tick)

    assert len(estimator._ofi_history["BTCUSDT"]) == 20


def test_update_ml_model(estimator: ExecutionEdgeEstimator) -> None:
    """Test updating ML model with outcomes."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # Update with outcome
    estimator.update_ml_model(signal, tick, Regime.NORMAL, actual_outcome=50.0)

    assert len(estimator._training_data) == 1
    assert "features" in estimator._training_data[0]
    assert "actual_edge" in estimator._training_data[0]
    assert estimator._training_data[0]["actual_edge"] == 50.0


def test_train_ml_model_requires_minimum_samples(estimator: ExecutionEdgeEstimator) -> None:
    """Test that training requires minimum samples."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # Add fewer than 100 samples
    for i in range(50):
        estimator.update_ml_model(signal, tick, Regime.NORMAL, actual_outcome=float(i))

    # Should not have trained yet
    assert estimator._ml_model is None
    assert estimator._ml_enabled is False


def test_train_ml_model_with_sufficient_samples(estimator: ExecutionEdgeEstimator) -> None:
    """Test that model trains with sufficient samples."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # Add sufficient samples
    for i in range(100):
        estimator.update_ml_model(signal, tick, Regime.NORMAL, actual_outcome=float(i))

    # Manually trigger training
    estimator._train_ml_model()

    # Model should be trained
    assert estimator._ml_model is not None
    assert "weights" in estimator._ml_model
    assert "bias" in estimator._ml_model
    assert estimator._feature_scaler is not None
    assert estimator._ml_enabled is True


def test_ml_prediction(estimator: ExecutionEdgeEstimator) -> None:
    """Test ML-based edge prediction."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # Train model first
    for i in range(100):
        estimator.update_ml_model(signal, tick, Regime.NORMAL, actual_outcome=100.0)
    estimator._train_ml_model()

    # Get ML prediction
    edge_ml = estimator.estimate_edge_ml(signal, tick, Regime.NORMAL)

    assert isinstance(edge_ml, float)
    # Should return a reasonable edge value
    assert -1000 < edge_ml < 1000


def test_ml_fallback_to_rule_based(estimator: ExecutionEdgeEstimator) -> None:
    """Test that ML falls back to rule-based when not trained."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # ML not trained, should use rule-based
    edge_ml = estimator.estimate_edge_ml(signal, tick, Regime.NORMAL)
    edge_rule = estimator.estimate_edge(signal, tick, Regime.NORMAL)

    assert edge_ml == edge_rule, "Should fallback to rule-based when ML not trained"


def test_save_load_model(estimator: ExecutionEdgeEstimator) -> None:
    """Test saving and loading ML model."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # Train model
    for i in range(100):
        estimator.update_ml_model(signal, tick, Regime.NORMAL, actual_outcome=float(i))
    estimator._train_ml_model()

    # Save model
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        model_path = f.name

    success = estimator.save_model(model_path)
    assert success
    assert Path(model_path).exists()

    # Create new estimator and load
    estimator2 = ExecutionEdgeEstimator()
    load_success = estimator2.load_model(model_path)
    assert load_success
    assert estimator2._ml_model is not None
    assert estimator2._ml_enabled is True

    # Cleanup
    Path(model_path).unlink()


def test_enable_disable_ml(estimator: ExecutionEdgeEstimator) -> None:
    """Test enabling/disabling ML predictions."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # Train model
    for i in range(100):
        estimator.update_ml_model(signal, tick, Regime.NORMAL, actual_outcome=float(i))
    estimator._train_ml_model()

    assert estimator._ml_enabled is True

    # Disable ML
    estimator.enable_ml(False)
    assert estimator._ml_enabled is False

    # Enable ML
    estimator.enable_ml(True)
    assert estimator._ml_enabled is True


def test_feature_normalization(estimator: ExecutionEdgeEstimator) -> None:
    """Test that features are normalized during training."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # Train model
    for i in range(100):
        estimator.update_ml_model(signal, tick, Regime.NORMAL, actual_outcome=float(i))
    estimator._train_ml_model()

    # Check that scaler was created
    assert estimator._feature_scaler is not None
    assert "mean" in estimator._feature_scaler
    assert "std" in estimator._feature_scaler

    mean = estimator._feature_scaler["mean"]
    std = estimator._feature_scaler["std"]

    assert mean.shape == (8,)  # 8 features
    assert std.shape == (8,)


def test_online_learning_improves_accuracy(estimator: ExecutionEdgeEstimator) -> None:
    """Test that online learning improves prediction accuracy."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    # Pattern: small spread -> high edge
    tick_good = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49999.0,  # Small spread
        ask=50001.0,
    )

    # Pattern: large spread -> low edge
    tick_bad = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49900.0,  # Large spread
        ask=50100.0,
    )

    # Train with pattern
    for _ in range(50):
        estimator.update_ml_model(signal, tick_good, Regime.NORMAL, actual_outcome=100.0)
        estimator.update_ml_model(signal, tick_bad, Regime.NORMAL, actual_outcome=-50.0)

    estimator._train_ml_model()

    # Test predictions
    edge_good = estimator.estimate_edge_ml(signal, tick_good, Regime.NORMAL)
    edge_bad = estimator.estimate_edge_ml(signal, tick_bad, Regime.NORMAL)

    # Model should have learned the pattern (good > bad)
    # Note: This is probabilistic, but should generally hold
    assert edge_good > edge_bad


def test_regime_features_extraction(estimator: ExecutionEdgeEstimator) -> None:
    """Test that regime is correctly encoded in features."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # Test IMPULSE regime
    features_impulse = estimator._extract_features(signal, tick, Regime.IMPULSE)
    assert features_impulse[4] == 1.0  # regime_impulse
    assert features_impulse[5] == 0.0  # regime_range

    # Test RANGE regime
    features_range = estimator._extract_features(signal, tick, Regime.RANGE)
    assert features_range[4] == 0.0  # regime_impulse
    assert features_range[5] == 1.0  # regime_range

    # Test NORMAL regime
    features_normal = estimator._extract_features(signal, tick, Regime.NORMAL)
    assert features_normal[4] == 0.0  # regime_impulse
    assert features_normal[5] == 0.0  # regime_range


def test_training_data_maxlen(estimator: ExecutionEdgeEstimator) -> None:
    """Test that training data respects maxlen."""
    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        take_profit=50500.0,
    )

    tick = Tick(
        symbol="BTCUSDT",
        price=50000.0,
        timestamp=datetime.utcnow(),
        volume=10.0,
        bid=49995.0,
        ask=50005.0,
    )

    # Add more than maxlen samples
    for i in range(11000):
        estimator.update_ml_model(signal, tick, Regime.NORMAL, actual_outcome=float(i))

    # Should be capped at maxlen (10000)
    assert len(estimator._training_data) == 10000

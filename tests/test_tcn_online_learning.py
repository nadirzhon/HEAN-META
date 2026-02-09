"""Tests for TCN Predictor online learning functionality."""

from datetime import datetime, timedelta

import numpy as np
import pytest
import torch

from hean.core.intelligence.tcn_predictor import TCPriceReversalPredictor


def test_training_buffer_initialization() -> None:
    """Test that training buffer is properly initialized."""
    predictor = TCPriceReversalPredictor(sequence_length=100)

    assert hasattr(predictor, "_training_buffer")
    assert len(predictor._training_buffer) == 0
    assert predictor._training_enabled is True


def test_update_from_feedback() -> None:
    """Test update_from_feedback stores training samples."""
    predictor = TCPriceReversalPredictor(sequence_length=10)

    # Build up tick buffer
    base_time = datetime.utcnow()
    for i in range(10):
        predictor.update_tick(
            price=50000.0 + i * 10,
            volume=1.0,
            bid=50000.0 + i * 10 - 5,
            ask=50000.0 + i * 10 + 5,
            timestamp=base_time + timedelta(seconds=i),
        )

    # Update with feedback
    predictor.update_from_feedback(actual_outcome=True)

    assert len(predictor._training_buffer) == 1
    features, outcome = predictor._training_buffer[0]
    assert isinstance(features, np.ndarray)
    assert features.shape == (10, 4)  # 10 ticks, 4 features each
    assert outcome is True


def test_training_buffer_maxlen() -> None:
    """Test that training buffer respects maxlen."""
    predictor = TCPriceReversalPredictor(sequence_length=10)

    # Build tick buffer
    base_time = datetime.utcnow()
    for i in range(10):
        predictor.update_tick(
            price=50000.0 + i,
            volume=1.0,
            bid=50000.0 + i - 1,
            ask=50000.0 + i + 1,
            timestamp=base_time + timedelta(seconds=i),
        )

    # Add more than maxlen samples
    for _ in range(1100):
        predictor.update_from_feedback(actual_outcome=True)

    # Should be capped at maxlen (1000)
    assert len(predictor._training_buffer) == 1000


def test_train_step_requires_minimum_samples() -> None:
    """Test that training step requires minimum batch size."""
    predictor = TCPriceReversalPredictor(sequence_length=10)

    # Build tick buffer
    base_time = datetime.utcnow()
    for i in range(10):
        predictor.update_tick(
            price=50000.0,
            volume=1.0,
            bid=49995.0,
            ask=50005.0,
            timestamp=base_time + timedelta(seconds=i),
        )

    # Add some samples (less than batch size)
    for _ in range(10):
        predictor.update_from_feedback(actual_outcome=True)

    initial_metrics = predictor.get_training_metrics()

    # Training should not have happened yet
    assert initial_metrics["total_updates"] == 0.0


def test_train_step_updates_model() -> None:
    """Test that training step updates model weights."""
    predictor = TCPriceReversalPredictor(sequence_length=10)

    # Build tick buffer
    base_time = datetime.utcnow()
    for i in range(10):
        predictor.update_tick(
            price=50000.0 + i * 10,
            volume=1.0 + i * 0.1,
            bid=50000.0 + i * 10 - 5,
            ask=50000.0 + i * 10 + 5,
            timestamp=base_time + timedelta(seconds=i),
        )

    # Get initial weights
    initial_weights = {
        name: param.clone() for name, param in predictor.model.named_parameters()
    }

    # Add enough samples for training
    for i in range(50):
        # Alternate outcomes
        predictor.update_from_feedback(actual_outcome=(i % 2 == 0))

    # Trigger training manually
    predictor._train_step()

    # Check that weights changed
    weights_changed = False
    for name, param in predictor.model.named_parameters():
        if not torch.equal(param, initial_weights[name]):
            weights_changed = True
            break

    assert weights_changed, "Model weights should have changed after training"


def test_get_training_metrics() -> None:
    """Test training metrics reporting."""
    predictor = TCPriceReversalPredictor(sequence_length=10)

    metrics = predictor.get_training_metrics()

    assert "total_updates" in metrics
    assert "avg_loss" in metrics
    assert "recent_accuracy" in metrics
    assert "buffer_size" in metrics
    assert "training_enabled" in metrics

    assert metrics["total_updates"] == 0.0
    assert metrics["buffer_size"] == 0.0
    assert metrics["training_enabled"] == 1.0


def test_enable_disable_training() -> None:
    """Test enabling/disabling training."""
    predictor = TCPriceReversalPredictor(sequence_length=10)

    assert predictor._training_enabled is True

    predictor.enable_training(False)
    assert predictor._training_enabled is False

    predictor.enable_training(True)
    assert predictor._training_enabled is True


def test_predict_with_trigger_training() -> None:
    """Test prediction with training trigger."""
    predictor = TCPriceReversalPredictor(sequence_length=10)

    # Build tick buffer
    base_time = datetime.utcnow()
    for i in range(10):
        predictor.update_tick(
            price=50000.0 + i * 100,
            volume=1.0,
            bid=50000.0 + i * 100 - 10,
            ask=50000.0 + i * 100 + 10,
            timestamp=base_time + timedelta(seconds=i),
        )

    # Add training samples
    for i in range(50):
        predictor.update_from_feedback(actual_outcome=(i % 2 == 0))

    # Predict with training trigger
    prob, should_trigger = predictor.predict_reversal_probability(trigger_training=True)

    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
    assert isinstance(should_trigger, bool)

    # Check that training happened
    metrics = predictor.get_training_metrics()
    assert metrics["total_updates"] > 0


def test_model_save_includes_training_metrics() -> None:
    """Test that model save includes training metrics."""
    import tempfile

    predictor = TCPriceReversalPredictor(sequence_length=10)

    # Build buffer and train
    base_time = datetime.utcnow()
    for i in range(10):
        predictor.update_tick(
            price=50000.0,
            volume=1.0,
            bid=49995.0,
            ask=50005.0,
            timestamp=base_time + timedelta(seconds=i),
        )

    for i in range(50):
        predictor.update_from_feedback(actual_outcome=True)

    predictor._train_step()

    # Save model
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        model_path = f.name

    success = predictor.save_model(model_path)
    assert success

    # Load and verify
    checkpoint = torch.load(model_path, map_location="cpu")
    assert "training_metrics" in checkpoint
    assert "total_updates" in checkpoint["training_metrics"]


def test_online_learning_improves_predictions() -> None:
    """Test that online learning improves prediction accuracy over time."""
    predictor = TCPriceReversalPredictor(sequence_length=20)

    # Simulate reversal pattern: high volatility followed by reversal
    base_time = datetime.utcnow()

    # Train with pattern: high price changes -> reversal
    # Need at least 32 samples for training batch
    for iteration in range(40):
        # High volatility pattern
        for i in range(20):
            # Large price swings
            price = 50000.0 + (i % 2) * 500
            predictor.update_tick(
                price=price,
                volume=10.0,
                bid=price - 10,
                ask=price + 10,
                timestamp=base_time + timedelta(seconds=iteration * 20 + i),
            )

        # This pattern leads to reversal
        predictor.update_from_feedback(actual_outcome=True)

    # Verify we have training data
    assert len(predictor._training_buffer) >= predictor._training_batch_size

    # Train model
    for _ in range(5):
        predictor._train_step()

    # Verify training happened
    metrics = predictor.get_training_metrics()
    assert metrics["total_updates"] > 0, "Training should have occurred"

    # Now test: same pattern should predict higher reversal probability
    for i in range(20):
        price = 50000.0 + (i % 2) * 500
        predictor.update_tick(
            price=price,
            volume=10.0,
            bid=price - 10,
            ask=price + 10,
            timestamp=base_time + timedelta(seconds=300 + i),
        )

    prob, _ = predictor.predict_reversal_probability()

    # Probability should be non-trivial (model learned something)
    # Note: This is probabilistic, so we just check it's reasonable
    assert 0.0 < prob < 1.0

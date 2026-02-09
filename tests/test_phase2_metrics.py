"""Tests for Phase 2 Metrics."""

import pytest
from datetime import datetime

from hean.observability.phase2_metrics import Phase2Metrics, phase2_metrics


class TestPhase2Metrics:
    """Tests for Phase2Metrics."""

    @pytest.fixture
    def metrics(self):
        """Create fresh metrics instance."""
        m = Phase2Metrics()
        m.reset()
        return m

    def test_initialization(self, metrics):
        """Test metrics initialization."""
        assert metrics.ml_predictions_made == 0
        assert metrics.ttl_current_ms == 0.0
        assert metrics.decay_signals_tracked == 0

    def test_record_ml_prediction(self, metrics):
        """Test recording ML predictions."""
        metrics.record_ml_prediction(
            quality_score=0.75,
            model_version=1,
            confidence_changed=True,
        )

        assert metrics.ml_predictions_made == 1
        assert metrics.ml_model_version == 1
        assert metrics.ml_confidence_adjustments == 1
        assert len(metrics.quality_score_history) == 1

    def test_record_ml_prediction_average(self, metrics):
        """Test ML prediction average calculation."""
        scores = [0.6, 0.7, 0.8, 0.9, 0.75]
        for score in scores:
            metrics.record_ml_prediction(score, 1, False)

        expected_avg = sum(scores) / len(scores)
        assert metrics.ml_avg_quality_score == pytest.approx(expected_avg)

    def test_record_ml_feature_importance(self, metrics):
        """Test recording feature importance."""
        importance = {
            "feature_1": 0.5,
            "feature_2": 0.3,
            "feature_3": 0.2,
        }

        metrics.record_ml_feature_importance(importance)

        assert len(metrics.ml_feature_importance_top3) == 3
        assert "feature_1" in metrics.ml_feature_importance_top3

    def test_record_ttl_adjustment(self, metrics):
        """Test recording TTL adjustments."""
        metrics.record_ttl_adjustment(600.0, "spread")
        metrics.record_ttl_adjustment(700.0, "hour")
        metrics.record_ttl_adjustment(800.0, "volatility")

        assert metrics.ttl_current_ms == 800.0
        assert metrics.ttl_spread_adjustments == 1
        assert metrics.ttl_hour_adjustments == 1
        assert metrics.ttl_volatility_adjustments == 1

    def test_record_ttl_learning(self, metrics):
        """Test recording TTL learning."""
        metrics.record_ttl_learning(
            fill_time_ms=350.0,
            samples=25,
        )

        assert metrics.ttl_avg_fill_time_ms == 350.0
        assert metrics.ttl_learning_samples == 25

    def test_record_decay_tracking(self, metrics):
        """Test recording decay tracking."""
        metrics.record_decay_tracking(
            signals_tracked=5,
            signals_expired=2,
            avg_age_seconds=120.0,
        )

        assert metrics.decay_signals_tracked == 5
        assert metrics.decay_signals_expired == 2
        assert metrics.decay_avg_age_seconds == 120.0

    def test_record_decay_urgent_execution(self, metrics):
        """Test recording decay urgent executions."""
        metrics.record_decay_urgent_execution()
        metrics.record_decay_urgent_execution()

        assert metrics.decay_urgent_executions == 2

    def test_record_decay_prevented_wait(self, metrics):
        """Test recording decay prevented waits."""
        metrics.record_decay_prevented_wait()

        assert metrics.decay_prevented_waits == 1

    def test_record_combined_confidence_change(self, metrics):
        """Test recording combined confidence changes."""
        metrics.record_combined_confidence_change(increased=True)
        metrics.record_combined_confidence_change(increased=True)
        metrics.record_combined_confidence_change(increased=False)

        assert metrics.combined_confidence_boost == 2
        assert metrics.combined_confidence_reduction == 1

    def test_get_summary(self, metrics):
        """Test getting metrics summary."""
        # Record some data
        metrics.record_ml_prediction(0.75, 1, True)
        metrics.record_ttl_adjustment(600.0, "spread")
        metrics.record_decay_tracking(3, 1, 150.0)

        summary = metrics.get_summary()

        # ML metrics
        assert summary["ml_predictions_made"] == 1
        assert summary["ml_model_version"] == 1
        assert summary["ml_avg_quality_score"] > 0

        # TTL metrics
        assert summary["ttl_current_ms"] == 600.0
        assert summary["ttl_total_adjustments"] == 1

        # Decay metrics
        assert summary["decay_signals_tracked"] == 3
        assert summary["decay_expiration_rate"] > 0

        # Meta
        assert summary["last_updated"] is not None

    def test_ml_adjustment_rate(self, metrics):
        """Test ML adjustment rate calculation."""
        metrics.record_ml_prediction(0.7, 1, True)
        metrics.record_ml_prediction(0.8, 1, False)
        metrics.record_ml_prediction(0.6, 1, True)

        summary = metrics.get_summary()

        # 2 out of 3 predictions changed confidence
        assert summary["ml_adjustment_rate"] == pytest.approx(2/3)

    def test_decay_expiration_rate(self, metrics):
        """Test decay expiration rate calculation."""
        metrics.record_decay_tracking(
            signals_tracked=10,
            signals_expired=3,
            avg_age_seconds=200.0,
        )

        summary = metrics.get_summary()
        assert summary["decay_expiration_rate"] == pytest.approx(0.3)

    def test_ttl_adjustment_distribution(self, metrics):
        """Test TTL adjustment distribution."""
        metrics.record_ttl_adjustment(500.0, "spread")
        metrics.record_ttl_adjustment(550.0, "spread")
        metrics.record_ttl_adjustment(600.0, "hour")
        metrics.record_ttl_adjustment(650.0, "volatility")

        summary = metrics.get_summary()

        assert summary["ttl_spread_adjustments"] == 2
        assert summary["ttl_hour_adjustments"] == 1
        assert summary["ttl_volatility_adjustments"] == 1
        assert summary["ttl_total_adjustments"] == 4

    def test_reset(self, metrics):
        """Test metrics reset."""
        # Record some data
        metrics.record_ml_prediction(0.75, 1, True)
        metrics.record_ttl_adjustment(600.0, "spread")
        metrics.record_decay_tracking(3, 1, 150.0)

        assert metrics.ml_predictions_made > 0

        # Reset
        metrics.reset()

        assert metrics.ml_predictions_made == 0
        assert metrics.ttl_current_ms == 0.0
        assert metrics.decay_signals_tracked == 0
        assert len(metrics.quality_score_history) == 0
        assert metrics.last_updated is None

    def test_quality_score_history_limit(self, metrics):
        """Test that quality score history is limited."""
        # Record more than maxlen (100)
        for i in range(150):
            metrics.record_ml_prediction(0.5 + (i % 50) / 100.0, 1, False)

        # Should be limited to 100
        assert len(metrics.quality_score_history) == 100

    def test_update_count(self, metrics):
        """Test update count tracking."""
        initial_count = metrics.update_count

        metrics.record_ml_prediction(0.75, 1, False)
        metrics.record_ttl_adjustment(600.0, "spread")
        metrics.record_decay_urgent_execution()

        # Should have incremented
        assert metrics.update_count > initial_count

    def test_last_updated_timestamp(self, metrics):
        """Test last updated timestamp."""
        before = datetime.utcnow()

        metrics.record_ml_prediction(0.75, 1, False)

        after = datetime.utcnow()

        assert metrics.last_updated is not None
        assert before <= metrics.last_updated <= after


class TestGlobalPhase2Metrics:
    """Tests for global phase2_metrics instance."""

    def test_global_instance(self):
        """Test that global instance exists."""
        assert phase2_metrics is not None
        assert isinstance(phase2_metrics, Phase2Metrics)

    def test_global_instance_recording(self):
        """Test recording to global instance."""
        # Reset first
        phase2_metrics.reset()

        initial_predictions = phase2_metrics.ml_predictions_made

        phase2_metrics.record_ml_prediction(0.75, 1, False)

        assert phase2_metrics.ml_predictions_made == initial_predictions + 1


@pytest.mark.asyncio
async def test_phase2_metrics_integration():
    """Integration test for Phase 2 metrics."""
    metrics = Phase2Metrics()
    metrics.reset()

    # Simulate a trading session with Phase 2 features
    for i in range(20):
        # ML predictions
        metrics.record_ml_prediction(
            quality_score=0.6 + (i % 10) / 25.0,
            model_version=1,
            confidence_changed=(i % 3 == 0),
        )

        # TTL adjustments
        if i % 5 == 0:
            metrics.record_ttl_adjustment(500.0 + i * 10, "spread")
        elif i % 5 == 1:
            metrics.record_ttl_adjustment(500.0 + i * 10, "hour")
        else:
            metrics.record_ttl_adjustment(500.0 + i * 10, "volatility")

    # Decay tracking
    metrics.record_decay_tracking(
        signals_tracked=15,
        signals_expired=5,
        avg_age_seconds=180.0,
    )

    # Decay events
    metrics.record_decay_urgent_execution()
    metrics.record_decay_urgent_execution()
    metrics.record_decay_prevented_wait()

    # Combined confidence changes
    for i in range(10):
        metrics.record_combined_confidence_change(increased=(i % 2 == 0))

    # TTL learning
    metrics.record_ttl_learning(fill_time_ms=425.0, samples=50)

    # Feature importance
    metrics.record_ml_feature_importance({
        "price_momentum_5m": 0.25,
        "volume_ratio_5m": 0.20,
        "orderbook_imbalance": 0.15,
    })

    # Get summary
    summary = metrics.get_summary()

    # Verify comprehensive data
    assert summary["ml_predictions_made"] == 20
    assert summary["ml_avg_quality_score"] > 0
    assert summary["ttl_total_adjustments"] == 20
    assert summary["ttl_learning_samples"] == 50
    assert summary["decay_signals_tracked"] == 15
    assert summary["decay_urgent_executions"] == 2
    assert summary["decay_prevented_waits"] == 1
    assert summary["combined_confidence_boost"] == 5
    assert summary["combined_confidence_reduction"] == 5
    assert len(summary["ml_top_features"]) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

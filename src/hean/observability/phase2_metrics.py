"""Phase 2 Performance Tracking Metrics.

Tracks ML Signal Quality, Enhanced Adaptive TTL, Signal Decay,
and other Phase 2 improvements.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Phase2Metrics:
    """Tracks Phase 2 enhancement metrics."""

    # ML Signal Quality metrics
    ml_predictions_made: int = 0
    ml_model_version: int = 0
    ml_avg_quality_score: float = 0.0
    ml_confidence_adjustments: int = 0  # Times ML changed final confidence
    ml_feature_importance_top3: dict[str, float] = field(default_factory=dict)

    # Enhanced Adaptive TTL metrics
    ttl_spread_adjustments: int = 0  # Times TTL adjusted for spread
    ttl_hour_adjustments: int = 0    # Times TTL adjusted for hour of day
    ttl_volatility_adjustments: int = 0  # Times TTL adjusted for volatility
    ttl_current_ms: float = 0.0
    ttl_avg_fill_time_ms: float = 0.0
    ttl_learning_samples: int = 0  # Samples used for learning

    # Signal Decay metrics
    decay_signals_tracked: int = 0
    decay_signals_expired: int = 0  # Signals that decayed to min confidence
    decay_avg_age_seconds: float = 0.0
    decay_urgent_executions: int = 0  # Executions triggered by decay urgency
    decay_prevented_waits: int = 0  # Times decay prevented waiting

    # Combined metrics (ML + Decay)
    combined_confidence_boost: int = 0  # Times combined factors increased confidence
    combined_confidence_reduction: int = 0  # Times combined factors decreased

    # Performance tracking
    quality_score_history: deque = field(default_factory=lambda: deque(maxlen=100))
    last_updated: datetime | None = None
    update_count: int = 0

    def record_ml_prediction(
        self,
        quality_score: float,
        model_version: int,
        confidence_changed: bool = False,
    ) -> None:
        """Record ML quality prediction.

        Args:
            quality_score: Predicted quality score (0.0 to 1.0)
            model_version: Current model version
            confidence_changed: Whether ML changed final confidence
        """
        self.ml_predictions_made += 1
        self.ml_model_version = model_version
        self.quality_score_history.append(quality_score)

        # Update running average
        if len(self.quality_score_history) > 0:
            self.ml_avg_quality_score = (
                sum(self.quality_score_history) / len(self.quality_score_history)
            )

        if confidence_changed:
            self.ml_confidence_adjustments += 1

        self.last_updated = datetime.utcnow()
        self.update_count += 1

        logger.debug(
            f"[Phase2Metrics] ML prediction: score={quality_score:.3f}, "
            f"avg={self.ml_avg_quality_score:.3f}"
        )

    def record_ml_feature_importance(self, feature_importance: dict[str, float]) -> None:
        """Record top features from ML model.

        Args:
            feature_importance: Dictionary of feature importances
        """
        # Get top 3 features
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        self.ml_feature_importance_top3 = dict(sorted_features)
        self.last_updated = datetime.utcnow()

    def record_ttl_adjustment(
        self,
        new_ttl_ms: float,
        adjustment_type: str,  # "spread" | "hour" | "volatility"
    ) -> None:
        """Record adaptive TTL adjustment.

        Args:
            new_ttl_ms: New TTL in milliseconds
            adjustment_type: Type of adjustment
        """
        self.ttl_current_ms = new_ttl_ms

        if adjustment_type == "spread":
            self.ttl_spread_adjustments += 1
        elif adjustment_type == "hour":
            self.ttl_hour_adjustments += 1
        elif adjustment_type == "volatility":
            self.ttl_volatility_adjustments += 1

        self.last_updated = datetime.utcnow()

        logger.debug(
            f"[Phase2Metrics] TTL adjusted: {new_ttl_ms:.0f}ms ({adjustment_type})"
        )

    def record_ttl_learning(
        self,
        fill_time_ms: float,
        samples: int,
    ) -> None:
        """Record TTL learning event.

        Args:
            fill_time_ms: Average fill time from learning
            samples: Number of samples used
        """
        self.ttl_avg_fill_time_ms = fill_time_ms
        self.ttl_learning_samples = samples
        self.last_updated = datetime.utcnow()

    def record_decay_tracking(
        self,
        signals_tracked: int,
        signals_expired: int,
        avg_age_seconds: float,
    ) -> None:
        """Record signal decay tracking stats.

        Args:
            signals_tracked: Number of signals being tracked
            signals_expired: Number of expired signals
            avg_age_seconds: Average age of tracked signals
        """
        self.decay_signals_tracked = signals_tracked
        self.decay_signals_expired = signals_expired
        self.decay_avg_age_seconds = avg_age_seconds
        self.last_updated = datetime.utcnow()

    def record_decay_urgent_execution(self) -> None:
        """Record execution triggered by decay urgency."""
        self.decay_urgent_executions += 1
        self.last_updated = datetime.utcnow()

        logger.info(
            f"[Phase2Metrics] Decay urgent execution #{self.decay_urgent_executions}"
        )

    def record_decay_prevented_wait(self) -> None:
        """Record time decay prevented waiting for better timing."""
        self.decay_prevented_waits += 1
        self.last_updated = datetime.utcnow()

    def record_combined_confidence_change(self, increased: bool) -> None:
        """Record combined confidence adjustment.

        Args:
            increased: True if confidence increased, False if decreased
        """
        if increased:
            self.combined_confidence_boost += 1
        else:
            self.combined_confidence_reduction += 1

        self.last_updated = datetime.utcnow()

    def get_summary(self) -> dict[str, Any]:
        """Get metrics summary for export.

        Returns:
            Dictionary of Phase 2 metrics
        """
        # Calculate ML effectiveness
        ml_adjustment_rate = 0.0
        if self.ml_predictions_made > 0:
            ml_adjustment_rate = self.ml_confidence_adjustments / self.ml_predictions_made

        # Calculate TTL adjustment distribution
        total_ttl_adjustments = (
            self.ttl_spread_adjustments
            + self.ttl_hour_adjustments
            + self.ttl_volatility_adjustments
        )

        # Calculate decay impact
        decay_expiration_rate = 0.0
        if self.decay_signals_tracked > 0:
            decay_expiration_rate = self.decay_signals_expired / self.decay_signals_tracked

        return {
            # ML metrics
            "ml_predictions_made": self.ml_predictions_made,
            "ml_model_version": self.ml_model_version,
            "ml_avg_quality_score": self.ml_avg_quality_score,
            "ml_confidence_adjustments": self.ml_confidence_adjustments,
            "ml_adjustment_rate": ml_adjustment_rate,
            "ml_top_features": self.ml_feature_importance_top3,

            # TTL metrics
            "ttl_current_ms": self.ttl_current_ms,
            "ttl_avg_fill_time_ms": self.ttl_avg_fill_time_ms,
            "ttl_learning_samples": self.ttl_learning_samples,
            "ttl_total_adjustments": total_ttl_adjustments,
            "ttl_spread_adjustments": self.ttl_spread_adjustments,
            "ttl_hour_adjustments": self.ttl_hour_adjustments,
            "ttl_volatility_adjustments": self.ttl_volatility_adjustments,

            # Decay metrics
            "decay_signals_tracked": self.decay_signals_tracked,
            "decay_signals_expired": self.decay_signals_expired,
            "decay_expiration_rate": decay_expiration_rate,
            "decay_avg_age_seconds": self.decay_avg_age_seconds,
            "decay_urgent_executions": self.decay_urgent_executions,
            "decay_prevented_waits": self.decay_prevented_waits,

            # Combined metrics
            "combined_confidence_boost": self.combined_confidence_boost,
            "combined_confidence_reduction": self.combined_confidence_reduction,

            # Meta
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "update_count": self.update_count,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.ml_predictions_made = 0
        self.ml_model_version = 0
        self.ml_avg_quality_score = 0.0
        self.ml_confidence_adjustments = 0
        self.ml_feature_importance_top3.clear()

        self.ttl_spread_adjustments = 0
        self.ttl_hour_adjustments = 0
        self.ttl_volatility_adjustments = 0
        self.ttl_current_ms = 0.0
        self.ttl_avg_fill_time_ms = 0.0
        self.ttl_learning_samples = 0

        self.decay_signals_tracked = 0
        self.decay_signals_expired = 0
        self.decay_avg_age_seconds = 0.0
        self.decay_urgent_executions = 0
        self.decay_prevented_waits = 0

        self.combined_confidence_boost = 0
        self.combined_confidence_reduction = 0

        self.quality_score_history.clear()
        self.last_updated = None
        self.update_count = 0


# Global Phase 2 metrics instance
phase2_metrics = Phase2Metrics()

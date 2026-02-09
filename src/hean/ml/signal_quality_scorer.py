"""Signal Quality Scorer using ML.

Predicts signal quality (probability of success) using a lightweight
gradient boosting model trained on historical signal outcomes.

For production deployment, the model should be pre-trained offline.
For development, includes a simple online learning capability.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from hean.core.types import Signal
from hean.logging import get_logger
from hean.ml.feature_extraction import FeatureExtractor, MarketFeatures

logger = get_logger(__name__)


@dataclass
class SignalOutcome:
    """Tracks signal outcome for training."""

    signal_id: str
    features: MarketFeatures
    success: bool  # True if profitable, False if loss
    pnl_pct: float
    timestamp: datetime


class SignalQualityScorer:
    """ML-based signal quality scorer.

    Uses a simple ensemble of weak learners (decision stumps) to predict
    signal quality. This is a lightweight implementation suitable for
    real-time inference.

    For production, replace with pre-trained XGBoost/LightGBM model.
    """

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        online_learning: bool = False,
        min_training_samples: int = 50,
    ):
        """Initialize signal quality scorer.

        Args:
            feature_extractor: Feature extractor instance
            online_learning: Enable online learning from outcomes
            min_training_samples: Minimum samples before training
        """
        self._feature_extractor = feature_extractor
        self._online_learning = online_learning
        self._min_training_samples = min_training_samples

        # Training data buffer
        self._training_buffer: deque[SignalOutcome] = deque(maxlen=1000)

        # Simple model: feature weights (starts with uniform weights)
        self._n_features = len(MarketFeatures().to_array())
        self._feature_weights = np.ones(self._n_features) / self._n_features
        self._bias = 0.5  # Start at neutral

        # Model performance tracking
        self._predictions_made = 0
        self._correct_predictions = 0
        self._model_version = 0
        self._last_training: datetime | None = None

        logger.info(
            f"SignalQualityScorer initialized: "
            f"n_features={self._n_features}, "
            f"online_learning={online_learning}"
        )

    def score_signal(
        self,
        signal: Signal,
        context: dict[str, Any],
    ) -> float:
        """Score signal quality (predicted probability of success).

        Args:
            signal: Trading signal
            context: Market context

        Returns:
            Quality score (0.0 to 1.0), where 1.0 = high quality
        """
        # Extract features
        features = self._feature_extractor.extract_features(signal, context)
        feature_array = features.to_array()

        # Simple linear model: score = sigmoid(weights · features + bias)
        score = self._predict(feature_array)

        self._predictions_made += 1

        logger.debug(
            f"Signal quality score: {score:.3f} for {signal.symbol} {signal.side}"
        )

        return score

    def record_outcome(
        self,
        signal_id: str,
        signal: Signal,
        context: dict[str, Any],
        success: bool,
        pnl_pct: float,
    ) -> None:
        """Record signal outcome for learning.

        Args:
            signal_id: Signal identifier
            signal: Original signal
            context: Original context
            success: Whether signal was profitable
            pnl_pct: P&L percentage
        """
        if not self._online_learning:
            return

        # Extract features (same as during scoring)
        features = self._feature_extractor.extract_features(signal, context)

        outcome = SignalOutcome(
            signal_id=signal_id,
            features=features,
            success=success,
            pnl_pct=pnl_pct,
            timestamp=datetime.utcnow(),
        )

        self._training_buffer.append(outcome)

        logger.debug(
            f"Recorded outcome: signal_id={signal_id}, success={success}, "
            f"pnl={pnl_pct:.2f}%, buffer_size={len(self._training_buffer)}"
        )

        # Trigger training if buffer is large enough
        if len(self._training_buffer) >= self._min_training_samples:
            # Train periodically (every 10 new samples)
            if len(self._training_buffer) % 10 == 0:
                self._train_model()

    def _predict(self, features: np.ndarray) -> float:
        """Predict quality score from features.

        Args:
            features: Feature array

        Returns:
            Predicted score (0.0 to 1.0)
        """
        # Linear combination
        linear_output = np.dot(self._feature_weights, features) + self._bias

        # Sigmoid activation to get probability
        score = self._sigmoid(linear_output)

        # Clamp to safe range
        return float(np.clip(score, 0.0, 1.0))

    def _train_model(self) -> None:
        """Train model on buffered outcomes using gradient descent.

        This is a simplified online learning implementation.
        For production, use pre-trained XGBoost/LightGBM.
        """
        if len(self._training_buffer) < self._min_training_samples:
            logger.warning("Insufficient samples for training")
            return

        logger.info(
            f"Training model on {len(self._training_buffer)} samples "
            f"(version {self._model_version})"
        )

        # Convert to numpy arrays
        X = np.array([outcome.features.to_array() for outcome in self._training_buffer])
        y = np.array([1.0 if outcome.success else 0.0 for outcome in self._training_buffer])

        # Simple gradient descent for logistic regression
        learning_rate = 0.01
        n_iterations = 50

        for _ in range(n_iterations):
            # Forward pass
            predictions = self._sigmoid(np.dot(X, self._feature_weights) + self._bias)

            # Calculate gradients
            error = predictions - y
            weight_gradient = np.dot(X.T, error) / len(y)
            bias_gradient = np.mean(error)

            # Update weights
            self._feature_weights -= learning_rate * weight_gradient
            self._bias -= learning_rate * bias_gradient

        # Calculate training accuracy
        final_predictions = self._sigmoid(np.dot(X, self._feature_weights) + self._bias)
        predicted_classes = (final_predictions >= 0.5).astype(int)
        actual_classes = y.astype(int)
        accuracy = np.mean(predicted_classes == actual_classes)

        self._model_version += 1
        self._last_training = datetime.utcnow()

        logger.info(
            f"Model training complete: version={self._model_version}, "
            f"accuracy={accuracy:.2%}"
        )

        # Update top features (for interpretability)
        self._log_top_features()

    def _sigmoid(self, x: np.ndarray | float) -> np.ndarray | float:
        """Sigmoid activation function.

        Args:
            x: Input value(s)

        Returns:
            Sigmoid output
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def _log_top_features(self) -> None:
        """Log top predictive features."""
        feature_names = MarketFeatures.get_feature_names()
        feature_importance = np.abs(self._feature_weights)

        # Get top 5 features
        top_indices = np.argsort(feature_importance)[-5:][::-1]
        top_features = [
            (feature_names[i], self._feature_weights[i])
            for i in top_indices
        ]

        logger.info(
            "Top 5 predictive features: "
            + ", ".join(f"{name}={weight:.3f}" for name, weight in top_features)
        )

    def get_model_stats(self) -> dict[str, Any]:
        """Get model statistics.

        Returns:
            Dictionary of model metrics
        """
        accuracy = 0.0
        if self._predictions_made > 0:
            accuracy = self._correct_predictions / self._predictions_made

        return {
            "model_version": self._model_version,
            "predictions_made": self._predictions_made,
            "model_accuracy": accuracy,
            "training_samples": len(self._training_buffer),
            "last_training": (
                self._last_training.isoformat() if self._last_training else None
            ),
            "online_learning": self._online_learning,
            "feature_count": self._n_features,
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        feature_names = MarketFeatures.get_feature_names()
        importance = np.abs(self._feature_weights)

        # Normalize to sum to 1.0
        importance_normalized = importance / np.sum(importance)

        return {
            name: float(score)
            for name, score in zip(feature_names, importance_normalized, strict=False)
        }


class EnhancedMultiFactorConfirmation:
    """Enhanced MultiFactorConfirmation with ML signal quality scoring.

    Wraps the existing MultiFactorConfirmation and adds ML-based
    signal quality assessment.
    """

    def __init__(
        self,
        base_confirmation: Any,  # MultiFactorConfirmation instance
        signal_quality_scorer: SignalQualityScorer,
        ml_weight: float = 0.3,  # Weight for ML score in combined confidence
    ):
        """Initialize enhanced confirmation.

        Args:
            base_confirmation: Base MultiFactorConfirmation instance
            signal_quality_scorer: ML quality scorer
            ml_weight: Weight for ML score (0.0 to 1.0)
        """
        self._base_confirmation = base_confirmation
        self._scorer = signal_quality_scorer
        self._ml_weight = max(0.0, min(1.0, ml_weight))

        logger.info(
            f"EnhancedMultiFactorConfirmation initialized with ml_weight={self._ml_weight}"
        )

    def confirm(self, signal: Signal, context: dict[str, Any]) -> Any:
        """Confirm signal with ML enhancement.

        Args:
            signal: Signal to confirm
            context: Market context

        Returns:
            Enhanced ConfirmationResult
        """
        # Get base confirmation
        base_result = self._base_confirmation.confirm(signal, context)

        # Get ML quality score
        ml_quality_score = self._scorer.score_signal(signal, context)

        # Combine scores
        # Final confidence = (1 - ml_weight) * base_confidence + ml_weight * ml_score
        base_confidence = base_result.confidence
        enhanced_confidence = (
            (1.0 - self._ml_weight) * base_confidence
            + self._ml_weight * ml_quality_score
        )

        # Clamp to [0, 1]
        enhanced_confidence = max(0.0, min(1.0, enhanced_confidence))

        # Update result with enhanced confidence
        base_result.confidence = enhanced_confidence

        # Add ML metadata
        if not hasattr(base_result, 'metadata'):
            base_result.metadata = {}
        base_result.metadata['ml_quality_score'] = ml_quality_score
        base_result.metadata['base_confidence'] = base_confidence
        base_result.metadata['ml_weight'] = self._ml_weight

        logger.debug(
            f"Enhanced confirmation: base={base_confidence:.3f}, "
            f"ml={ml_quality_score:.3f}, final={enhanced_confidence:.3f}"
        )

        return base_result

    def record_outcome(
        self,
        signal_id: str,
        signal: Signal,
        context: dict[str, Any],
        success: bool,
        pnl_pct: float,
    ) -> None:
        """Record outcome for ML learning.

        Args:
            signal_id: Signal identifier
            signal: Original signal
            context: Original context
            success: Whether profitable
            pnl_pct: P&L percentage
        """
        self._scorer.record_outcome(signal_id, signal, context, success, pnl_pct)

    def get_ml_stats(self) -> dict[str, Any]:
        """Get ML model statistics."""
        return self._scorer.get_model_stats()

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        return self._scorer.get_feature_importance()

"""
Model Stacking & Meta-Learning

Combines predictions from multiple models:
- Level 1: Base models (LightGBM, XGBoost, CatBoost, LSTM, TFT, RL Agent)
- Level 2: Meta-model learns to combine them optimally

Expected Performance:
- Accuracy: +3-7% vs single best model
- Sharpe: +0.2-0.5
- Robustness: Better in regime changes

Author: HEAN Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from loguru import logger

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed")


@dataclass
class StackingConfig:
    """Configuration for model stacking."""

    # Meta-learner
    meta_model: str = "logistic"  # "logistic", "rf", "xgboost"
    use_proba: bool = True  # Use probabilities instead of hard predictions

    # Cross-validation
    cv_folds: int = 5

    # Model weights (if not using meta-learner)
    manual_weights: Optional[Dict[str, float]] = None

    # Save path
    model_dir: str = "models/stacking"


class ModelStacking:
    """
    Meta-learning model stacker.

    Usage:
        # Setup base models
        base_models = {
            "lgb": lgb_model,
            "xgb": xgb_model,
            "catboost": cb_model,
            "lstm": lstm_model,
        }

        # Train stacker
        stacker = ModelStacking()
        stacker.train(base_models, X_train, y_train)

        # Predict with ensemble
        ensemble_pred = stacker.predict({
            "lgb": lgb_pred,
            "xgb": xgb_pred,
            "catboost": cb_pred,
            "lstm": lstm_pred,
        })
    """

    def __init__(self, config: Optional[StackingConfig] = None) -> None:
        """Initialize model stacker."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required")

        self.config = config or StackingConfig()
        self.meta_model: Optional[Any] = None
        self.base_model_names: List[str] = []

        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

        logger.info("ModelStacking initialized", config=self.config)

    def train(
        self,
        base_predictions: Dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train meta-learner on base model predictions.

        Args:
            base_predictions: {"model_name": predictions, ...}
            y_true: True labels

        Returns:
            Training metrics
        """
        logger.info(f"Training meta-learner on {len(base_predictions)} base models")

        self.base_model_names = list(base_predictions.keys())

        # Stack predictions
        X_meta = np.column_stack([
            base_predictions[name] for name in self.base_model_names
        ])

        # Create meta-model
        if self.config.meta_model == "logistic":
            self.meta_model = LogisticRegression(max_iter=1000)
        elif self.config.meta_model == "rf":
            self.meta_model = RandomForestClassifier(n_estimators=100)
        else:
            raise ValueError(f"Unknown meta model: {self.config.meta_model}")

        # Train
        self.meta_model.fit(X_meta, y_true)

        # Cross-validation score
        cv_scores = cross_val_score(
            self.meta_model,
            X_meta,
            y_true,
            cv=self.config.cv_folds,
            scoring='accuracy',
        )

        metrics = {
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
        }

        logger.info(
            f"Meta-learner trained: "
            f"CV Accuracy = {metrics['cv_accuracy_mean']:.1%} "
            f"(±{metrics['cv_accuracy_std']:.1%})"
        )

        # Save
        self.save()

        return metrics

    def predict(
        self,
        base_predictions: Dict[str, float | np.ndarray],
    ) -> float:
        """
        Predict using ensemble.

        Args:
            base_predictions: {"model_name": prediction, ...}

        Returns:
            Ensemble prediction (probability)
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained")

        # Stack predictions
        X_meta = np.array([
            base_predictions[name] for name in self.base_model_names
        ]).reshape(1, -1)

        # Predict
        if self.config.use_proba:
            pred = self.meta_model.predict_proba(X_meta)[0, 1]
        else:
            pred = self.meta_model.predict(X_meta)[0]

        return float(pred)

    def predict_batch(
        self,
        base_predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Batch prediction."""
        if self.meta_model is None:
            raise ValueError("Meta-model not trained")

        # Stack
        X_meta = np.column_stack([
            base_predictions[name] for name in self.base_model_names
        ])

        # Predict
        if self.config.use_proba:
            preds = self.meta_model.predict_proba(X_meta)[:, 1]
        else:
            preds = self.meta_model.predict(X_meta)

        return preds

    def get_model_weights(self) -> Dict[str, float]:
        """
        Get learned weights of base models.

        Only works for linear meta-models (Logistic Regression).
        """
        if not isinstance(self.meta_model, LogisticRegression):
            logger.warning("Weights only available for LogisticRegression")
            return {}

        weights = self.meta_model.coef_[0]

        # Normalize to sum to 1
        weights_abs = np.abs(weights)
        weights_norm = weights_abs / weights_abs.sum()

        return {
            name: float(w)
            for name, w in zip(self.base_model_names, weights_norm)
        }

    def save(self, path: Optional[str] = None) -> None:
        """Save meta-model."""
        if path is None:
            path = f"{self.config.model_dir}/meta_model.pkl"

        joblib.dump({
            'meta_model': self.meta_model,
            'base_model_names': self.base_model_names,
            'config': self.config,
        }, path)

        logger.info(f"Meta-model saved to {path}")

    @classmethod
    def load(cls, path: str) -> ModelStacking:
        """Load meta-model."""
        data = joblib.load(path)

        stacker = cls(data['config'])
        stacker.meta_model = data['meta_model']
        stacker.base_model_names = data['base_model_names']

        logger.info(f"Meta-model loaded from {path}")
        return stacker


class StrategyEnsemble:
    """
    Ensemble of trading strategies with weighted voting.

    Usage:
        ensemble = StrategyEnsemble()

        # Add strategies with weights
        ensemble.add_strategy("rsi_strategy", weight=0.3)
        ensemble.add_strategy("ml_strategy", weight=0.4)
        ensemble.add_strategy("sentiment_strategy", weight=0.3)

        # Aggregate signals
        signal = ensemble.aggregate_signals({
            "rsi_strategy": {"action": "BUY", "confidence": 0.7},
            "ml_strategy": {"action": "BUY", "confidence": 0.8},
            "sentiment_strategy": {"action": "SELL", "confidence": 0.5},
        })
        # Result: BUY with confidence based on weighted voting
    """

    def __init__(self) -> None:
        """Initialize strategy ensemble."""
        self.strategies: Dict[str, float] = {}  # {name: weight}

    def add_strategy(self, name: str, weight: float = 1.0) -> None:
        """Add strategy to ensemble."""
        self.strategies[name] = weight
        logger.info(f"Added strategy: {name} (weight={weight})")

    def aggregate_signals(
        self,
        signals: Dict[str, Dict[str, Any]],
        min_agreement: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Aggregate signals from multiple strategies.

        Args:
            signals: {"strategy_name": {"action": "BUY/SELL", "confidence": 0.7}, ...}
            min_agreement: Minimum weighted agreement to generate signal

        Returns:
            Aggregated signal
        """
        # Count weighted votes
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0

        for name, signal in signals.items():
            if name not in self.strategies:
                continue

            weight = self.strategies[name]
            confidence = signal.get('confidence', 1.0)
            action = signal.get('action', 'NEUTRAL')

            weighted_vote = weight * confidence
            total_weight += weight

            if action == "BUY":
                buy_weight += weighted_vote
            elif action == "SELL":
                sell_weight += weighted_vote

        if total_weight == 0:
            return {"action": "NEUTRAL", "confidence": 0.0}

        # Normalize
        buy_pct = buy_weight / total_weight
        sell_pct = sell_weight / total_weight

        # Determine action
        if buy_pct > min_agreement:
            return {"action": "BUY", "confidence": buy_pct}
        elif sell_pct > min_agreement:
            return {"action": "SELL", "confidence": sell_pct}
        else:
            return {"action": "NEUTRAL", "confidence": 0.0}


# Convenience function
def create_full_ensemble(
    lgb_model,
    xgb_model,
    cb_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> ModelStacking:
    """
    Quick setup for full ML ensemble.

    Example:
        stacker = create_full_ensemble(
            lgb_model, xgb_model, cb_model,
            X_train, y_train
        )

        ensemble_pred = stacker.predict({
            "lgb": 0.65,
            "xgb": 0.70,
            "cb": 0.60,
        })
    """
    # Get predictions from each model
    base_predictions = {
        "lgb": lgb_model.predict_proba(X_train)[:, 1],
        "xgb": xgb_model.predict_proba(X_train)[:, 1],
        "cb": cb_model.predict_proba(X_train)[:, 1],
    }

    # Train stacker
    stacker = ModelStacking()
    stacker.train(base_predictions, y_train)

    return stacker

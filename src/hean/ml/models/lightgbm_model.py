"""LightGBM Model for Bitcoin Price Prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import json
from pathlib import Path

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")


class LightGBMModel:
    """LightGBM classifier for binary price direction prediction."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize LightGBM model."""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")

        self.config = config or {}

        # Default parameters
        self.params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': self.config.get('num_leaves', 31),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'feature_fraction': self.config.get('feature_fraction', 0.8),
            'bagging_fraction': self.config.get('bagging_fraction', 0.8),
            'bagging_freq': self.config.get('bagging_freq', 5),
            'max_depth': self.config.get('max_depth', -1),
            'min_data_in_leaf': self.config.get('min_data_in_leaf', 20),
            'verbose': -1
        }

        self.model = None
        self.best_iteration = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        num_boost_round: int = 1000
    ) -> Dict[str, Any]:
        """Train LightGBM model."""
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        # Train model
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        self.best_iteration = self.model.best_iteration

        return {
            'best_iteration': self.best_iteration,
            'best_score': self.model.best_score
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        proba = self.predict_proba(X)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X, num_iteration=self.best_iteration)

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model must be trained first")

        importance = self.model.feature_importance(importance_type='gain')
        feature_names = self.model.feature_name()

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })

        df = df.sort_values('importance', ascending=False)
        return df.head(top_n)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_model(str(path_obj / 'model.txt'))

        # Save metadata
        metadata = {
            'params': self.params,
            'best_iteration': self.best_iteration
        }

        with open(path_obj / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str) -> None:
        """Load model from disk."""
        path_obj = Path(path)

        # Load model
        self.model = lgb.Booster(model_file=str(path_obj / 'model.txt'))

        # Load metadata
        with open(path_obj / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.params = metadata['params']
        self.best_iteration = metadata.get('best_iteration')

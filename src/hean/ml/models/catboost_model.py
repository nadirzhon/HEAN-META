"""CatBoost Model for Bitcoin Price Prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import json
from pathlib import Path

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")


class CatBoostModel:
    """CatBoost classifier for binary price direction prediction."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize CatBoost model."""
        if not CATBOOST_AVAILABLE:
            raise ImportError("CatBoost is required. Install with: pip install catboost")

        self.config = config or {}

        # Default parameters
        params = {
            'iterations': self.config.get('iterations', 1000),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'depth': self.config.get('depth', 6),
            'l2_leaf_reg': self.config.get('l2_leaf_reg', 3),
            'random_strength': self.config.get('random_strength', 1),
            'bagging_temperature': self.config.get('bagging_temperature', 1),
            'border_count': self.config.get('border_count', 128),
            'loss_function': 'Logloss',
            'eval_metric': 'Logloss',
            'early_stopping_rounds': 50,
            'verbose': 100,
            'random_seed': 42,
            'task_type': 'CPU'  # Change to 'GPU' if CUDA is available
        }

        self.model = CatBoostClassifier(**params)
        self.best_iteration = None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train CatBoost model."""
        # Create Pool
        train_pool = Pool(X_train, y_train)

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = Pool(X_val, y_val)

        # Train model
        self.model.fit(
            train_pool,
            eval_set=eval_set,
            use_best_model=True,
            plot=False
        )

        self.best_iteration = self.model.best_iteration_

        # Get best score
        best_score = {}
        if eval_set is not None:
            best_score['valid'] = self.model.best_score_['validation']
        best_score['train'] = self.model.best_score_.get('learn', {})

        return {
            'best_iteration': self.best_iteration,
            'best_score': best_score
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        if self.model is None or not self.model.is_fitted():
            raise ValueError("Model must be trained before prediction")

        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None or not self.model.is_fitted():
            raise ValueError("Model must be trained before prediction")

        # Get probability for class 1 (price going up)
        proba = self.model.predict_proba(X)
        return proba[:, 1]

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None or not self.model.is_fitted():
            raise ValueError("Model must be trained first")

        importance = self.model.get_feature_importance()
        feature_names = self.model.feature_names_

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })

        df = df.sort_values('importance', ascending=False)
        return df.head(top_n)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None or not self.model.is_fitted():
            raise ValueError("No model to save")

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_model(str(path_obj / 'model.cbm'))

        # Save metadata
        metadata = {
            'best_iteration': self.best_iteration,
            'params': self.model.get_all_params()
        }

        with open(path_obj / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str) -> None:
        """Load model from disk."""
        path_obj = Path(path)

        # Load model
        self.model = CatBoostClassifier()
        self.model.load_model(str(path_obj / 'model.cbm'))

        # Load metadata
        with open(path_obj / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.best_iteration = metadata.get('best_iteration')

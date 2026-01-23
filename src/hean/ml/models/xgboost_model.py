"""XGBoost Model for Bitcoin Price Prediction."""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
import json
import pickle
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")


class XGBoostModel:
    """XGBoost classifier for binary price direction prediction."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize XGBoost model."""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")

        self.config = config or {}

        # Default parameters
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': self.config.get('max_depth', 6),
            'learning_rate': self.config.get('learning_rate', 0.05),
            'subsample': self.config.get('subsample', 0.8),
            'colsample_bytree': self.config.get('colsample_bytree', 0.8),
            'min_child_weight': self.config.get('min_child_weight', 3),
            'gamma': self.config.get('gamma', 0.1),
            'reg_alpha': self.config.get('reg_alpha', 0.1),
            'reg_lambda': self.config.get('reg_lambda', 1.0),
            'tree_method': 'hist',
            'random_state': 42
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
        """Train XGBoost model."""
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train)

        evals = [(dtrain, 'train')]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'valid'))

        # Train model
        evals_result = {}

        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=50,
            evals_result=evals_result,
            verbose_eval=100
        )

        self.best_iteration = self.model.best_iteration

        # Get best score
        best_score = {}
        for eval_name, metrics in evals_result.items():
            best_score[eval_name] = {
                metric: values[self.best_iteration]
                for metric, values in metrics.items()
            }

        return {
            'best_iteration': self.best_iteration,
            'best_score': best_score
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

        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix, iteration_range=(0, self.best_iteration + 1))

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance."""
        if self.model is None:
            raise ValueError("Model must be trained first")

        importance = self.model.get_score(importance_type='gain')

        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in importance.items()
        ])

        df = df.sort_values('importance', ascending=False)
        return df.head(top_n)

    def save(self, path: str) -> None:
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save_model(str(path_obj / 'model.json'))

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
        self.model = xgb.Booster()
        self.model.load_model(str(path_obj / 'model.json'))

        # Load metadata
        with open(path_obj / 'metadata.json', 'r') as f:
            metadata = json.load(f)

        self.params = metadata['params']
        self.best_iteration = metadata.get('best_iteration')

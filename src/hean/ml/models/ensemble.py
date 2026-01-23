"""
Ensemble Model for Bitcoin Price Prediction

Combines predictions from:
- LightGBM
- XGBoost
- CatBoost

Uses weighted voting for final prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import json
import pickle
from pathlib import Path

from .lightgbm_model import LightGBMModel
from .xgboost_model import XGBoostModel
from .catboost_model import CatBoostModel


class EnsembleModel:
    """
    Ensemble model combining LightGBM, XGBoost, and CatBoost.

    Uses weighted voting based on individual model performance.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize ensemble model.

        Args:
            config: Configuration dictionary for models
        """
        self.config = config or {}

        # Initialize individual models
        self.lightgbm = LightGBMModel(self.config.get('lightgbm', {}))
        self.xgboost = XGBoostModel(self.config.get('xgboost', {}))
        self.catboost = CatBoostModel(self.config.get('catboost', {}))

        # Model weights (will be updated based on validation performance)
        self.weights = {
            'lightgbm': 0.33,
            'xgboost': 0.33,
            'catboost': 0.34
        }

        # Trained flag
        self.is_trained = False

        # Feature names
        self.feature_names: List[str] = []

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Train all models in the ensemble.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dictionary with training results for each model
        """
        self.feature_names = list(X_train.columns)

        results = {}

        # Train LightGBM
        print("Training LightGBM...")
        lgb_result = self.lightgbm.train(X_train, y_train, X_val, y_val)
        results['lightgbm'] = lgb_result

        # Train XGBoost
        print("Training XGBoost...")
        xgb_result = self.xgboost.train(X_train, y_train, X_val, y_val)
        results['xgboost'] = xgb_result

        # Train CatBoost
        print("Training CatBoost...")
        cat_result = self.catboost.train(X_train, y_train, X_val, y_val)
        results['catboost'] = cat_result

        # Update weights based on validation performance
        if X_val is not None and y_val is not None:
            self._update_weights(X_val, y_val)
            results['weights'] = self.weights

        self.is_trained = True

        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Features to predict on

        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Get predictions from each model
        lgb_pred = self.lightgbm.predict(X)
        xgb_pred = self.xgboost.predict(X)
        cat_pred = self.catboost.predict(X)

        # Weighted average
        ensemble_pred = (
            self.weights['lightgbm'] * lgb_pred +
            self.weights['xgboost'] * xgb_pred +
            self.weights['catboost'] * cat_pred
        )

        # Convert probabilities to binary predictions
        return (ensemble_pred > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities using the ensemble.

        Args:
            X: Features to predict on

        Returns:
            Probability predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Get probability predictions from each model
        lgb_proba = self.lightgbm.predict_proba(X)
        xgb_proba = self.xgboost.predict_proba(X)
        cat_proba = self.catboost.predict_proba(X)

        # Weighted average of probabilities
        ensemble_proba = (
            self.weights['lightgbm'] * lgb_proba +
            self.weights['xgboost'] * xgb_proba +
            self.weights['catboost'] * cat_proba
        )

        return ensemble_proba

    def _update_weights(self, X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Update model weights based on validation performance.

        Uses accuracy as the metric for weighting.
        """
        from sklearn.metrics import accuracy_score

        # Get predictions from each model
        lgb_pred = self.lightgbm.predict(X_val)
        xgb_pred = self.xgboost.predict(X_val)
        cat_pred = self.catboost.predict(X_val)

        # Calculate accuracies
        lgb_acc = accuracy_score(y_val, lgb_pred)
        xgb_acc = accuracy_score(y_val, xgb_pred)
        cat_acc = accuracy_score(y_val, cat_pred)

        # Update weights proportional to accuracy
        total_acc = lgb_acc + xgb_acc + cat_acc

        self.weights = {
            'lightgbm': lgb_acc / total_acc,
            'xgboost': xgb_acc / total_acc,
            'catboost': cat_acc / total_acc
        }

        print(f"Updated weights: {self.weights}")

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from all models.

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary with feature importance DataFrames for each model
        """
        importance = {}

        importance['lightgbm'] = self.lightgbm.get_feature_importance(top_n)
        importance['xgboost'] = self.xgboost.get_feature_importance(top_n)
        importance['catboost'] = self.catboost.get_feature_importance(top_n)

        # Combined importance (weighted average)
        combined = pd.DataFrame({
            'feature': self.feature_names
        })

        for model_name, model in [
            ('lightgbm', self.lightgbm),
            ('xgboost', self.xgboost),
            ('catboost', self.catboost)
        ]:
            imp = model.get_feature_importance(len(self.feature_names))
            combined = combined.merge(
                imp[['feature', 'importance']].rename(
                    columns={'importance': f'{model_name}_importance'}
                ),
                on='feature',
                how='left'
            )

        # Calculate weighted importance
        combined['weighted_importance'] = (
            combined['lightgbm_importance'] * self.weights['lightgbm'] +
            combined['xgboost_importance'] * self.weights['xgboost'] +
            combined['catboost_importance'] * self.weights['catboost']
        )

        combined = combined.sort_values('weighted_importance', ascending=False)
        importance['ensemble'] = combined.head(top_n)

        return importance

    def save(self, path: str) -> None:
        """
        Save ensemble model to disk.

        Args:
            path: Directory path to save models
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save individual models
        self.lightgbm.save(str(path_obj / 'lightgbm'))
        self.xgboost.save(str(path_obj / 'xgboost'))
        self.catboost.save(str(path_obj / 'catboost'))

        # Save ensemble metadata
        metadata = {
            'weights': self.weights,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'config': self.config
        }

        with open(path_obj / 'ensemble_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load(self, path: str) -> None:
        """
        Load ensemble model from disk.

        Args:
            path: Directory path to load models from
        """
        path_obj = Path(path)

        # Load individual models
        self.lightgbm.load(str(path_obj / 'lightgbm'))
        self.xgboost.load(str(path_obj / 'xgboost'))
        self.catboost.load(str(path_obj / 'catboost'))

        # Load ensemble metadata
        with open(path_obj / 'ensemble_metadata.json', 'r') as f:
            metadata = json.load(f)

        self.weights = metadata['weights']
        self.is_trained = metadata['is_trained']
        self.feature_names = metadata['feature_names']
        self.config = metadata.get('config', {})

    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model.

        Useful for debugging and analysis.

        Args:
            X: Features to predict on

        Returns:
            Dictionary with predictions from each model
        """
        return {
            'lightgbm': self.lightgbm.predict_proba(X),
            'xgboost': self.xgboost.predict_proba(X),
            'catboost': self.catboost.predict_proba(X),
            'ensemble': self.predict_proba(X)
        }

"""
Bitcoin Price Predictor using Ensemble Models.

Ensemble: LightGBM + XGBoost + CatBoost
Target: Predict 5-minute price movement (up/down)
Features: 50+ engineered features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import pickle
import json
from datetime import datetime

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

from ml.features import FeatureEngineering, FeatureConfig


@dataclass
class PredictorConfig:
    """Configuration for Bitcoin price predictor."""

    # Prediction horizon (minutes)
    prediction_horizon: int = 5

    # Ensemble weights
    lgb_weight: float = 0.35
    xgb_weight: float = 0.35
    catboost_weight: float = 0.30

    # Model hyperparameters
    lgb_params: Optional[Dict[str, Any]] = None
    xgb_params: Optional[Dict[str, Any]] = None
    catboost_params: Optional[Dict[str, Any]] = None

    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42

    # Prediction threshold
    threshold: float = 0.5

    def __post_init__(self):
        if self.lgb_params is None:
            self.lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_child_samples': 20,
                'verbose': -1,
                'n_estimators': 500,
                'early_stopping_rounds': 50,
            }

        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 1,
                'gamma': 0,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': self.random_state,
                'verbosity': 0,
                'early_stopping_rounds': 50,
            }

        if self.catboost_params is None:
            self.catboost_params = {
                'loss_function': 'Logloss',
                'eval_metric': 'Logloss',
                'iterations': 500,
                'learning_rate': 0.05,
                'depth': 6,
                'l2_leaf_reg': 3,
                'random_seed': self.random_state,
                'verbose': False,
                'early_stopping_rounds': 50,
            }


class BitcoinPricePredictor:
    """
    Ensemble predictor for Bitcoin price movements.

    Uses LightGBM + XGBoost + CatBoost to predict 5-minute price direction.
    """

    def __init__(
        self,
        config: Optional[PredictorConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
    ):
        """
        Initialize Bitcoin price predictor.

        Args:
            config: Predictor configuration
            feature_config: Feature engineering configuration
        """
        self.config = config or PredictorConfig()
        self.feature_config = feature_config or FeatureConfig()
        self.feature_engineering = FeatureEngineering(self.feature_config)

        # Models
        self.lgb_model = None
        self.xgb_model = None
        self.catboost_model = None

        # Feature importance
        self.feature_importance: Dict[str, float] = {}
        self.feature_names: List[str] = []

        # Training history
        self.training_history: List[Dict[str, Any]] = []

        # Model status
        self.is_trained = False
        self.last_training_time: Optional[datetime] = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training/prediction.

        Args:
            df: OHLCV dataframe
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data

        Returns:
            (features_df, target_series) tuple
        """
        # Generate features
        features = self.feature_engineering.generate_features(
            df,
            orderbook_data,
            sentiment_data
        )

        # Create target: 1 if price goes up in next 5 minutes, 0 otherwise
        # Assuming 5-minute candles, target is next candle's direction
        target = (features['close'].shift(-1) > features['close']).astype(int)

        # Remove last row (no target)
        features = features.iloc[:-1]
        target = target.iloc[:-1]

        # Get feature columns (exclude metadata)
        feature_cols = self.feature_engineering.get_feature_names()
        self.feature_names = feature_cols

        return features[feature_cols], target

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Train ensemble models.

        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)

        Returns:
            Training metrics dictionary
        """
        if not LIGHTGBM_AVAILABLE or not XGBOOST_AVAILABLE or not CATBOOST_AVAILABLE:
            raise ImportError(
                "LightGBM, XGBoost, and CatBoost are required. "
                "Install with: pip install lightgbm xgboost catboost"
            )

        metrics = {}

        # Train LightGBM
        print("Training LightGBM...")
        lgb_train = lgb.Dataset(X, y)

        if X_val is not None and y_val is not None:
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            self.lgb_model = lgb.train(
                self.config.lgb_params,
                lgb_train,
                valid_sets=[lgb_val],
                callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
            )
        else:
            self.lgb_model = lgb.train(
                self.config.lgb_params,
                lgb_train,
            )

        # Train XGBoost
        print("Training XGBoost...")
        if X_val is not None and y_val is not None:
            self.xgb_model = xgb.XGBClassifier(**self.config.xgb_params)
            self.xgb_model.fit(
                X, y,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.xgb_model = xgb.XGBClassifier(**self.config.xgb_params)
            self.xgb_model.fit(X, y, verbose=False)

        # Train CatBoost
        print("Training CatBoost...")
        self.catboost_model = cb.CatBoostClassifier(**self.config.catboost_params)

        if X_val is not None and y_val is not None:
            self.catboost_model.fit(
                X, y,
                eval_set=(X_val, y_val),
                verbose=False
            )
        else:
            self.catboost_model.fit(X, y, verbose=False)

        # Calculate feature importance
        self._calculate_feature_importance()

        # Update status
        self.is_trained = True
        self.last_training_time = datetime.now()

        # Store training history
        training_record = {
            'timestamp': self.last_training_time.isoformat(),
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'metrics': metrics,
        }
        self.training_history.append(training_record)

        print(f"Training completed at {self.last_training_time}")
        print(f"Models trained on {len(X)} samples with {len(self.feature_names)} features")

        return metrics

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of price going up.

        Args:
            X: Features dataframe

        Returns:
            Array of probabilities (0-1)
        """
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")

        # Get predictions from each model
        lgb_pred = self.lgb_model.predict(X)
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        catboost_pred = self.catboost_model.predict_proba(X)[:, 1]

        # Weighted ensemble
        ensemble_pred = (
            self.config.lgb_weight * lgb_pred +
            self.config.xgb_weight * xgb_pred +
            self.config.catboost_weight * catboost_pred
        )

        return ensemble_pred

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict price direction (0 or 1).

        Args:
            X: Features dataframe

        Returns:
            Array of predictions (0=down, 1=up)
        """
        proba = self.predict_proba(X)
        return (proba >= self.config.threshold).astype(int)

    def predict_single(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        Predict for single timestamp (latest in dataframe).

        Args:
            df: OHLCV dataframe (should contain enough history for features)
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data

        Returns:
            Dictionary with prediction results
        """
        # Prepare features
        features, _ = self.prepare_data(df, orderbook_data, sentiment_data)

        # Get latest features
        latest_features = features.iloc[[-1]]

        # Predict
        proba = self.predict_proba(latest_features)[0]
        prediction = int(proba >= self.config.threshold)

        # Get individual model predictions
        lgb_pred = self.lgb_model.predict(latest_features)[0]
        xgb_pred = self.xgb_model.predict_proba(latest_features)[0, 1]
        catboost_pred = self.catboost_model.predict_proba(latest_features)[0, 1]

        return {
            'prediction': prediction,
            'probability': float(proba),
            'direction': 'UP' if prediction == 1 else 'DOWN',
            'confidence': float(abs(proba - 0.5) * 2),  # 0-1, higher is more confident
            'individual_predictions': {
                'lightgbm': float(lgb_pred),
                'xgboost': float(xgb_pred),
                'catboost': float(catboost_pred),
            },
            'timestamp': datetime.now().isoformat(),
        }

    def _calculate_feature_importance(self) -> None:
        """Calculate and store feature importance from all models."""
        if not self.is_trained:
            return

        # Get feature importance from each model
        lgb_importance = dict(zip(self.feature_names, self.lgb_model.feature_importance(importance_type='gain')))
        xgb_importance = dict(zip(self.feature_names, self.xgb_model.feature_importances_))
        catboost_importance = dict(zip(self.feature_names, self.catboost_model.feature_importances_))

        # Normalize and average
        for feature in self.feature_names:
            lgb_imp = lgb_importance.get(feature, 0)
            xgb_imp = xgb_importance.get(feature, 0)
            cb_imp = catboost_importance.get(feature, 0)

            # Weighted average
            self.feature_importance[feature] = (
                self.config.lgb_weight * lgb_imp +
                self.config.xgb_weight * xgb_imp +
                self.config.catboost_weight * cb_imp
            )

        # Normalize to sum to 1
        total_importance = sum(self.feature_importance.values())
        if total_importance > 0:
            self.feature_importance = {
                k: v / total_importance
                for k, v in self.feature_importance.items()
            }

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top N most important features.

        Args:
            top_n: Number of top features to return

        Returns:
            Dictionary of feature names and their importance scores
        """
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_features[:top_n])

    def save_model(self, model_dir: str) -> None:
        """
        Save trained models to disk.

        Args:
            model_dir: Directory to save models
        """
        if not self.is_trained:
            raise ValueError("No trained models to save")

        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save LightGBM
        self.lgb_model.save_model(str(model_path / "lgb_model.txt"))

        # Save XGBoost
        self.xgb_model.save_model(str(model_path / "xgb_model.json"))

        # Save CatBoost
        self.catboost_model.save_model(str(model_path / "catboost_model.cbm"))

        # Save metadata
        metadata = {
            'config': {
                'prediction_horizon': self.config.prediction_horizon,
                'lgb_weight': self.config.lgb_weight,
                'xgb_weight': self.config.xgb_weight,
                'catboost_weight': self.config.catboost_weight,
                'threshold': self.config.threshold,
            },
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_history': self.training_history,
        }

        with open(model_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Models saved to {model_dir}")

    def load_model(self, model_dir: str) -> None:
        """
        Load trained models from disk.

        Args:
            model_dir: Directory containing saved models
        """
        model_path = Path(model_dir)

        if not model_path.exists():
            raise ValueError(f"Model directory not found: {model_dir}")

        # Load LightGBM
        self.lgb_model = lgb.Booster(model_file=str(model_path / "lgb_model.txt"))

        # Load XGBoost
        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(model_path / "xgb_model.json"))

        # Load CatBoost
        self.catboost_model = cb.CatBoostClassifier()
        self.catboost_model.load_model(str(model_path / "catboost_model.cbm"))

        # Load metadata
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        self.feature_names = metadata['feature_names']
        self.feature_importance = metadata['feature_importance']
        self.training_history = metadata.get('training_history', [])

        if metadata.get('last_training_time'):
            self.last_training_time = datetime.fromisoformat(metadata['last_training_time'])

        self.is_trained = True
        print(f"Models loaded from {model_dir}")

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained models."""
        return {
            'is_trained': self.is_trained,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'n_features': len(self.feature_names),
            'ensemble_weights': {
                'lightgbm': self.config.lgb_weight,
                'xgboost': self.config.xgb_weight,
                'catboost': self.config.catboost_weight,
            },
            'prediction_horizon_minutes': self.config.prediction_horizon,
            'threshold': self.config.threshold,
            'training_history_count': len(self.training_history),
        }

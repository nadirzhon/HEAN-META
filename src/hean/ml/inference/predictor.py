"""
Production ML Predictor

Real-time prediction service for Bitcoin price movements.
Optimized for low-latency production use.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from ..models.ensemble import EnsembleModel
from ..features.feature_engineer import FeatureEngineer


class MLPredictor:
    """
    Production-ready ML predictor for real-time Bitcoin price predictions.

    Provides:
    - Fast inference
    - Feature engineering
    - Prediction confidence
    - Model health monitoring
    """

    def __init__(self, model_path: str, config: Optional[Dict] = None):
        """
        Initialize predictor.

        Args:
            model_path: Path to saved model
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Load model
        self.model = EnsembleModel()
        self.model.load(model_path)

        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(self.config.get('features', {}))

        # Prediction cache
        self.last_prediction = None
        self.last_prediction_time = None

        # Performance tracking
        self.prediction_count = 0
        self.total_inference_time = 0.0

        self.logger.info(f"MLPredictor initialized with model from {model_path}")

    def predict(
        self,
        ohlcv_data: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[Dict] = None,
        return_probabilities: bool = True
    ) -> Dict[str, Any]:
        """
        Make prediction on latest data.

        Args:
            ohlcv_data: Recent OHLCV data (needs at least 200 candles for features)
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data
            return_probabilities: Whether to return probabilities

        Returns:
            Dictionary with prediction and metadata
        """
        start_time = datetime.now()

        try:
            # 1. Engineer features
            df = self.feature_engineer.engineer_features(
                ohlcv_data,
                orderbook_data,
                sentiment_data
            )

            # Handle missing orderbook/sentiment data
            if orderbook_data is None:
                from ..features.orderbook_features import OrderbookFeatures
                ob_features = OrderbookFeatures()
                df = ob_features.create_synthetic_orderbook(df)

            if sentiment_data is None:
                from ..features.sentiment_features import SentimentFeatures
                sent_features = SentimentFeatures()
                df = sent_features.create_synthetic_sentiment(df)

            # 2. Get features for latest candle
            feature_cols = self.feature_engineer.get_feature_names(df)
            X = df[feature_cols].iloc[[-1]]  # Get last row

            # 3. Make prediction
            prediction = self.model.predict(X)[0]

            result = {
                'prediction': int(prediction),
                'direction': 'UP' if prediction == 1 else 'DOWN',
                'timestamp': datetime.now().isoformat()
            }

            # 4. Add probabilities if requested
            if return_probabilities:
                probability = self.model.predict_proba(X)[0]
                result['probability'] = float(probability)
                result['confidence'] = float(abs(probability - 0.5) * 2)  # 0-1 scale

                # Get individual model predictions for transparency
                model_predictions = self.model.get_model_predictions(X)
                result['model_probabilities'] = {
                    'lightgbm': float(model_predictions['lightgbm'][0]),
                    'xgboost': float(model_predictions['xgboost'][0]),
                    'catboost': float(model_predictions['catboost'][0])
                }

            # 5. Calculate inference time
            inference_time = (datetime.now() - start_time).total_seconds()
            result['inference_time_ms'] = inference_time * 1000

            # Update tracking
            self.prediction_count += 1
            self.total_inference_time += inference_time

            result['prediction_count'] = self.prediction_count
            result['avg_inference_time_ms'] = (self.total_inference_time / self.prediction_count) * 1000

            # Cache prediction
            self.last_prediction = result
            self.last_prediction_time = datetime.now()

            return result

        except Exception as e:
            self.logger.error(f"Prediction error: {e}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def predict_batch(
        self,
        ohlcv_data: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Make predictions on batch of data.

        Useful for backtesting or analyzing historical predictions.

        Args:
            ohlcv_data: OHLCV data
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data

        Returns:
            DataFrame with predictions and probabilities
        """
        # Engineer features
        df = self.feature_engineer.engineer_features(
            ohlcv_data,
            orderbook_data,
            sentiment_data
        )

        # Handle missing data
        if orderbook_data is None:
            from ..features.orderbook_features import OrderbookFeatures
            ob_features = OrderbookFeatures()
            df = ob_features.create_synthetic_orderbook(df)

        if sentiment_data is None:
            from ..features.sentiment_features import SentimentFeatures
            sent_features = SentimentFeatures()
            df = sent_features.create_synthetic_sentiment(df)

        # Get features
        feature_cols = self.feature_engineer.get_feature_names(df)
        X = df[feature_cols]

        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        # Create results DataFrame
        results = pd.DataFrame({
            'prediction': predictions,
            'probability': probabilities,
            'confidence': np.abs(probabilities - 0.5) * 2
        })

        # Add timestamp if available
        if 'timestamp' in df.columns:
            results['timestamp'] = df['timestamp'].values

        return results

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        importance = self.model.get_feature_importance(top_n=top_n)
        return importance['ensemble']

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the predictor.

        Returns:
            Health status dictionary
        """
        health = {
            'status': 'healthy',
            'model_loaded': self.model.is_trained,
            'prediction_count': self.prediction_count,
            'avg_inference_time_ms': (
                (self.total_inference_time / self.prediction_count) * 1000
                if self.prediction_count > 0 else 0
            ),
            'last_prediction_time': (
                self.last_prediction_time.isoformat()
                if self.last_prediction_time else None
            )
        }

        # Check if model is too old (needs retraining)
        if self.last_prediction_time:
            hours_since_prediction = (
                (datetime.now() - self.last_prediction_time).total_seconds() / 3600
            )
            if hours_since_prediction > 24:
                health['status'] = 'warning'
                health['warning'] = f'No predictions in {hours_since_prediction:.1f} hours'

        return health

    def validate_input(
        self,
        ohlcv_data: pd.DataFrame,
        min_candles: int = 200
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input data for prediction.

        Args:
            ohlcv_data: OHLCV data to validate
            min_candles: Minimum number of candles required

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in ohlcv_data.columns]

        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"

        # Check minimum data length
        if len(ohlcv_data) < min_candles:
            return False, f"Need at least {min_candles} candles, got {len(ohlcv_data)}"

        # Check for NaN values
        if ohlcv_data[required_cols].isna().any().any():
            return False, "Input data contains NaN values"

        # Check for valid price data
        if (ohlcv_data[['open', 'high', 'low', 'close']] <= 0).any().any():
            return False, "Invalid price data (non-positive values)"

        return True, None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get predictor statistics.

        Returns:
            Statistics dictionary
        """
        return {
            'prediction_count': self.prediction_count,
            'avg_inference_time_ms': (
                (self.total_inference_time / self.prediction_count) * 1000
                if self.prediction_count > 0 else 0
            ),
            'last_prediction': self.last_prediction,
            'model_weights': self.model.weights
        }

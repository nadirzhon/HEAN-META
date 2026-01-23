"""
Machine Learning module for Bitcoin price prediction.

Provides ensemble models (LightGBM + XGBoost + CatBoost) for predicting
5-minute price movements with 50+ engineered features.
"""

from ml.price_predictor import BitcoinPricePredictor
from ml.features import FeatureEngineering
from ml.trainer import ModelTrainer
from ml.backtester import Backtester

__all__ = [
    "BitcoinPricePredictor",
    "FeatureEngineering",
    "ModelTrainer",
    "Backtester",
]

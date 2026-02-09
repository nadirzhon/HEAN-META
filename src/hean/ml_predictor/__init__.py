"""
ML Price Predictor - LSTM Neural Network for Price Prediction

Predicts future price movements using:
- Historical price data (OHLCV)
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Sentiment scores (from sentiment analysis)
- Google Trends data
- Funding rates

Architecture:
- LSTM (Long Short-Term Memory) neural network
- Multi-feature input
- Multi-step prediction (predict next 1h, 4h, 24h)

Expected performance:
- Accuracy: 60-70% (direction prediction)
- Profit improvement: +30-50%

Usage:
    from hean.ml_predictor import PricePredictor

    predictor = PricePredictor()
    await predictor.load_model("btc_lstm_v1.h5")

    prediction = await predictor.predict("BTCUSDT")
    print(f"Next 1h: {prediction.direction_1h} ({prediction.confidence_1h:.0%})")
"""

from .features import FeatureEngineering
from .lstm_model import LSTMPriceModel
from .models import ModelMetrics, PredictionDirection, PricePrediction, TrainingConfig
from .predictor import PricePredictor
from .strategy import MLPredictorStrategy
from .trainer import ModelTrainer

__all__ = [
    "PricePrediction",
    "PredictionDirection",
    "ModelMetrics",
    "TrainingConfig",
    "FeatureEngineering",
    "LSTMPriceModel",
    "PricePredictor",
    "ModelTrainer",
    "MLPredictorStrategy",
]

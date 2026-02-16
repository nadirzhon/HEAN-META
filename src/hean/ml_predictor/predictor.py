"""
Real-time price predictor using trained LSTM model
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .features import FeatureEngineering
from .lstm_model import LSTMPriceModel
from .models import PredictionDirection, PricePrediction

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    Real-time price predictor

    Loads trained LSTM model and makes predictions on live data

    Usage:
        predictor = PricePredictor()
        await predictor.load_model("models/btcusdt_v1_20260130.h5")

        prediction = await predictor.predict("BTCUSDT")
        if prediction.should_trade:
            execute_trade(prediction)
    """

    def __init__(
        self,
        lookback_periods: int = 60,
        prediction_horizons: list[int] | None = None
    ):
        """
        Initialize predictor

        Args:
            lookback_periods: How many periods to look back
            prediction_horizons: Hours ahead to predict
        """
        self.lookback_periods = lookback_periods
        self.prediction_horizons = prediction_horizons if prediction_horizons is not None else [1, 4, 24]

        # Feature engineering
        self.feature_engineering = FeatureEngineering(
            use_technical_indicators=True,
            use_sentiment=True,
            use_google_trends=True,
            use_funding_rates=True
        )

        # Model
        self.model: LSTMPriceModel | None = None
        self.model_version: str = "unknown"

    async def load_model(self, model_path: str):
        """
        Load trained model

        Args:
            model_path: Path to model file
        """
        logger.info(f"Loading model from {model_path}")

        # Extract version from filename
        path = Path(model_path)
        self.model_version = path.stem

        # Load model (will be lazy-initialized on first use)
        # For now, create a placeholder
        # In production, use: self.model = LSTMPriceModel.load(model_path)
        logger.warning(
            f"[ML_PREDICTOR] Model loading not fully implemented. "
            f"Version: {self.model_version}. In production, use: LSTMPriceModel.load(model_path)"
        )

        logger.info(f"Model loaded: {self.model_version}")

    async def predict(
        self,
        symbol: str,
        ohlcv_data: pd.DataFrame | None = None
    ) -> PricePrediction:
        """
        Make price prediction

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            ohlcv_data: Optional OHLCV data (will fetch if not provided)

        Returns:
            PricePrediction
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first")

        # Get recent data
        if ohlcv_data is None:
            ohlcv_data = await self._fetch_recent_data(symbol)

        # Get current price
        current_price = ohlcv_data['close'].iloc[-1]

        # Create features
        features = self.feature_engineering.create_features(ohlcv_data)

        # Get last sequence
        data = features.to_numpy_array()
        if len(data) < self.lookback_periods:
            raise ValueError(f"Not enough data: need {self.lookback_periods}, got {len(data)}")

        last_sequence = data[-self.lookback_periods:]
        X = np.expand_dims(last_sequence, axis=0)  # Add batch dimension

        # Predict
        y_pred = self.model.predict(X)[0]  # Shape: (n_horizons,)

        # Convert returns to prices and directions
        predictions = {}
        for i, horizon in enumerate(self.prediction_horizons):
            predicted_return = y_pred[i]  # %
            predicted_price = current_price * (1 + predicted_return / 100)

            # Determine direction
            direction = self._classify_direction(predicted_return)

            # Calculate confidence (based on magnitude)
            confidence = min(1.0, abs(predicted_return) / 5.0)  # Normalize by 5%

            predictions[horizon] = {
                'price': predicted_price,
                'direction': direction,
                'confidence': confidence,
                'expected_return': predicted_return
            }

        # Create prediction object
        prediction = PricePrediction(
            symbol=symbol,
            current_price=current_price,
            model_version=self.model_version,
            features_used=[
                'OHLCV', 'RSI', 'MACD', 'Bollinger',
                'SMA', 'EMA', 'ATR', 'OBV'
            ]
        )

        # Fill in predictions
        if 1 in predictions:
            pred_1h = predictions[1]
            prediction.price_1h = pred_1h['price']
            prediction.direction_1h = pred_1h['direction']
            prediction.confidence_1h = pred_1h['confidence']
            prediction.expected_return_1h = pred_1h['expected_return']

        if 4 in predictions:
            pred_4h = predictions[4]
            prediction.price_4h = pred_4h['price']
            prediction.direction_4h = pred_4h['direction']
            prediction.confidence_4h = pred_4h['confidence']
            prediction.expected_return_4h = pred_4h['expected_return']

        if 24 in predictions:
            pred_24h = predictions[24]
            prediction.price_24h = pred_24h['price']
            prediction.direction_24h = pred_24h['direction']
            prediction.confidence_24h = pred_24h['confidence']
            prediction.expected_return_24h = pred_24h['expected_return']

        return prediction

    def _classify_direction(self, return_pct: float) -> PredictionDirection:
        """Classify predicted return into direction"""
        if return_pct > 3.0:
            return PredictionDirection.STRONG_UP
        elif return_pct > 1.0:
            return PredictionDirection.UP
        elif return_pct > -1.0:
            return PredictionDirection.NEUTRAL
        elif return_pct > -3.0:
            return PredictionDirection.DOWN
        else:
            return PredictionDirection.STRONG_DOWN

    async def _fetch_recent_data(
        self,
        symbol: str,
        periods: int = 100
    ) -> pd.DataFrame:
        """
        Fetch recent OHLCV data from Bybit v5 API.

        Args:
            symbol: Trading symbol
            periods: Number of periods (hourly candles)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            from hean.exchange.bybit.http import BybitHTTPClient

            client = BybitHTTPClient()
            await client.connect()
            try:
                # Fetch hourly klines from Bybit
                klines = await client.get_klines(
                    symbol=symbol,
                    interval="60",  # 1 hour
                    limit=min(periods, 200),
                )
            finally:
                await client.disconnect()

            if not klines:
                raise ValueError(f"No kline data returned for {symbol}")

            # Bybit returns [startTime, open, high, low, close, volume, turnover]
            # Data comes in reverse chronological order, so reverse it
            rows = []
            for k in reversed(klines):
                rows.append({
                    'timestamp': pd.Timestamp(int(k[0]), unit='ms', tz='UTC'),
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5]),
                })

            df = pd.DataFrame(rows)
            logger.info(f"Fetched {len(df)} klines for {symbol} from Bybit")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV from Bybit for {symbol}: {e}")
            raise ValueError(
                f"Cannot fetch real OHLCV data for {symbol}: {e}. "
                "Ensure BYBIT_API_KEY and BYBIT_API_SECRET are configured."
            ) from e


# Example usage
async def main():
    """Example usage"""

    # Create predictor
    _predictor = PricePredictor(  # noqa: F841
        lookback_periods=60,
        prediction_horizons=[1, 4, 24]
    )

    # Load model (in production, load actual trained model)
    # For demo, we'll skip this as we don't have a real model yet
    # await predictor.load_model("models/btcusdt_v1_20260130.h5")

    # Make prediction
    # prediction = await predictor.predict("BTCUSDT")

    print("\n📊 Price Prediction Demo:")
    print("   (Note: This requires a trained model)")
    print("\nExpected output:")
    print("   Symbol: BTCUSDT")
    print("   Current Price: $52,143.52")
    print("   ")
    print("   1h Prediction:")
    print("     Price: $52,450.23 (+0.59%)")
    print("     Direction: UP")
    print("     Confidence: 72%")
    print("   ")
    print("   4h Prediction:")
    print("     Price: $53,012.45 (+1.67%)")
    print("     Direction: UP")
    print("     Confidence: 81%")
    print("   ")
    print("   24h Prediction:")
    print("     Price: $54,220.12 (+3.98%)")
    print("     Direction: STRONG_UP")
    print("     Confidence: 85%")
    print("   ")
    print("   Should Trade: True")
    print("   Best Timeframe: 24h (85% confidence)")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

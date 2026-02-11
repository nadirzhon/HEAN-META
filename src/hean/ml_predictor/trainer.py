"""
Model trainer - complete training pipeline
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import MarketDataLoader
from .features import FeatureEngineering
from .lstm_model import LSTMPriceModel
from .models import ModelMetrics, TrainingConfig

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Complete training pipeline for LSTM price predictor

    Handles:
    - Data loading and preprocessing
    - Feature engineering
    - Model training
    - Validation and metrics
    - Model saving

    Usage:
        trainer = ModelTrainer(config)
        await trainer.load_data("BTCUSDT", start_date, end_date)
        metrics = await trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        use_real_data: bool = True,
        testnet: bool = False
    ):
        """
        Initialize trainer

        Args:
            config: Training configuration
            use_real_data: If True, load real data from exchange (default: True)
            testnet: If True, use testnet for real data (default: False)
        """
        self.config = config
        self.use_real_data = use_real_data
        self.testnet = testnet

        self.feature_engineering = FeatureEngineering(
            use_technical_indicators=config.use_technical_indicators,
            use_sentiment=config.use_sentiment,
            use_google_trends=config.use_google_trends,
            use_funding_rates=config.use_funding_rates
        )

        # Data loader for real data
        if use_real_data:
            self.data_loader = MarketDataLoader(testnet=testnet)
        else:
            self.data_loader = None

        # Data
        self.ohlcv_data: pd.DataFrame | None = None
        self.sentiment_data: pd.DataFrame | None = None
        self.trends_data: pd.DataFrame | None = None
        self.funding_data: pd.DataFrame | None = None

        # Processed data
        self.X_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.X_val: np.ndarray | None = None
        self.y_val: np.ndarray | None = None
        self.X_test: np.ndarray | None = None
        self.y_test: np.ndarray | None = None

        # Model
        self.model: LSTMPriceModel | None = None

        # Training history
        self.history: dict | None = None

    async def load_data(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None
    ):
        """
        Load historical data

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
            start_date: Start date (default: 3 months ago)
            end_date: End date (default: now)
        """
        # Default dates
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=90)

        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")

        # Load OHLCV data (real or sample)
        if self.use_real_data and self.data_loader:
            logger.info("Loading real market data from exchange...")
            self.ohlcv_data = await self.data_loader.load_ohlcv(
                symbol=symbol,
                interval="60",  # 1 hour candles
                start_date=start_date,
                end_date=end_date
            )
            if self.ohlcv_data.empty:
                raise ValueError(
                    f"No real OHLCV data available for {symbol}. "
                    "Cannot train on sample data — model would learn noise."
                )
        else:
            raise ValueError(
                "use_real_data=False is not allowed in production. "
                "Training on sample data produces unreliable models."
            )

        # Load funding rates (real data only)
        if self.config.use_funding_rates:
            if self.use_real_data and self.data_loader:
                self.funding_data = await self.data_loader.load_funding_rates(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                if self.funding_data.empty:
                    raise ValueError(
                        "No real funding rate data available. Cannot train on sample data."
                    )
            else:
                raise ValueError(
                    "use_real_data=False is not allowed. "
                    "Funding rate training requires real exchange data."
                )

        # Sentiment and trends: skip instead of using fake data
        if self.config.use_sentiment:
            logger.warning("Sentiment data API not integrated — skipping sentiment features")
            self.sentiment_data = None

        if self.config.use_google_trends:
            logger.warning("Google Trends API not integrated — skipping trends features")
            self.trends_data = None

        logger.info(f"Data loaded: {len(self.ohlcv_data)} OHLCV records")

    async def prepare_data(self):
        """Prepare data for training"""
        logger.info("Preparing features...")

        # Create features
        features = self.feature_engineering.create_features(
            self.ohlcv_data,
            self.sentiment_data,
            self.trends_data,
            self.funding_data
        )

        # Create sequences
        X, y = self.feature_engineering.create_sequences(
            features,
            lookback_periods=self.config.lookback_periods,
            prediction_horizons=self.config.prediction_horizons
        )

        logger.info(f"Created {len(X)} sequences")

        # Split into train/val/test
        n_samples = len(X)
        n_train = int(n_samples * 0.7)
        n_val = int(n_samples * 0.15)

        self.X_train = X[:n_train]
        self.y_train = y[:n_train]

        self.X_val = X[n_train:n_train + n_val]
        self.y_val = y[n_train:n_train + n_val]

        self.X_test = X[n_train + n_val:]
        self.y_test = y[n_train + n_val:]

        logger.info(
            f"Data split: train={len(self.X_train)}, "
            f"val={len(self.X_val)}, test={len(self.X_test)}"
        )

        # Create model
        n_features = X.shape[2]
        n_outputs = len(self.config.prediction_horizons)

        self.model = LSTMPriceModel(
            n_features=n_features,
            n_outputs=n_outputs,
            lstm_units=self.config.lstm_units,
            dropout_rate=self.config.dropout_rate,
            learning_rate=self.config.learning_rate
        )

        logger.info("Model created")
        self.model.summary()

    async def train(self) -> ModelMetrics:
        """
        Train the model

        Returns:
            ModelMetrics with training results
        """
        if self.model is None:
            raise ValueError("Data not prepared. Call prepare_data() first")

        logger.info("Starting training...")

        start_time = datetime.utcnow()

        # Train
        self.history = self.model.train(
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            early_stopping_patience=self.config.early_stopping_patience,
            verbose=1
        )

        training_time = (datetime.utcnow() - start_time).total_seconds()

        logger.info(f"Training completed in {training_time:.1f}s")

        # Evaluate on test set
        test_loss, test_mae, test_mape = self.model.evaluate(self.X_test, self.y_test)

        # Calculate direction accuracy
        direction_accuracy = self.model.calculate_direction_accuracy(
            self.X_test,
            self.y_test
        )

        # Calculate timeframe-specific accuracies
        y_pred = self.model.predict(self.X_test)

        accuracy_1h = (np.sign(y_pred[:, 0]) == np.sign(self.y_test[:, 0])).mean()
        accuracy_4h = (np.sign(y_pred[:, 1]) == np.sign(self.y_test[:, 1])).mean() if y_pred.shape[1] > 1 else None
        accuracy_24h = (np.sign(y_pred[:, 2]) == np.sign(self.y_test[:, 2])).mean() if y_pred.shape[1] > 2 else None

        # Create metrics
        metrics = ModelMetrics(
            direction_accuracy=direction_accuracy,
            mae=test_mae,
            mse=test_loss,
            rmse=np.sqrt(test_loss),
            mape=test_mape,
            accuracy_1h=accuracy_1h,
            accuracy_4h=accuracy_4h,
            accuracy_24h=accuracy_24h,
            training_samples=len(self.X_train),
            validation_samples=len(self.X_val),
            epochs=len(self.history['loss']),
            training_time_seconds=training_time
        )

        logger.info(
            f"Model Metrics:\n"
            f"  Direction Accuracy: {metrics.direction_accuracy:.1%}\n"
            f"  MAE: {metrics.mae:.2f}\n"
            f"  RMSE: {metrics.rmse:.2f}\n"
            f"  MAPE: {metrics.mape:.2f}%\n"
            f"  1h Accuracy: {metrics.accuracy_1h:.1%}\n"
            f"  4h Accuracy: {metrics.accuracy_4h:.1%}\n"
            f"  24h Accuracy: {metrics.accuracy_24h:.1%}"
        )

        return metrics

    async def save_model(self, symbol: str, version: str = "v1"):
        """
        Save trained model

        Args:
            symbol: Trading symbol
            version: Model version
        """
        if self.model is None:
            raise ValueError("No model to save")

        # Create save directory
        save_dir = Path(self.config.model_save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        filename = f"{symbol.lower()}_{version}_{datetime.utcnow().strftime('%Y%m%d')}.h5"
        filepath = save_dir / filename

        self.model.save(str(filepath))

        logger.info(f"Model saved to {filepath}")

        return str(filepath)


# Example usage
async def main():
    """Example: Train a model for BTC"""

    # Create config
    config = TrainingConfig(
        lookback_periods=60,
        prediction_horizons=[1, 4, 24],
        lstm_units=[128, 64, 32],
        dropout_rate=0.2,
        learning_rate=0.001,
        epochs=50,
        batch_size=32,
        use_technical_indicators=True,
        use_sentiment=True,
        use_google_trends=True,
        use_funding_rates=True
    )

    # Create trainer
    trainer = ModelTrainer(config)

    # Load data
    print("\n1. Loading data...")
    await trainer.load_data(
        "BTCUSDT",
        start_date=datetime.utcnow() - timedelta(days=90),
        end_date=datetime.utcnow()
    )

    # Prepare data
    print("\n2. Preparing features...")
    await trainer.prepare_data()

    # Train
    print("\n3. Training model...")
    metrics = await trainer.train()

    # Save
    print("\n4. Saving model...")
    filepath = await trainer.save_model("BTCUSDT", version="v1")

    print("\n✅ Training complete!")
    print(f"   Model saved: {filepath}")
    print(f"   Direction Accuracy: {metrics.direction_accuracy:.1%}")
    print(f"   Is Good Model: {metrics.is_good_model}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

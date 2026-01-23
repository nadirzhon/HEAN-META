"""
Model training script with auto-retraining capabilities.

Features:
- Train ensemble models on historical data
- Auto-retraining every 24 hours
- Model versioning and checkpointing
- Performance monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta
import time
import asyncio
import logging
from dataclasses import dataclass

from ml.price_predictor import BitcoinPricePredictor, PredictorConfig
from ml.features import FeatureConfig
from ml.data_loader import DataLoader
from ml.metrics import MetricsCalculator

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for model trainer."""

    # Training schedule
    retrain_interval_hours: int = 24
    auto_retrain: bool = True

    # Data configuration
    train_size: float = 0.7
    val_size: float = 0.15
    test_size: float = 0.15

    # Model directory
    model_dir: str = "models/bitcoin_predictor"
    checkpoint_dir: str = "models/checkpoints"

    # Data sources
    data_source: str = "synthetic"  # "csv", "bybit", "binance", "synthetic"
    csv_path: Optional[str] = None
    orderbook_path: Optional[str] = None
    sentiment_path: Optional[str] = None

    # Performance thresholds
    min_accuracy: float = 0.52  # Minimum accuracy to accept model
    min_f1_score: float = 0.50

    # Logging
    log_interval: int = 10  # Log every N epochs


class ModelTrainer:
    """
    Model trainer with auto-retraining capabilities.

    Trains Bitcoin price prediction models on schedule.
    """

    def __init__(
        self,
        trainer_config: Optional[TrainerConfig] = None,
        predictor_config: Optional[PredictorConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
    ):
        """
        Initialize model trainer.

        Args:
            trainer_config: Trainer configuration
            predictor_config: Predictor configuration
            feature_config: Feature engineering configuration
        """
        self.trainer_config = trainer_config or TrainerConfig()
        self.predictor_config = predictor_config or PredictorConfig()
        self.feature_config = feature_config or FeatureConfig()

        self.predictor = BitcoinPricePredictor(
            config=self.predictor_config,
            feature_config=self.feature_config,
        )

        self.data_loader = DataLoader(symbol="BTCUSDT")

        # Training state
        self.is_running = False
        self.last_training_time: Optional[datetime] = None
        self.training_count = 0

        # Performance tracking
        self.performance_history: list[Dict[str, Any]] = []

    def train_model(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
        save_model: bool = True,
    ) -> Dict[str, Any]:
        """
        Train the model on provided data.

        Args:
            df: OHLCV dataframe
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data
            save_model: Whether to save the trained model

        Returns:
            Training metrics dictionary
        """
        logger.info(f"Starting training with {len(df)} samples")

        # Prepare data
        X, y = self.predictor.prepare_data(df, orderbook_data, sentiment_data)

        logger.info(f"Generated {len(X)} samples with {len(X.columns)} features")

        # Split data
        n = len(X)
        train_end = int(n * self.trainer_config.train_size)
        val_end = int(n * (self.trainer_config.train_size + self.trainer_config.val_size))

        X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
        X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

        logger.info(
            f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
        )

        # Train models
        training_metrics = self.predictor.train(X_train, y_train, X_val, y_val)

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = self._evaluate_model(X_test, y_test, df.iloc[val_end:]['close'].values)

        # Combine metrics
        all_metrics = {
            'training_time': datetime.now().isoformat(),
            'n_train_samples': len(X_train),
            'n_val_samples': len(X_val),
            'n_test_samples': len(X_test),
            'n_features': len(X.columns),
            **test_metrics,
        }

        # Check performance thresholds
        if test_metrics['accuracy'] < self.trainer_config.min_accuracy:
            logger.warning(
                f"Model accuracy {test_metrics['accuracy']:.4f} below threshold "
                f"{self.trainer_config.min_accuracy:.4f}"
            )

        if test_metrics['f1_score'] < self.trainer_config.min_f1_score:
            logger.warning(
                f"Model F1 score {test_metrics['f1_score']:.4f} below threshold "
                f"{self.trainer_config.min_f1_score:.4f}"
            )

        # Save model
        if save_model:
            self._save_model(all_metrics)

        # Update state
        self.last_training_time = datetime.now()
        self.training_count += 1
        self.performance_history.append(all_metrics)

        # Print metrics
        MetricsCalculator.print_metrics_report(test_metrics)

        return all_metrics

    def _evaluate_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        prices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.

        Args:
            X: Test features
            y: Test labels
            prices: Test prices

        Returns:
            Metrics dictionary
        """
        # Predict
        y_pred = self.predictor.predict(X)
        y_proba = self.predictor.predict_proba(X)

        # Calculate metrics
        metrics = MetricsCalculator.calculate_all_metrics(
            y.values,
            y_pred,
            y_proba,
            prices,
            transaction_cost=0.001,
        )

        return metrics

    def _save_model(self, metrics: Dict[str, Any]) -> None:
        """
        Save trained model with versioning.

        Args:
            metrics: Training metrics
        """
        # Create model directory
        model_path = Path(self.trainer_config.model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        # Save latest model
        self.predictor.save_model(str(model_path))

        # Save checkpoint with timestamp
        checkpoint_path = Path(self.trainer_config.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_subdir = checkpoint_path / f"checkpoint_{timestamp}"
        checkpoint_subdir.mkdir(parents=True, exist_ok=True)

        self.predictor.save_model(str(checkpoint_subdir))

        logger.info(f"Model saved to {model_path}")
        logger.info(f"Checkpoint saved to {checkpoint_subdir}")

    def load_data(self) -> pd.DataFrame:
        """
        Load training data based on configuration.

        Returns:
            OHLCV dataframe
        """
        if self.trainer_config.data_source == "csv":
            if self.trainer_config.csv_path is None:
                raise ValueError("csv_path must be specified for CSV data source")
            return self.data_loader.load_from_csv(self.trainer_config.csv_path)

        elif self.trainer_config.data_source == "bybit":
            return self.data_loader.load_from_exchange(
                exchange="bybit",
                start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                interval="5m",
            )

        elif self.trainer_config.data_source == "binance":
            return self.data_loader.load_from_exchange(
                exchange="binance",
                start_date=(datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
                interval="5m",
            )

        elif self.trainer_config.data_source == "synthetic":
            logger.info("Generating synthetic data for testing...")
            return self.data_loader.generate_synthetic_data(n_samples=10000)

        else:
            raise ValueError(f"Unknown data source: {self.trainer_config.data_source}")

    def train_once(self) -> Dict[str, Any]:
        """
        Perform single training run.

        Returns:
            Training metrics
        """
        logger.info("Loading data...")
        df = self.load_data()

        # Load optional data
        orderbook_data = None
        if self.trainer_config.orderbook_path:
            orderbook_data = self.data_loader.load_orderbook_data(
                self.trainer_config.orderbook_path
            )

        sentiment_data = None
        if self.trainer_config.sentiment_path:
            sentiment_data = self.data_loader.load_sentiment_data(
                self.trainer_config.sentiment_path
            )

        # Train
        metrics = self.train_model(df, orderbook_data, sentiment_data)

        return metrics

    async def start_auto_retraining(self) -> None:
        """
        Start auto-retraining loop.

        Retrains model every N hours (configured in trainer_config).
        """
        if not self.trainer_config.auto_retrain:
            logger.info("Auto-retraining is disabled")
            return

        self.is_running = True
        logger.info(
            f"Starting auto-retraining loop (interval: {self.trainer_config.retrain_interval_hours}h)"
        )

        while self.is_running:
            try:
                # Train model
                logger.info(f"Starting scheduled training run #{self.training_count + 1}")
                metrics = self.train_once()

                logger.info(
                    f"Training completed. Accuracy: {metrics.get('accuracy', 0):.4f}, "
                    f"F1: {metrics.get('f1_score', 0):.4f}"
                )

                # Wait for next training cycle
                wait_seconds = self.trainer_config.retrain_interval_hours * 3600
                logger.info(
                    f"Next training in {self.trainer_config.retrain_interval_hours} hours "
                    f"({datetime.now() + timedelta(seconds=wait_seconds)})"
                )

                await asyncio.sleep(wait_seconds)

            except Exception as e:
                logger.error(f"Error during auto-retraining: {e}", exc_info=True)
                # Wait 1 hour before retrying
                await asyncio.sleep(3600)

    def stop_auto_retraining(self) -> None:
        """Stop auto-retraining loop."""
        logger.info("Stopping auto-retraining loop")
        self.is_running = False

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get top N most important features.

        Args:
            top_n: Number of top features

        Returns:
            Dictionary of features and importance scores
        """
        return self.predictor.get_feature_importance(top_n)

    def get_training_history(self) -> list[Dict[str, Any]]:
        """Get training history."""
        return self.performance_history

    def get_status(self) -> Dict[str, Any]:
        """Get trainer status."""
        return {
            'is_running': self.is_running,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'training_count': self.training_count,
            'auto_retrain_enabled': self.trainer_config.auto_retrain,
            'retrain_interval_hours': self.trainer_config.retrain_interval_hours,
            'model_is_trained': self.predictor.is_trained,
        }


# CLI interface for manual training
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Bitcoin price prediction model")
    parser.add_argument(
        "--data-source",
        choices=["csv", "bybit", "binance", "synthetic"],
        default="synthetic",
        help="Data source"
    )
    parser.add_argument("--csv-path", type=str, help="Path to CSV file (if data-source=csv)")
    parser.add_argument("--model-dir", type=str, default="models/bitcoin_predictor", help="Model directory")
    parser.add_argument("--auto-retrain", action="store_true", help="Enable auto-retraining")
    parser.add_argument("--retrain-interval", type=int, default=24, help="Retrain interval (hours)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create trainer config
    trainer_config = TrainerConfig(
        data_source=args.data_source,
        csv_path=args.csv_path,
        model_dir=args.model_dir,
        auto_retrain=args.auto_retrain,
        retrain_interval_hours=args.retrain_interval,
    )

    # Create trainer
    trainer = ModelTrainer(trainer_config=trainer_config)

    # Run training
    if args.auto_retrain:
        # Start auto-retraining loop
        asyncio.run(trainer.start_auto_retraining())
    else:
        # Single training run
        metrics = trainer.train_once()
        print("\nTraining completed successfully!")
        print(f"Model saved to: {args.model_dir}")

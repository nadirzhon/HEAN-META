"""
Auto-Retraining System

Automatically retrains ensemble model on fresh data every 24 hours.

Features:
- Scheduled retraining (configurable interval)
- Data fetching from exchange/database
- Model versioning
- Performance monitoring
- Automatic rollback if performance degrades

Author: HEAN Team
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from loguru import logger

from hean.ml.price_predictor import (
    EnsemblePredictor,
    PredictorConfig,
    prepare_target,
)


@dataclass
class RetrainerConfig:
    """Configuration for auto-retrainer."""

    # Scheduling
    retrain_interval_hours: int = 24
    retrain_at_startup: bool = True

    # Data
    training_window_days: int = 90  # Use last 90 days
    min_samples: int = 10000  # Minimum samples required

    # Model versioning
    keep_n_models: int = 5  # Keep last 5 models
    model_dir: str = "models/ensemble"

    # Performance monitoring
    min_accuracy: float = 0.53  # Rollback if accuracy < 53%
    min_auc: float = 0.55  # Rollback if AUC < 55%

    # Predictor config
    predictor_config: PredictorConfig = None

    def __post_init__(self) -> None:
        """Initialize predictor config."""
        if self.predictor_config is None:
            self.predictor_config = PredictorConfig()


class AutoRetrainer:
    """
    Automatic model retraining system.

    Usage:
        config = RetrainerConfig(retrain_interval_hours=24)
        retrainer = AutoRetrainer(config, data_source=my_data_source)

        # Start background retraining
        await retrainer.start()

        # Manual retrain
        await retrainer.retrain_now()

        # Get latest model
        predictor = retrainer.get_latest_predictor()
    """

    def __init__(
        self,
        config: RetrainerConfig,
        data_source: Optional[Any] = None,
    ) -> None:
        """
        Initialize auto-retrainer.

        Args:
            config: Retrainer configuration
            data_source: Source for fetching training data
                        (must implement fetch_ohlcv() method)
        """
        self.config = config
        self.data_source = data_source
        self.current_predictor: Optional[EnsemblePredictor] = None
        self.running = False
        self.last_retrain: Optional[datetime] = None
        self.retrain_history: list[Dict[str, Any]] = []

        Path(config.model_dir).mkdir(parents=True, exist_ok=True)

        logger.info("AutoRetrainer initialized", config=self.config)

    async def start(self) -> None:
        """Start automatic retraining loop."""
        if self.running:
            logger.warning("AutoRetrainer already running")
            return

        self.running = True
        logger.info("AutoRetrainer started")

        # Initial training
        if self.config.retrain_at_startup:
            await self.retrain_now()

        # Background loop
        asyncio.create_task(self._retrain_loop())

    async def stop(self) -> None:
        """Stop automatic retraining."""
        self.running = False
        logger.info("AutoRetrainer stopped")

    async def retrain_now(self) -> EnsemblePredictor:
        """
        Manually trigger retraining.

        Returns:
            Newly trained predictor
        """
        logger.info("Starting manual retrain")
        return await self._perform_retrain()

    def get_latest_predictor(self) -> Optional[EnsemblePredictor]:
        """Get currently active predictor."""
        return self.current_predictor

    async def _retrain_loop(self) -> None:
        """Background retraining loop."""
        while self.running:
            try:
                # Check if retrain is needed
                if self._should_retrain():
                    await self._perform_retrain()

                # Sleep until next check (every hour)
                await asyncio.sleep(3600)

            except Exception as e:
                logger.error(f"Error in retrain loop: {e}")
                await asyncio.sleep(3600)

    def _should_retrain(self) -> bool:
        """Check if retraining is needed."""
        if self.last_retrain is None:
            return True

        time_since_retrain = datetime.now() - self.last_retrain
        hours_since = time_since_retrain.total_seconds() / 3600

        return hours_since >= self.config.retrain_interval_hours

    async def _perform_retrain(self) -> EnsemblePredictor:
        """Perform model retraining."""
        try:
            logger.info("=== Starting Retraining ===")
            start_time = datetime.now()

            # 1. Fetch training data
            logger.info("Fetching training data...")
            df = await self._fetch_training_data()

            if len(df) < self.config.min_samples:
                raise ValueError(
                    f"Insufficient data: {len(df)} < {self.config.min_samples}"
                )

            logger.info(f"Fetched {len(df)} samples")

            # 2. Prepare features and target
            logger.info("Preparing features...")
            features_df = await self._prepare_features(df)
            features_df = prepare_target(
                features_df,
                horizon=self.config.predictor_config.prediction_horizon,
                threshold=self.config.predictor_config.classification_threshold,
            )

            # 3. Train new model
            logger.info("Training new model...")
            predictor = EnsemblePredictor(self.config.predictor_config)
            metrics = predictor.train(features_df)

            # 4. Validate performance
            logger.info("Validating performance...")
            if not self._validate_performance(metrics):
                logger.warning("Performance below threshold, keeping old model")
                return self.current_predictor

            # 5. Save new model
            model_path = predictor.save(
                f"{self.config.model_dir}/ensemble_{start_time.strftime('%Y%m%d_%H%M%S')}.pkl"
            )

            # 6. Update current predictor
            old_predictor = self.current_predictor
            self.current_predictor = predictor
            self.last_retrain = datetime.now()

            # 7. Record history
            self.retrain_history.append({
                'timestamp': start_time,
                'duration': (datetime.now() - start_time).total_seconds(),
                'metrics': metrics,
                'model_path': model_path,
                'samples': len(df),
            })

            # 8. Cleanup old models
            self._cleanup_old_models()

            duration = (datetime.now() - start_time).total_seconds()
            logger.info(
                f"Retraining complete in {duration:.1f}s",
                metrics=metrics
            )

            return predictor

        except Exception as e:
            logger.error(f"Retraining failed: {e}", exc_info=True)
            raise

    async def _fetch_training_data(self) -> pd.DataFrame:
        """Fetch training data from source."""
        if self.data_source is None:
            # Fallback: generate synthetic data for testing
            logger.warning("No data source, generating synthetic data")
            return self._generate_synthetic_data()

        # Fetch from data source
        end_time = datetime.now()
        start_time = end_time - timedelta(days=self.config.training_window_days)

        if hasattr(self.data_source, 'fetch_ohlcv'):
            df = await self.data_source.fetch_ohlcv(
                symbol='BTCUSDT',
                timeframe='5m',
                start_time=start_time,
                end_time=end_time,
            )
        else:
            raise ValueError("Data source must have fetch_ohlcv() method")

        return df

    async def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features from OHLCV data."""
        # Import here to avoid circular dependency
        from hean.features import TALibFeatures, FeatureConfig

        config = FeatureConfig(
            rsi_periods=[7, 14, 21, 28],
            macd_params=[(12, 26, 9), (5, 35, 5)],
            enable_patterns=True,
            enable_statistical=True,
        )

        ta = TALibFeatures(config)
        features = ta.generate_features(df, include_patterns=True)

        # Add timestamp for reference
        if 'timestamp' in df.columns:
            features['timestamp'] = df['timestamp']

        return features

    def _validate_performance(self, metrics: Dict[str, Any]) -> bool:
        """Validate model performance against thresholds."""
        accuracy = metrics.get('ensemble_accuracy', 0.0)
        auc = metrics.get('ensemble_auc', 0.0)

        if accuracy < self.config.min_accuracy:
            logger.warning(
                f"Accuracy {accuracy:.3f} < {self.config.min_accuracy}"
            )
            return False

        if auc < self.config.min_auc:
            logger.warning(
                f"AUC {auc:.3f} < {self.config.min_auc}"
            )
            return False

        return True

    def _cleanup_old_models(self) -> None:
        """Delete old model files, keeping only recent N."""
        model_dir = Path(self.config.model_dir)
        models = sorted(model_dir.glob("ensemble_*.pkl"), key=lambda p: p.stat().st_mtime)

        if len(models) > self.config.keep_n_models:
            for model_path in models[:-self.config.keep_n_models]:
                logger.info(f"Deleting old model: {model_path}")
                model_path.unlink()

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        import numpy as np

        n = 20000
        dates = pd.date_range(
            end=datetime.now(),
            periods=n,
            freq='5min'
        )

        # Random walk with drift
        returns = np.random.randn(n) * 0.001 + 0.0001
        price = 50000 * (1 + returns).cumprod()

        df = pd.DataFrame({
            'timestamp': dates,
            'open': price,
            'high': price * 1.002,
            'low': price * 0.998,
            'close': price,
            'volume': np.random.randint(1000, 10000, n).astype(float),
        })

        return df

    def get_retrain_history(self) -> list[Dict[str, Any]]:
        """Get history of retraining runs."""
        return self.retrain_history

    def get_next_retrain_time(self) -> Optional[datetime]:
        """Get scheduled time for next retrain."""
        if self.last_retrain is None:
            return datetime.now()

        return self.last_retrain + timedelta(
            hours=self.config.retrain_interval_hours
        )

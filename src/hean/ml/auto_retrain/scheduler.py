"""
Auto-Retraining Scheduler

Automatically retrains ML models every 24 hours on fresh data.
Monitors model performance and triggers retraining when needed.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json

from ..training.trainer import ModelTrainer
from ..inference.predictor import MLPredictor


class RetrainingScheduler:
    """
    Manages automatic model retraining on a schedule.

    Features:
    - Scheduled retraining (default: every 24 hours)
    - Performance-based retraining (trigger on degradation)
    - Safe model replacement (validation before deployment)
    - Training history tracking
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize retraining scheduler.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Scheduling parameters
        self.retrain_interval_hours = self.config.get('retrain_interval_hours', 24)
        self.min_samples_required = self.config.get('min_samples_required', 10000)

        # Performance thresholds for triggering retraining
        self.min_accuracy = self.config.get('min_accuracy', 0.55)
        self.max_performance_drop = self.config.get('max_performance_drop', 0.05)

        # Paths
        self.model_save_path = self.config.get('model_save_path', 'models/bitcoin_predictor')
        self.backup_path = self.config.get('backup_path', 'models/backups')

        # State
        self.is_running = False
        self.last_training_time = None
        self.next_training_time = None
        self.training_history = []

        # Callbacks
        self.on_training_start: Optional[Callable] = None
        self.on_training_complete: Optional[Callable] = None
        self.on_training_error: Optional[Callable] = None

        # Data provider (function that returns training data)
        self.data_provider: Optional[Callable] = None

        self.logger.info(f"RetrainingScheduler initialized (interval: {self.retrain_interval_hours}h)")

    async def start(self, data_provider: Callable) -> None:
        """
        Start the retraining scheduler.

        Args:
            data_provider: Async function that returns (ohlcv_data, orderbook_data, sentiment_data)
        """
        self.data_provider = data_provider
        self.is_running = True

        # Calculate next training time
        self.next_training_time = datetime.now() + timedelta(hours=self.retrain_interval_hours)

        self.logger.info(f"Retraining scheduler started. Next training: {self.next_training_time}")

        # Start scheduler loop
        await self._scheduler_loop()

    async def stop(self) -> None:
        """Stop the retraining scheduler."""
        self.is_running = False
        self.logger.info("Retraining scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self.is_running:
            try:
                # Check if it's time to retrain
                if datetime.now() >= self.next_training_time:
                    await self._execute_retraining()

                    # Schedule next training
                    self.next_training_time = datetime.now() + timedelta(
                        hours=self.retrain_interval_hours
                    )

                # Sleep for 1 minute before next check
                await asyncio.sleep(60)

            except Exception as e:
                self.logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                await asyncio.sleep(300)  # Sleep 5 minutes on error

    async def _execute_retraining(self) -> None:
        """Execute the retraining process."""
        self.logger.info("Starting automatic model retraining...")

        try:
            # Notify training start
            if self.on_training_start:
                await self._safe_callback(self.on_training_start)

            # 1. Get fresh data
            self.logger.info("Fetching training data...")
            ohlcv_data, orderbook_data, sentiment_data = await self.data_provider()

            # Validate data
            if len(ohlcv_data) < self.min_samples_required:
                raise ValueError(
                    f"Insufficient data: {len(ohlcv_data)} samples "
                    f"(required: {self.min_samples_required})"
                )

            # 2. Backup current model
            self._backup_current_model()

            # 3. Train new model
            trainer = ModelTrainer(self.config.get('trainer', {}))

            self.logger.info("Training new model...")
            results = trainer.train(
                ohlcv_data,
                orderbook_data,
                sentiment_data,
                split_method='time_series'
            )

            # 4. Validate new model
            is_valid = self._validate_new_model(results)

            if not is_valid:
                self.logger.warning("New model failed validation, keeping old model")
                self._restore_backup()
                return

            # 5. Save new model
            self.logger.info("Saving new model...")
            trainer.save_model(self.model_save_path)

            # 6. Update tracking
            self.last_training_time = datetime.now()

            training_record = {
                'timestamp': self.last_training_time.isoformat(),
                'metrics': results['metrics'],
                'duration_seconds': results['duration_seconds'],
                'data_stats': results['data_stats']
            }

            self.training_history.append(training_record)
            self._save_training_history()

            self.logger.info(f"Model retraining completed successfully")
            self.logger.info(f"Test Accuracy: {results['metrics']['test']['accuracy']:.4f}")
            self.logger.info(f"Test F1: {results['metrics']['test']['f1']:.4f}")

            # Notify training complete
            if self.on_training_complete:
                await self._safe_callback(self.on_training_complete, results)

        except Exception as e:
            self.logger.error(f"Retraining failed: {e}", exc_info=True)

            # Notify training error
            if self.on_training_error:
                await self._safe_callback(self.on_training_error, e)

            # Restore backup
            self._restore_backup()

    def _validate_new_model(self, results: Dict[str, Any]) -> bool:
        """
        Validate new model before deployment.

        Args:
            results: Training results

        Returns:
            True if model passes validation
        """
        test_metrics = results['metrics']['test']

        # Check minimum accuracy
        if test_metrics['accuracy'] < self.min_accuracy:
            self.logger.warning(
                f"Model accuracy {test_metrics['accuracy']:.4f} below threshold "
                f"{self.min_accuracy:.4f}"
            )
            return False

        # Check for performance drop (if we have history)
        if self.training_history:
            last_accuracy = self.training_history[-1]['metrics']['test']['accuracy']
            accuracy_drop = last_accuracy - test_metrics['accuracy']

            if accuracy_drop > self.max_performance_drop:
                self.logger.warning(
                    f"Model accuracy dropped by {accuracy_drop:.4f} "
                    f"(max allowed: {self.max_performance_drop:.4f})"
                )
                return False

        return True

    def _backup_current_model(self) -> None:
        """Backup current model before replacing."""
        import shutil

        model_path = Path(self.model_save_path)
        if not model_path.exists():
            return

        backup_dir = Path(self.backup_path)
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped backup
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"model_backup_{timestamp}"

        backup_path = backup_dir / backup_name

        try:
            shutil.copytree(model_path, backup_path)
            self.logger.info(f"Current model backed up to {backup_path}")
        except Exception as e:
            self.logger.error(f"Failed to backup model: {e}")

    def _restore_backup(self) -> None:
        """Restore most recent backup."""
        import shutil

        backup_dir = Path(self.backup_path)
        if not backup_dir.exists():
            return

        # Find most recent backup
        backups = sorted(backup_dir.glob('model_backup_*'))
        if not backups:
            return

        latest_backup = backups[-1]

        try:
            model_path = Path(self.model_save_path)
            if model_path.exists():
                shutil.rmtree(model_path)

            shutil.copytree(latest_backup, model_path)
            self.logger.info(f"Restored model from backup: {latest_backup}")
        except Exception as e:
            self.logger.error(f"Failed to restore backup: {e}")

    def _save_training_history(self) -> None:
        """Save training history to disk."""
        history_path = Path(self.model_save_path) / 'training_history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save training history: {e}")

    async def trigger_manual_retraining(self) -> Dict[str, Any]:
        """
        Manually trigger retraining.

        Returns:
            Training results
        """
        self.logger.info("Manual retraining triggered")
        await self._execute_retraining()

        if self.training_history:
            return self.training_history[-1]
        else:
            return {'error': 'Retraining failed'}

    def get_status(self) -> Dict[str, Any]:
        """
        Get scheduler status.

        Returns:
            Status dictionary
        """
        return {
            'is_running': self.is_running,
            'last_training_time': (
                self.last_training_time.isoformat()
                if self.last_training_time else None
            ),
            'next_training_time': (
                self.next_training_time.isoformat()
                if self.next_training_time else None
            ),
            'retrain_interval_hours': self.retrain_interval_hours,
            'training_count': len(self.training_history),
            'last_training_metrics': (
                self.training_history[-1]['metrics']
                if self.training_history else None
            )
        }

    async def _safe_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Safely execute callback with error handling."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Callback error: {e}", exc_info=True)

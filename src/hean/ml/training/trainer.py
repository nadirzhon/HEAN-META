"""
Model Training Module

Handles:
- Data splitting (train/val/test)
- Model training
- Hyperparameter optimization
- Model persistence
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json

from ..models.ensemble import EnsembleModel
from ..features.feature_engineer import FeatureEngineer
from ..metrics.evaluator import ModelEvaluator
from .data_splitter import DataSplitter


class ModelTrainer:
    """
    Orchestrates the complete training pipeline.

    Handles data preparation, feature engineering, model training, and evaluation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize trainer.

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}

        # Initialize components
        self.feature_engineer = FeatureEngineer(self.config.get('features', {}))
        self.model = EnsembleModel(self.config.get('model', {}))
        self.evaluator = ModelEvaluator()
        self.data_splitter = DataSplitter(self.config.get('data_split', {}))

        # Training state
        self.is_trained = False
        self.training_history = []

    def prepare_data(
        self,
        ohlcv_data: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Prepare data with feature engineering.

        Args:
            ohlcv_data: OHLCV data
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data

        Returns:
            DataFrame with all features and target
        """
        print("Engineering features...")

        # Engineer features
        df = self.feature_engineer.engineer_features(
            ohlcv_data,
            orderbook_data,
            sentiment_data
        )

        # If no orderbook data, create synthetic
        if orderbook_data is None:
            print("Creating synthetic orderbook features...")
            from ..features.orderbook_features import OrderbookFeatures
            ob_features = OrderbookFeatures()
            df = ob_features.create_synthetic_orderbook(df)

        # If no sentiment data, create synthetic
        if sentiment_data is None:
            print("Creating synthetic sentiment features...")
            from ..features.sentiment_features import SentimentFeatures
            sent_features = SentimentFeatures()
            df = sent_features.create_synthetic_sentiment(df)

        # Validate features
        is_valid, issues = self.feature_engineer.validate_features(df)
        if not is_valid:
            print(f"Warning: Feature validation issues: {issues}")

        print(f"Prepared {len(df)} samples with {len(self.feature_engineer.get_feature_names(df))} features")

        return df

    def split_data(
        self,
        df: pd.DataFrame,
        split_method: str = 'time_series'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: DataFrame with features and target
            split_method: Split method ('time_series' or 'random')

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if split_method == 'time_series':
            return self.data_splitter.time_series_split(df)
        elif split_method == 'random':
            return self.data_splitter.random_split(df)
        else:
            raise ValueError(f"Unknown split method: {split_method}")

    def train(
        self,
        ohlcv_data: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[Dict] = None,
        split_method: str = 'time_series'
    ) -> Dict[str, Any]:
        """
        Complete training pipeline.

        Args:
            ohlcv_data: OHLCV data
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment data
            split_method: Data split method

        Returns:
            Training results
        """
        start_time = datetime.now()

        # 1. Prepare data
        df = self.prepare_data(ohlcv_data, orderbook_data, sentiment_data)

        # 2. Split data
        train_df, val_df, test_df = self.split_data(df, split_method)

        print(f"\nData split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val:   {len(val_df)} samples")
        print(f"  Test:  {len(test_df)} samples")

        # 3. Prepare features and targets
        feature_cols = self.feature_engineer.get_feature_names(df)

        X_train = train_df[feature_cols]
        y_train = train_df['target']

        X_val = val_df[feature_cols]
        y_val = val_df['target']

        X_test = test_df[feature_cols]
        y_test = test_df['target']

        # 4. Train model
        print("\nTraining ensemble model...")
        training_results = self.model.train(X_train, y_train, X_val, y_val)

        # 5. Evaluate on all sets
        print("\nEvaluating model...")

        # Train set
        y_train_pred = self.model.predict(X_train)
        y_train_proba = self.model.predict_proba(X_train)
        train_metrics = self.evaluator.evaluate(
            y_train, y_train_pred, y_train_proba, "train"
        )

        # Validation set
        y_val_pred = self.model.predict(X_val)
        y_val_proba = self.model.predict_proba(X_val)
        val_metrics = self.evaluator.evaluate(
            y_val, y_val_pred, y_val_proba, "validation"
        )

        # Test set
        y_test_pred = self.model.predict(X_test)
        y_test_proba = self.model.predict_proba(X_test)
        test_metrics = self.evaluator.evaluate(
            y_test, y_test_pred, y_test_proba, "test"
        )

        # Print evaluation results
        self.evaluator.print_evaluation(train_metrics)
        self.evaluator.print_evaluation(val_metrics)
        self.evaluator.print_evaluation(test_metrics)

        # 6. Feature importance
        print("\nTop 20 Important Features:")
        importance = self.model.get_feature_importance(top_n=20)
        print(importance['ensemble'][['feature', 'weighted_importance']].to_string(index=False))

        # Training complete
        self.is_trained = True
        end_time = datetime.now()

        # Compile results
        results = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': (end_time - start_time).total_seconds(),
            'data_stats': {
                'total_samples': len(df),
                'train_samples': len(train_df),
                'val_samples': len(val_df),
                'test_samples': len(test_df),
                'num_features': len(feature_cols)
            },
            'training_results': training_results,
            'metrics': {
                'train': train_metrics,
                'validation': val_metrics,
                'test': test_metrics
            },
            'feature_importance': importance['ensemble'].to_dict('records')
        }

        # Store in history
        self.training_history.append(results)

        return results

    def save_model(self, path: str, include_metadata: bool = True) -> None:
        """
        Save trained model and metadata.

        Args:
            path: Directory to save model
            include_metadata: Whether to save training metadata
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)

        # Save model
        self.model.save(str(path_obj / 'model'))

        # Save metadata
        if include_metadata and self.training_history:
            metadata = {
                'config': self.config,
                'training_history': self.training_history,
                'last_trained': datetime.now().isoformat()
            }

            with open(path_obj / 'training_metadata.json', 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json.dump(metadata, f, indent=2, default=self._json_serializer)

        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load trained model and metadata.

        Args:
            path: Directory to load model from
        """
        path_obj = Path(path)

        # Load model
        self.model.load(str(path_obj / 'model'))

        # Load metadata
        metadata_path = path_obj / 'training_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            self.config = metadata.get('config', {})
            self.training_history = metadata.get('training_history', [])

        self.is_trained = True
        print(f"Model loaded from {path}")

    @staticmethod
    def _json_serializer(obj):
        """Helper to serialize numpy types to JSON."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

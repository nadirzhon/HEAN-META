#!/usr/bin/env python3
"""
Train Bitcoin Price Prediction Model

This script trains the ensemble ML model on historical data.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hean.ml.training.trainer import ModelTrainer
from hean.ml.backtesting.backtester import Backtester


def load_sample_data() -> pd.DataFrame:
    """
    Load or generate sample OHLCV data.

    In production, replace this with actual data loading from exchange API.
    """
    # Generate synthetic OHLCV data for demonstration
    print("Generating sample data...")

    n_samples = 20000  # ~70 days of 5-minute data
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='5min')

    # Simulate Bitcoin-like price movement
    np.random.seed(42)
    price_base = 45000
    returns = np.random.normal(0.0001, 0.01, n_samples)
    prices = price_base * (1 + returns).cumprod()

    # Generate OHLCV
    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices * (1 + np.random.normal(0, 0.002, n_samples)),
        'volume': np.random.lognormal(10, 2, n_samples)
    }

    df = pd.DataFrame(data)

    print(f"Generated {len(df)} samples")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


def main():
    """Main training script."""
    print("="*80)
    print("Bitcoin Price Prediction Model Training")
    print("="*80)

    # 1. Load data
    ohlcv_data = load_sample_data()

    # 2. Configure trainer
    config = {
        'features': {
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'bb_period': 20
        },
        'model': {
            'lightgbm': {
                'num_leaves': 31,
                'learning_rate': 0.05
            },
            'xgboost': {
                'max_depth': 6,
                'learning_rate': 0.05
            },
            'catboost': {
                'depth': 6,
                'learning_rate': 0.05
            }
        },
        'data_split': {
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15
        }
    }

    # 3. Train model
    trainer = ModelTrainer(config)

    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    results = trainer.train(
        ohlcv_data,
        orderbook_data=None,  # Will use synthetic
        sentiment_data=None,   # Will use synthetic
        split_method='time_series'
    )

    # 4. Run backtest
    print("\n" + "="*80)
    print("Running backtest...")
    print("="*80)

    # Get test predictions
    test_metrics = results['metrics']['test']
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")

    # Backtest on test data (would need actual predictions here)
    # For now, just show the metrics

    # 5. Save model
    model_path = 'models/bitcoin_predictor'
    print(f"\n" + "="*80)
    print(f"Saving model to {model_path}...")
    print("="*80)

    trainer.save_model(model_path, include_metadata=True)

    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)

    print(f"\nModel saved to: {model_path}")
    print(f"Training duration: {results['duration_seconds']:.2f} seconds")
    print(f"\nYou can now use the model with: python scripts/run_ml_predictions.py")


if __name__ == '__main__':
    main()

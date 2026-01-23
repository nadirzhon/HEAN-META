#!/usr/bin/env python3
"""
Run ML Predictions

Make real-time predictions using trained model.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hean.ml.inference.predictor import MLPredictor


def load_live_data() -> pd.DataFrame:
    """
    Load live OHLCV data for prediction.

    In production, replace this with actual live data from exchange.
    """
    print("Loading live data...")

    # Generate recent data (last 200 candles needed for features)
    n_samples = 250
    timestamps = pd.date_range(end=pd.Timestamp.now(), periods=n_samples, freq='5min')

    # Simulate price movement
    np.random.seed(int(datetime.now().timestamp()))
    price_base = 45000
    returns = np.random.normal(0.0001, 0.01, n_samples)
    prices = price_base * (1 + returns).cumprod()

    data = {
        'timestamp': timestamps,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices * (1 + np.random.normal(0, 0.002, n_samples)),
        'volume': np.random.lognormal(10, 2, n_samples)
    }

    return pd.DataFrame(data)


def main():
    """Main prediction script."""
    print("="*80)
    print("Bitcoin Price Prediction - ML Inference")
    print("="*80)

    # 1. Load model
    model_path = 'models/bitcoin_predictor'

    if not Path(model_path).exists():
        print(f"\nError: Model not found at {model_path}")
        print("Please train the model first using: python scripts/train_ml_model.py")
        return

    print(f"\nLoading model from {model_path}...")
    predictor = MLPredictor(model_path)

    # 2. Check model health
    health = predictor.health_check()
    print(f"\nModel Health: {health['status']}")
    print(f"  Model Loaded: {health['model_loaded']}")
    print(f"  Predictions Made: {health['prediction_count']}")

    # 3. Load live data
    ohlcv_data = load_live_data()

    # Validate input
    is_valid, error = predictor.validate_input(ohlcv_data)
    if not is_valid:
        print(f"\nError: Invalid input data - {error}")
        return

    print(f"\nLoaded {len(ohlcv_data)} candles")
    print(f"Latest price: ${ohlcv_data['close'].iloc[-1]:.2f}")

    # 4. Make prediction
    print("\n" + "="*80)
    print("Making Prediction...")
    print("="*80)

    result = predictor.predict(
        ohlcv_data,
        orderbook_data=None,
        sentiment_data=None,
        return_probabilities=True
    )

    # 5. Display results
    if 'error' in result:
        print(f"\nPrediction Error: {result['error']}")
        return

    print(f"\nPrediction Results:")
    print(f"  Direction: {result['direction']}")
    print(f"  Probability: {result['probability']:.2%}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Inference Time: {result['inference_time_ms']:.2f}ms")

    print(f"\nModel Ensemble Breakdown:")
    print(f"  LightGBM: {result['model_probabilities']['lightgbm']:.4f}")
    print(f"  XGBoost:  {result['model_probabilities']['xgboost']:.4f}")
    print(f"  CatBoost: {result['model_probabilities']['catboost']:.4f}")

    # 6. Show top features
    print("\n" + "="*80)
    print("Top 10 Most Important Features:")
    print("="*80)

    importance = predictor.get_feature_importance(top_n=10)
    for i, row in importance.iterrows():
        print(f"  {i+1}. {row['feature']:30s} {row['weighted_importance']:.2f}")

    # 7. Trading recommendation
    print("\n" + "="*80)
    print("Trading Recommendation:")
    print("="*80)

    if result['direction'] == 'UP':
        if result['confidence'] > 0.7:
            print("  🟢 STRONG BUY signal")
        elif result['confidence'] > 0.5:
            print("  🟢 BUY signal")
        else:
            print("  🟡 WEAK BUY signal")
    else:
        if result['confidence'] > 0.7:
            print("  🔴 STRONG SELL signal")
        elif result['confidence'] > 0.5:
            print("  🔴 SELL signal")
        else:
            print("  🟡 WEAK SELL signal")

    print(f"\n  Confidence Level: {result['confidence']:.1%}")
    print(f"  Expected Movement: {result['direction']}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()

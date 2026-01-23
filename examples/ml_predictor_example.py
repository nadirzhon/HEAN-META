"""
Example: ML Ensemble Price Predictor

Shows how to:
1. Train ensemble model
2. Make predictions
3. Integrate with trading strategies
4. Auto-retrain on fresh data

Author: HEAN Team
"""

import asyncio
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger

from hean.features import FeatureConfig, TALibFeatures
from hean.ml import EnsemblePredictor, PredictorConfig, PredictionResult
from hean.ml.auto_retrainer import AutoRetrainer, RetrainerConfig
from hean.ml.price_predictor import prepare_target


async def example_train_and_predict() -> None:
    """Example: Train model and make predictions."""
    logger.info("=== Train and Predict ===")

    # 1. Generate sample data
    n = 5000
    dates = pd.date_range(end=datetime.now(), periods=n, freq='5min')

    # Realistic price movement (trend + noise)
    trend = np.linspace(50000, 52000, n)
    noise = np.random.randn(n).cumsum() * 50
    price = trend + noise

    df = pd.DataFrame({
        'timestamp': dates,
        'open': price,
        'high': price * 1.001,
        'low': price * 0.999,
        'close': price,
        'volume': np.random.randint(1000, 5000, n).astype(float),
    })

    # 2. Generate features
    logger.info("Generating features...")
    ta_config = FeatureConfig(
        rsi_periods=[14, 21],
        enable_patterns=True,
        enable_all_patterns=False,
    )
    ta = TALibFeatures(ta_config)
    features = ta.generate_features(df)

    # 3. Prepare target (predict 1 hour ahead = 12 candles)
    logger.info("Preparing target...")
    features = prepare_target(features, horizon=12, threshold=0.002)

    # 4. Train ensemble
    logger.info("Training ensemble...")
    config = PredictorConfig(
        prediction_horizon=12,
        classification_threshold=0.002,
    )
    predictor = EnsemblePredictor(config)
    metrics = predictor.train(features)

    logger.info("Training complete!")
    logger.info(f"Ensemble Accuracy: {metrics['ensemble_accuracy']:.3f}")
    logger.info(f"Ensemble AUC: {metrics['ensemble_auc']:.3f}")

    # 5. Make predictions on new data
    logger.info("\n=== Making Predictions ===")
    latest_features = features.iloc[-100:]

    for i in range(5):
        result = predictor.predict(latest_features.iloc[-(i+1)])
        logger.info(
            f"Prediction {i+1}: {result.direction.value} "
            f"(confidence: {result.confidence:.2%})"
        )
        logger.info(f"  Model votes: {result.model_votes}")

    # 6. Feature importance
    logger.info("\n=== Feature Importance ===")
    importance = predictor.get_feature_importance(top_n=10)

    logger.info("Top 10 features (LightGBM):")
    for idx, row in importance['lgb'].head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.2f}")


async def example_trading_integration() -> None:
    """Example: Integration with trading strategy."""
    logger.info("\n=== Trading Strategy Integration ===")

    # Generate data
    n = 3000
    price = 50000 + np.random.randn(n).cumsum() * 20
    df = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=n, freq='5min'),
        'open': price,
        'high': price * 1.002,
        'low': price * 0.998,
        'close': price,
        'volume': np.random.randint(1000, 5000, n).astype(float),
    })

    # Prepare features & train
    ta = TALibFeatures()
    features = ta.generate_features(df)
    features = prepare_target(features, horizon=12)

    predictor = EnsemblePredictor()
    predictor.train(features)

    # === Trading Strategy with ML Predictions ===
    signals = []
    positions = []
    pnl = []
    equity = 10000  # Starting equity

    for i in range(500, len(features)):
        row = features.iloc[i]
        price_current = df.iloc[i]['close']

        # Get ML prediction
        result = predictor.predict(features.iloc[i])

        # Traditional indicators
        rsi = row['rsi_14']
        macd_bullish = row['macd_12_26_9'] > row['macd_signal_12_26_9']

        # === Signal Generation ===
        # LONG: ML predicts UP + RSI oversold + high confidence
        if (result.direction.value == "UP" and
            result.confidence > 0.60 and
            rsi < 40 and
            macd_bullish):

            signals.append({
                'timestamp': row['timestamp'],
                'signal': 'LONG',
                'price': price_current,
                'confidence': result.confidence,
                'rsi': rsi,
            })

        # SHORT: ML predicts DOWN + RSI overbought + high confidence
        elif (result.direction.value == "DOWN" and
              result.confidence > 0.60 and
              rsi > 60 and
              not macd_bullish):

            signals.append({
                'timestamp': row['timestamp'],
                'signal': 'SHORT',
                'price': price_current,
                'confidence': result.confidence,
                'rsi': rsi,
            })

    logger.info(f"Generated {len(signals)} trading signals")

    if signals:
        logger.info("\nSample signals:")
        for sig in signals[:5]:
            logger.info(
                f"  {sig['signal']} @ ${sig['price']:.0f} "
                f"(conf: {sig['confidence']:.1%}, RSI: {sig['rsi']:.1f})"
            )

        # Calculate win rate (simplified)
        long_signals = [s for s in signals if s['signal'] == 'LONG']
        logger.info(f"\nLong signals: {len(long_signals)}")
        logger.info(f"Avg confidence: {np.mean([s['confidence'] for s in signals]):.1%}")


async def example_auto_retrain() -> None:
    """Example: Auto-retraining system."""
    logger.info("\n=== Auto-Retraining System ===")

    # Configure auto-retrainer
    config = RetrainerConfig(
        retrain_interval_hours=24,
        retrain_at_startup=True,
        training_window_days=90,
        min_accuracy=0.53,
    )

    # Start retrainer (will use synthetic data in this example)
    retrainer = AutoRetrainer(config, data_source=None)

    logger.info("Starting auto-retrainer...")
    await retrainer.start()

    # Wait a bit
    await asyncio.sleep(5)

    # Get latest predictor
    predictor = retrainer.get_latest_predictor()
    if predictor:
        logger.info("Latest predictor available!")
        logger.info(f"Metrics: {predictor.metrics}")

    # Manual retrain
    logger.info("\nTriggering manual retrain...")
    predictor = await retrainer.retrain_now()
    logger.info("Manual retrain complete!")

    # Stop retrainer
    await retrainer.stop()


async def example_model_persistence() -> None:
    """Example: Save and load models."""
    logger.info("\n=== Model Persistence ===")

    # Train a model
    n = 2000
    price = 50000 + np.random.randn(n).cumsum() * 10
    df = pd.DataFrame({
        'open': price, 'high': price * 1.001,
        'low': price * 0.999, 'close': price,
        'volume': np.random.randint(1000, 5000, n).astype(float),
    })

    ta = TALibFeatures()
    features = ta.generate_features(df)
    features = prepare_target(features)

    predictor = EnsemblePredictor()
    predictor.train(features)

    # Save model
    model_path = predictor.save("models/my_ensemble.pkl")
    logger.info(f"Model saved to: {model_path}")

    # Load model
    loaded_predictor = EnsemblePredictor.load(model_path)
    logger.info("Model loaded successfully!")

    # Test prediction
    result = loaded_predictor.predict(features.iloc[-1])
    logger.info(f"Prediction: {result.direction.value} ({result.confidence:.1%})")


async def main() -> None:
    """Run all examples."""
    await example_train_and_predict()
    await example_trading_integration()
    await example_auto_retrain()
    await example_model_persistence()

    logger.info("\n" + "="*60)
    logger.info("✅ All ML examples completed successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())

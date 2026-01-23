"""
Example: TA-Lib Integration with HEAN Trading System

This example shows how to:
1. Generate 200+ technical indicators
2. Integrate with existing strategies
3. Use features for ML models
4. Pattern recognition for signals

Author: HEAN Team
"""

import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from loguru import logger

from hean.features import FeatureConfig, TALibFeatures


async def example_basic_usage() -> None:
    """Basic TA-Lib usage example."""
    logger.info("=== Basic TA-Lib Usage ===")

    # Create sample OHLCV data (normally from exchange)
    np.random.seed(42)
    n = 1000
    base_price = 50000

    dates = pd.date_range(end=datetime.now(), periods=n, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': base_price + np.cumsum(np.random.randn(n) * 100),
        'high': base_price + np.cumsum(np.random.randn(n) * 100) + 50,
        'low': base_price + np.cumsum(np.random.randn(n) * 100) - 50,
        'close': base_price + np.cumsum(np.random.randn(n) * 100),
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    # Initialize TA-Lib
    config = FeatureConfig(
        rsi_periods=[14, 21, 28],
        enable_patterns=True,
        enable_all_patterns=False,  # Use only important patterns
    )
    ta = TALibFeatures(config)

    # Generate all features
    features = ta.generate_features(df, include_patterns=True)

    logger.info(f"Generated {len(features.columns)} features")
    logger.info(f"Feature columns: {list(features.columns[:20])}...")

    # Show last row with features
    logger.info("\nLatest indicators:")
    indicators = {
        'RSI_14': features['rsi_14'].iloc[-1],
        'MACD': features['macd_12_26_9'].iloc[-1],
        'BB_Position': features['bb_position_20'].iloc[-1],
        'ADX': features['adx'].iloc[-1],
        'Pattern_Score': features['pattern_score'].iloc[-1],
    }
    for name, value in indicators.items():
        logger.info(f"  {name}: {value:.2f}")


async def example_pattern_detection() -> None:
    """Pattern recognition example."""
    logger.info("\n=== Pattern Detection ===")

    # Generate sample data with patterns
    n = 100
    df = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 102,
        'low': np.random.randn(n).cumsum() + 98,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    ta = TALibFeatures()
    patterns = ta.detect_patterns(df, threshold=0)

    logger.info(f"Detected {len(patterns)} pattern types")
    for pattern_name, indices in list(patterns.items())[:5]:
        logger.info(f"  {pattern_name}: {len(indices)} occurrences at {indices[:3]}...")


async def example_strategy_integration() -> None:
    """Example: Integrating TA-Lib with a trading strategy."""
    logger.info("\n=== Strategy Integration ===")

    # Simulated market data
    n = 500
    df = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=n, freq='1min'),
        'open': 50000 + np.cumsum(np.random.randn(n) * 10),
        'high': 50000 + np.cumsum(np.random.randn(n) * 10) + 20,
        'low': 50000 + np.cumsum(np.random.randn(n) * 10) - 20,
        'close': 50000 + np.cumsum(np.random.randn(n) * 10),
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    ta = TALibFeatures()
    features = ta.generate_features(df, include_patterns=True)

    # === Strategy: RSI + MACD + Pattern Confirmation ===
    signals = []

    for i in range(50, len(features)):
        row = features.iloc[i]

        # RSI oversold/overbought
        rsi_oversold = row['rsi_14'] < 30
        rsi_overbought = row['rsi_14'] > 70

        # MACD crossover
        macd_bullish = row['macd_12_26_9'] > row['macd_signal_12_26_9']
        macd_bearish = row['macd_12_26_9'] < row['macd_signal_12_26_9']

        # Pattern confirmation
        bullish_pattern = row['pattern_score'] > 0
        bearish_pattern = row['pattern_score'] < 0

        # Trend filter (ADX > 25 = strong trend)
        strong_trend = row['adx'] > 25

        # Generate signals
        if rsi_oversold and macd_bullish and bullish_pattern and strong_trend:
            signals.append({
                'timestamp': row['timestamp'],
                'signal': 'BUY',
                'price': df.iloc[i]['close'],
                'rsi': row['rsi_14'],
                'macd': row['macd_12_26_9'],
                'pattern_score': row['pattern_score'],
            })
        elif rsi_overbought and macd_bearish and bearish_pattern and strong_trend:
            signals.append({
                'timestamp': row['timestamp'],
                'signal': 'SELL',
                'price': df.iloc[i]['close'],
                'rsi': row['rsi_14'],
                'macd': row['macd_12_26_9'],
                'pattern_score': row['pattern_score'],
            })

    logger.info(f"Generated {len(signals)} trading signals")
    if signals:
        logger.info("Sample signals:")
        for sig in signals[:3]:
            logger.info(f"  {sig}")


async def example_feature_importance() -> None:
    """Example: Feature importance analysis."""
    logger.info("\n=== Feature Importance ===")

    # Generate data with returns
    n = 1000
    df = pd.DataFrame({
        'open': 50000 + np.cumsum(np.random.randn(n) * 10),
        'high': 50000 + np.cumsum(np.random.randn(n) * 10) + 20,
        'low': 50000 + np.cumsum(np.random.randn(n) * 10) - 20,
        'close': 50000 + np.cumsum(np.random.randn(n) * 10),
        'volume': np.random.randint(100, 1000, n).astype(float),
    })

    ta = TALibFeatures()
    features = ta.generate_features(df, include_patterns=False)

    # Get feature importance
    importance = ta.get_feature_importance_proxy(features)

    logger.info("Top 20 features by correlation with returns:")
    for feature, corr in importance.head(20).items():
        logger.info(f"  {feature}: {corr:.4f}")


async def example_ml_features() -> None:
    """Example: Preparing features for ML models."""
    logger.info("\n=== ML Feature Preparation ===")

    # Generate data
    n = 2000
    df = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=n, freq='5min'),
        'open': 50000 + np.cumsum(np.random.randn(n) * 20),
        'high': 50000 + np.cumsum(np.random.randn(n) * 20) + 50,
        'low': 50000 + np.cumsum(np.random.randn(n) * 20) - 50,
        'close': 50000 + np.cumsum(np.random.randn(n) * 20),
        'volume': np.random.randint(1000, 10000, n).astype(float),
    })

    # Generate features
    config = FeatureConfig(
        rsi_periods=[7, 14, 21, 28],
        macd_params=[(12, 26, 9), (5, 35, 5), (19, 39, 9)],
        enable_patterns=True,
        enable_statistical=True,
    )
    ta = TALibFeatures(config)
    features = ta.generate_features(df, include_patterns=True, include_cycles=False)

    # Prepare for ML
    # 1. Create target (future returns)
    features['target_1h'] = features['close'].shift(-12).pct_change()  # 1h forward
    features['target_direction'] = (features['target_1h'] > 0).astype(int)

    # 2. Remove NaN
    features_clean = features.dropna()

    # 3. Select feature columns
    feature_cols = [
        col for col in features_clean.columns
        if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                      'target_1h', 'target_direction']
    ]

    X = features_clean[feature_cols]
    y = features_clean['target_direction']

    logger.info(f"ML Dataset prepared:")
    logger.info(f"  Samples: {len(X)}")
    logger.info(f"  Features: {len(feature_cols)}")
    logger.info(f"  Target distribution: {y.value_counts().to_dict()}")

    # Feature statistics
    logger.info(f"\nFeature statistics:")
    logger.info(f"  Missing values: {X.isna().sum().sum()}")
    logger.info(f"  Infinite values: {np.isinf(X.values).sum()}")


async def main() -> None:
    """Run all examples."""
    await example_basic_usage()
    await example_pattern_detection()
    await example_strategy_integration()
    await example_feature_importance()
    await example_ml_features()


if __name__ == "__main__":
    asyncio.run(main())

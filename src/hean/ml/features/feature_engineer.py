"""
Feature Engineering Pipeline for Bitcoin Price Prediction

Generates 50+ features including:
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Volume-based features
- Orderbook imbalance features
- Sentiment indicators
- Price action features
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

from .technical_indicators import TechnicalIndicators
from .volume_features import VolumeFeatures
from .orderbook_features import OrderbookFeatures
from .sentiment_features import SentimentFeatures


class FeatureEngineer:
    """
    Main feature engineering pipeline that orchestrates all feature generators.

    Produces 50+ features for ML models to predict Bitcoin price movements.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer with configuration.

        Args:
            config: Configuration dictionary for feature parameters
        """
        self.config = config or {}

        # Initialize feature generators
        self.technical = TechnicalIndicators(self.config)
        self.volume = VolumeFeatures(self.config)
        self.orderbook = OrderbookFeatures(self.config)
        self.sentiment = SentimentFeatures(self.config)

        # Feature groups
        self.feature_groups = [
            'technical',
            'volume',
            'orderbook',
            'sentiment',
            'price_action',
            'volatility',
            'momentum'
        ]

    def engineer_features(
        self,
        ohlcv_data: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Generate all features from input data.

        Args:
            ohlcv_data: OHLCV (Open, High, Low, Close, Volume) dataframe
            orderbook_data: Optional orderbook data
            sentiment_data: Optional sentiment indicators

        Returns:
            DataFrame with all engineered features
        """
        df = ohlcv_data.copy()

        # 1. Technical Indicators (20+ features)
        df = self.technical.add_all_indicators(df)

        # 2. Volume Features (10+ features)
        df = self.volume.add_volume_features(df)

        # 3. Orderbook Features (10+ features)
        if orderbook_data is not None:
            df = self.orderbook.add_orderbook_features(df, orderbook_data)

        # 4. Sentiment Features (5+ features)
        if sentiment_data is not None:
            df = self.sentiment.add_sentiment_features(df, sentiment_data)

        # 5. Price Action Features (5+ features)
        df = self._add_price_action_features(df)

        # 6. Additional Features
        df = self._add_volatility_features(df)
        df = self._add_momentum_features(df)
        df = self._add_time_features(df)

        # 7. Create target variable (price movement in 5 minutes)
        df = self._create_target(df, periods=1)  # 1 period = 5 minutes for 5m candles

        # Drop NaN values from indicator calculations
        df = df.dropna()

        return df

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features."""
        # High-Low range
        df['hl_range'] = df['high'] - df['low']
        df['hl_range_pct'] = (df['hl_range'] / df['close']) * 100

        # Body size (close - open)
        df['body_size'] = abs(df['close'] - df['open'])
        df['body_size_pct'] = (df['body_size'] / df['close']) * 100

        # Upper and lower shadows
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']

        # Shadow ratios
        df['shadow_ratio'] = df['upper_shadow'] / (df['lower_shadow'] + 1e-8)

        # Bullish/bearish candles
        df['is_bullish'] = (df['close'] > df['open']).astype(int)

        # Gap detection
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = (df['gap'] / df['close'].shift(1)) * 100

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        # Rolling standard deviation
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['close'].rolling(window=period).std()
            df[f'volatility_{period}_pct'] = (df[f'volatility_{period}'] / df['close']) * 100

        # Average True Range (ATR) - already added in technical indicators
        # True Range
        df['true_range'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )

        # Historical volatility
        df['hist_volatility_10'] = df['close'].pct_change().rolling(window=10).std() * np.sqrt(10)
        df['hist_volatility_20'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(20)

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        # Rate of change
        for period in [3, 5, 10]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                                   df['close'].shift(period)) * 100

        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)

        # Acceleration
        df['acceleration_5'] = df['momentum_5'] - df['momentum_5'].shift(1)

        # Velocity (smoothed momentum)
        df['velocity_5'] = df['close'].diff(5).rolling(window=3).mean()

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            # Cyclical encoding for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

            # Cyclical encoding for day of week
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    def _create_target(self, df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Create binary target variable for price movement prediction.

        Args:
            df: DataFrame with price data
            periods: Number of periods ahead to predict (1 period = 5 minutes)

        Returns:
            DataFrame with target column added
        """
        # Future price
        df['future_close'] = df['close'].shift(-periods)

        # Binary target: 1 if price goes up, 0 if down
        df['target'] = (df['future_close'] > df['close']).astype(int)

        # Also add continuous target for analysis
        df['price_change'] = df['future_close'] - df['close']
        df['price_change_pct'] = ((df['future_close'] - df['close']) / df['close']) * 100

        return df

    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of all feature column names (excluding target and metadata).

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        exclude_cols = [
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'target', 'future_close', 'price_change', 'price_change_pct'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature groups for importance analysis.

        Returns:
            Dictionary mapping group names to feature prefixes
        """
        return {
            'technical': ['rsi', 'macd', 'bb', 'sma', 'ema', 'stoch', 'adx'],
            'volume': ['volume', 'obv', 'vwap', 'mfi'],
            'orderbook': ['bid_ask', 'order_imbalance', 'spread'],
            'sentiment': ['sentiment', 'fear_greed'],
            'price_action': ['hl_range', 'body_size', 'shadow', 'gap'],
            'volatility': ['volatility', 'atr', 'true_range'],
            'momentum': ['roc', 'momentum', 'acceleration', 'velocity'],
            'time': ['hour', 'day', 'weekend']
        }

    def validate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate that all required features are present and valid.

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []

        # Check for target column
        if 'target' not in df.columns:
            issues.append("Missing 'target' column")

        # Check for NaN values
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            issues.append(f"NaN values found in columns: {nan_cols}")

        # Check for infinite values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_cols = [col for col in numeric_cols if np.isinf(df[col]).any()]
        if inf_cols:
            issues.append(f"Infinite values found in columns: {inf_cols}")

        # Check minimum number of features
        feature_cols = self.get_feature_names(df)
        if len(feature_cols) < 50:
            issues.append(f"Only {len(feature_cols)} features generated, expected 50+")

        is_valid = len(issues) == 0
        return is_valid, issues

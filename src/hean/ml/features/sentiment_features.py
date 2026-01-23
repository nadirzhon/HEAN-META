"""
Sentiment-based Features for ML

Implements sentiment-related indicators:
- Fear & Greed Index
- Social media sentiment
- News sentiment
- Market sentiment proxies
- And more...
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class SentimentFeatures:
    """Generator for sentiment-based features."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration."""
        self.config = config or {}

    def add_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment_data: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Add sentiment-based features.

        Args:
            df: Main price dataframe
            sentiment_data: Dictionary containing sentiment indicators
        """
        if sentiment_data:
            df = self._add_external_sentiment(df, sentiment_data)

        # Add market-based sentiment proxies
        df = self._add_market_sentiment_proxies(df)
        df = self._add_momentum_sentiment(df)
        df = self._add_volatility_sentiment(df)

        return df

    def _add_external_sentiment(
        self,
        df: pd.DataFrame,
        sentiment_data: Dict
    ) -> pd.DataFrame:
        """Add external sentiment data."""
        # Fear & Greed Index (0-100)
        if 'fear_greed_index' in sentiment_data:
            df['fear_greed_index'] = sentiment_data['fear_greed_index']

            # Sentiment categories
            df['extreme_fear'] = (df['fear_greed_index'] < 25).astype(int)
            df['fear'] = ((df['fear_greed_index'] >= 25) &
                          (df['fear_greed_index'] < 45)).astype(int)
            df['neutral'] = ((df['fear_greed_index'] >= 45) &
                            (df['fear_greed_index'] < 55)).astype(int)
            df['greed'] = ((df['fear_greed_index'] >= 55) &
                          (df['fear_greed_index'] < 75)).astype(int)
            df['extreme_greed'] = (df['fear_greed_index'] >= 75).astype(int)

            # Sentiment change
            df['sentiment_change'] = df['fear_greed_index'].diff()

            # Sentiment momentum
            df['sentiment_momentum'] = df['fear_greed_index'].diff(5)

        # Social media sentiment (if available)
        if 'social_sentiment' in sentiment_data:
            df['social_sentiment'] = sentiment_data['social_sentiment']
            df['social_sentiment_sma_5'] = df['social_sentiment'].rolling(window=5).mean()

        # News sentiment (if available)
        if 'news_sentiment' in sentiment_data:
            df['news_sentiment'] = sentiment_data['news_sentiment']
            df['news_sentiment_positive'] = (df['news_sentiment'] > 0).astype(int)

        return df

    def _add_market_sentiment_proxies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add market-based sentiment proxies.

        These are derived from price and volume action to estimate market sentiment.
        """
        # Price momentum as sentiment
        for period in [5, 10, 20]:
            df[f'price_momentum_sentiment_{period}'] = (
                df['close'].pct_change(period) * 100
            )

        # Bullish/Bearish candle ratio
        for period in [10, 20]:
            bullish_count = (df['close'] > df['open']).rolling(window=period).sum()
            df[f'bullish_ratio_{period}'] = bullish_count / period

        # Consecutive ups/downs (trend strength as sentiment)
        df['consecutive_ups'] = self._count_consecutive(df['close'] > df['close'].shift(1))
        df['consecutive_downs'] = self._count_consecutive(df['close'] < df['close'].shift(1))

        # Higher highs and higher lows (bullish structure)
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['higher_low'] = (df['low'] > df['low'].shift(1)).astype(int)
        df['lower_high'] = (df['high'] < df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)

        # Bullish/Bearish structure score
        for period in [5, 10]:
            hh_count = df['higher_high'].rolling(window=period).sum()
            hl_count = df['higher_low'].rolling(window=period).sum()
            lh_count = df['lower_high'].rolling(window=period).sum()
            ll_count = df['lower_low'].rolling(window=period).sum()

            df[f'bullish_structure_{period}'] = (hh_count + hl_count) / period
            df[f'bearish_structure_{period}'] = (lh_count + ll_count) / period

        return df

    def _add_momentum_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based sentiment indicators."""
        # Strong momentum as positive sentiment
        for period in [5, 10]:
            momentum = df['close'].diff(period)
            momentum_std = momentum.rolling(window=20).std()
            df[f'strong_momentum_{period}'] = (
                abs(momentum) > momentum_std
            ).astype(int)

        # Acceleration as sentiment shift
        df['momentum_acceleration'] = df['close'].diff(5).diff()

        # Directional sentiment from momentum
        df['momentum_bullish'] = (df['close'].diff(5) > 0).astype(int)
        df['momentum_bearish'] = (df['close'].diff(5) < 0).astype(int)

        return df

    def _add_volatility_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based sentiment.

        High volatility often indicates fear/uncertainty.
        """
        # Volatility levels
        volatility = df['close'].pct_change().rolling(window=20).std()
        volatility_mean = volatility.rolling(window=50).mean()
        volatility_std = volatility.rolling(window=50).std()

        # High volatility (fear)
        df['high_volatility_fear'] = (
            volatility > volatility_mean + volatility_std
        ).astype(int)

        # Low volatility (complacency)
        df['low_volatility_complacency'] = (
            volatility < volatility_mean - volatility_std
        ).astype(int)

        # Volatility change (sentiment shift)
        df['volatility_increasing'] = (
            volatility > volatility.shift(5)
        ).astype(int)

        return df

    def _count_consecutive(self, series: pd.Series) -> pd.Series:
        """Count consecutive True values in a boolean series."""
        # Create groups of consecutive values
        groups = (series != series.shift()).cumsum()

        # Count consecutive True values
        result = series.groupby(groups).cumsum()

        # Set False values to 0
        result[~series] = 0

        return result

    def create_synthetic_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic sentiment indicators from price/volume data.

        Useful when external sentiment data is not available.
        """
        # Composite sentiment score from multiple factors
        factors = []

        # 1. Price momentum factor
        price_momentum = df['close'].pct_change(10) * 100
        price_momentum_norm = (price_momentum - price_momentum.rolling(50).mean()) / (
            price_momentum.rolling(50).std() + 1e-8
        )
        factors.append(price_momentum_norm)

        # 2. Volume factor (high volume = strong sentiment)
        if 'volume' in df.columns:
            volume_ratio = df['volume'] / df['volume'].rolling(20).mean()
            volume_norm = (volume_ratio - 1) / (volume_ratio.rolling(50).std() + 1e-8)
            factors.append(volume_norm)

        # 3. Volatility factor (high vol = fear)
        volatility = df['close'].pct_change().rolling(20).std()
        volatility_norm = -(volatility - volatility.rolling(50).mean()) / (
            volatility.rolling(50).std() + 1e-8
        )
        factors.append(volatility_norm)

        # 4. Trend factor
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        trend_factor = ((sma_20 - sma_50) / sma_50) * 100
        trend_norm = trend_factor / (abs(trend_factor).rolling(50).mean() + 1e-8)
        factors.append(trend_norm)

        # Combine factors into composite sentiment (0-100 scale)
        if factors:
            composite = sum(factors) / len(factors)

            # Normalize to 0-100 range (50 = neutral)
            composite_clipped = composite.clip(-3, 3)  # Clip to ~99.7% of data
            df['synthetic_sentiment'] = ((composite_clipped + 3) / 6) * 100

            # Sentiment categories
            df['synthetic_extreme_fear'] = (df['synthetic_sentiment'] < 25).astype(int)
            df['synthetic_fear'] = (
                (df['synthetic_sentiment'] >= 25) &
                (df['synthetic_sentiment'] < 45)
            ).astype(int)
            df['synthetic_greed'] = (
                (df['synthetic_sentiment'] >= 55) &
                (df['synthetic_sentiment'] < 75)
            ).astype(int)
            df['synthetic_extreme_greed'] = (
                df['synthetic_sentiment'] >= 75
            ).astype(int)

        return df

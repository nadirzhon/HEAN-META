"""
Volume-based Features for ML

Implements volume-based indicators:
- Volume trends and changes
- On-Balance Volume (OBV)
- Volume-Weighted Average Price (VWAP)
- Money Flow Index (MFI)
- Volume oscillators
- And more...
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class VolumeFeatures:
    """Generator for volume-based features."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration."""
        self.config = config or {}

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all volume-based features."""
        df = self._add_volume_basics(df)
        df = self._add_obv(df)
        df = self._add_vwap(df)
        df = self._add_mfi(df)
        df = self._add_volume_oscillators(df)
        df = self._add_volume_ratios(df)

        return df

    def _add_volume_basics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic volume features."""
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_change_abs'] = df['volume'].diff()

        # Volume moving averages
        for period in [5, 10, 20, 50]:
            df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_sma_{period}'] + 1e-8)

        # Volume standard deviation
        df['volume_std_20'] = df['volume'].rolling(window=20).std()

        # Volume z-score (normalized volume)
        df['volume_zscore'] = (
            (df['volume'] - df['volume_sma_20']) /
            (df['volume_std_20'] + 1e-8)
        )

        # High/Low volume flags
        df['high_volume'] = (df['volume'] > df['volume_sma_20'] * 1.5).astype(int)
        df['low_volume'] = (df['volume'] < df['volume_sma_20'] * 0.5).astype(int)

        # Volume trend
        df['volume_trend_5'] = df['volume'].rolling(window=5).apply(
            lambda x: 1 if x[-1] > x[0] else 0
        )

        return df

    def _add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume (OBV).

        OBV is a cumulative indicator that uses volume flow to predict
        changes in stock price.
        """
        # Calculate OBV
        obv = [0]
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i - 1]:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['close'].iloc[i] < df['close'].iloc[i - 1]:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])

        df['obv'] = obv

        # OBV moving averages
        df['obv_sma_10'] = df['obv'].rolling(window=10).mean()
        df['obv_sma_20'] = df['obv'].rolling(window=20).mean()

        # OBV slope (trend)
        df['obv_slope'] = df['obv'].diff(5) / 5

        # OBV divergence from price
        price_change = df['close'].pct_change(10)
        obv_change = df['obv'].pct_change(10)
        df['obv_price_divergence'] = obv_change - price_change

        return df

    def _add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Volume-Weighted Average Price (VWAP).

        VWAP is the average price weighted by volume.
        """
        # Typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # VWAP calculation
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()

        # Rolling VWAP
        for period in [20, 50]:
            df[f'vwap_{period}'] = (
                (df['typical_price'] * df['volume']).rolling(window=period).sum() /
                df['volume'].rolling(window=period).sum()
            )

        # Distance from VWAP
        df['dist_vwap'] = ((df['close'] - df['vwap']) / df['vwap']) * 100
        df['dist_vwap_20'] = ((df['close'] - df['vwap_20']) / df['vwap_20']) * 100

        # Price above/below VWAP
        df['above_vwap'] = (df['close'] > df['vwap']).astype(int)

        return df

    def _add_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Money Flow Index (MFI).

        MFI is a momentum indicator that uses price and volume.
        """
        # Typical price
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        # Money flow
        money_flow = typical_price * df['volume']

        # Positive and negative money flow
        positive_flow = []
        negative_flow = []

        for i in range(len(df)):
            if i == 0:
                positive_flow.append(0)
                negative_flow.append(0)
            elif typical_price.iloc[i] > typical_price.iloc[i - 1]:
                positive_flow.append(money_flow.iloc[i])
                negative_flow.append(0)
            elif typical_price.iloc[i] < typical_price.iloc[i - 1]:
                positive_flow.append(0)
                negative_flow.append(money_flow.iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)

        df['positive_mf'] = positive_flow
        df['negative_mf'] = negative_flow

        # Money Flow Ratio
        positive_mf_sum = pd.Series(positive_flow).rolling(window=period).sum()
        negative_mf_sum = pd.Series(negative_flow).rolling(window=period).sum()

        mfr = positive_mf_sum / (negative_mf_sum + 1e-8)

        # Money Flow Index
        df['mfi'] = 100 - (100 / (1 + mfr))

        # MFI signals
        df['mfi_oversold'] = (df['mfi'] < 20).astype(int)
        df['mfi_overbought'] = (df['mfi'] > 80).astype(int)

        return df

    def _add_volume_oscillators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume oscillators."""
        # Volume Rate of Change
        for period in [5, 10]:
            df[f'volume_roc_{period}'] = (
                (df['volume'] - df['volume'].shift(period)) /
                (df['volume'].shift(period) + 1e-8) * 100
            )

        # Volume Momentum
        df['volume_momentum_5'] = df['volume'] - df['volume'].shift(5)
        df['volume_momentum_10'] = df['volume'] - df['volume'].shift(10)

        # Volume Acceleration
        df['volume_acceleration'] = df['volume_momentum_5'].diff()

        return df

    def _add_volume_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume ratio features."""
        # Volume to price ratio
        df['volume_price_ratio'] = df['volume'] / (df['close'] + 1e-8)

        # Volume to range ratio
        df['volume_range_ratio'] = df['volume'] / ((df['high'] - df['low']) + 1e-8)

        # Relative volume (compared to average)
        df['relative_volume'] = df['volume'] / (df['volume_sma_20'] + 1e-8)

        # Volume concentration (last N bars vs previous N bars)
        for period in [5, 10]:
            recent_vol = df['volume'].rolling(window=period).sum()
            previous_vol = df['volume'].shift(period).rolling(window=period).sum()
            df[f'volume_concentration_{period}'] = recent_vol / (previous_vol + 1e-8)

        return df

"""
Technical Indicators for ML Feature Engineering

Implements 20+ technical indicators:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Moving Averages (SMA, EMA)
- Stochastic Oscillator
- ADX (Average Directional Index)
- ATR (Average True Range)
- And more...
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class TechnicalIndicators:
    """Generator for technical analysis indicators."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration."""
        self.config = config or {}

        # Default parameters
        self.rsi_period = self.config.get('rsi_period', 14)
        self.macd_fast = self.config.get('macd_fast', 12)
        self.macd_slow = self.config.get('macd_slow', 26)
        self.macd_signal = self.config.get('macd_signal', 9)
        self.bb_period = self.config.get('bb_period', 20)
        self.bb_std = self.config.get('bb_std', 2)

    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe."""
        df = self.add_rsi(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_moving_averages(df)
        df = self.add_stochastic(df)
        df = self.add_adx(df)
        df = self.add_atr(df)
        df = self.add_cci(df)
        df = self.add_williams_r(df)
        df = self.add_ichimoku(df)

        return df

    def add_rsi(self, df: pd.DataFrame, periods: Optional[int] = None) -> pd.DataFrame:
        """
        Add Relative Strength Index.

        Args:
            df: Input dataframe
            periods: RSI period (default: 14)
        """
        periods = periods or self.rsi_period

        # Calculate price changes
        delta = df['close'].diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gains and losses
        avg_gain = gain.rolling(window=periods).mean()
        avg_loss = loss.rolling(window=periods).mean()

        # Calculate RS and RSI
        rs = avg_gain / (avg_loss + 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))

        # RSI variations
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)

        # RSI momentum
        df['rsi_change'] = df['rsi'].diff()

        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD (Moving Average Convergence Divergence)."""
        # Calculate EMAs
        ema_fast = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.macd_slow, adjust=False).mean()

        # MACD line
        df['macd'] = ema_fast - ema_slow

        # Signal line
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()

        # MACD histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # MACD cross signals
        df['macd_cross_above'] = ((df['macd'] > df['macd_signal']) &
                                   (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_below'] = ((df['macd'] < df['macd_signal']) &
                                   (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands."""
        # Middle band (SMA)
        df['bb_middle'] = df['close'].rolling(window=self.bb_period).mean()

        # Standard deviation
        std = df['close'].rolling(window=self.bb_period).std()

        # Upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (std * self.bb_std)
        df['bb_lower'] = df['bb_middle'] - (std * self.bb_std)

        # Bandwidth
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # %B (position within bands)
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)

        # Touch signals
        df['bb_touch_upper'] = (df['close'] >= df['bb_upper']).astype(int)
        df['bb_touch_lower'] = (df['close'] <= df['bb_lower']).astype(int)

        return df

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add various moving averages."""
        periods = [5, 10, 20, 50, 100, 200]

        for period in periods:
            # Simple Moving Average
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

            # Exponential Moving Average
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

            # Distance from MA
            df[f'dist_sma_{period}'] = ((df['close'] - df[f'sma_{period}']) /
                                         df[f'sma_{period}']) * 100

        # Golden/Death cross signals
        df['golden_cross'] = ((df['sma_50'] > df['sma_200']) &
                               (df['sma_50'].shift(1) <= df['sma_200'].shift(1))).astype(int)
        df['death_cross'] = ((df['sma_50'] < df['sma_200']) &
                              (df['sma_50'].shift(1) >= df['sma_200'].shift(1))).astype(int)

        return df

    def add_stochastic(self, df: pd.DataFrame, k_period: int = 14,
                      d_period: int = 3) -> pd.DataFrame:
        """Add Stochastic Oscillator."""
        # %K line
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()

        df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-8))

        # %D line (SMA of %K)
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()

        # Overbought/Oversold
        df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
        df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)

        return df

    def add_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average Directional Index (ADX)."""
        # True Range
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))

        tr = np.maximum(high_low, np.maximum(high_close, low_close))

        # Directional Movement
        up_move = df['high'] - df['high'].shift(1)
        down_move = df['low'].shift(1) - df['low']

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed TR and DM
        atr = pd.Series(tr).rolling(window=period).mean()
        plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)

        # ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        df['adx'] = dx.rolling(window=period).mean()

        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Trend strength
        df['adx_strong_trend'] = (df['adx'] > 25).astype(int)

        return df

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Average True Range (ATR)."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))

        df['atr'] = pd.Series(true_range).rolling(window=period).mean()
        df['atr_pct'] = (df['atr'] / df['close']) * 100

        return df

    def add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Commodity Channel Index (CCI)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3

        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())

        df['cci'] = (typical_price - sma_tp) / (0.015 * mad + 1e-8)

        return df

    def add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Add Williams %R."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()

        df['williams_r'] = -100 * ((high_max - df['close']) / (high_max - low_min + 1e-8))

        return df

    def add_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku Cloud indicators."""
        # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['ichimoku_tenkan'] = (high_9 + low_9) / 2

        # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['ichimoku_kijun'] = (high_26 + low_26) / 2

        # Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)

        # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)

        # Cloud signals
        df['ichimoku_above_cloud'] = (
            (df['close'] > df['ichimoku_senkou_a']) &
            (df['close'] > df['ichimoku_senkou_b'])
        ).astype(int)

        return df

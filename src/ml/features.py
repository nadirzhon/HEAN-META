"""
Feature Engineering Pipeline for Bitcoin Price Prediction.

Generates 50+ features including:
- Technical Indicators (RSI, MACD, Bollinger Bands, etc.)
- Volume-based features
- Orderbook imbalance features
- Price action features
- Volatility features
- Momentum features
- Sentiment indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    # RSI periods
    rsi_periods: List[int] = None
    # MACD parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    # Bollinger Bands
    bb_period: int = 20
    bb_std: float = 2.0
    # Moving averages
    ma_periods: List[int] = None
    # EMA periods
    ema_periods: List[int] = None
    # Volume periods
    volume_periods: List[int] = None
    # ATR period
    atr_period: int = 14
    # Stochastic
    stoch_k: int = 14
    stoch_d: int = 3

    def __post_init__(self):
        if self.rsi_periods is None:
            self.rsi_periods = [7, 14, 21]
        if self.ma_periods is None:
            self.ma_periods = [5, 10, 20, 50, 100, 200]
        if self.ema_periods is None:
            self.ema_periods = [9, 12, 26, 50]
        if self.volume_periods is None:
            self.volume_periods = [5, 10, 20]


class FeatureEngineering:
    """
    Feature engineering pipeline for Bitcoin price prediction.

    Generates 50+ features from OHLCV data and orderbook.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engineering pipeline.

        Args:
            config: Feature configuration (uses defaults if None)
        """
        self.config = config or FeatureConfig()
        self.feature_names: List[str] = []

    def generate_features(
        self,
        df: pd.DataFrame,
        orderbook_data: Optional[pd.DataFrame] = None,
        sentiment_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Generate all features from input data.

        Args:
            df: OHLCV dataframe with columns: timestamp, open, high, low, close, volume
            orderbook_data: Optional orderbook data with bid/ask levels
            sentiment_data: Optional sentiment scores

        Returns:
            DataFrame with all engineered features
        """
        features = df.copy()

        # Price action features
        features = self._add_price_action_features(features)

        # Technical indicators
        features = self._add_technical_indicators(features)

        # Volume features
        features = self._add_volume_features(features)

        # Volatility features
        features = self._add_volatility_features(features)

        # Momentum features
        features = self._add_momentum_features(features)

        # Orderbook features (if available)
        if orderbook_data is not None:
            features = self._add_orderbook_features(features, orderbook_data)

        # Sentiment features (if available)
        if sentiment_data is not None:
            features = self._add_sentiment_features(features, sentiment_data)

        # Time-based features
        features = self._add_time_features(features)

        # Remove NaN values
        features = features.dropna()

        # Store feature names (excluding original columns)
        self.feature_names = [col for col in features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        return features

    def _add_price_action_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price action features (returns, ranges, etc.)."""
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_1h'] = df['close'].pct_change(12)  # 12 * 5min = 1h
        df['returns_4h'] = df['close'].pct_change(48)
        df['returns_1d'] = df['close'].pct_change(288)  # 288 * 5min = 1d

        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['oc_range'] = (df['open'] - df['close']) / df['close']

        # Price position within candle
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)

        # Gap from previous close
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        # Log returns
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators (RSI, MACD, Bollinger Bands, etc.)."""
        # RSI for multiple periods
        for period in self.config.rsi_periods:
            df[f'rsi_{period}'] = self._calculate_rsi(df['close'], period)

        # MACD
        macd, signal, hist = self._calculate_macd(
            df['close'],
            self.config.macd_fast,
            self.config.macd_slow,
            self.config.macd_signal
        )
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(
            df['close'],
            self.config.bb_period,
            self.config.bb_std
        )
        df['bb_upper'] = bb_upper
        df['bb_middle'] = bb_middle
        df['bb_lower'] = bb_lower
        df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)

        # Moving Averages
        for period in self.config.ma_periods:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ma_{period}_dist'] = (df['close'] - df[f'ma_{period}']) / df[f'ma_{period}']

        # EMAs
        for period in self.config.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema_{period}_dist'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']

        # ATR (Average True Range)
        df['atr'] = self._calculate_atr(df, self.config.atr_period)
        df['atr_pct'] = df['atr'] / df['close']

        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = self._calculate_stochastic(
            df,
            self.config.stoch_k,
            self.config.stoch_d
        )

        # Commodity Channel Index (CCI)
        df['cci'] = self._calculate_cci(df, 20)

        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        # Volume change
        df['volume_change'] = df['volume'].pct_change()

        # Volume moving averages
        for period in self.config.volume_periods:
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / (df[f'volume_ma_{period}'] + 1e-10)

        # Volume-weighted average price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_dist'] = (df['close'] - df['vwap']) / df['vwap']

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20, adjust=False).mean()

        # Volume Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['returns']).cumsum()

        # Money Flow Index (MFI)
        df['mfi'] = self._calculate_mfi(df, 14)

        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility features."""
        # Historical volatility (rolling std of returns)
        for period in [5, 10, 20, 50]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()

        # Parkinson's volatility (using high-low)
        df['parkinson_vol'] = np.sqrt(
            (1 / (4 * np.log(2))) * np.log(df['high'] / df['low']) ** 2
        ).rolling(window=20).mean()

        # Garman-Klass volatility
        df['gk_vol'] = np.sqrt(
            0.5 * np.log(df['high'] / df['low']) ** 2 -
            (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
        ).rolling(window=20).mean()

        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features."""
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) /
                                   df['close'].shift(period))

        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df, 14)

        # Momentum
        df['momentum'] = df['close'] - df['close'].shift(10)

        # Ultimate Oscillator
        df['ultimate_osc'] = self._calculate_ultimate_oscillator(df)

        return df

    def _add_orderbook_features(
        self,
        df: pd.DataFrame,
        orderbook_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add orderbook imbalance features."""
        # Merge orderbook data
        df = df.merge(orderbook_data, on='timestamp', how='left')

        # Bid-ask spread
        df['bid_ask_spread'] = (df['ask_price'] - df['bid_price']) / df['bid_price']

        # Order imbalance
        df['order_imbalance'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-10)

        # Weighted mid price
        df['weighted_mid'] = (df['bid_price'] * df['ask_volume'] + df['ask_price'] * df['bid_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-10)

        # Orderbook depth ratio
        df['depth_ratio'] = df['bid_volume'] / (df['ask_volume'] + 1e-10)

        return df

    def _add_sentiment_features(
        self,
        df: pd.DataFrame,
        sentiment_data: pd.DataFrame
    ) -> pd.DataFrame:
        """Add sentiment features."""
        # Merge sentiment data
        df = df.merge(sentiment_data, on='timestamp', how='left')

        # Sentiment score features already in sentiment_data
        # (e.g., 'sentiment_score', 'sentiment_volume', 'fear_greed_index')

        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        if 'timestamp' in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
            df['day_of_month'] = pd.to_datetime(df['timestamp']).dt.day
            df['month'] = pd.to_datetime(df['timestamp']).dt.month

            # Cyclical encoding for time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        return df

    # Technical indicator calculation methods

    @staticmethod
    def _calculate_rsi(series: pd.Series, period: int) -> pd.Series:
        """Calculate RSI (Relative Strength Index)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _calculate_macd(
        series: pd.Series,
        fast: int,
        slow: int,
        signal: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram

    @staticmethod
    def _calculate_bollinger_bands(
        series: pd.Series,
        period: int,
        std_dev: float
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate ATR (Average True Range)."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr

    @staticmethod
    def _calculate_stochastic(
        df: pd.DataFrame,
        k_period: int,
        d_period: int
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        k = 100 * (df['close'] - low_min) / (high_max - low_min + 1e-10)
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def _calculate_cci(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate CCI (Commodity Channel Index)."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma) / (0.015 * mad + 1e-10)
        return cci

    @staticmethod
    def _calculate_mfi(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate MFI (Money Flow Index)."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']
        mf_sign = np.sign(tp.diff())

        positive_mf = mf.where(mf_sign > 0, 0).rolling(window=period).sum()
        negative_mf = mf.where(mf_sign < 0, 0).abs().rolling(window=period).sum()

        mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-10)))
        return mfi

    @staticmethod
    def _calculate_williams_r(df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Williams %R."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        wr = -100 * (high_max - df['close']) / (high_max - low_min + 1e-10)
        return wr

    @staticmethod
    def _calculate_ultimate_oscillator(df: pd.DataFrame) -> pd.Series:
        """Calculate Ultimate Oscillator."""
        bp = df['close'] - pd.concat([df['low'], df['close'].shift()], axis=1).min(axis=1)
        tr = pd.concat([
            df['high'] - df['low'],
            np.abs(df['high'] - df['close'].shift()),
            np.abs(df['low'] - df['close'].shift())
        ], axis=1).max(axis=1)

        avg7 = bp.rolling(7).sum() / (tr.rolling(7).sum() + 1e-10)
        avg14 = bp.rolling(14).sum() / (tr.rolling(14).sum() + 1e-10)
        avg28 = bp.rolling(28).sum() / (tr.rolling(28).sum() + 1e-10)

        uo = 100 * ((4 * avg7 + 2 * avg14 + avg28) / 7)
        return uo

    def get_feature_names(self) -> List[str]:
        """Get list of generated feature names."""
        return self.feature_names

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category for importance analysis."""
        groups = {
            'price_action': [],
            'technical': [],
            'volume': [],
            'volatility': [],
            'momentum': [],
            'orderbook': [],
            'sentiment': [],
            'time': [],
        }

        for feature in self.feature_names:
            if any(x in feature for x in ['returns', 'range', 'gap', 'position']):
                groups['price_action'].append(feature)
            elif any(x in feature for x in ['rsi', 'macd', 'bb_', 'ma_', 'ema_', 'atr', 'stoch', 'cci']):
                groups['technical'].append(feature)
            elif any(x in feature for x in ['volume', 'vwap', 'obv', 'vpt', 'mfi']):
                groups['volume'].append(feature)
            elif any(x in feature for x in ['volatility', 'vol']):
                groups['volatility'].append(feature)
            elif any(x in feature for x in ['roc', 'williams', 'momentum', 'ultimate']):
                groups['momentum'].append(feature)
            elif any(x in feature for x in ['bid', 'ask', 'order', 'depth']):
                groups['orderbook'].append(feature)
            elif any(x in feature for x in ['sentiment', 'fear', 'greed']):
                groups['sentiment'].append(feature)
            elif any(x in feature for x in ['hour', 'day', 'month']):
                groups['time'].append(feature)

        return groups

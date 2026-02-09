"""
Feature engineering for ML price prediction

Calculates technical indicators and prepares features for model
"""

import logging

import numpy as np
import pandas as pd

from .models import FeatureSet

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Feature engineering for price prediction

    Calculates technical indicators and prepares data for LSTM model

    Features include:
    - Raw OHLCV data
    - Technical indicators (RSI, MACD, Bollinger Bands, etc.)
    - Sentiment scores (optional)
    - Google Trends data (optional)
    - Funding rates (optional)
    """

    def __init__(
        self,
        use_technical_indicators: bool = True,
        use_sentiment: bool = True,
        use_google_trends: bool = True,
        use_funding_rates: bool = True
    ):
        """
        Initialize feature engineering

        Args:
            use_technical_indicators: Calculate technical indicators
            use_sentiment: Include sentiment scores
            use_google_trends: Include Google Trends data
            use_funding_rates: Include funding rates
        """
        self.use_technical_indicators = use_technical_indicators
        self.use_sentiment = use_sentiment
        self.use_google_trends = use_google_trends
        self.use_funding_rates = use_funding_rates

    def create_features(
        self,
        ohlcv_data: pd.DataFrame,
        sentiment_data: pd.DataFrame | None = None,
        trends_data: pd.DataFrame | None = None,
        funding_data: pd.DataFrame | None = None
    ) -> FeatureSet:
        """
        Create feature set from raw data

        Args:
            ohlcv_data: DataFrame with columns: timestamp, open, high, low, close, volume
            sentiment_data: Optional DataFrame with: timestamp, sentiment_score
            trends_data: Optional DataFrame with: timestamp, interest_score
            funding_data: Optional DataFrame with: timestamp, funding_rate

        Returns:
            FeatureSet
        """
        # Ensure data is sorted by timestamp
        ohlcv_data = ohlcv_data.sort_values('timestamp').reset_index(drop=True)

        # Extract raw OHLCV
        open_prices = ohlcv_data['open'].tolist()
        high_prices = ohlcv_data['high'].tolist()
        low_prices = ohlcv_data['low'].tolist()
        close_prices = ohlcv_data['close'].tolist()
        volumes = ohlcv_data['volume'].tolist()
        timestamps = pd.to_datetime(ohlcv_data['timestamp']).tolist()

        # Calculate technical indicators
        tech_indicators = {}
        if self.use_technical_indicators:
            tech_indicators = self._calculate_technical_indicators(ohlcv_data)

        # Merge external data
        sentiment_scores = None
        if self.use_sentiment and sentiment_data is not None:
            sentiment_scores = self._merge_external_data(
                ohlcv_data, sentiment_data, 'sentiment_score'
            )

        google_trends = None
        if self.use_google_trends and trends_data is not None:
            google_trends = self._merge_external_data(
                ohlcv_data, trends_data, 'interest_score'
            )

        funding_rates = None
        if self.use_funding_rates and funding_data is not None:
            funding_rates = self._merge_external_data(
                ohlcv_data, funding_data, 'funding_rate'
            )

        return FeatureSet(
            open_prices=open_prices,
            high_prices=high_prices,
            low_prices=low_prices,
            close_prices=close_prices,
            volumes=volumes,
            timestamps=timestamps,
            **tech_indicators,
            sentiment_scores=sentiment_scores,
            google_trends=google_trends,
            funding_rates=funding_rates
        )

    def _calculate_technical_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate all technical indicators"""
        indicators = {}

        # RSI (Relative Strength Index)
        indicators['rsi'] = self._calculate_rsi(df['close']).tolist()

        # MACD
        macd, signal = self._calculate_macd(df['close'])
        indicators['macd'] = macd.tolist()
        indicators['macd_signal'] = signal.tolist()

        # Bollinger Bands
        upper, lower = self._calculate_bollinger_bands(df['close'])
        indicators['bollinger_upper'] = upper.tolist()
        indicators['bollinger_lower'] = lower.tolist()

        # Moving Averages
        indicators['sma_20'] = df['close'].rolling(window=20).mean().tolist()
        indicators['sma_50'] = df['close'].rolling(window=50).mean().tolist()
        indicators['ema_12'] = df['close'].ewm(span=12).mean().tolist()
        indicators['ema_26'] = df['close'].ewm(span=26).mean().tolist()

        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr(df).tolist()

        # OBV (On-Balance Volume)
        indicators['obv'] = self._calculate_obv(df).tolist()

        return indicators

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # Fill NaN with neutral value

    def _calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate MACD and signal line"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()

        return macd.fillna(0), signal_line.fillna(0)

    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)

        return upper.fillna(method='bfill'), lower.fillna(method='bfill')

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.fillna(method='bfill')

    def _calculate_obv(self, df: pd.DataFrame) -> pd.Series:
        """Calculate On-Balance Volume"""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv

    def _merge_external_data(
        self,
        ohlcv_data: pd.DataFrame,
        external_data: pd.DataFrame,
        value_column: str
    ) -> list[float]:
        """
        Merge external data (sentiment, trends, funding) with OHLCV data

        Args:
            ohlcv_data: Main DataFrame with timestamps
            external_data: External DataFrame with timestamps and values
            value_column: Column name with values to merge

        Returns:
            List of values aligned with OHLCV timestamps
        """
        # Merge on timestamp (forward fill for missing values)
        merged = pd.merge_asof(
            ohlcv_data[['timestamp']].sort_values('timestamp'),
            external_data[['timestamp', value_column]].sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )

        # Fill remaining NaN with 0 or forward fill
        values = merged[value_column].fillna(method='ffill').fillna(0).tolist()

        return values

    def normalize_features(self, feature_set: FeatureSet) -> FeatureSet:
        """
        Normalize features to 0-1 range

        Args:
            feature_set: Raw features

        Returns:
            Normalized features
        """
        # Convert to numpy array
        data = feature_set.to_numpy_array()

        # Normalize each feature (column) to 0-1
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        _normalized = scaler.fit_transform(data)  # noqa: F841

        # NOTE: Feature normalization bypassed - proper field mapping not yet implemented
        # The normalized array cannot be safely mapped back to FeatureSet fields without
        # explicit field-to-column mapping. Using raw features until mapping is implemented.
        logger.warning(
            "[ML_FEATURES] Feature normalization bypassed: "
            "proper field mapping not yet implemented. Using raw features."
        )
        return feature_set

    def create_sequences(
        self,
        feature_set: FeatureSet,
        lookback_periods: int = 60,
        prediction_horizons: list[int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training

        Args:
            feature_set: Feature set
            lookback_periods: Number of periods to look back
            prediction_horizons: Hours ahead to predict

        Returns:
            (X, y) where:
            - X: shape (n_samples, lookback_periods, n_features)
            - y: shape (n_samples, len(prediction_horizons))
        """
        if prediction_horizons is None:
            prediction_horizons = [1, 4, 24]
        data = feature_set.to_numpy_array()
        prices = np.array(feature_set.close_prices)

        X = []
        y = []

        # Create sequences
        for i in range(lookback_periods, len(data) - max(prediction_horizons)):
            # Input sequence
            X.append(data[i - lookback_periods:i])

            # Target: future prices (normalized returns)
            current_price = prices[i]
            future_prices = [
                prices[i + h] if i + h < len(prices) else current_price
                for h in prediction_horizons
            ]

            # Calculate returns (%)
            returns = [
                ((fp - current_price) / current_price) * 100
                for fp in future_prices
            ]

            y.append(returns)

        return np.array(X), np.array(y)


# Example usage
async def main():
    """Example usage - demo script with synthetic data"""
    # Sample OHLCV data (synthetic for demonstration)
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=1000, freq='H'),
        'open': np.random.randn(1000).cumsum() + 50000,
        'high': np.random.randn(1000).cumsum() + 50100,
        'low': np.random.randn(1000).cumsum() + 49900,
        'close': np.random.randn(1000).cumsum() + 50000,
        'volume': np.random.randint(100, 1000, 1000)  # Demo data: random volume
    }
    df = pd.DataFrame(data)

    # Create features
    fe = FeatureEngineering(
        use_technical_indicators=True,
        use_sentiment=False,  # No sentiment data in this example
        use_google_trends=False,
        use_funding_rates=False
    )

    features = fe.create_features(df)

    print("\nFeature Set Created:")
    print(f"  Number of features: {features.n_features}")
    print(f"  Number of timesteps: {len(features.close_prices)}")
    print("  Features: OHLCV + RSI, MACD, Bollinger, SMA, EMA, ATR, OBV")

    # Create sequences
    X, y = fe.create_sequences(features, lookback_periods=60, prediction_horizons=[1, 4, 24])

    print("\nSequences Created:")
    print(f"  X shape: {X.shape}")  # (samples, lookback, features)
    print(f"  y shape: {y.shape}")  # (samples, horizons)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

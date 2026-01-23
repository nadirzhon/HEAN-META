"""
Data loader for Bitcoin historical data.

Supports loading from:
- CSV files
- Exchange APIs (Bybit, Binance, etc.)
- Database
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class DataLoader:
    """Load and prepare Bitcoin historical data for training."""

    def __init__(self, symbol: str = "BTCUSDT"):
        """
        Initialize data loader.

        Args:
            symbol: Trading symbol (default: BTCUSDT)
        """
        self.symbol = symbol

    def load_from_csv(
        self,
        file_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load OHLCV data from CSV file.

        Args:
            file_path: Path to CSV file
            start_date: Start date filter (format: YYYY-MM-DD)
            end_date: End date filter (format: YYYY-MM-DD)

        Returns:
            OHLCV dataframe
        """
        df = pd.read_csv(file_path)

        # Ensure required columns exist
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Filter by date range
        if start_date:
            df = df[df['timestamp'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['timestamp'] <= pd.to_datetime(end_date)]

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df

    def load_from_exchange(
        self,
        exchange: str = "bybit",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = "5m",
    ) -> pd.DataFrame:
        """
        Load data from exchange API.

        Args:
            exchange: Exchange name (bybit, binance)
            start_date: Start date (format: YYYY-MM-DD)
            end_date: End date (format: YYYY-MM-DD)
            interval: Candle interval (1m, 5m, 15m, 1h, etc.)

        Returns:
            OHLCV dataframe
        """
        if exchange == "bybit":
            return self._load_from_bybit(start_date, end_date, interval)
        elif exchange == "binance":
            return self._load_from_binance(start_date, end_date, interval)
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")

    def _load_from_bybit(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str,
    ) -> pd.DataFrame:
        """Load data from Bybit API."""
        try:
            from pybit.unified_trading import HTTP
        except ImportError:
            raise ImportError("pybit is required for Bybit data loading. Install with: pip install pybit")

        session = HTTP(testnet=False)

        # Convert dates to timestamps
        if start_date:
            start_ts = int(pd.to_datetime(start_date).timestamp() * 1000)
        else:
            start_ts = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

        if end_date:
            end_ts = int(pd.to_datetime(end_date).timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        # Fetch data
        all_data = []
        current_ts = start_ts

        while current_ts < end_ts:
            response = session.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=interval,
                start=current_ts,
                limit=1000
            )

            if response['retCode'] != 0:
                raise ValueError(f"Bybit API error: {response['retMsg']}")

            data = response['result']['list']
            if not data:
                break

            all_data.extend(data)

            # Update timestamp for next batch
            current_ts = int(data[-1][0]) + 1

        # Convert to DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        )

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def _load_from_binance(
        self,
        start_date: Optional[str],
        end_date: Optional[str],
        interval: str,
    ) -> pd.DataFrame:
        """Load data from Binance API."""
        try:
            from binance.client import Client
        except ImportError:
            raise ImportError("python-binance is required. Install with: pip install python-binance")

        client = Client()

        # Convert interval format
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY,
        }

        if interval not in interval_map:
            raise ValueError(f"Unsupported interval: {interval}")

        # Fetch data
        klines = client.get_historical_klines(
            self.symbol,
            interval_map[interval],
            start_str=start_date or "30 days ago UTC",
            end_str=end_date or "now UTC",
        )

        # Convert to DataFrame
        df = pd.DataFrame(
            klines,
            columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ]
        )

        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    def generate_synthetic_data(
        self,
        n_samples: int = 10000,
        start_price: float = 50000.0,
        volatility: float = 0.02,
    ) -> pd.DataFrame:
        """
        Generate synthetic Bitcoin price data for testing.

        Args:
            n_samples: Number of samples to generate
            start_price: Starting price
            volatility: Price volatility (std dev of returns)

        Returns:
            OHLCV dataframe
        """
        np.random.seed(42)

        # Generate timestamps (5-minute intervals)
        start_time = datetime.now() - timedelta(minutes=n_samples * 5)
        timestamps = [start_time + timedelta(minutes=i * 5) for i in range(n_samples)]

        # Generate prices using geometric Brownian motion
        returns = np.random.normal(0, volatility, n_samples)
        prices = start_price * np.exp(np.cumsum(returns))

        # Generate OHLC from prices
        data = []
        for i, (ts, close) in enumerate(zip(timestamps, prices)):
            # Add some intrabar volatility
            intrabar_range = abs(np.random.normal(0, volatility * 0.5))
            high = close * (1 + intrabar_range)
            low = close * (1 - intrabar_range)
            open_price = (high + low) / 2 + np.random.normal(0, (high - low) * 0.1)

            # Volume (correlated with price movement)
            volume = abs(np.random.normal(100, 50)) * (1 + abs(returns[i]) * 10)

            data.append({
                'timestamp': ts,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume,
            })

        df = pd.DataFrame(data)
        return df

    def split_data(
        self,
        df: pd.DataFrame,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Uses chronological split (no shuffling to preserve time series order).

        Args:
            df: Input dataframe
            train_size: Training set size (0-1)
            val_size: Validation set size (0-1)
            test_size: Test set size (0-1)

        Returns:
            (train_df, val_df, test_df) tuple
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
            "train_size + val_size + test_size must equal 1.0"

        n = len(df)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))

        train_df = df.iloc[:train_end].reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].reset_index(drop=True)
        test_df = df.iloc[val_end:].reset_index(drop=True)

        return train_df, val_df, test_df

    def load_orderbook_data(
        self,
        file_path: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load orderbook data.

        Args:
            file_path: Path to orderbook CSV file

        Returns:
            Orderbook dataframe or None if not available
        """
        if file_path is None or not Path(file_path).exists():
            return None

        df = pd.read_csv(file_path)

        # Expected columns: timestamp, bid_price, bid_volume, ask_price, ask_volume
        required_cols = ['timestamp', 'bid_price', 'bid_volume', 'ask_price', 'ask_volume']
        if not all(col in df.columns for col in required_cols):
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def load_sentiment_data(
        self,
        file_path: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Load sentiment data.

        Args:
            file_path: Path to sentiment CSV file

        Returns:
            Sentiment dataframe or None if not available
        """
        if file_path is None or not Path(file_path).exists():
            return None

        df = pd.read_csv(file_path)

        # Expected columns: timestamp, sentiment_score
        if 'timestamp' not in df.columns or 'sentiment_score' not in df.columns:
            return None

        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df

    def prepare_training_data(
        self,
        ohlcv_path: str,
        orderbook_path: Optional[str] = None,
        sentiment_path: Optional[str] = None,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
    ) -> Tuple[
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
        Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    ]:
        """
        Load and prepare all data for training.

        Args:
            ohlcv_path: Path to OHLCV data
            orderbook_path: Optional path to orderbook data
            sentiment_path: Optional path to sentiment data
            train_size: Training set proportion
            val_size: Validation set proportion
            test_size: Test set proportion

        Returns:
            ((train_ohlcv, val_ohlcv, test_ohlcv),
             (train_orderbook, val_orderbook, test_orderbook),
             (train_sentiment, val_sentiment, test_sentiment))
        """
        # Load OHLCV
        ohlcv_df = self.load_from_csv(ohlcv_path)
        train_ohlcv, val_ohlcv, test_ohlcv = self.split_data(
            ohlcv_df, train_size, val_size, test_size
        )

        # Load orderbook (optional)
        orderbook_splits = None
        if orderbook_path:
            orderbook_df = self.load_orderbook_data(orderbook_path)
            if orderbook_df is not None:
                orderbook_splits = self.split_data(
                    orderbook_df, train_size, val_size, test_size
                )

        # Load sentiment (optional)
        sentiment_splits = None
        if sentiment_path:
            sentiment_df = self.load_sentiment_data(sentiment_path)
            if sentiment_df is not None:
                sentiment_splits = self.split_data(
                    sentiment_df, train_size, val_size, test_size
                )

        return (
            (train_ohlcv, val_ohlcv, test_ohlcv),
            orderbook_splits,
            sentiment_splits,
        )

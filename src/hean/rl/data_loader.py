"""
Data loader for historical Bitcoin OHLCV data.

Supports loading from multiple sources:
- CSV files
- Binance API
- Bybit API
- Synthetic data generation for testing
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and prepare historical trading data for RL training."""

    @staticmethod
    def load_csv(
        file_path: str | Path,
        date_column: str = 'timestamp',
        columns: Optional[list[str]] = None,
    ) -> np.ndarray:
        """
        Load OHLCV data from CSV file.

        Args:
            file_path: Path to CSV file
            date_column: Name of timestamp column
            columns: Column names for [open, high, low, close, volume]

        Returns:
            NumPy array of shape [N x 5] with OHLCV data
        """
        df = pd.read_csv(file_path)

        if columns is None:
            # Try to auto-detect columns
            columns = ['open', 'high', 'low', 'close', 'volume']

        # Convert to lowercase for matching
        df.columns = df.columns.str.lower()

        # Extract OHLCV columns
        try:
            data = df[columns].values
        except KeyError as e:
            available_cols = df.columns.tolist()
            raise ValueError(
                f"Could not find required columns {columns}. "
                f"Available columns: {available_cols}"
            ) from e

        logger.info(f"Loaded {len(data)} candles from {file_path}")
        return data.astype(np.float32)

    @staticmethod
    def load_binance(
        symbol: str = 'BTCUSDT',
        interval: str = '1h',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 1000,
    ) -> np.ndarray:
        """
        Load data from Binance API.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Candle interval (e.g., '1h', '4h', '1d')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            limit: Maximum number of candles

        Returns:
            NumPy array of shape [N x 5] with OHLCV data
        """
        try:
            from binance.client import Client
        except ImportError:
            raise ImportError(
                "Binance client not installed. "
                "Install with: pip install python-binance"
            )

        client = Client()

        # Fetch klines
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_str=start_date,
            end_str=end_date,
            limit=limit,
        )

        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])

        # Extract OHLCV
        data = df[['open', 'high', 'low', 'close', 'volume']].astype(float).values

        logger.info(f"Loaded {len(data)} candles from Binance for {symbol}")
        return data.astype(np.float32)

    @staticmethod
    def load_bybit(
        symbol: str = 'BTCUSDT',
        interval: str = '60',  # Minutes
        start_time: Optional[int] = None,
        limit: int = 200,
    ) -> np.ndarray:
        """
        Load data from Bybit API.

        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Candle interval in minutes
            start_time: Start timestamp in milliseconds
            limit: Maximum number of candles

        Returns:
            NumPy array of shape [N x 5] with OHLCV data
        """
        try:
            import aiohttp
            import asyncio
        except ImportError:
            raise ImportError("aiohttp not installed. Install with: pip install aiohttp")

        async def fetch_klines():
            url = "https://api.bybit.com/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': interval,
                'limit': limit,
            }
            if start_time:
                params['start'] = start_time

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    data = await response.json()
                    return data

        # Run async function
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(fetch_klines())

        if result.get('retCode') != 0:
            raise ValueError(f"Bybit API error: {result.get('retMsg')}")

        klines = result['result']['list']

        # Parse klines: [timestamp, open, high, low, close, volume, turnover]
        data = []
        for kline in reversed(klines):  # Bybit returns newest first
            data.append([
                float(kline[1]),  # open
                float(kline[2]),  # high
                float(kline[3]),  # low
                float(kline[4]),  # close
                float(kline[5]),  # volume
            ])

        data = np.array(data, dtype=np.float32)
        logger.info(f"Loaded {len(data)} candles from Bybit for {symbol}")
        return data

    @staticmethod
    def generate_synthetic(
        n_candles: int = 10000,
        initial_price: float = 30000.0,
        trend: float = 0.0001,
        volatility: float = 0.02,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate synthetic OHLCV data for testing.

        Uses geometric Brownian motion with configurable drift and volatility.

        Args:
            n_candles: Number of candles to generate
            initial_price: Starting price
            trend: Drift term (daily return)
            volatility: Volatility (daily std)
            seed: Random seed for reproducibility

        Returns:
            NumPy array of shape [N x 5] with synthetic OHLCV data
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate price series using GBM
        dt = 1.0  # 1 time unit per candle
        prices = [initial_price]

        for _ in range(n_candles):
            drift = trend * dt
            shock = volatility * np.sqrt(dt) * np.random.randn()
            price = prices[-1] * np.exp(drift + shock)
            prices.append(price)

        prices = np.array(prices)

        # Generate OHLCV data
        data = []
        for i in range(n_candles):
            open_price = prices[i]
            close_price = prices[i + 1]

            # Generate high/low with some randomness
            high_price = max(open_price, close_price) * (1 + abs(np.random.randn()) * 0.01)
            low_price = min(open_price, close_price) * (1 - abs(np.random.randn()) * 0.01)

            # Generate random volume
            volume = np.random.lognormal(mean=5, sigma=1)

            data.append([open_price, high_price, low_price, close_price, volume])

        data = np.array(data, dtype=np.float32)
        logger.info(f"Generated {len(data)} synthetic candles")
        return data

    @staticmethod
    def split_data(
        data: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets.

        Args:
            data: Full dataset
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        n = len(data)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_data = data[:train_end]
        val_data = data[train_end:val_end]
        test_data = data[val_end:]

        logger.info(
            f"Split data: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )

        return train_data, val_data, test_data

    @staticmethod
    def normalize_data(
        data: np.ndarray,
        method: str = 'none',
    ) -> Tuple[np.ndarray, dict]:
        """
        Normalize OHLCV data.

        Args:
            data: Raw OHLCV data
            method: Normalization method ('none', 'minmax', 'standard')

        Returns:
            Tuple of (normalized_data, normalization_params)
        """
        if method == 'none':
            return data, {}

        normalized = data.copy()
        params = {}

        if method == 'minmax':
            # Min-max normalization per column
            for i in range(data.shape[1]):
                col_min = data[:, i].min()
                col_max = data[:, i].max()
                normalized[:, i] = (data[:, i] - col_min) / (col_max - col_min + 1e-8)
                params[f'col_{i}_min'] = col_min
                params[f'col_{i}_max'] = col_max

        elif method == 'standard':
            # Standardization per column
            for i in range(data.shape[1]):
                col_mean = data[:, i].mean()
                col_std = data[:, i].std()
                normalized[:, i] = (data[:, i] - col_mean) / (col_std + 1e-8)
                params[f'col_{i}_mean'] = col_mean
                params[f'col_{i}_std'] = col_std

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        logger.info(f"Normalized data using method: {method}")
        return normalized, params


def load_sample_data(source: str = 'synthetic', **kwargs) -> np.ndarray:
    """
    Convenient function to load sample data.

    Args:
        source: Data source ('synthetic', 'csv', 'binance', 'bybit')
        **kwargs: Additional arguments for specific loader

    Returns:
        NumPy array of OHLCV data

    Examples:
        >>> data = load_sample_data('synthetic', n_candles=10000)
        >>> data = load_sample_data('csv', file_path='btc_data.csv')
        >>> data = load_sample_data('binance', symbol='BTCUSDT', limit=5000)
    """
    loader = DataLoader()

    if source == 'synthetic':
        return loader.generate_synthetic(**kwargs)
    elif source == 'csv':
        return loader.load_csv(**kwargs)
    elif source == 'binance':
        return loader.load_binance(**kwargs)
    elif source == 'bybit':
        return loader.load_bybit(**kwargs)
    else:
        raise ValueError(f"Unknown data source: {source}")

"""
Data loader for ML price predictor - fetches real market data
"""

import asyncio
import logging
from datetime import datetime, timedelta

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from pybit.unified_trading import HTTP
    PYBIT_AVAILABLE = True
except ImportError:
    PYBIT_AVAILABLE = False
    logger.warning(
        "pybit not installed. Real data loading will not work. "
        "Install with: pip install pybit --break-system-packages"
    )


class MarketDataLoader:
    """
    Loader for real market data from Bybit

    Handles:
    - OHLCV data (klines)
    - Funding rates
    - Market metadata
    """

    def __init__(
        self,
        testnet: bool = False,
        api_key: str | None = None,
        api_secret: str | None = None
    ):
        """
        Initialize data loader

        Args:
            testnet: Use testnet instead of mainnet
            api_key: API key (optional, for authenticated endpoints)
            api_secret: API secret (optional)
        """
        self.testnet = testnet
        self.client = None

        if PYBIT_AVAILABLE:
            self.client = HTTP(
                testnet=testnet,
                api_key=api_key,
                api_secret=api_secret
            )
        else:
            logger.error("pybit not available - cannot fetch real data")

    async def load_ohlcv(
        self,
        symbol: str,
        interval: str = "60",  # 1 hour
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Load OHLCV data from Bybit

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval in minutes (1, 5, 15, 60, 240, D)
            start_date: Start date
            end_date: End date
            limit: Max number of candles per request (max 1000)

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        if not self.client:
            logger.error("Client not initialized - returning empty DataFrame")
            return pd.DataFrame()

        # Default dates
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=90)

        logger.info(f"Loading OHLCV data for {symbol} from {start_date} to {end_date}")

        try:
            # Convert to timestamps (milliseconds)
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            all_data = []
            current_start = start_ts

            # Fetch data in chunks (Bybit limits to 1000 candles per request)
            while current_start < end_ts:
                # Run synchronous pybit call in executor to avoid blocking
                # Capture loop variables to avoid late binding issue
                _current_start = current_start
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda cs=_current_start: self.client.get_kline(
                        category="linear",
                        symbol=symbol,
                        interval=interval,
                        start=cs,
                        end=end_ts,
                        limit=limit
                    )
                )

                if result['retCode'] != 0:
                    logger.error(f"Error fetching klines: {result['retMsg']}")
                    break

                klines = result['result']['list']
                if not klines:
                    break

                all_data.extend(klines)

                # Update start time for next batch
                # Bybit returns descending order, so last item is oldest
                last_timestamp = int(klines[-1][0])
                current_start = last_timestamp + 1

                # Avoid hitting rate limits
                await asyncio.sleep(0.1)

                logger.debug(f"Fetched {len(klines)} candles, total: {len(all_data)}")

            if not all_data:
                logger.warning(f"No data fetched for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            # Bybit kline format: [startTime, openPrice, highPrice, lowPrice, closePrice, volume, turnover]
            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])

            # Convert types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'turnover']:
                df[col] = pd.to_numeric(df[col])

            # Sort by timestamp ascending
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Drop turnover column (not needed for OHLCV)
            df = df.drop('turnover', axis=1)

            logger.info(f"Loaded {len(df)} OHLCV records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error loading OHLCV data: {e}", exc_info=True)
            return pd.DataFrame()

    async def load_funding_rates(
        self,
        symbol: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 200
    ) -> pd.DataFrame:
        """
        Load funding rate history from Bybit

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            start_date: Start date
            end_date: End date
            limit: Max records per request

        Returns:
            DataFrame with columns: timestamp, funding_rate
        """
        if not self.client:
            logger.error("Client not initialized - returning empty DataFrame")
            return pd.DataFrame()

        # Default dates
        if end_date is None:
            end_date = datetime.utcnow()
        if start_date is None:
            start_date = end_date - timedelta(days=90)

        logger.info(f"Loading funding rates for {symbol}")

        try:
            # Convert to timestamps (milliseconds)
            start_ts = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)

            # Run in executor
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.get_funding_rate_history(
                    category="linear",
                    symbol=symbol,
                    startTime=start_ts,
                    endTime=end_ts,
                    limit=limit
                )
            )

            if result['retCode'] != 0:
                logger.error(f"Error fetching funding rates: {result['retMsg']}")
                return pd.DataFrame()

            data = result['result']['list']
            if not data:
                return pd.DataFrame()

            # Convert to DataFrame
            # Format: [symbol, fundingRate, fundingRateTimestamp]
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['fundingRateTimestamp'].astype(int), unit='ms')
            df['funding_rate'] = pd.to_numeric(df['fundingRate'])

            # Keep only relevant columns
            df = df[['timestamp', 'funding_rate']]

            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)

            logger.info(f"Loaded {len(df)} funding rate records for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error loading funding rates: {e}", exc_info=True)
            return pd.DataFrame()


# Example usage
async def main():
    """Example usage"""
    loader = MarketDataLoader(testnet=True)

    # Load OHLCV
    ohlcv = await loader.load_ohlcv("BTCUSDT", interval="60", limit=100)
    print(f"Loaded {len(ohlcv)} OHLCV records")
    if not ohlcv.empty:
        print(ohlcv.head())

    # Load funding rates
    funding = await loader.load_funding_rates("BTCUSDT")
    print(f"\nLoaded {len(funding)} funding rate records")
    if not funding.empty:
        print(funding.head())


if __name__ == "__main__":
    asyncio.run(main())

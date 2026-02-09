"""
Binance funding rate client
"""

import logging
from datetime import UTC, datetime

import aiohttp

from .models import ExchangeFundingRate, ExchangeName, FundingHistory

logger = logging.getLogger(__name__)


class BinanceFundingClient:
    """
    Fetch funding rates from Binance Futures

    Binance API Docs: https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize Binance funding client

        Args:
            testnet: Use testnet API (default True for safety)
        """
        self.testnet = testnet
        if testnet:
            self.base_url = "https://testnet.binancefuture.com"
        else:
            self.base_url = "https://fapi.binance.com"

        self._session: aiohttp.ClientSession | None = None
        self._history_cache: dict[str, FundingHistory] = {}

    async def initialize(self):
        """Initialize HTTP session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        logger.info(f"Binance funding client initialized (testnet={self.testnet})")

    async def close(self):
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None

    async def get_funding_rate(self, symbol: str) -> ExchangeFundingRate | None:
        """
        Get current funding rate for symbol

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            ExchangeFundingRate or None if error
        """
        if not self._session:
            await self.initialize()

        try:
            # Fetch current funding rate
            url = f"{self.base_url}/fapi/v1/fundingRate"
            params = {
                "symbol": symbol,
                "limit": 1
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Binance API error: {resp.status}")
                    text = await resp.text()
                    logger.error(f"Response: {text}")
                    return None

                data = await resp.json()

                if not data:
                    logger.warning(f"No funding data for {symbol}")
                    return None

                item = data[0] if isinstance(data, list) else data

                # Get next funding time
                next_funding_time = await self._get_next_funding_time(symbol)

                # Get mark price
                mark_price = await self._get_mark_price(symbol)

                # Get predicted rate
                predicted_rate = await self._predict_next_rate(symbol)

                return ExchangeFundingRate(
                    exchange=ExchangeName.BINANCE,
                    symbol=symbol,
                    rate=float(item["fundingRate"]),
                    next_funding_time=next_funding_time,
                    predicted_rate=predicted_rate,
                    mark_price=mark_price,
                    timestamp=datetime.fromtimestamp(int(item["fundingTime"]) / 1000, tz=UTC)
                )

        except Exception as e:
            logger.error(f"Error fetching Binance funding rate for {symbol}: {e}")
            return None

    async def _get_next_funding_time(self, symbol: str) -> datetime:
        """Get next funding timestamp"""
        try:
            url = f"{self.base_url}/fapi/v1/premiumIndex"
            params = {"symbol": symbol}

            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    timestamp_ms = int(data.get("nextFundingTime", 0))
                    if timestamp_ms:
                        return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

        except Exception as e:
            logger.warning(f"Could not get next funding time: {e}")

        # Default: Binance funding every 8 hours (00:00, 08:00, 16:00 UTC)
        from datetime import timedelta
        now = datetime.now(UTC)
        hour = now.hour
        next_funding_hour = ((hour // 8) + 1) * 8 % 24
        next_funding = now.replace(hour=next_funding_hour, minute=0, second=0, microsecond=0)
        if next_funding <= now:
            next_funding += timedelta(days=1)
        return next_funding

    async def _get_mark_price(self, symbol: str) -> float | None:
        """Get current mark price"""
        try:
            url = f"{self.base_url}/fapi/v1/premiumIndex"
            params = {"symbol": symbol}

            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("markPrice", 0))

        except Exception as e:
            logger.warning(f"Could not get mark price: {e}")

        return None

    async def get_funding_history(
        self,
        symbol: str,
        limit: int = 50
    ) -> FundingHistory | None:
        """
        Get historical funding rates

        Args:
            symbol: Trading pair
            limit: Number of historical records (max 1000)

        Returns:
            FundingHistory or None
        """
        if not self._session:
            await self.initialize()

        try:
            url = f"{self.base_url}/fapi/v1/fundingRate"
            params = {
                "symbol": symbol,
                "limit": min(limit, 1000)
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()

                if not data:
                    return None

                rates = [float(item["fundingRate"]) for item in data]
                timestamps = [
                    datetime.fromtimestamp(int(item["fundingTime"]) / 1000, tz=UTC)
                    for item in data
                ]

                history = FundingHistory(
                    exchange=ExchangeName.BINANCE,
                    symbol=symbol,
                    rates=rates,
                    timestamps=timestamps
                )

                # Cache for prediction
                self._history_cache[symbol] = history

                return history

        except Exception as e:
            logger.error(f"Error fetching funding history: {e}")
            return None

    async def _predict_next_rate(self, symbol: str) -> float | None:
        """Predict next funding rate using historical data"""
        # Check cache first
        if symbol in self._history_cache:
            return self._history_cache[symbol].predict_next()

        # Fetch history
        history = await self.get_funding_history(symbol, limit=10)
        if history:
            return history.predict_next()

        return None


# Example usage
async def main():
    """Example usage"""
    client = BinanceFundingClient(testnet=False)  # Binance testnet doesn't have funding rates
    await client.initialize()

    # Get current funding rate
    funding = await client.get_funding_rate("BTCUSDT")

    if funding:
        print("\nBinance Funding Rate for BTC:")
        print(f"  Rate: {funding.rate_percent:.4f}%")
        print(f"  Annual Rate: {funding.annual_rate:.2f}%")
        print(f"  Next Funding: {funding.next_funding_time}")
        print(f"  Predicted Next Rate: {funding.predicted_rate}")
        print(f"  Mark Price: ${funding.mark_price:,.2f}")

    # Get history
    history = await client.get_funding_history("BTCUSDT", limit=20)
    if history:
        print("\nFunding History (last 20):")
        print(f"  Average: {history.average_rate * 100:.4f}%")
        print(f"  Volatility: {history.volatility * 100:.4f}%")
        print(f"  Prediction: {history.predict_next() * 100:.4f}%")

    await client.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

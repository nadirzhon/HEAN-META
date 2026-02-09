"""
Bybit funding rate client
"""

import logging
from datetime import UTC, datetime

import aiohttp

from .models import ExchangeFundingRate, ExchangeName, FundingHistory

logger = logging.getLogger(__name__)


class BybitFundingClient:
    """
    Fetch funding rates from Bybit

    Bybit API Docs: https://bybit-exchange.github.io/docs/v5/market/funding-rate
    """

    def __init__(self, testnet: bool = True):
        """
        Initialize Bybit funding client

        Args:
            testnet: Use testnet API (default True for safety)
        """
        self.testnet = testnet
        if testnet:
            self.base_url = "https://api-testnet.bybit.com"
        else:
            self.base_url = "https://api.bybit.com"

        self._session: aiohttp.ClientSession | None = None
        self._history_cache: dict[str, FundingHistory] = {}

    async def initialize(self):
        """Initialize HTTP session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        logger.info(f"Bybit funding client initialized (testnet={self.testnet})")

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
            url = f"{self.base_url}/v5/market/funding/history"
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": 1
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Bybit API error: {resp.status}")
                    return None

                data = await resp.json()

                if data.get("retCode") != 0:
                    logger.error(f"Bybit API error: {data.get('retMsg')}")
                    return None

                result = data.get("result", {})
                items = result.get("list", [])

                if not items:
                    logger.warning(f"No funding data for {symbol}")
                    return None

                item = items[0]

                # Get next funding time
                next_funding_time = await self._get_next_funding_time(symbol)

                # Get mark price
                mark_price = await self._get_mark_price(symbol)

                # Get predicted rate (from history)
                predicted_rate = await self._predict_next_rate(symbol)

                return ExchangeFundingRate(
                    exchange=ExchangeName.BYBIT,
                    symbol=symbol,
                    rate=float(item["fundingRate"]),
                    next_funding_time=next_funding_time,
                    predicted_rate=predicted_rate,
                    mark_price=mark_price,
                    timestamp=datetime.utcnow()
                )

        except Exception as e:
            logger.error(f"Error fetching Bybit funding rate for {symbol}: {e}")
            return None

    async def _get_next_funding_time(self, symbol: str) -> datetime:
        """Get next funding timestamp"""
        try:
            url = f"{self.base_url}/v5/market/tickers"
            params = {
                "category": "linear",
                "symbol": symbol
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    items = result.get("list", [])
                    if items:
                        timestamp_ms = int(items[0].get("nextFundingTime", 0))
                        if timestamp_ms:
                            return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

        except Exception as e:
            logger.warning(f"Could not get next funding time: {e}")

        # Default: assume 8 hour funding cycle, next is in max 8 hours
        # Bybit funding times: 00:00, 08:00, 16:00 UTC
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
            url = f"{self.base_url}/v5/market/tickers"
            params = {
                "category": "linear",
                "symbol": symbol
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    result = data.get("result", {})
                    items = result.get("list", [])
                    if items:
                        return float(items[0].get("markPrice", 0))

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
            limit: Number of historical records (max 200)

        Returns:
            FundingHistory or None
        """
        if not self._session:
            await self.initialize()

        try:
            url = f"{self.base_url}/v5/market/funding/history"
            params = {
                "category": "linear",
                "symbol": symbol,
                "limit": min(limit, 200)
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()
                if data.get("retCode") != 0:
                    return None

                result = data.get("result", {})
                items = result.get("list", [])

                if not items:
                    return None

                rates = [float(item["fundingRate"]) for item in items]
                timestamps = [
                    datetime.fromtimestamp(int(item["fundingRateTimestamp"]) / 1000, tz=UTC)
                    for item in items
                ]

                history = FundingHistory(
                    exchange=ExchangeName.BYBIT,
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
    client = BybitFundingClient(testnet=True)
    await client.initialize()

    # Get current funding rate
    funding = await client.get_funding_rate("BTCUSDT")

    if funding:
        print("\nBybit Funding Rate for BTC:")
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

"""
OKX funding rate client
"""

import logging
from datetime import UTC, datetime

import aiohttp

from .models import ExchangeFundingRate, ExchangeName, FundingHistory

logger = logging.getLogger(__name__)


class OKXFundingClient:
    """
    Fetch funding rates from OKX (formerly OKEx)

    OKX API Docs: https://www.okx.com/docs-v5/en/#rest-api-public-data-get-funding-rate-history
    """

    def __init__(self, testnet: bool = False):
        """
        Initialize OKX funding client

        Args:
            testnet: Use demo trading (default False, OKX has limited testnet)
        """
        self.testnet = testnet
        if testnet:
            # OKX demo trading
            self.base_url = "https://www.okx.com"
            logger.warning("OKX testnet not fully supported, using production API")
        else:
            self.base_url = "https://www.okx.com"

        self._session: aiohttp.ClientSession | None = None
        self._history_cache: dict[str, FundingHistory] = {}

    async def initialize(self):
        """Initialize HTTP session"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        logger.info("OKX funding client initialized")

    async def close(self):
        """Close HTTP session"""
        if self._session:
            await self._session.close()
            self._session = None

    def _convert_symbol(self, symbol: str) -> str:
        """
        Convert symbol format to OKX format

        Args:
            symbol: Standard format (e.g., "BTCUSDT")

        Returns:
            OKX format (e.g., "BTC-USDT-SWAP")
        """
        # Remove "USDT" suffix and add "-USDT-SWAP"
        if symbol.endswith("USDT"):
            base = symbol[:-4]
            return f"{base}-USDT-SWAP"
        return symbol

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
            okx_symbol = self._convert_symbol(symbol)

            # Fetch current funding rate
            url = f"{self.base_url}/api/v5/public/funding-rate"
            params = {
                "instId": okx_symbol
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"OKX API error: {resp.status}")
                    text = await resp.text()
                    logger.error(f"Response: {text}")
                    return None

                data = await resp.json()

                if data.get("code") != "0":
                    logger.error(f"OKX API error: {data.get('msg')}")
                    return None

                items = data.get("data", [])
                if not items:
                    logger.warning(f"No funding data for {symbol}")
                    return None

                item = items[0]

                # Parse next funding time
                next_funding_ms = int(item.get("nextFundingTime", 0))
                next_funding_time = datetime.fromtimestamp(next_funding_ms / 1000, tz=UTC)

                # Get mark price
                mark_price = await self._get_mark_price(okx_symbol)

                # Get predicted rate
                predicted_rate = await self._predict_next_rate(symbol)

                # Parse funding timestamp
                funding_time_ms = int(item.get("fundingTime", 0))
                timestamp = datetime.fromtimestamp(funding_time_ms / 1000, tz=UTC)

                return ExchangeFundingRate(
                    exchange=ExchangeName.OKX,
                    symbol=symbol,
                    rate=float(item["fundingRate"]),
                    next_funding_time=next_funding_time,
                    predicted_rate=predicted_rate,
                    mark_price=mark_price,
                    timestamp=timestamp
                )

        except Exception as e:
            logger.error(f"Error fetching OKX funding rate for {symbol}: {e}")
            return None

    async def _get_mark_price(self, okx_symbol: str) -> float | None:
        """Get current mark price"""
        try:
            url = f"{self.base_url}/api/v5/public/mark-price"
            params = {
                "instType": "SWAP",
                "instId": okx_symbol
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data.get("code") == "0":
                        items = data.get("data", [])
                        if items:
                            return float(items[0].get("markPx", 0))

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
            limit: Number of historical records (max 100)

        Returns:
            FundingHistory or None
        """
        if not self._session:
            await self.initialize()

        try:
            okx_symbol = self._convert_symbol(symbol)

            url = f"{self.base_url}/api/v5/public/funding-rate-history"
            params = {
                "instId": okx_symbol,
                "limit": str(min(limit, 100))
            }

            async with self._session.get(url, params=params) as resp:
                if resp.status != 200:
                    return None

                data = await resp.json()

                if data.get("code") != "0":
                    return None

                items = data.get("data", [])
                if not items:
                    return None

                rates = [float(item["fundingRate"]) for item in items]
                timestamps = [
                    datetime.fromtimestamp(int(item["fundingTime"]) / 1000, tz=UTC)
                    for item in items
                ]

                history = FundingHistory(
                    exchange=ExchangeName.OKX,
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
    client = OKXFundingClient(testnet=False)
    await client.initialize()

    # Get current funding rate
    funding = await client.get_funding_rate("BTCUSDT")

    if funding:
        print("\nOKX Funding Rate for BTC:")
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

"""Bybit price feed implementation using WebSocket."""

import asyncio

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, FundingRate
from hean.exchange.bybit.http import BybitHTTPClient
from hean.exchange.bybit.ws_public import BybitPublicWebSocket
from hean.exchange.models import PriceFeed
from hean.logging import get_logger

logger = get_logger(__name__)


class BybitPriceFeed(PriceFeed):
    """Bybit price feed using WebSocket for real-time data."""

    def __init__(self, bus: EventBus, symbols: list[str]) -> None:
        """Initialize Bybit price feed.

        Args:
            bus: Event bus for publishing events
            symbols: List of symbols to subscribe to
        """
        self._bus = bus
        self._symbols = symbols
        self._ws_public: BybitPublicWebSocket | None = None
        self._http_client: BybitHTTPClient | None = None
        self._running = False
        self._funding_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the price feed."""
        if not settings.is_live and not settings.paper_use_live_feed:
            logger.warning("BybitPriceFeed: Live feed disabled for paper mode")
            return

        try:
            # Initialize HTTP client for funding rate if credentials are available
            if settings.bybit_api_key and settings.bybit_api_secret:
                self._http_client = BybitHTTPClient()
                await self._http_client.connect()
            else:
                logger.info("Bybit HTTP client disabled (missing API keys). Funding rates skipped.")

            # Initialize and connect WebSocket
            self._ws_public = BybitPublicWebSocket(self._bus)
            await self._ws_public.connect()

            # Subscribe to tickers for all symbols
            for symbol in self._symbols:
                await self._ws_public.subscribe_ticker(symbol)

            # Subscribe to a small orderbook set for OFI/iceberg visuals
            orderbook_symbols = [s for s in self._symbols if s in {"BTCUSDT", "ETHUSDT"}]
            if not orderbook_symbols:
                orderbook_symbols = self._symbols[:2]
            for symbol in orderbook_symbols:
                await self._ws_public.subscribe_orderbook(symbol, depth=25)

            # Start funding rate polling if HTTP client is available
            self._running = True
            if self._http_client:
                self._funding_task = asyncio.create_task(self._poll_funding_rates())

            mode_label = "live" if settings.is_live else "paper"
            logger.info(f"Bybit price feed started for {mode_label} mode: {self._symbols}")

        except Exception as e:
            logger.error(f"Failed to start Bybit price feed: {e}", exc_info=True)
            raise

    async def stop(self) -> None:
        """Stop the price feed."""
        self._running = False

        if self._funding_task:
            self._funding_task.cancel()
            try:
                await self._funding_task
            except asyncio.CancelledError:
                pass

        if self._ws_public:
            await self._ws_public.disconnect()
            self._ws_public = None

        if self._http_client:
            await self._http_client.disconnect()
            self._http_client = None

        logger.info("Bybit price feed stopped")

    async def subscribe(self, symbol: str) -> None:
        """Subscribe to price updates for a symbol.

        Args:
            symbol: Trading symbol
        """
        if self._ws_public:
            await self._ws_public.subscribe_ticker(symbol)
            if symbol not in self._symbols:
                self._symbols.append(symbol)

    async def _poll_funding_rates(self) -> None:
        """Poll funding rates periodically (every 8 hours or on demand)."""
        while self._running:
            try:
                for symbol in self._symbols:
                    try:
                        funding_data = await self._http_client.get_funding_rate(symbol)
                        if funding_data:
                            rate = float(funding_data.get("fundingRate", 0))
                            next_funding_time = funding_data.get("nextFundingTime")

                            from datetime import datetime

                            funding = FundingRate(
                                symbol=symbol,
                                rate=rate,
                                timestamp=datetime.utcnow(),
                                next_funding_time=datetime.fromtimestamp(
                                    int(next_funding_time) / 1000
                                )
                                if next_funding_time
                                else None,
                            )

                            await self._bus.publish(
                                Event(event_type=EventType.FUNDING, data={"funding": funding})
                            )

                            logger.debug(f"Published funding rate for {symbol}: {rate:.6f}")
                    except Exception as e:
                        logger.error(f"Failed to get funding rate for {symbol}: {e}")

                # Poll every 8 hours (funding happens every 8 hours)
                await asyncio.sleep(8 * 60 * 60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in funding rate polling: {e}")
                await asyncio.sleep(60)  # Wait before retry

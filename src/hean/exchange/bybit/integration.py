"""Bybit integration helper for easy setup and testing."""

from hean.config import settings
from hean.core.bus import EventBus
from hean.exchange.bybit.http import BybitHTTPClient
from hean.exchange.bybit.ws_private import BybitPrivateWebSocket
from hean.exchange.bybit.ws_public import BybitPublicWebSocket
from hean.logging import get_logger

logger = get_logger(__name__)


class BybitIntegration:
    """Helper class for Bybit integration setup and testing."""

    def __init__(self, bus: EventBus) -> None:
        """Initialize Bybit integration.

        Args:
            bus: Event bus for publishing events
        """
        self._bus = bus
        self._http_client: BybitHTTPClient | None = None
        self._ws_public: BybitPublicWebSocket | None = None
        self._ws_private: BybitPrivateWebSocket | None = None
        self._connected = False

    async def connect(self, symbols: list[str] | None = None) -> bool:
        """Connect to Bybit API and WebSockets.

        Args:
            symbols: List of symbols to subscribe to (default: ["BTCUSDT", "ETHUSDT"])

        Returns:
            True if connection successful, False otherwise
        """
        if not settings.is_live:
            logger.info("Not in live mode, skipping Bybit connection")
            return False

        if not settings.bybit_api_key or not settings.bybit_api_secret:
            logger.warning("Bybit API credentials not configured")
            return False

        symbols = symbols or ["BTCUSDT", "ETHUSDT"]

        try:
            # Connect HTTP client
            self._http_client = BybitHTTPClient()
            await self._http_client.connect()
            logger.info("Bybit HTTP client connected")

            # Connect public WebSocket
            self._ws_public = BybitPublicWebSocket(self._bus)
            await self._ws_public.connect()
            for symbol in symbols:
                await self._ws_public.subscribe_ticker(symbol)
            logger.info(f"Bybit public WebSocket connected and subscribed to {symbols}")

            # Connect private WebSocket
            self._ws_private = BybitPrivateWebSocket(self._bus)
            await self._ws_private.connect()
            await self._ws_private.subscribe_all()
            logger.info("Bybit private WebSocket connected and subscribed")

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Bybit: {e}", exc_info=True)
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from Bybit."""
        if self._ws_private:
            await self._ws_private.disconnect()
            self._ws_private = None

        if self._ws_public:
            await self._ws_public.disconnect()
            self._ws_public = None

        if self._http_client:
            await self._http_client.disconnect()
            self._http_client = None

        self._connected = False
        logger.info("Bybit integration disconnected")

    async def test_connection(self) -> bool:
        """Test connection to Bybit API.

        Returns:
            True if connection test successful
        """
        if not self._http_client:
            return False

        try:
            account_info = await self._http_client.get_account_info()
            logger.info(f"Bybit connection test successful: {account_info}")
            return True
        except Exception as e:
            logger.error(f"Bybit connection test failed: {e}")
            return False

    async def test_order_placement(self, symbol: str = "BTCUSDT") -> bool:
        """Test order placement (dry run).

        Args:
            symbol: Symbol to test with

        Returns:
            True if test successful
        """
        if not self._http_client:
            return False

        try:
            # Get current ticker to get price
            ticker = await self._http_client.get_ticker(symbol)
            current_price = float(ticker.get("lastPrice", 0))

            if current_price == 0:
                logger.error("Could not get current price for test")
                return False

            logger.info(f"Test order placement: {symbol} @ {current_price}")
            # Note: Actual order placement would require OrderRequest
            # This is just a connection/price test
            return True

        except Exception as e:
            logger.error(f"Order placement test failed: {e}")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if connected to Bybit."""
        return self._connected

    @property
    def http_client(self) -> BybitHTTPClient | None:
        """Get HTTP client."""
        return self._http_client

    @property
    def ws_public(self) -> BybitPublicWebSocket | None:
        """Get public WebSocket client."""
        return self._ws_public

    @property
    def ws_private(self) -> BybitPrivateWebSocket | None:
        """Get private WebSocket client."""
        return self._ws_private

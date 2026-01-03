"""Bybit public WebSocket client for market data."""

import asyncio
import json
import ssl

try:
    import websockets
    import websockets.exceptions
except ImportError:
    websockets = None  # type: ignore
    logger.warning("websockets library not installed. Bybit WebSocket will not work.")
from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


class BybitPublicWebSocket:
    """Bybit public WebSocket client for market data."""

    def __init__(self, bus: EventBus) -> None:
        """Initialize the Bybit public WebSocket client.

        Args:
            bus: Event bus for publishing market data
        """
        self._bus = bus
        self._connected = False
        self._testnet = settings.bybit_testnet
        self._websocket: websockets.WebSocketClientProtocol | None = None
        self._task: asyncio.Task[None] | None = None
        self._subscribed_symbols: set[str] = set()

        # WebSocket URLs
        if self._testnet:
            self._ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self._ws_url = "wss://stream.bybit.com/v5/public/linear"

    async def connect(self) -> None:
        """Connect to Bybit public WebSocket."""
        if not settings.is_live:
            logger.info("Bybit public WebSocket: Paper mode - using synthetic feed")
            return

        if websockets is None:
            raise ImportError(
                "websockets library not installed. Install with: pip install websockets"
            )

        try:
            # Create SSL context that doesn't verify certificates (for macOS compatibility)
            # WARNING: In production, use proper certificate verification
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            self._websocket = await websockets.connect(
                self._ws_url,
                ssl=ssl_context
            )  # type: ignore
            self._connected = True
            self._task = asyncio.create_task(self._listen())
            logger.info(
                f"Bybit public WebSocket connected to {'testnet' if self._testnet else 'mainnet'}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Bybit public WebSocket: {e}")
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from Bybit public WebSocket."""
        self._connected = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._websocket:
            await self._websocket.close()
            self._websocket = None

        logger.info("Bybit public WebSocket disconnected")

    async def _listen(self) -> None:
        """Listen for WebSocket messages with reconnection."""
        if not self._websocket:
            return

        reconnect_delay = 5.0
        max_reconnect_attempts = 10
        reconnect_attempts = 0

        while self._connected:
            try:
                async for message in self._websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                        reconnect_attempts = 0  # Reset on successful message
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse WebSocket message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
            except Exception as e:
                error_name = type(e).__name__
                if "ConnectionClosed" in error_name or "ConnectionClosed" in str(e):
                    if self._connected and reconnect_attempts < max_reconnect_attempts:
                        reconnect_attempts += 1
                        logger.warning(
                            f"WebSocket connection closed, reconnecting (attempt {reconnect_attempts}/{max_reconnect_attempts})..."
                        )
                        await asyncio.sleep(reconnect_delay)
                        try:
                            await self.connect()
                            # Re-subscribe to all symbols
                            for symbol in self._subscribed_symbols.copy():
                                await self.subscribe_ticker(symbol)
                        except Exception as e:
                            logger.error(f"Reconnection failed: {e}")
                            reconnect_delay = min(reconnect_delay * 1.5, 60.0)
                    else:
                        logger.error("Max reconnection attempts reached or connection disabled")
                        self._connected = False
                        break
            except asyncio.CancelledError:
                logger.debug("WebSocket listen task cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket listen error: {e}")
                if self._connected and reconnect_attempts < max_reconnect_attempts:
                    reconnect_attempts += 1
                    await asyncio.sleep(reconnect_delay)
                    try:
                        await self.connect()
                    except Exception:
                        pass
                else:
                    self._connected = False
                    break

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming WebSocket message.

        Args:
            data: Parsed JSON message
        """
        topic = data.get("topic", "")

        if "tickers" in topic:
            await self._handle_ticker(data)
        elif "orderbook" in topic:
            await self._handle_orderbook(data)
        elif "trade" in topic:
            await self._handle_trade(data)

    async def _handle_ticker(self, data: dict) -> None:
        """Handle ticker update.

        Args:
            data: Ticker data from WebSocket
        """
        ticker_data = data.get("data", {})
        if not ticker_data:
            return

        symbol = ticker_data.get("symbol", "")
        if not symbol:
            return

        try:
            # Safely convert price - handle empty strings, None, and invalid values
            last_price_raw = ticker_data.get("lastPrice")
            if last_price_raw is None or last_price_raw == "":
                logger.warning(f"Invalid lastPrice for {symbol}: {last_price_raw}, skipping tick")
                return
            
            try:
                price = float(last_price_raw)
            except (ValueError, TypeError):
                logger.warning(f"Failed to convert lastPrice to float for {symbol}: {last_price_raw}, skipping tick")
                return

            # Skip ticks with zero or negative price
            if price <= 0:
                logger.warning(f"Invalid price (<=0) for {symbol}: {price}, skipping tick")
                return

            bid = None
            if ticker_data.get("bid1Price"):
                try:
                    bid = float(ticker_data.get("bid1Price"))
                    if bid <= 0:
                        bid = None
                except (ValueError, TypeError):
                    bid = None

            ask = None
            if ticker_data.get("ask1Price"):
                try:
                    ask = float(ticker_data.get("ask1Price"))
                    if ask <= 0:
                        ask = None
                except (ValueError, TypeError):
                    ask = None

            from datetime import datetime

            tick = Tick(
                symbol=symbol,
                price=price,
                timestamp=datetime.utcnow(),
                bid=bid,
                ask=ask,
            )

            await self._bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse ticker data: {e}")

    async def _handle_orderbook(self, data: dict) -> None:
        """Handle order book update.

        Args:
            data: Order book data from WebSocket
        """
        # TODO: Implement order book handling if needed
        pass

    async def _handle_trade(self, data: dict) -> None:
        """Handle trade update.

        Args:
            data: Trade data from WebSocket
        """
        # TODO: Implement trade handling if needed
        pass

    async def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to ticker updates for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTCUSDT")
        """
        if not self._connected or not self._websocket:
            logger.warning("WebSocket not connected, cannot subscribe")
            return

        topic = f"tickers.{symbol}"
        subscribe_msg = {
            "op": "subscribe",
            "args": [topic],
        }

        try:
            await self._websocket.send(json.dumps(subscribe_msg))
            self._subscribed_symbols.add(symbol)
            logger.info(f"Subscribed to ticker for {symbol}")
        except Exception as e:
            logger.error(f"Failed to subscribe to ticker for {symbol}: {e}")

    async def subscribe_orderbook(self, symbol: str, depth: int = 25) -> None:
        """Subscribe to order book updates.

        Args:
            symbol: Trading symbol
            depth: Order book depth (1, 25, 50, 100, 200)
        """
        if not self._connected or not self._websocket:
            logger.warning("WebSocket not connected, cannot subscribe")
            return

        topic = f"orderbook.{depth}.{symbol}"
        subscribe_msg = {
            "op": "subscribe",
            "args": [topic],
        }

        try:
            await self._websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to orderbook for {symbol} (depth: {depth})")
        except Exception as e:
            logger.error(f"Failed to subscribe to orderbook for {symbol}: {e}")

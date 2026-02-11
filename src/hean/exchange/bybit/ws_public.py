"""Bybit public WebSocket client for market data."""

import asyncio
import json
import ssl
import time

from hean.logging import get_logger

logger = get_logger(__name__)

try:
    import websockets
    import websockets.exceptions
except ImportError:
    websockets = None  # type: ignore
    logger.warning("websockets library not installed. Bybit WebSocket will not work.")

from hean.config import settings  # noqa: E402
from hean.core.bus import EventBus  # noqa: E402
from hean.core.types import Event, EventType, Tick  # noqa: E402


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
        self._subscribed_orderbooks: dict[str, int] = {}
        self._missing_price_warned: set[str] = set()

        # WebSocket URLs
        if self._testnet:
            self._ws_url = "wss://stream-testnet.bybit.com/v5/public/linear"
        else:
            self._ws_url = "wss://stream.bybit.com/v5/public/linear"

        # Phase 16: Dynamic endpoint switching support
        self._dynamic_ws_url: str | None = None

        # Cache ticker state to handle delta updates
        self._ticker_cache: dict[str, dict] = {}

    async def connect(self, _from_listen: bool = False) -> None:
        """Connect to Bybit public WebSocket."""
        if not settings.is_live and not settings.paper_use_live_feed:
            logger.info("Bybit public WebSocket: Paper mode without live feed - no ticks will be generated")
            return

        if websockets is None:
            raise ImportError(
                "websockets library not installed. Install with: pip install websockets"
            )

        try:
            # Phase 16: Use dynamic endpoint if set (from API Scouter)
            ws_url = self._dynamic_ws_url if self._dynamic_ws_url else self._ws_url

            # Create SSL context that doesn't verify certificates (for macOS compatibility)
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            self._websocket = await asyncio.wait_for(
                websockets.connect(ws_url, ssl=ssl_context),
                timeout=15,
            )  # type: ignore
            self._connected = True
            # Only create listen task on initial connect, not reconnects from within _listen
            if not _from_listen:
                self._task = asyncio.create_task(self._listen())
            logger.info(
                f"Bybit public WebSocket connected to {ws_url} ({'testnet' if self._testnet else 'mainnet'})"
            )
        except Exception as e:
            logger.error(f"Failed to connect to Bybit public WebSocket: {e}")
            self._connected = False
            if not _from_listen:
                raise RuntimeError("Failed to connect to Bybit public WebSocket") from e

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
        reconnect_attempts = 0
        last_message_time = time.time()
        PING_INTERVAL = 20.0  # Send ping every 20 seconds
        CONNECTION_TIMEOUT = 60.0  # Timeout after 60 seconds of no messages

        while self._connected:
            try:
                # Use asyncio.wait_for to add timeout for receiving messages
                try:
                    message = await asyncio.wait_for(
                        self._websocket.recv(),
                        timeout=min(PING_INTERVAL, CONNECTION_TIMEOUT)
                    )
                    last_message_time = time.time()

                    try:
                        data = json.loads(message)
                        # Log all non-pong messages for debugging
                        if data.get("op") not in ("pong",):
                            logger.debug(f"WebSocket received: {data}")
                        await self._handle_message(data)
                        reconnect_attempts = 0  # Reset on successful message
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse WebSocket message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                except TimeoutError as te:
                    # Check if we need to send ping or if connection is dead
                    time_since_last = time.time() - last_message_time
                    if time_since_last >= CONNECTION_TIMEOUT:
                        logger.warning(f"WebSocket connection timeout ({time_since_last:.1f}s), reconnecting...")
                        raise ConnectionError("Connection timeout") from te
                    # Send ping to keep connection alive
                    try:
                        ping_msg = {"op": "ping"}
                        await self._websocket.send(json.dumps(ping_msg))
                        logger.debug("Sent ping to keep connection alive")
                    except Exception as e:
                        logger.warning(f"Failed to send ping: {e}")
                        raise ConnectionError("Failed to send ping") from e
            except (TimeoutError, ConnectionError) as e:
                if not self._connected:
                    break
                reconnect_attempts += 1
                logger.warning(
                    f"WebSocket disconnected ({type(e).__name__}), reconnect attempt {reconnect_attempts}..."
                )
                await self._reconnect(reconnect_delay)
                if self._websocket and self._connected:
                    reconnect_delay = 5.0
                    reconnect_attempts = 0
                else:
                    reconnect_delay = min(reconnect_delay * 1.5, 60.0)
            except asyncio.CancelledError:
                logger.debug("WebSocket listen task cancelled")
                break
            except Exception as e:
                if not self._connected:
                    break
                logger.warning(f"WebSocket error ({type(e).__name__}): {e}, reconnecting...")
                reconnect_attempts += 1
                await self._reconnect(reconnect_delay)
                if self._websocket and self._connected:
                    reconnect_delay = 5.0
                    reconnect_attempts = 0
                else:
                    reconnect_delay = min(reconnect_delay * 1.5, 60.0)

    async def _reconnect(self, delay: float) -> None:
        """Reconnect WebSocket with delay, re-subscribing to all topics."""
        await asyncio.sleep(delay)
        try:
            self._connected = True  # Keep connected flag so _listen loop continues
            await self.connect(_from_listen=True)
            for symbol in self._subscribed_symbols.copy():
                await self.subscribe_ticker(symbol)
            for symbol, depth in self._subscribed_orderbooks.items():
                await self.subscribe_orderbook(symbol, depth=depth)
            logger.info("WebSocket reconnected and re-subscribed successfully")
        except Exception as err:
            logger.error(f"Reconnection failed: {err}")
            self._connected = True  # Stay in loop to retry

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming WebSocket message.

        Args:
            data: Parsed JSON message
        """
        # Handle ping/pong - Bybit sends "ping" and expects "pong" response
        if data.get("op") == "ping":
            try:
                pong_msg = {"op": "pong"}
                if self._websocket:
                    await self._websocket.send(json.dumps(pong_msg))
                    logger.debug("Responded to server ping with pong")
            except Exception as e:
                logger.warning(f"Failed to send pong response: {e}")
            return

        # Handle pong response to our pings
        if data.get("op") == "pong":
            logger.debug("Received pong response from server")
            return

        # Handle subscription success/error
        if data.get("op") == "subscribe":
            if data.get("success"):
                logger.debug(f"Subscription confirmed: {data.get('req_id', 'unknown')}")
            else:
                logger.warning(f"Subscription failed: {data}")
            return

        topic = data.get("topic", "")

        if "tickers" in topic:
            await self._handle_ticker(data)
        elif "orderbook" in topic:
            await self._handle_orderbook(data)
        elif "trade" in topic:
            await self._handle_trade(data)

    async def _handle_ticker(self, data: dict) -> None:
        """Handle ticker update (snapshot or delta).

        Bybit sends two types of ticker messages:
        - snapshot: Full ticker state (sent on subscription and periodically)
        - delta: Partial update with only changed fields

        We cache the last snapshot and merge deltas to always have complete state.

        Args:
            data: Ticker data from WebSocket
        """
        ticker_data = data.get("data", {})
        if not ticker_data:
            return

        symbol = ticker_data.get("symbol", "")
        if not symbol:
            return

        message_type = data.get("type", "")

        try:
            # Handle snapshot vs delta
            if message_type == "snapshot":
                # Full state - replace cache
                self._ticker_cache[symbol] = ticker_data.copy()
            elif message_type == "delta":
                # Partial update - merge with cached state
                if symbol not in self._ticker_cache:
                    # No cached state yet, skip this delta
                    logger.debug(f"Skipping delta for {symbol} (no cached snapshot yet)")
                    return
                # Merge delta into cache
                self._ticker_cache[symbol].update(ticker_data)
            else:
                # Unknown message type, treat as snapshot
                self._ticker_cache[symbol] = ticker_data.copy()

            # Get current state from cache
            current_state = self._ticker_cache.get(symbol, {})
            if not current_state:
                return

            # Extract price from current state
            last_price_raw = current_state.get("lastPrice")
            price = None
            if last_price_raw not in (None, ""):
                try:
                    price = float(last_price_raw)
                except (ValueError, TypeError):
                    price = None

            # Fallback to mid price if lastPrice missing
            bid_fallback = current_state.get("bid1Price")
            ask_fallback = current_state.get("ask1Price")
            bid_val = None
            ask_val = None
            try:
                bid_val = float(bid_fallback) if bid_fallback not in (None, "") else None
            except (ValueError, TypeError):
                bid_val = None
            try:
                ask_val = float(ask_fallback) if ask_fallback not in (None, "") else None
            except (ValueError, TypeError):
                ask_val = None

            if price is None:
                if bid_val and ask_val:
                    price = (bid_val + ask_val) / 2
                elif bid_val:
                    price = bid_val
                elif ask_val:
                    price = ask_val

            # Skip ticks with zero or negative price
            if price is None or price <= 0:
                # Don't spam warnings for deltas without price updates
                if message_type == "snapshot" and symbol not in self._missing_price_warned:
                    logger.warning(f"Invalid price for {symbol}: lastPrice={last_price_raw}, bid={bid_fallback}, ask={ask_fallback}, skipping tick")
                    self._missing_price_warned.add(symbol)
                return

            # Clear warning flag if we got valid price
            self._missing_price_warned.discard(symbol)

            bid = bid_val
            ask = ask_val

            from datetime import datetime

            tick = Tick(
                symbol=symbol,
                price=price,
                timestamp=datetime.utcnow(),
                bid=bid,
                ask=ask,
            )

            # Update last tick timestamp for heartbeat monitoring
            self._last_tick_timestamp = datetime.utcnow()

            await self._bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to parse ticker data: {e}")

    async def _handle_orderbook(self, data: dict) -> None:
        """Handle order book update.

        Args:
            data: Order book data from WebSocket
        """
        orderbook_data = data.get("data", {})
        if not orderbook_data:
            return

        symbol = orderbook_data.get("s", "")
        if not symbol:
            return

        # Extract bids and asks (L2 format)
        bids = orderbook_data.get("b", [])  # List of [price, qty]
        asks = orderbook_data.get("a", [])  # List of [price, qty]

        orderbook = {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "b": bids,  # Backward compatibility
            "a": asks,  # Backward compatibility
            "timestamp_ns": orderbook_data.get("ts", 0),
            "update_id": orderbook_data.get("u", 0),
            "ts": orderbook_data.get("ts", 0),
            "u": orderbook_data.get("u", 0),
        }

        # Publish orderbook update event
        await self._bus.publish(
            Event(
                event_type=EventType.ORDER_BOOK_UPDATE,
                data={"orderbook": orderbook},
            )
        )

    async def _handle_trade(self, data: dict) -> None:
        """Handle trade update from Bybit publicTrade topic.

        Bybit sends public trade data in the format:
        {"topic": "publicTrade.BTCUSDT", "data": [{"i": "...", "T": ts, "p": "price",
         "v": "qty", "S": "Buy"/"Sell", "s": "BTCUSDT", "BT": false}]}

        Each trade is published as a TICK event so strategies receive real trade prices.

        Args:
            data: Trade data from WebSocket
        """
        from datetime import datetime

        trade_list = data.get("data", [])
        if not trade_list:
            return

        for trade in trade_list:
            symbol = trade.get("s", "")
            if not symbol:
                continue

            try:
                price = float(trade.get("p", 0))
            except (ValueError, TypeError):
                continue

            if price <= 0:
                continue

            try:
                qty = float(trade.get("v", 0))
            except (ValueError, TypeError):
                qty = 0.0

            side = trade.get("S", "")  # "Buy" or "Sell"
            trade_ts = trade.get("T", 0)  # Unix timestamp in ms

            tick = Tick(
                symbol=symbol,
                price=price,
                timestamp=datetime.utcfromtimestamp(trade_ts / 1000) if trade_ts else datetime.utcnow(),
                bid=None,
                ask=None,
            )

            await self._bus.publish(
                Event(
                    event_type=EventType.TICK,
                    data={
                        "tick": tick,
                        "trade_qty": qty,
                        "trade_side": side,
                        "source": "publicTrade",
                    },
                )
            )

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
            raise RuntimeError(f"Failed to subscribe to ticker for {symbol}") from e

    async def subscribe_orderbook(self, symbol: str, depth: int = 50) -> None:
        """Subscribe to order book updates.

        Args:
            symbol: Trading symbol
            depth: Order book depth (1, 50, 200, 500 for Bybit V5)
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
            self._subscribed_orderbooks[symbol] = depth
            logger.info(f"Subscribed to orderbook for {symbol} (depth: {depth})")
        except Exception as e:
            logger.error(f"Failed to subscribe to orderbook for {symbol}: {e}")
            raise RuntimeError(f"Failed to subscribe to orderbook for {symbol}") from e

    async def switch_endpoint(self, ws_url: str) -> None:
        """Phase 16: Switch WebSocket endpoint dynamically (called by API Scouter).

        Disconnects from current endpoint and reconnects to new endpoint,
        preserving subscriptions.

        Args:
            ws_url: New WebSocket URL
        """
        if ws_url == (self._dynamic_ws_url or self._ws_url):
            logger.debug(f"Endpoint already set to {ws_url}, skipping switch")
            return

        logger.info(f"Phase 16: Switching WebSocket endpoint to: {ws_url}")

        # Save current subscriptions
        subscribed_symbols = self._subscribed_symbols.copy()

        # Disconnect from current endpoint
        await self.disconnect()

        # Set new endpoint
        self._dynamic_ws_url = ws_url

        # Reconnect to new endpoint
        await self.connect()

        # Re-subscribe to all symbols
        for symbol in subscribed_symbols:
            await self.subscribe_ticker(symbol)

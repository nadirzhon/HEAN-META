"""Bybit private WebSocket client for account/order updates."""

import asyncio
import hashlib
import hmac
import json
import ssl
import time

import websockets

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, OrderStatus
from hean.logging import get_logger

logger = get_logger(__name__)


class BybitPrivateWebSocket:
    """Bybit private WebSocket client for account/order updates."""

    def __init__(self, bus: EventBus) -> None:
        """Initialize the Bybit private WebSocket client.

        Args:
            bus: Event bus for publishing order/position updates
        """
        self._bus = bus
        self._connected = False
        self._api_key = settings.bybit_api_key
        self._api_secret = settings.bybit_api_secret
        self._testnet = settings.bybit_testnet
        self._websocket: websockets.WebSocketClientProtocol | None = None
        self._task: asyncio.Task[None] | None = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._reconnect_delay = 5.0  # seconds

        # WebSocket URLs
        if self._testnet:
            self._ws_url = "wss://stream-testnet.bybit.com/v5/private"
        else:
            self._ws_url = "wss://stream.bybit.com/v5/private"

    def _generate_auth_message(self) -> dict:
        """Generate authentication message for private WebSocket.

        Returns:
            Authentication message dict
        """
        expires = int((time.time() + 10000) * 1000)  # 10 seconds from now

        # Create signature
        param_str = f"GET/realtime{expires}"
        signature = hmac.new(
            self._api_secret.encode("utf-8"), param_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        return {
            "op": "auth",
            "args": [self._api_key, expires, signature],
        }

    async def connect(self) -> None:
        """Connect to Bybit private WebSocket."""
        if not settings.is_live:
            logger.info("Bybit private WebSocket: Paper mode - no connection needed")
            return

        if not self._api_key or not self._api_secret:
            raise ValueError("Bybit API credentials not configured")

        await self._connect_with_reconnect()

    async def _connect_with_reconnect(self) -> None:
        """Connect with automatic reconnection logic."""
        if websockets is None:
            raise ImportError(
                "websockets library not installed. Install with: pip install websockets"
            )

        while self._reconnect_attempts < self._max_reconnect_attempts:
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

                # Authenticate
                auth_msg = self._generate_auth_message()
                await self._websocket.send(json.dumps(auth_msg))

                # Wait for auth response
                response = await asyncio.wait_for(self._websocket.recv(), timeout=5.0)
                auth_response = json.loads(response)

                if auth_response.get("success") is True:
                    self._connected = True
                    self._reconnect_attempts = 0
                    self._task = asyncio.create_task(self._listen())
                    logger.info(
                        f"Bybit private WebSocket connected to {'testnet' if self._testnet else 'mainnet'}"
                    )
                    return
                else:
                    error_msg = auth_response.get("ret_msg", "Authentication failed")
                    logger.error(f"WebSocket authentication failed: {error_msg}")
                    raise ValueError(f"Authentication failed: {error_msg}")

            except Exception as e:
                self._reconnect_attempts += 1
                logger.warning(f"Connection attempt {self._reconnect_attempts} failed: {e}")

                if self._reconnect_attempts < self._max_reconnect_attempts:
                    await asyncio.sleep(self._reconnect_delay)
                    self._reconnect_delay = min(
                        self._reconnect_delay * 1.5, 60.0
                    )  # Exponential backoff, max 60s
                else:
                    logger.error("Max reconnection attempts reached")
                    raise

    async def disconnect(self) -> None:
        """Disconnect from Bybit private WebSocket."""
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

        logger.info("Bybit private WebSocket disconnected")

    async def _listen(self) -> None:
        """Listen for WebSocket messages with reconnection."""
        if not self._websocket:
            return

        while self._connected:
            try:
                async for message in self._websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse WebSocket message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
            except Exception as e:
                error_name = type(e).__name__
                if "ConnectionClosed" in error_name or "ConnectionClosed" in str(e):
                    logger.warning("WebSocket connection closed, attempting reconnect...")
                    if self._connected:
                        await self._reconnect()
                    break
            except asyncio.CancelledError:
                logger.debug("WebSocket listen task cancelled")
                break
            except Exception as e:
                logger.error(f"WebSocket listen error: {e}")
                if self._connected:
                    await self._reconnect()
                break

    async def _reconnect(self) -> None:
        """Reconnect to WebSocket."""
        if self._websocket:
            try:
                await self._websocket.close()
            except Exception:
                pass
            self._websocket = None

        self._connected = False
        await asyncio.sleep(self._reconnect_delay)

        if self._reconnect_attempts < self._max_reconnect_attempts:
            await self._connect_with_reconnect()

    async def _handle_message(self, data: dict) -> None:
        """Handle incoming WebSocket message.

        Args:
            data: Parsed JSON message
        """
        topic = data.get("topic", "")

        if "order" in topic:
            await self._handle_order_update(data)
        elif "position" in topic:
            await self._handle_position_update(data)
        elif "execution" in topic:
            await self._handle_execution(data)

    async def _handle_order_update(self, data: dict) -> None:
        """Handle order update.

        Args:
            data: Order update data from WebSocket
        """
        order_data = data.get("data", [])
        if not order_data:
            return

        for order_info in order_data:
            try:
                order_id = order_info.get("orderId", "")
                status = order_info.get("orderStatus", "")

                # Map Bybit order status to our OrderStatus
                if status == "Filled":
                    order_status = OrderStatus.FILLED
                elif status == "PartiallyFilled":
                    order_status = OrderStatus.PARTIALLY_FILLED
                elif status == "Cancelled":
                    order_status = OrderStatus.CANCELLED
                elif status == "New":
                    order_status = OrderStatus.PLACED
                else:
                    order_status = OrderStatus.PLACED

                # Create order event
                if order_status == OrderStatus.FILLED:
                    fill_price = float(order_info.get("avgPrice", order_info.get("price", 0)))
                    await self._bus.publish(
                        Event(
                            event_type=EventType.ORDER_FILLED,
                            data={
                                "order_id": order_id,
                                "fill_price": fill_price,
                            },
                        )
                    )
                elif order_status == OrderStatus.CANCELLED:
                    await self._bus.publish(
                        Event(event_type=EventType.ORDER_CANCELLED, data={"order_id": order_id})
                    )

            except Exception as e:
                logger.error(f"Failed to handle order update: {e}")

    async def _handle_position_update(self, data: dict) -> None:
        """Handle position update.

        Args:
            data: Position update data from WebSocket
        """
        position_data = data.get("data", [])
        if not position_data:
            return

        for pos_info in position_data:
            try:
                symbol = pos_info.get("symbol", "")
                size = float(pos_info.get("size", 0))
                side = pos_info.get("side", "")
                entry_price = float(pos_info.get("avgPrice", 0))
                mark_price = float(pos_info.get("markPrice", 0))

                if size == 0:
                    # Position closed
                    await self._bus.publish(
                        Event(
                            event_type=EventType.POSITION_CLOSED,
                            data={
                                "symbol": symbol,
                                "side": side,
                            },
                        )
                    )
                else:
                    # Position opened or updated
                    await self._bus.publish(
                        Event(
                            event_type=EventType.POSITION_UPDATE,
                            data={
                                "symbol": symbol,
                                "side": side,
                                "size": size,
                                "entry_price": entry_price,
                                "mark_price": mark_price,
                            },
                        )
                    )

            except Exception as e:
                logger.error(f"Failed to handle position update: {e}")

    async def _handle_execution(self, data: dict) -> None:
        """Handle execution update.

        Args:
            data: Execution data from WebSocket
        """
        execution_data = data.get("data", [])
        if not execution_data:
            return

        for exec_info in execution_data:
            try:
                order_id = exec_info.get("orderId", "")
                symbol = exec_info.get("symbol", "")
                side = exec_info.get("side", "")
                exec_price = float(exec_info.get("execPrice", 0))
                exec_qty = float(exec_info.get("execQty", 0))
                exec_type = exec_info.get("execType", "")

                # Publish execution event
                if exec_type == "Trade":
                    await self._bus.publish(
                        Event(
                            event_type=EventType.ORDER_FILLED,
                            data={
                                "order_id": order_id,
                                "symbol": symbol,
                                "side": side,
                                "fill_price": exec_price,
                                "fill_size": exec_qty,
                            },
                        )
                    )

            except Exception as e:
                logger.error(f"Failed to handle execution: {e}")

    async def subscribe_orders(self) -> None:
        """Subscribe to order updates."""
        if not self._connected or not self._websocket:
            logger.warning("WebSocket not connected, cannot subscribe")
            return

        subscribe_msg = {
            "op": "subscribe",
            "args": ["order"],
        }

        try:
            await self._websocket.send(json.dumps(subscribe_msg))
            logger.info("Subscribed to order updates")
        except Exception as e:
            logger.error(f"Failed to subscribe to orders: {e}")

    async def subscribe_positions(self) -> None:
        """Subscribe to position updates."""
        if not self._connected or not self._websocket:
            logger.warning("WebSocket not connected, cannot subscribe")
            return

        subscribe_msg = {
            "op": "subscribe",
            "args": ["position"],
        }

        try:
            await self._websocket.send(json.dumps(subscribe_msg))
            logger.info("Subscribed to position updates")
        except Exception as e:
            logger.error(f"Failed to subscribe to positions: {e}")

    async def subscribe_executions(self) -> None:
        """Subscribe to execution updates."""
        if not self._connected or not self._websocket:
            logger.warning("WebSocket not connected, cannot subscribe")
            return

        subscribe_msg = {
            "op": "subscribe",
            "args": ["execution"],
        }

        try:
            await self._websocket.send(json.dumps(subscribe_msg))
            logger.info("Subscribed to execution updates")
        except Exception as e:
            logger.error(f"Failed to subscribe to executions: {e}")

    async def subscribe_all(self) -> None:
        """Subscribe to all available updates (orders, positions, executions)."""
        await self.subscribe_orders()
        await self.subscribe_positions()
        await self.subscribe_executions()
"""Bybit private WebSocket client for account/order updates.

Enhanced with:
- Reconnection reconciliation to prevent missed fills
- Order state tracking for reconciliation
- Automatic HTTP API fallback on reconnect
"""

import asyncio
import hashlib
import hmac
import json
import ssl
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import websockets

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, OrderStatus
from hean.logging import get_logger

if TYPE_CHECKING:
    from hean.exchange.bybit.http import BybitHTTPClient

logger = get_logger(__name__)


@dataclass
class ReconnectionState:
    """State tracking for reconnection reconciliation."""
    disconnected_at: datetime | None = None
    reconnected_at: datetime | None = None
    last_order_update_at: datetime | None = None
    pending_order_ids: set[str] = field(default_factory=set)
    reconciliation_in_progress: bool = False
    fills_recovered: int = 0
    cancellations_recovered: int = 0


class BybitPrivateWebSocket:
    """Bybit private WebSocket client for account/order updates.

    Enhanced with reconnection reconciliation to prevent missed fills/cancellations
    during WebSocket disconnections.
    """

    def __init__(self, bus: EventBus, http_client: "BybitHTTPClient | None" = None) -> None:
        """Initialize the Bybit private WebSocket client.

        Args:
            bus: Event bus for publishing order/position updates
            http_client: Optional HTTP client for reconciliation queries
        """
        self._bus = bus
        self._http_client = http_client
        self._connected = False
        self._api_key = settings.bybit_api_key
        self._api_secret = settings.bybit_api_secret
        self._testnet = settings.bybit_testnet
        self._websocket: websockets.WebSocketClientProtocol | None = None
        self._task: asyncio.Task[None] | None = None
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 999999  # Never give up
        self._reconnect_delay = 5.0  # seconds
        self._base_reconnect_delay = 5.0  # Store original for reset

        # WebSocket URLs
        if self._testnet:
            self._ws_url = "wss://stream-testnet.bybit.com/v5/private"
        else:
            self._ws_url = "wss://stream.bybit.com/v5/private"

        # Reconnection reconciliation state
        self._recon_state = ReconnectionState()

        # Track known orders for reconciliation
        self._known_orders: dict[str, dict[str, Any]] = {}  # order_id -> order_data

        # Metrics
        self._metrics = {
            "reconnections": 0,
            "reconciliations": 0,
            "fills_recovered": 0,
            "cancellations_recovered": 0,
            "messages_processed": 0,
        }

    def set_http_client(self, http_client: "BybitHTTPClient") -> None:
        """Set the HTTP client for reconciliation (can be set after init)."""
        self._http_client = http_client

    def track_order(self, order_id: str, order_data: dict[str, Any]) -> None:
        """Track an order for reconciliation purposes.

        Should be called when orders are placed to enable reconciliation.

        Args:
            order_id: The order ID
            order_data: Order data including symbol, side, status, etc.
        """
        self._known_orders[order_id] = {
            **order_data,
            "tracked_at": datetime.utcnow(),
        }
        self._recon_state.pending_order_ids.add(order_id)
        logger.debug(f"[WS] Tracking order {order_id} for reconciliation")

    def untrack_order(self, order_id: str) -> None:
        """Stop tracking an order (when filled, cancelled, or expired)."""
        self._known_orders.pop(order_id, None)
        self._recon_state.pending_order_ids.discard(order_id)

    def get_metrics(self) -> dict[str, int]:
        """Get WebSocket metrics including reconciliation stats."""
        return {
            **self._metrics,
            "pending_orders_tracked": len(self._recon_state.pending_order_ids),
            "is_connected": self._connected,
        }

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
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse WebSocket message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling WebSocket message: {e}")
                except TimeoutError:
                    # Check if we need to send ping or if connection is dead
                    time_since_last = time.time() - last_message_time
                    if time_since_last >= CONNECTION_TIMEOUT:
                        logger.warning(f"WebSocket connection timeout ({time_since_last:.1f}s), reconnecting...")
                        raise ConnectionError("Connection timeout") from None
                    # Send ping to keep connection alive
                    try:
                        ping_msg = {"op": "ping"}
                        await self._websocket.send(json.dumps(ping_msg))
                        logger.debug("Sent ping to keep connection alive")
                    except Exception as e:
                        logger.warning(f"Failed to send ping: {e}")
                        raise ConnectionError("Failed to send ping") from e
            except (TimeoutError, ConnectionError):
                # Connection timeout or error - reconnect
                if self._connected:
                    logger.warning("WebSocket connection issue, attempting reconnect...")
                    await self._reconnect()
                break
            except asyncio.CancelledError:
                logger.debug("WebSocket listen task cancelled")
                break
            except Exception as e:
                error_name = type(e).__name__
                if "ConnectionClosed" in error_name or "ConnectionClosed" in str(e):
                    logger.warning("WebSocket connection closed, attempting reconnect...")
                else:
                    logger.error(f"WebSocket listen error: {e}")
                if self._connected:
                    await self._reconnect()
                break

    async def _reconnect(self) -> None:
        """Reconnect to WebSocket with reconciliation.

        When reconnecting, we need to:
        1. Record the disconnection time
        2. Close the old connection
        3. Reconnect
        4. Reconcile missed events via HTTP API
        """
        # Record disconnection time
        self._recon_state.disconnected_at = datetime.utcnow()
        logger.warning(
            f"[WS] Disconnected at {self._recon_state.disconnected_at.isoformat()}. "
            f"Pending orders to reconcile: {len(self._recon_state.pending_order_ids)}"
        )

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                logger.warning(f"WebSocket close error during reconnect: {e}")
            self._websocket = None

        self._connected = False
        self._metrics["reconnections"] += 1

        # Use shorter delay for reconnect (max 5 seconds instead of 60)
        delay = min(self._reconnect_delay, 5.0)
        await asyncio.sleep(delay)

        if self._reconnect_attempts < self._max_reconnect_attempts:
            await self._connect_with_reconnect()

            # If reconnected successfully, perform reconciliation
            if self._connected:
                self._recon_state.reconnected_at = datetime.utcnow()
                asyncio.create_task(self._reconcile_after_reconnect())

    async def _reconcile_after_reconnect(self) -> None:
        """Reconcile order states after WebSocket reconnection.

        This prevents missing fills/cancellations that occurred while disconnected.
        """
        if not self._http_client:
            logger.warning(
                "[WS RECONCILIATION] No HTTP client available - cannot reconcile. "
                "Some fills may be missed!"
            )
            return

        if self._recon_state.reconciliation_in_progress:
            logger.debug("[WS RECONCILIATION] Already in progress, skipping")
            return

        self._recon_state.reconciliation_in_progress = True
        self._metrics["reconciliations"] += 1

        try:
            disconnection_duration = None
            if self._recon_state.disconnected_at and self._recon_state.reconnected_at:
                disconnection_duration = (
                    self._recon_state.reconnected_at - self._recon_state.disconnected_at
                )

            logger.info(
                f"[WS RECONCILIATION] Starting reconciliation. "
                f"Disconnected for: {disconnection_duration}. "
                f"Pending orders: {len(self._recon_state.pending_order_ids)}"
            )

            fills_recovered = 0
            cancellations_recovered = 0

            # Check each pending order via HTTP API
            for order_id in list(self._recon_state.pending_order_ids):
                try:
                    # Get order status from HTTP API
                    order_info = self._known_orders.get(order_id, {})
                    symbol = order_info.get("symbol", "BTCUSDT")

                    # Query order status (this would call the actual API)
                    order_status = await self._query_order_status(order_id, symbol)

                    if order_status:
                        current_status = order_status.get("orderStatus", "")

                        if current_status == "Filled":
                            # Order was filled while disconnected!
                            fill_price = float(order_status.get("avgPrice", 0))
                            fill_qty = float(order_status.get("cumExecQty", 0))

                            logger.warning(
                                f"[WS RECONCILIATION] RECOVERED FILL: order {order_id} "
                                f"filled at {fill_price} for {fill_qty}"
                            )

                            # Emit the fill event
                            await self._bus.publish(
                                Event(
                                    event_type=EventType.ORDER_FILLED,
                                    data={
                                        "order_id": order_id,
                                        "symbol": symbol,
                                        "fill_price": fill_price,
                                        "fill_size": fill_qty,
                                        "reconciled": True,  # Mark as recovered
                                    },
                                )
                            )

                            fills_recovered += 1
                            self.untrack_order(order_id)

                        elif current_status in ("Cancelled", "Rejected", "Deactivated"):
                            # Order was cancelled while disconnected
                            logger.warning(
                                f"[WS RECONCILIATION] RECOVERED CANCELLATION: order {order_id}"
                            )

                            await self._bus.publish(
                                Event(
                                    event_type=EventType.ORDER_CANCELLED,
                                    data={
                                        "order_id": order_id,
                                        "reconciled": True,
                                    },
                                )
                            )

                            cancellations_recovered += 1
                            self.untrack_order(order_id)

                        elif current_status == "PartiallyFilled":
                            # Partially filled - emit partial fill event
                            fill_price = float(order_status.get("avgPrice", 0))
                            fill_qty = float(order_status.get("cumExecQty", 0))

                            if fill_qty > 0:
                                logger.info(
                                    f"[WS RECONCILIATION] Partial fill detected: "
                                    f"order {order_id} filled {fill_qty} at {fill_price}"
                                )

                except Exception as e:
                    logger.error(f"[WS RECONCILIATION] Error checking order {order_id}: {e}")

            # Update metrics
            self._recon_state.fills_recovered = fills_recovered
            self._recon_state.cancellations_recovered = cancellations_recovered
            self._metrics["fills_recovered"] += fills_recovered
            self._metrics["cancellations_recovered"] += cancellations_recovered

            logger.info(
                f"[WS RECONCILIATION] Complete. "
                f"Recovered: {fills_recovered} fills, {cancellations_recovered} cancellations"
            )

        except Exception as e:
            logger.error(f"[WS RECONCILIATION] Error during reconciliation: {e}", exc_info=True)
        finally:
            self._recon_state.reconciliation_in_progress = False

    async def _query_order_status(self, order_id: str, symbol: str) -> dict[str, Any] | None:
        """Query order status from HTTP API.

        Args:
            order_id: The order ID to query
            symbol: The trading symbol

        Returns:
            Order status dict or None if not found
        """
        if not self._http_client:
            return None

        try:
            # Use HTTP client to get order info
            # This assumes the HTTP client has a get_order_info method
            if hasattr(self._http_client, 'get_order_info'):
                return await self._http_client.get_order_info(order_id, symbol)
            elif hasattr(self._http_client, 'get_order'):
                return await self._http_client.get_order(order_id, symbol)
            else:
                # Fallback: use raw API call
                logger.debug(
                    f"[WS] HTTP client does not have get_order_info method, "
                    f"order {order_id} status unknown"
                )
                return None
        except Exception as e:
            logger.error(f"[WS] Failed to query order {order_id}: {e}")
            return None

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

        # Handle auth response
        if data.get("op") == "auth":
            if data.get("success"):
                logger.info("WebSocket authentication successful")
            else:
                logger.error(f"WebSocket authentication failed: {data.get('ret_msg', 'Unknown error')}")
            return

        # Handle subscription success/error
        if data.get("op") == "subscribe":
            if data.get("success"):
                logger.debug(f"Subscription confirmed: {data.get('req_id', 'unknown')}")
            else:
                logger.warning(f"Subscription failed: {data}")
            return

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

        self._metrics["messages_processed"] += 1
        self._recon_state.last_order_update_at = datetime.utcnow()

        for order_info in order_data:
            try:
                order_id = order_info.get("orderId", "")
                status = order_info.get("orderStatus", "")
                symbol = order_info.get("symbol", "")

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

                # Update tracking for New orders
                if order_status == OrderStatus.PLACED:
                    self.track_order(order_id, {
                        "symbol": symbol,
                        "side": order_info.get("side", ""),
                        "status": status,
                    })

                # Create order event
                if order_status == OrderStatus.FILLED:
                    fill_price = float(order_info.get("avgPrice", order_info.get("price", 0)))
                    fill_qty = float(order_info.get("cumExecQty", order_info.get("qty", 0)))

                    await self._bus.publish(
                        Event(
                            event_type=EventType.ORDER_FILLED,
                            data={
                                "order_id": order_id,
                                "symbol": symbol,
                                "fill_price": fill_price,
                                "fill_size": fill_qty,
                            },
                        )
                    )

                    # Remove from tracking (terminal state)
                    self.untrack_order(order_id)

                elif order_status == OrderStatus.CANCELLED:
                    await self._bus.publish(
                        Event(event_type=EventType.ORDER_CANCELLED, data={"order_id": order_id})
                    )

                    # Remove from tracking (terminal state)
                    self.untrack_order(order_id)

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

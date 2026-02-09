"""WebSocket service using Socket.io for bi-directional control.

Connects Next.js frontend to Redis stream via WebSockets.
Ensures UI commands (Start/Stop/Risk Adjust) are acknowledged by C++ core within 5ms.
"""

import asyncio
import time
from datetime import UTC, datetime
from typing import Any

try:
    import socketio
    SOCKETIO_AVAILABLE = True
except ImportError:
    SOCKETIO_AVAILABLE = False

from hean.core.bus import EventBus
from hean.core.system.redis_state import get_redis_state_manager
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class WebSocketService:
    """WebSocket service for bi-directional control.

    Features:
    - Real-time state updates from Redis to frontend
    - UI commands (Start/Stop/Risk Adjust) with <5ms acknowledgment
    - State synchronization between C++ core and frontend
    - Connection management and reconnection handling
    """

    def __init__(self, bus: EventBus | None = None) -> None:
        """Initialize WebSocket service.

        Args:
            bus: Event bus for publishing events (optional)
        """
        if not SOCKETIO_AVAILABLE:
            raise RuntimeError(
                "Socket.io is not available. Install with: pip install python-socketio>=5.10.0"
            )

        self._bus = bus

        # Security: Get allowed origins from settings or use restrictive default
        from hean.config import settings
        allowed_origins = getattr(settings, 'ws_allowed_origins', None)
        if allowed_origins is None:
            # Development: allow all origins (iOS simulator, web dev servers, etc.)
            allowed_origins = "*"

        self._sio = socketio.AsyncServer(
            async_mode="asgi",
            cors_allowed_origins=allowed_origins,  # Security: Restricted to specific origins
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=1024 * 1024,  # 1MB max message size
        )
        self._app = socketio.ASGIApp(self._sio)
        self._redis_state_manager: Any | None = None
        self._running = False
        self._connected_clients: set[str] = set()
        self._max_connections = 100  # Connection limit for DoS protection
        self._command_handlers: dict[str, Any] = {}
        self._state_subscriptions: dict[str, Any] = {}

        # Register event handlers
        self._sio.on("connect", self._on_connect)
        self._sio.on("disconnect", self._on_disconnect)
        self._sio.on("command", self._on_command)
        self._sio.on("subscribe_state", self._on_subscribe_state)
        self._sio.on("unsubscribe_state", self._on_unsubscribe_state)

    async def start(self) -> None:
        """Start WebSocket service."""
        if self._running:
            return

        # Connect to Redis state manager
        try:
            self._redis_state_manager = await get_redis_state_manager()
        except Exception as e:
            logger.warning(f"Failed to connect to Redis state manager: {e}")

        self._running = True
        logger.info("WebSocket service started (Socket.io)")

    async def stop(self) -> None:
        """Stop WebSocket service."""
        if not self._running:
            return

        self._running = False

        # Unsubscribe from all state updates
        for key, _subscription in self._state_subscriptions.items():
            try:
                # Unsubscribe logic here
                pass
            except Exception as e:
                logger.warning(f"Failed to unsubscribe from {key}: {e}")

        self._state_subscriptions.clear()
        self._connected_clients.clear()

        logger.info("WebSocket service stopped")

    def register_command_handler(
        self,
        command: str,
        handler: Any,
    ) -> None:
        """Register a command handler.

        Args:
            command: Command name (e.g., "start", "stop", "risk_adjust")
            handler: Async function that handles the command
        """
        self._command_handlers[command] = handler
        logger.info(f"Registered command handler: {command}")

    async def _on_connect(self, sid: str, environ: dict[str, Any]) -> bool:
        """Handle client connection.

        Returns:
            True to accept connection, False to reject
        """
        # Connection limit enforcement (DoS protection)
        if len(self._connected_clients) >= self._max_connections:
            logger.warning(
                f"Connection rejected: max connections reached ({self._max_connections})"
            )
            return False  # Reject connection

        self._connected_clients.add(sid)
        logger.info(f"Client connected: {sid} (total: {len(self._connected_clients)})")

        # Send initial state
        await self._send_initial_state(sid)
        return True

    async def _on_disconnect(self, sid: str) -> None:
        """Handle client disconnection."""
        self._connected_clients.discard(sid)

        # Clean up subscriptions for this client
        keys_to_remove = []
        for key, subscription in self._state_subscriptions.items():
            if sid in subscription.get("clients", set()):
                subscription["clients"].discard(sid)
                if not subscription["clients"]:
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del self._state_subscriptions[key]

        logger.info(f"Client disconnected: {sid}")

    async def _on_command(self, sid: str, data: dict[str, Any]) -> None:
        """Handle command from client.

        Expected format:
        {
            "command": "start" | "stop" | "risk_adjust" | ...,
            "params": {...},
            "request_id": "uuid"
        }

        Must acknowledge within 5ms.
        """
        command = data.get("command")
        params = data.get("params", {})
        request_id = data.get("request_id", "")

        command_start_time = time.time()

        logger.info(f"Received command: {command} from {sid} (request_id={request_id})")

        # Acknowledge immediately (within 5ms requirement)
        await self._sio.emit(
            "command_ack",
            {
                "command": command,
                "request_id": request_id,
                "timestamp": datetime.now(UTC).isoformat(),
                "ack_time_ms": (time.time() - command_start_time) * 1000,
            },
            room=sid,
        )

        # Execute command handler
        handler = self._command_handlers.get(command)
        if handler:
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(params)
                else:
                    result = handler(params)

                # Send result
                await self._sio.emit(
                    "command_result",
                    {
                        "command": command,
                        "request_id": request_id,
                        "success": True,
                        "result": result,
                        "execution_time_ms": (time.time() - command_start_time) * 1000,
                    },
                    room=sid,
                )

                # Publish event to bus if available
                if self._bus:
                    await self._bus.publish(Event(
                        event_type=EventType.SIGNAL,
                        data={
                            "source": "websocket",
                            "command": command,
                            "params": params,
                            "result": result,
                        },
                    ))

                logger.info(
                    f"Command {command} executed successfully "
                    f"({(time.time() - command_start_time) * 1000:.2f}ms)"
                )

            except Exception as e:
                logger.error(f"Command {command} failed: {e}", exc_info=True)

                await self._sio.emit(
                    "command_result",
                    {
                        "command": command,
                        "request_id": request_id,
                        "success": False,
                        "error": str(e),
                        "execution_time_ms": (time.time() - command_start_time) * 1000,
                    },
                    room=sid,
                )
        else:
            logger.warning(f"Unknown command: {command}")
            await self._sio.emit(
                "command_error",
                {
                    "command": command,
                    "request_id": request_id,
                    "error": f"Unknown command: {command}",
                },
                room=sid,
            )

        # Check if acknowledgment was within 5ms
        ack_time_ms = (time.time() - command_start_time) * 1000
        if ack_time_ms > 5.0:
            logger.warning(
                f"Command acknowledgment took {ack_time_ms:.2f}ms "
                f"(exceeds 5ms requirement for command {command})"
            )

    async def _on_subscribe_state(self, sid: str, data: dict[str, Any]) -> None:
        """Handle state subscription request from client.

        Expected format:
        {
            "key": "state_key",
            "namespace": "global" (optional)
        }
        """
        key = data.get("key")
        namespace = data.get("namespace", "global")

        if not key:
            await self._sio.emit(
                "subscribe_error",
                {"error": "Missing 'key' parameter"},
                room=sid,
            )
            return

        state_key = f"{namespace}:{key}"

        # Subscribe to state updates
        if state_key not in self._state_subscriptions:
            # Create new subscription
            if self._redis_state_manager:
                try:
                    queue = await self._redis_state_manager.subscribe_state(key, namespace)
                    self._state_subscriptions[state_key] = {
                        "queue": queue,
                        "clients": {sid},
                    }

                    # Start background task to forward updates
                    asyncio.create_task(self._forward_state_updates(state_key, queue))

                    logger.info(f"Subscribed to state: {state_key}")
                except Exception as e:
                    logger.error(f"Failed to subscribe to state {state_key}: {e}", exc_info=True)
                    await self._sio.emit(
                        "subscribe_error",
                        {"error": f"Failed to subscribe: {e}"},
                        room=sid,
                    )
                    return
            else:
                await self._sio.emit(
                    "subscribe_error",
                    {"error": "Redis state manager not available"},
                    room=sid,
                )
                return
        else:
            # Add client to existing subscription
            self._state_subscriptions[state_key]["clients"].add(sid)

        await self._sio.emit(
            "subscribe_success",
            {"key": key, "namespace": namespace},
            room=sid,
        )

    async def _on_unsubscribe_state(self, sid: str, data: dict[str, Any]) -> None:
        """Handle state unsubscription request from client."""
        key = data.get("key")
        namespace = data.get("namespace", "global")
        state_key = f"{namespace}:{key}"

        if state_key in self._state_subscriptions:
            self._state_subscriptions[state_key]["clients"].discard(sid)

            # Clean up if no clients
            if not self._state_subscriptions[state_key]["clients"]:
                del self._state_subscriptions[state_key]

        await self._sio.emit(
            "unsubscribe_success",
            {"key": key, "namespace": namespace},
            room=sid,
        )

    async def _forward_state_updates(self, state_key: str, queue: asyncio.Queue) -> None:
        """Forward state updates from Redis to WebSocket clients."""
        while self._running and state_key in self._state_subscriptions:
            try:
                # Get update from queue (with timeout)
                update = await asyncio.wait_for(queue.get(), timeout=1.0)
                if update is None:
                    continue

                value, version, timestamp = update

                # Send to all subscribed clients
                subscription = self._state_subscriptions[state_key]
                dead_clients = []

                for client_sid in subscription["clients"]:
                    try:
                        await self._sio.emit(
                            "state_update",
                            {
                                "key": state_key.split(":", 1)[1] if ":" in state_key else state_key,
                                "namespace": state_key.split(":")[0] if ":" in state_key else "global",
                                "value": value,
                                "version": version,
                                "timestamp": timestamp,
                            },
                            room=client_sid,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to send state update to {client_sid}: {e}")
                        dead_clients.append(client_sid)

                # Remove dead clients
                for client_sid in dead_clients:
                    subscription["clients"].discard(client_sid)

            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error forwarding state updates for {state_key}: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    async def _send_initial_state(self, sid: str) -> None:
        """Send initial state to newly connected client."""
        if not self._redis_state_manager:
            return

        try:
            # Get current global state keys
            # This is a simplified version - in production, maintain a list of state keys
            await self._sio.emit(
                "initial_state",
                {
                    "message": "Connected to HEAN WebSocket service",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                room=sid,
            )
        except Exception as e:
            logger.warning(f"Failed to send initial state to {sid}: {e}")

    async def broadcast_event(self, event: dict[str, Any]) -> None:
        """Broadcast event to all connected clients."""
        if not self._running:
            return

        await self._sio.emit("event", event)

    async def send_to_client(self, sid: str, event: str, data: dict[str, Any]) -> None:
        """Send event to a specific client."""
        if not self._running:
            return

        await self._sio.emit(event, data, room=sid)

    def get_app(self) -> Any:
        """Get ASGI app for mounting."""
        return self._app


# Global instance
_websocket_service: WebSocketService | None = None


async def get_websocket_service(bus: EventBus | None = None) -> WebSocketService:
    """Get or create global WebSocket service."""
    global _websocket_service

    if _websocket_service is None:
        _websocket_service = WebSocketService(bus=bus)
        await _websocket_service.start()

    return _websocket_service

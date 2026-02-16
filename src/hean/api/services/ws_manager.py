"""WebSocket connection manager for HEAN API.

Manages WebSocket connections, topic-based pub/sub, and message delivery.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket

from hean.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections and topic subscriptions."""

    def __init__(self) -> None:
        self.active_connections: dict[str, WebSocket] = {}
        self.topic_subscriptions: dict[str, set[str]] = {}  # topic -> set of connection_ids
        self.connection_topics: dict[str, set[str]] = {}  # connection_id -> set of topics
        self.redis_pubsub_tasks: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        *,
        initial_state_provider: Any = None,
    ) -> None:
        """Accept a WebSocket connection and send initial state.

        Args:
            websocket: The WebSocket to accept.
            connection_id: Unique ID for this connection.
            initial_state_provider: Optional async callable that returns initial state dict.
        """
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_topics[connection_id] = set()
        logger.info(f"WebSocket client connected: {connection_id}")

        # Send initial state if provider is given
        if initial_state_provider:
            try:
                state = await initial_state_provider()
                await self.send_to_connection(connection_id, {
                    "topic": "system_status",
                    "data": state,
                    "timestamp": datetime.now(UTC).isoformat(),
                })
            except Exception as e:
                logger.warning(f"Failed to send initial state to {connection_id}: {e}")

    async def disconnect(self, connection_id: str) -> None:
        """Handle disconnection and cleanup."""
        if connection_id == "all":
            for cid in list(self.active_connections.keys()):
                await self.disconnect(cid)
            return

        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        # Unsubscribe from all topics
        if connection_id in self.connection_topics:
            for topic in list(self.connection_topics[connection_id]):
                self.unsubscribe(connection_id, topic)
            del self.connection_topics[connection_id]

        # Cancel Redis pubsub task if exists
        if connection_id in self.redis_pubsub_tasks:
            task = self.redis_pubsub_tasks[connection_id]
            task.cancel()
            del self.redis_pubsub_tasks[connection_id]

        logger.info(f"WebSocket client disconnected: {connection_id}")

    def subscribe(self, connection_id: str, topic: str) -> None:
        """Subscribe a connection to a topic."""
        if topic not in self.topic_subscriptions:
            self.topic_subscriptions[topic] = set()
        self.topic_subscriptions[topic].add(connection_id)

        if connection_id not in self.connection_topics:
            self.connection_topics[connection_id] = set()
        self.connection_topics[connection_id].add(topic)

        logger.info(f"Connection {connection_id} subscribed to topic: {topic}")

    def unsubscribe(self, connection_id: str, topic: str) -> None:
        """Unsubscribe a connection from a topic."""
        if topic in self.topic_subscriptions:
            self.topic_subscriptions[topic].discard(connection_id)
            if not self.topic_subscriptions[topic]:
                del self.topic_subscriptions[topic]

        if connection_id in self.connection_topics:
            self.connection_topics[connection_id].discard(topic)

        logger.debug(f"Connection {connection_id} unsubscribed from topic: {topic}")

    async def broadcast_to_topic(self, topic: str, data: dict[str, Any]) -> None:
        """Broadcast data to all subscribers of a topic."""
        if topic not in self.topic_subscriptions:
            return

        disconnected = []
        message = {
            "topic": topic,
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        for connection_id in list(self.topic_subscriptions[topic]):
            if connection_id in self.active_connections:
                try:
                    websocket = self.active_connections[connection_id]
                    await asyncio.wait_for(websocket.send_json(message), timeout=5.0)
                except TimeoutError:
                    logger.warning(f"Send timeout for {connection_id}, marking as disconnected")
                    disconnected.append(connection_id)
                except Exception as e:
                    logger.debug(f"Failed to send to {connection_id}: {e}")
                    disconnected.append(connection_id)
            else:
                disconnected.append(connection_id)

        # Clean up disconnected clients
        for connection_id in disconnected:
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
            self.unsubscribe(connection_id, topic)
            if connection_id in self.connection_topics:
                for t in list(self.connection_topics[connection_id]):
                    if t != topic:
                        self.unsubscribe(connection_id, t)
                del self.connection_topics[connection_id]

    async def send_to_connection(self, connection_id: str, data: dict[str, Any]) -> None:
        """Send data to a specific connection."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await asyncio.wait_for(websocket.send_json(data), timeout=5.0)
            except (TimeoutError, Exception) as e:
                logger.warning(f"Failed to send to {connection_id}: {e}, marking for cleanup")
                await self.disconnect(connection_id)

    def active_count(self) -> int:
        """Return active WebSocket client count."""
        return len(self.active_connections)

"""Redis Streams for persistent event storage and replay."""

import asyncio
import json
from datetime import datetime
from typing import Any, AsyncIterator, Optional

import redis.asyncio as aioredis

from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class RedisEventStream:
    """Redis Streams implementation for persistent event storage.

    Features:
    - Event persistence for replay/audit
    - Multi-consumer groups
    - At-least-once delivery
    - Automatic trimming to prevent memory issues
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_prefix: str = "hean:events",
        max_len: int = 100000,  # Max events per stream
        consumer_group: str = "trading-system",
    ) -> None:
        """Initialize Redis Streams.

        Args:
            redis_url: Redis connection URL
            stream_prefix: Prefix for stream keys
            max_len: Maximum stream length (older events auto-trimmed)
            consumer_group: Consumer group name
        """
        self._redis_url = redis_url
        self._stream_prefix = stream_prefix
        self._max_len = max_len
        self._consumer_group = consumer_group
        self._redis: Optional[aioredis.Redis] = None
        self._running = False

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            self._redis = await aioredis.from_url(
                self._redis_url, encoding="utf-8", decode_responses=True
            )
            await self._redis.ping()
            logger.info(f"Connected to Redis at {self._redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}", exc_info=True)
            raise

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis:
            await self._redis.close()
            logger.info("Disconnected from Redis")

    async def publish(self, event: Event) -> str:
        """Publish event to Redis Stream.

        Args:
            event: Event to publish

        Returns:
            Message ID from Redis
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        stream_key = f"{self._stream_prefix}:{event.event_type.value}"

        # Serialize event data
        message = {
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "data": json.dumps(event.data),
        }

        try:
            # XADD with MAXLEN to auto-trim
            message_id = await self._redis.xadd(
                stream_key, message, maxlen=self._max_len, approximate=True
            )
            logger.debug(f"Published {event.event_type} to Redis: {message_id}")
            return message_id

        except Exception as e:
            logger.error(f"Failed to publish event to Redis: {e}", exc_info=True)
            raise

    async def create_consumer_group(
        self, event_type: EventType, group_name: Optional[str] = None
    ) -> None:
        """Create consumer group for event type.

        Args:
            event_type: Event type to consume
            group_name: Optional group name (default: self._consumer_group)
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        stream_key = f"{self._stream_prefix}:{event_type.value}"
        group = group_name or self._consumer_group

        try:
            # Create group starting from beginning ($)
            await self._redis.xgroup_create(
                stream_key, group, id="0", mkstream=True
            )
            logger.info(f"Created consumer group {group} for {event_type}")
        except aioredis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.debug(f"Consumer group {group} already exists for {event_type}")
            else:
                logger.error(f"Failed to create consumer group: {e}", exc_info=True)
                raise

    async def consume(
        self,
        event_type: EventType,
        consumer_name: str,
        block_ms: int = 5000,
        count: int = 10,
        group_name: Optional[str] = None,
    ) -> AsyncIterator[tuple[str, Event]]:
        """Consume events from Redis Stream.

        Args:
            event_type: Event type to consume
            consumer_name: Unique consumer name
            block_ms: Block for N milliseconds if no messages
            count: Max messages to read per batch
            group_name: Optional group name (default: self._consumer_group)

        Yields:
            Tuple of (message_id, event)
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        stream_key = f"{self._stream_prefix}:{event_type.value}"
        group = group_name or self._consumer_group

        # Ensure consumer group exists
        await self.create_consumer_group(event_type, group)

        self._running = True
        while self._running:
            try:
                # XREADGROUP to read from consumer group
                messages = await self._redis.xreadgroup(
                    groupname=group,
                    consumername=consumer_name,
                    streams={stream_key: ">"},
                    count=count,
                    block=block_ms,
                )

                if not messages:
                    continue

                # Process messages
                for stream_name, stream_messages in messages:
                    for message_id, message_data in stream_messages:
                        try:
                            # Deserialize event
                            event = Event(
                                event_type=EventType(message_data["event_type"]),
                                timestamp=datetime.fromisoformat(
                                    message_data["timestamp"]
                                ),
                                data=json.loads(message_data["data"]),
                            )

                            yield (message_id, event)

                        except Exception as e:
                            logger.error(
                                f"Failed to deserialize event {message_id}: {e}",
                                exc_info=True,
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error consuming events: {e}", exc_info=True)
                await asyncio.sleep(1)  # Backoff on error

        self._running = False

    async def acknowledge(self, event_type: EventType, message_id: str) -> None:
        """Acknowledge message processing.

        Args:
            event_type: Event type
            message_id: Message ID to acknowledge
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        stream_key = f"{self._stream_prefix}:{event_type.value}"

        try:
            await self._redis.xack(stream_key, self._consumer_group, message_id)
            logger.debug(f"Acknowledged {message_id} from {event_type}")
        except Exception as e:
            logger.error(f"Failed to acknowledge message: {e}", exc_info=True)

    async def replay_events(
        self, event_type: EventType, start_id: str = "0", end_id: str = "+"
    ) -> AsyncIterator[Event]:
        """Replay events from stream (for debugging/audit).

        Args:
            event_type: Event type to replay
            start_id: Start message ID (default: beginning)
            end_id: End message ID (default: end)

        Yields:
            Events from stream
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        stream_key = f"{self._stream_prefix}:{event_type.value}"

        try:
            # XRANGE to read messages
            messages = await self._redis.xrange(stream_key, start_id, end_id)

            for message_id, message_data in messages:
                try:
                    event = Event(
                        event_type=EventType(message_data["event_type"]),
                        timestamp=datetime.fromisoformat(message_data["timestamp"]),
                        data=json.loads(message_data["data"]),
                    )
                    yield event

                except Exception as e:
                    logger.error(
                        f"Failed to deserialize event {message_id}: {e}",
                        exc_info=True,
                    )

        except Exception as e:
            logger.error(f"Error replaying events: {e}", exc_info=True)
            raise

    async def get_stream_info(self, event_type: EventType) -> dict[str, Any]:
        """Get stream information.

        Args:
            event_type: Event type

        Returns:
            Stream info dict
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        stream_key = f"{self._stream_prefix}:{event_type.value}"

        try:
            info = await self._redis.xinfo_stream(stream_key)
            return {
                "length": info["length"],
                "first_entry": info["first-entry"],
                "last_entry": info["last-entry"],
                "groups": info["groups"],
            }
        except Exception as e:
            logger.error(f"Failed to get stream info: {e}", exc_info=True)
            return {}

    async def trim_stream(self, event_type: EventType, max_len: int) -> int:
        """Manually trim stream to max length.

        Args:
            event_type: Event type
            max_len: Maximum length

        Returns:
            Number of entries removed
        """
        if not self._redis:
            raise RuntimeError("Not connected to Redis")

        stream_key = f"{self._stream_prefix}:{event_type.value}"

        try:
            removed = await self._redis.xtrim(stream_key, maxlen=max_len, approximate=True)
            logger.info(f"Trimmed {removed} entries from {event_type}")
            return removed
        except Exception as e:
            logger.error(f"Failed to trim stream: {e}", exc_info=True)
            raise


class HybridEventBus:
    """Hybrid EventBus combining in-memory (fast) and Redis Streams (persistent).

    Strategy:
    - High-frequency events (TICK): In-memory only
    - Important events (SIGNAL, ORDER_*, PNL): Both in-memory + Redis
    - Audit events (ERROR): Redis only
    """

    def __init__(
        self,
        in_memory_bus: Any,  # EventBus from core.bus
        redis_stream: RedisEventStream,
    ) -> None:
        """Initialize hybrid bus.

        Args:
            in_memory_bus: Fast in-memory event bus
            redis_stream: Redis Streams for persistence
        """
        self.in_memory = in_memory_bus
        self.redis = redis_stream

        # Define which events to persist
        self._persist_events = {
            EventType.SIGNAL,
            EventType.ORDER_REQUEST,
            EventType.ORDER_PLACED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
            EventType.ORDER_REJECTED,
            EventType.PNL_UPDATE,
            EventType.ERROR,
            EventType.KILLSWITCH_TRIGGERED,
        }

    async def start(self) -> None:
        """Start both buses."""
        await self.in_memory.start()
        await self.redis.connect()
        logger.info("Hybrid EventBus started")

    async def stop(self) -> None:
        """Stop both buses."""
        await self.in_memory.stop()
        await self.redis.disconnect()
        logger.info("Hybrid EventBus stopped")

    async def publish(self, event: Event) -> None:
        """Publish event to appropriate bus(es).

        Args:
            event: Event to publish
        """
        # Always publish to in-memory bus (fast path)
        await self.in_memory.publish(event)

        # Persist important events to Redis
        if event.event_type in self._persist_events:
            try:
                await self.redis.publish(event)
            except Exception as e:
                logger.error(f"Failed to persist event to Redis: {e}", exc_info=True)
                # Don't fail on Redis errors (graceful degradation)

    def subscribe(self, event_type: EventType, handler: Any) -> None:
        """Subscribe to in-memory events.

        Args:
            event_type: Event type
            handler: Event handler function
        """
        self.in_memory.subscribe(event_type, handler)

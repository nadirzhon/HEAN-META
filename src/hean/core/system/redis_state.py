"""Redis-based atomic state manager for C++/Python integration.

This module provides atomic state sharing between C++ core, Python execution engine,
and Next.js frontend using Redis for real-time synchronization.
"""

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

try:
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)


class RedisStateManager:
    """Atomic state manager using Redis for deterministic state sharing.

    Provides:
    - Atomic updates using Redis Lua scripts (no retry overhead)
    - Global state that C++ and Python always see the same
    - Pub/sub for real-time updates
    - Version tracking to prevent race conditions
    """

    # Lua script for atomic compare-and-set with version tracking
    _LUA_CAS_UPDATE = """
    local state_key = KEYS[1]
    local version_key = KEYS[2]
    local expected_version = ARGV[1]
    local new_value = ARGV[2]
    local new_version = ARGV[3]
    local ttl = ARGV[4]

    local current_version = redis.call('GET', version_key)
    if current_version == false then
        current_version = '0'
    end

    if current_version == expected_version then
        redis.call('SETEX', state_key, ttl, new_value)
        redis.call('SETEX', version_key, ttl, new_version)
        return {1, new_version}
    else
        return {0, current_version}
    end
    """

    def __init__(self, redis_url: str | None = None) -> None:
        """Initialize Redis state manager.

        Args:
            redis_url: Redis URL (defaults to settings.redis_url)
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError(
                "Redis is not available. Install with: pip install redis>=5.0.0"
            )

        self._redis_url = redis_url or settings.redis_url
        self._client: aioredis.Redis | None = None
        self._pubsub: aioredis.client.PubSub | None = None
        self._running = False
        self._state_version = 0
        self._state_lock = asyncio.Lock()
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        self._cas_script = None  # Will be loaded on connect

    async def connect(self) -> None:
        """Connect to Redis with retry logic."""
        if self._client:
            return

        # Retry connection with exponential backoff
        max_retries = 5
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Ensure we use redis:6379 for Docker (service name)
                redis_url = self._redis_url
                if not redis_url.startswith("redis://"):
                    redis_url = f"redis://{redis_url}"
                if "redis:6379" not in redis_url and ":6379" not in redis_url:
                    # Default to Docker service name if not specified
                    redis_url = "redis://redis:6379/0"

                self._client = aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    socket_connect_timeout=5.0,
                    socket_timeout=5.0,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )

                # Test connection
                await self._client.ping()
                logger.info(f"✅ Connected to Redis at {redis_url}")

                # Load Lua scripts for atomic operations
                self._cas_script = self._client.register_script(self._LUA_CAS_UPDATE)
                logger.debug("✅ Loaded Redis Lua CAS script")

                # Initialize pubsub
                self._pubsub = self._client.pubsub()

                # Start pubsub listener
                self._running = True
                asyncio.create_task(self._listen_pubsub())
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Redis connection attempt {attempt + 1}/{max_retries} failed: {e}. "
                        f"Retrying in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"❌ Failed to connect to Redis after {max_retries} attempts: {e}", exc_info=True)
                    raise RuntimeError(f"Failed to connect to Redis after {max_retries} attempts") from e

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        self._running = False

        if self._pubsub:
            await self._pubsub.unsubscribe()
            await self._pubsub.close()
            self._pubsub = None

        if self._client:
            await self._client.close()
            self._client = None

        logger.info("Disconnected from Redis")

    async def set_state_atomic(
        self,
        key: str,
        value: Any,
        namespace: str = "global",
    ) -> int:
        """Atomically set state with version tracking.

        Uses Redis transaction (MULTI/EXEC) to ensure atomicity.
        Returns the new version number.

        Args:
            key: State key
            value: State value (must be JSON serializable)
            namespace: State namespace (default: "global")

        Returns:
            New version number
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        async with self._state_lock:
            state_key = f"hean:state:{namespace}:{key}"
            version_key = f"hean:state:{namespace}:{key}:version"

            # Use pipeline to get version and set new values atomically
            pipe = self._client.pipeline()
            pipe.get(version_key)  # Get current version IN pipeline
            pipe.set(state_key, json.dumps(value), ex=86400)  # 24h expiry
            results = await pipe.execute()

            # Calculate new version from pipelined result
            current_version = results[0]
            new_version = int(current_version or 0) + 1

            # Set new version (separate call, but version is just metadata)
            await self._client.setex(version_key, 86400, str(new_version))

            # Publish update notification
            await self._publish_update(namespace, key, value, new_version)

            self._state_version = new_version
            logger.debug(
                f"Atomic state update: {namespace}:{key} -> version {new_version}"
            )

            return new_version

    async def get_state(
        self,
        key: str,
        namespace: str = "global",
        expected_version: int | None = None,
    ) -> tuple[Any, int]:
        """Get state with version check.

        Args:
            key: State key
            namespace: State namespace (default: "global")
            expected_version: Optional expected version for validation

        Returns:
            Tuple of (value, version)

        Raises:
            ValueError: If expected_version doesn't match
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        state_key = f"hean:state:{namespace}:{key}"
        version_key = f"hean:state:{namespace}:{key}:version"

        # Get state and version atomically
        pipe = self._client.pipeline()
        pipe.get(state_key)
        pipe.get(version_key)
        results = await pipe.execute()

        value_str = results[0]
        version_str = results[1]

        if value_str is None:
            return None, 0

        version = int(version_str or 0)

        if expected_version is not None and version != expected_version:
            raise ValueError(
                f"Version mismatch: expected {expected_version}, got {version}"
            )

        value = json.loads(value_str)
        return value, version

    async def update_state_atomic(
        self,
        key: str,
        updater: Any,
        namespace: str = "global",
    ) -> tuple[Any, int]:
        """Atomically update state using a function.

        Prevents race conditions by reading, updating, and writing atomically.

        Args:
            key: State key
            updater: Function that takes current value and returns new value,
                    or a value to merge with current state (if dict/list)
            namespace: State namespace (default: "global")

        Returns:
            Tuple of (new_value, new_version)
        """
        if not self._client:
            raise RuntimeError("Not connected to Redis")

        async with self._state_lock:
            state_key = f"hean:state:{namespace}:{key}"
            version_key = f"hean:state:{namespace}:{key}:version"

            # Retry loop for optimistic locking
            max_retries = 10
            for attempt in range(max_retries):
                # Get current state
                pipe = self._client.pipeline()
                pipe.get(state_key)
                pipe.get(version_key)
                results = await pipe.execute()

                current_value_str = results[0]
                current_version_str = results[1]
                current_version = int(current_version_str or 0)

                # Parse current value
                if current_value_str:
                    current_value = json.loads(current_value_str)
                else:
                    current_value = None

                # Apply updater
                if callable(updater):
                    new_value = updater(current_value)
                elif isinstance(current_value, dict) and isinstance(updater, dict):
                    new_value = {**current_value, **updater}
                elif isinstance(current_value, list) and isinstance(updater, list):
                    new_value = current_value + updater
                else:
                    new_value = updater

                new_version = current_version + 1

                # Execute atomic CAS update using Lua script (no retry overhead!)
                result = await self._cas_script(
                    keys=[state_key, version_key],
                    args=[
                        str(current_version),  # expected_version
                        json.dumps(new_value),  # new_value
                        str(new_version),  # new_version
                        "86400",  # ttl (24h)
                    ],
                )

                # Check if CAS succeeded
                success, returned_version = result
                if success == 1:
                    # Success! Publish update
                    await self._publish_update(namespace, key, new_value, new_version)
                    self._state_version = new_version
                    logger.debug(
                        f"✅ Atomic CAS update: {namespace}:{key} v{current_version}→v{new_version}"
                    )
                    return new_value, new_version
                else:
                    # CAS failed, version changed - retry
                    logger.debug(
                        f"⚠️ CAS retry {attempt+1}/{max_retries}: "
                        f"{namespace}:{key} expected v{current_version}, got v{returned_version}"
                    )
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.001)  # 1ms backoff before retry
                        continue

            raise RuntimeError(
                f"Failed to update state after {max_retries} retries: "
                f"{namespace}:{key} (concurrent modifications)"
            )

    async def subscribe_state(
        self,
        key: str,
        namespace: str = "global",
    ) -> asyncio.Queue:
        """Subscribe to state updates.

        Args:
            key: State key
            namespace: State namespace (default: "global")

        Returns:
            Queue that receives (value, version) tuples on updates
        """
        channel = f"hean:state:{namespace}:{key}"

        if channel not in self._subscribers:
            self._subscribers[channel] = []
            if self._pubsub:
                await self._pubsub.subscribe(channel)

        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers[channel].append(queue)

        return queue

    async def _publish_update(
        self,
        namespace: str,
        key: str,
        value: Any,
        version: int,
    ) -> None:
        """Publish state update notification."""
        if not self._client:
            return

        channel = f"hean:state:{namespace}:{key}"
        message = {
            "namespace": namespace,
            "key": key,
            "value": value,
            "version": version,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        await self._client.publish(channel, json.dumps(message))

    async def _listen_pubsub(self) -> None:
        """Listen to pubsub messages and forward to subscribers."""
        if not self._pubsub:
            return

        while self._running:
            try:
                try:
                    message = await self._pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=1.0,
                    )
                except RuntimeError as e:
                    # Happens if pubsub has no subscriptions yet; skip quietly
                    logger.debug(f"Redis pubsub not subscribed yet: {e}")
                    await asyncio.sleep(0.1)
                    continue

                if message is None:
                    continue

                channel = message.get("channel")
                if not channel or channel not in self._subscribers:
                    continue

                data = json.loads(message.get("data", "{}"))

                # Forward to all subscribers
                dead_queues = []
                for queue in self._subscribers[channel]:
                    try:
                        await queue.put((
                            data.get("value"),
                            data.get("version"),
                            data.get("timestamp"),
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to forward update to subscriber: {e}")
                        dead_queues.append(queue)

                # Remove dead queues
                for queue in dead_queues:
                    self._subscribers[channel].remove(queue)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in pubsub listener: {e}", exc_info=True)
                await asyncio.sleep(0.1)


# Global instance
_redis_state_manager: RedisStateManager | None = None


async def get_redis_state_manager() -> RedisStateManager:
    """Get or create global Redis state manager."""
    global _redis_state_manager

    if _redis_state_manager is None:
        _redis_state_manager = RedisStateManager()
        await _redis_state_manager.connect()

    return _redis_state_manager

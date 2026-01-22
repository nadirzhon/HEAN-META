"""Redis-based atomic state manager for C++/Python integration.

This module provides atomic state sharing between C++ core, Python execution engine,
and Next.js frontend using Redis for real-time synchronization.
"""

import asyncio
import json
import time
from typing import Any, Optional
from datetime import datetime, timezone

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
    - Atomic updates using Redis transactions (MULTI/EXEC)
    - Global state that C++ and Python always see the same
    - Pub/sub for real-time updates
    - Version tracking to prevent race conditions
    """
    
    def __init__(self, redis_url: Optional[str] = None) -> None:
        """Initialize Redis state manager.
        
        Args:
            redis_url: Redis URL (defaults to settings.redis_url)
        """
        if not REDIS_AVAILABLE:
            raise RuntimeError(
                "Redis is not available. Install with: pip install redis>=5.0.0"
            )
        
        self._redis_url = redis_url or settings.redis_url
        self._client: Optional[aioredis.Redis] = None
        self._pubsub: Optional[aioredis.client.PubSub] = None
        self._running = False
        self._state_version = 0
        self._state_lock = asyncio.Lock()
        self._subscribers: dict[str, list[asyncio.Queue]] = {}
        
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
                    raise
    
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
            
            # Use Redis transaction for atomic update
            pipe = self._client.pipeline()
            
            # Get current version
            current_version = await self._client.get(version_key)
            new_version = int(current_version or 0) + 1
            
            # Set state and version atomically
            pipe.set(state_key, json.dumps(value), ex=86400)  # 24h expiry
            pipe.set(version_key, str(new_version), ex=86400)
            
            # Execute transaction
            await pipe.execute()
            
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
        expected_version: Optional[int] = None,
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
                
                # Try to update atomically using version check
                pipe = self._client.pipeline()
                pipe.watch(state_key, version_key)  # Watch for changes
                
                # Check version hasn't changed
                check_pipe = self._client.pipeline()
                check_pipe.get(version_key)
                check_results = await check_pipe.execute()
                
                if check_results[0] != current_version_str:
                    # Version changed, retry
                    continue
                
                # Atomic update
                pipe.multi()
                pipe.set(state_key, json.dumps(new_value), ex=86400)
                pipe.set(version_key, str(new_version), ex=86400)
                
                try:
                    await pipe.execute()
                    
                    # Publish update
                    await self._publish_update(namespace, key, new_value, new_version)
                    
                    self._state_version = new_version
                    return new_value, new_version
                    
                except aioredis.WatchError:
                    # Concurrent modification, retry
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.001)  # 1ms backoff
                        continue
                    raise RuntimeError(
                        f"Failed to update state after {max_retries} retries"
                    )
            
            raise RuntimeError(f"Failed to update state after {max_retries} retries")
    
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
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
_redis_state_manager: Optional[RedisStateManager] = None


async def get_redis_state_manager() -> RedisStateManager:
    """Get or create global Redis state manager."""
    global _redis_state_manager
    
    if _redis_state_manager is None:
        _redis_state_manager = RedisStateManager()
        await _redis_state_manager.connect()
    
    return _redis_state_manager

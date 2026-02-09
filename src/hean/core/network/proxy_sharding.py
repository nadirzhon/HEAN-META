"""
Phase 19: API Proxy Sharding System
Implements rotating proxy manager that cycles through residential and SOCKS5 proxies.
Distributes WebSocket traffic to stay below 20% of exchange's rate-limit per IP.

The system rotates proxies intelligently based on:
- Rate limit usage per proxy
- Connection health
- Latency to exchange
- Proxy rotation schedule
"""

import asyncio
import random
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum

from hean.logging import get_logger

logger = get_logger(__name__)


class ProxyType(IntEnum):
    """Proxy type."""
    RESIDENTIAL = 1
    SOCKS5 = 2
    HTTP = 3
    HTTPS = 4


@dataclass
class ProxyConfig:
    """Configuration for a proxy."""
    id: str
    type: ProxyType
    host: str
    port: int
    username: str | None = None
    password: str | None = None
    enabled: bool = True
    max_rate_limit_per_second: int = 100  # Exchange rate limit (requests/second)
    safety_threshold: float = 0.2  # Use only 20% of rate limit per proxy

    def get_url(self) -> str:
        """Get proxy URL."""
        if self.username and self.password:
            return f"{self.type.name.lower()}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.type.name.lower()}://{self.host}:{self.port}"


@dataclass
class ProxyState:
    """State of a proxy connection."""
    proxy: ProxyConfig
    is_active: bool = False
    is_healthy: bool = True
    request_count: int = 0
    error_count: int = 0
    last_request_time: float = 0.0
    last_error_time: float = 0.0
    latency_ms: float = float('inf')
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=20))
    rate_limit_usage: float = 0.0  # 0.0 to 1.0 (percentage of rate limit used)
    rotation_cooldown_until: float = 0.0  # Timestamp when proxy can be used again
    consecutive_failures: int = 0
    total_requests: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0

    def update_request(self, success: bool = True, latency_ms: float | None = None) -> None:
        """Update proxy state after a request."""
        current_time = time.time()
        self.last_request_time = current_time
        self.total_requests += 1

        if success:
            self.consecutive_failures = 0
            if latency_ms is not None:
                self.latency_ms = latency_ms
                self.recent_latencies.append(latency_ms)
                self.latency_ms = statistics.mean(self.recent_latencies) if self.recent_latencies else float('inf')
            self.request_count += 1
        else:
            self.error_count += 1
            self.last_error_time = current_time
            self.consecutive_failures += 1
            self.is_healthy = self.consecutive_failures < 3

        # Calculate rate limit usage (requests per second over last minute)
        self._calculate_rate_limit_usage()

    def _calculate_rate_limit_usage(self) -> None:
        """Calculate rate limit usage percentage."""
        # Simple calculation: request count / (max_rate * safety_threshold * time_window)
        # For now, use a simplified rolling window
        max_allowed_rate = self.proxy.max_rate_limit_per_second * self.proxy.safety_threshold

        # Estimate based on recent activity (simplified)
        if self.request_count > 0:
            # Assume average request rate over recent period
            # In production, use a proper sliding window
            self.rate_limit_usage = min(1.0, self.request_count / (max_allowed_rate * 60))
        else:
            self.rate_limit_usage = 0.0

    def is_available(self) -> bool:
        """Check if proxy is available for use."""
        if not self.proxy.enabled:
            return False

        if not self.is_healthy:
            return False

        if time.time() < self.rotation_cooldown_until:
            return False

        # Check if rate limit is exceeded
        if self.rate_limit_usage >= 0.95:  # 95% threshold
            return False

        return True

    def get_score(self) -> float:
        """Get proxy score for selection (higher = better)."""
        if not self.is_available():
            return -1.0

        # Score based on:
        # - Low latency (better)
        # - Low rate limit usage (better - more headroom)
        # - High success rate (better)
        # - Low consecutive failures (better)

        latency_score = 1.0 / (1.0 + self.latency_ms / 100.0)  # Normalize latency
        rate_score = 1.0 - self.rate_limit_usage  # More headroom = better
        health_score = 1.0 / (1.0 + self.consecutive_failures)

        total_requests = self.total_requests
        success_rate = (total_requests - self.error_count) / total_requests if total_requests > 0 else 0.5

        score = (
            latency_score * 0.3 +
            rate_score * 0.3 +
            health_score * 0.2 +
            success_rate * 0.2
        )

        return score


class ProxyShardingManager:
    """
    API Proxy Sharding Manager.

    Rotates through a pool of residential and SOCKS5 proxies to distribute
    WebSocket traffic and stay below 20% of exchange rate-limit per IP.
    """

    def __init__(
        self,
        proxies: list[ProxyConfig],
        rotation_interval_ms: float = 30000.0,  # Rotate every 30s
        health_check_interval_ms: float = 5000.0,  # Health check every 5s
        cooldown_duration_ms: float = 60000.0,  # 1min cooldown after rotation
    ):
        """Initialize the Proxy Sharding Manager.

        Args:
            proxies: List of proxy configurations
            rotation_interval_ms: Interval between proxy rotations (ms)
            health_check_interval_ms: Interval for health checks (ms)
            cooldown_duration_ms: Cooldown duration after rotating away from a proxy (ms)
        """
        self._proxies = proxies
        self._rotation_interval_ms = rotation_interval_ms
        self._health_check_interval_ms = health_check_interval_ms
        self._cooldown_duration_ms = cooldown_duration_ms

        # Proxy states
        self._proxy_states: dict[str, ProxyState] = {
            proxy.id: ProxyState(proxy=proxy) for proxy in proxies
        }

        # Current active proxies (one per connection type)
        self._active_proxy_id: str | None = None
        self._last_rotation_time = time.time()

        # Rotation lock (prevent concurrent rotations)
        self._rotation_lock = asyncio.Lock()

        # Running state
        self._running = False
        self._rotation_task: asyncio.Task | None = None
        self._health_check_task: asyncio.Task | None = None

        # Statistics
        self._total_requests = 0
        self._total_rotations = 0
        self._rate_limit_violations = 0

    async def start(self) -> None:
        """Start the proxy sharding manager."""
        if self._running:
            return

        self._running = True

        # Select initial proxy
        await self._select_best_proxy()

        # Start rotation task
        self._rotation_task = asyncio.create_task(self._rotation_loop())

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            f"Proxy Sharding Manager started with {len(self._proxies)} proxies "
            f"(rotation_interval={self._rotation_interval_ms}ms)"
        )

    async def stop(self) -> None:
        """Stop the proxy sharding manager."""
        if not self._running:
            return

        self._running = False

        # Cancel tasks
        if self._rotation_task:
            self._rotation_task.cancel()
            try:
                await self._rotation_task
            except asyncio.CancelledError:
                pass

        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        logger.info("Proxy Sharding Manager stopped")

    async def _rotation_loop(self) -> None:
        """Main rotation loop."""
        while self._running:
            try:
                await asyncio.sleep(self._rotation_interval_ms / 1000.0)

                # Check if rotation is needed
                if self._should_rotate():
                    await self._rotate_proxy()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in rotation loop: {e}")

    async def _health_check_loop(self) -> None:
        """Health check loop."""
        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval_ms / 1000.0)

                # Perform health checks on all proxies
                await self._check_all_proxies()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")

    def _should_rotate(self) -> bool:
        """Check if proxy rotation is needed."""
        # Always rotate on schedule
        current_time = time.time()
        elapsed_ms = (current_time - self._last_rotation_time) * 1000.0

        if elapsed_ms >= self._rotation_interval_ms:
            return True

        # Rotate if current proxy is unhealthy
        if self._active_proxy_id:
            proxy_state = self._proxy_states.get(self._active_proxy_id)
            if proxy_state and not proxy_state.is_available():
                logger.warning(f"Current proxy {self._active_proxy_id} is unavailable, rotating")
                return True

        # Rotate if rate limit is approaching threshold
        if self._active_proxy_id:
            proxy_state = self._proxy_states.get(self._active_proxy_id)
            if proxy_state and proxy_state.rate_limit_usage >= 0.8:  # 80% threshold
                logger.warning(f"Current proxy {self._active_proxy_id} rate limit at {proxy_state.rate_limit_usage:.2%}, rotating")
                return True

        return False

    async def _rotate_proxy(self) -> None:
        """Rotate to a new proxy."""
        async with self._rotation_lock:
            try:
                # Mark current proxy for cooldown
                if self._active_proxy_id:
                    current_state = self._proxy_states.get(self._active_proxy_id)
                    if current_state:
                        current_state.is_active = False
                        current_state.rotation_cooldown_until = (
                            time.time() + self._cooldown_duration_ms / 1000.0
                        )
                        logger.info(
                            f"Rotated away from proxy {self._active_proxy_id}, "
                            f"cooldown until {current_state.rotation_cooldown_until:.2f}"
                        )

                # Select new best proxy
                await self._select_best_proxy()

                self._last_rotation_time = time.time()
                self._total_rotations += 1

                if self._active_proxy_id:
                    logger.info(f"Rotated to proxy {self._active_proxy_id}")

            except Exception as e:
                logger.error(f"Error during proxy rotation: {e}")

    async def _select_best_proxy(self) -> None:
        """Select the best available proxy."""
        # Get all available proxies with scores
        available_proxies = [
            (proxy_id, state)
            for proxy_id, state in self._proxy_states.items()
            if state.is_available()
        ]

        if not available_proxies:
            logger.warning("No available proxies found, using first proxy as fallback")
            if self._proxy_states:
                self._active_proxy_id = next(iter(self._proxy_states.keys()))
                self._proxy_states[self._active_proxy_id].is_active = True
            return

        # Sort by score (highest first)
        available_proxies.sort(key=lambda x: x[1].get_score(), reverse=True)

        # Select best proxy (or randomly from top 3 for load balancing)
        top_n = min(3, len(available_proxies))
        selected = random.choice(available_proxies[:top_n])

        proxy_id, state = selected
        self._active_proxy_id = proxy_id
        state.is_active = True

        logger.info(
            f"Selected proxy {proxy_id} (score={state.get_score():.3f}, "
            f"latency={state.latency_ms:.2f}ms, rate_usage={state.rate_limit_usage:.2%})"
        )

    async def _check_all_proxies(self) -> None:
        """Check health of all proxies."""
        # In production, perform actual connectivity/latency tests
        # For now, update based on recent activity
        for proxy_id, state in self._proxy_states.items():
            if state.proxy.enabled:
                # Check if proxy is still healthy based on recent failures
                if state.consecutive_failures >= 3:
                    state.is_healthy = False
                    logger.warning(f"Proxy {proxy_id} marked as unhealthy (consecutive_failures={state.consecutive_failures})")
                elif state.consecutive_failures == 0 and not state.is_healthy:
                    state.is_healthy = True
                    logger.info(f"Proxy {proxy_id} marked as healthy again")

    def get_active_proxy(self) -> ProxyConfig | None:
        """Get currently active proxy configuration.

        Returns:
            Active proxy config or None if no proxy available
        """
        if self._active_proxy_id:
            state = self._proxy_states.get(self._active_proxy_id)
            if state and state.is_available():
                return state.proxy

        # Fallback: select best available
        if self._running:
            asyncio.create_task(self._select_best_proxy())

        return None

    async def record_request(
        self,
        success: bool = True,
        latency_ms: float | None = None,
        bytes_sent: int = 0,
        bytes_received: int = 0,
    ) -> None:
        """Record a request to update proxy statistics.

        Args:
            success: Whether the request was successful
            latency_ms: Request latency in milliseconds
            bytes_sent: Bytes sent in request
            bytes_received: Bytes received in response
        """
        self._total_requests += 1

        if self._active_proxy_id:
            state = self._proxy_states.get(self._active_proxy_id)
            if state:
                state.update_request(success=success, latency_ms=latency_ms)
                state.total_bytes_sent += bytes_sent
                state.total_bytes_received += bytes_received

                # Check for rate limit violations
                if state.rate_limit_usage >= 1.0:
                    self._rate_limit_violations += 1
                    logger.error(
                        f"Rate limit violation on proxy {self._active_proxy_id} "
                        f"(usage={state.rate_limit_usage:.2%})"
                    )

                    # Trigger immediate rotation
                    if self._running:
                        asyncio.create_task(self._rotate_proxy())

    def get_stats(self) -> dict:
        """Get statistics about proxy usage.

        Returns:
            Dictionary with proxy statistics
        """
        return {
            "active_proxy_id": self._active_proxy_id,
            "total_proxies": len(self._proxies),
            "available_proxies": sum(1 for s in self._proxy_states.values() if s.is_available()),
            "total_requests": self._total_requests,
            "total_rotations": self._total_rotations,
            "rate_limit_violations": self._rate_limit_violations,
            "proxies": {
                proxy_id: {
                    "type": state.proxy.type.name,
                    "is_active": state.is_active,
                    "is_healthy": state.is_healthy,
                    "is_available": state.is_available(),
                    "latency_ms": state.latency_ms,
                    "rate_limit_usage": state.rate_limit_usage,
                    "success_rate": (
                        (state.total_requests - state.error_count) / state.total_requests
                        if state.total_requests > 0 else 0.0
                    ),
                    "total_requests": state.total_requests,
                    "error_count": state.error_count,
                }
                for proxy_id, state in self._proxy_states.items()
            },
        }

    def add_proxy(self, proxy: ProxyConfig) -> None:
        """Add a new proxy to the pool.

        Args:
            proxy: Proxy configuration to add
        """
        if proxy.id not in self._proxy_states:
            self._proxies.append(proxy)
            self._proxy_states[proxy.id] = ProxyState(proxy=proxy)
            logger.info(f"Added proxy {proxy.id} ({proxy.type.name})")
        else:
            logger.warning(f"Proxy {proxy.id} already exists, updating configuration")
            self._proxy_states[proxy.id].proxy = proxy

    def remove_proxy(self, proxy_id: str) -> None:
        """Remove a proxy from the pool.

        Args:
            proxy_id: ID of proxy to remove
        """
        if proxy_id in self._proxy_states:
            # If it's the active proxy, rotate away first
            if self._active_proxy_id == proxy_id:
                asyncio.create_task(self._rotate_proxy())

            del self._proxy_states[proxy_id]
            self._proxies = [p for p in self._proxies if p.id != proxy_id]
            logger.info(f"Removed proxy {proxy_id}")

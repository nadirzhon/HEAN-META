"""
Phase 16: Dynamic API Scouter
Maintains a live list of the 5 fastest Bybit API nodes based on TCP-handshake latency.
Forces the Executor to switch its WebSocket/REST connection to the fastest node every 60 seconds.
"""

import asyncio
import socket
import statistics
import time
from collections import deque
from dataclasses import dataclass, field

import aiohttp

from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class NodeLatency:
    """Represents a Bybit API node with its latency measurements."""
    host: str
    ws_url: str
    rest_url: str
    latency_ms: float = float('inf')
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=10))
    last_check: float = 0.0
    success_count: int = 0
    failure_count: int = 0

    def update_latency(self, latency_ms: float, success: bool = True) -> None:
        """Update latency measurement."""
        if success:
            self.latency_ms = latency_ms
            self.recent_latencies.append(latency_ms)
            self.success_count += 1
        else:
            self.failure_count += 1

        self.last_check = time.time()

        # Calculate average latency from recent measurements
        if len(self.recent_latencies) > 0:
            self.latency_ms = statistics.mean(self.recent_latencies)

    def get_avg_latency(self) -> float:
        """Get average latency from recent measurements."""
        if len(self.recent_latencies) == 0:
            return float('inf')
        return statistics.mean(self.recent_latencies)

    def get_reliability(self) -> float:
        """Get reliability score (0.0 to 1.0)."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total


class DynamicAPIScouter:
    """
    Dynamic API Scouter that finds the fastest Bybit API nodes.

    Continuously probes Bybit API endpoints and ranks them by TCP handshake latency.
    Maintains a list of the top 5 fastest nodes and forces Executor to switch every 60 seconds.
    """

    # Known Bybit API endpoints (can be expanded)
    BYBIT_NODES = [
        {
            "host": "api.bybit.com",
            "ws_url": "wss://stream.bybit.com/v5/public/linear",
            "rest_url": "https://api.bybit.com",
        },
        {
            "host": "api-testnet.bybit.com",
            "ws_url": "wss://stream-testnet.bybit.com/v5/public/linear",
            "rest_url": "https://api-testnet.bybit.com",
        },
        # Additional nodes (if Bybit has geo-distributed endpoints, add them here)
        # Note: Bybit typically uses DNS-based load balancing, but we can still measure latency
    ]

    def __init__(
        self,
        check_interval: float = 10.0,
        switch_interval: float = 60.0,
        top_n: int = 5,
        testnet: bool = False
    ):
        """Initialize the API Scouter.

        Args:
            check_interval: How often to check latency (seconds)
            switch_interval: How often to force endpoint switch (seconds)
            top_n: Number of top nodes to maintain
            testnet: Whether to use testnet endpoints
        """
        self._check_interval = check_interval
        self._switch_interval = switch_interval
        self._top_n = top_n
        self._testnet = testnet

        # Initialize nodes
        self._nodes: list[NodeLatency] = []
        for node_config in self.BYBIT_NODES:
            # Filter by testnet if specified
            if testnet and "testnet" not in node_config["host"]:
                continue
            if not testnet and "testnet" in node_config["host"]:
                continue

            self._nodes.append(NodeLatency(
                host=node_config["host"],
                ws_url=node_config["ws_url"],
                rest_url=node_config["rest_url"],
            ))

        # Current fastest node
        self._current_fastest: NodeLatency | None = None
        self._last_switch_time = time.time()

        # Callback for notifying Executor of endpoint change
        self._endpoint_change_callback: callable | None = None

        # Running state
        self._running = False
        self._task: asyncio.Task | None = None

        # HTTP session for REST latency checks
        self._http_session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Start the scouter."""
        if self._running:
            return

        self._running = True
        self._http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5.0),
            connector=aiohttp.TCPConnector(limit=10, ttl_dns_cache=300)
        )

        # Initial latency check
        await self._check_all_nodes()

        # Start background task
        self._task = asyncio.create_task(self._scout_loop())
        logger.info(f"Dynamic API Scouter started (check_interval={self._check_interval}s, switch_interval={self._switch_interval}s)")

    async def stop(self) -> None:
        """Stop the scouter."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._http_session:
            await self._http_session.close()

        logger.info("Dynamic API Scouter stopped")

    def set_endpoint_change_callback(self, callback: callable) -> None:
        """Set callback for when fastest endpoint changes.

        Args:
            callback: Function called with (ws_url, rest_url) when endpoint changes
        """
        self._endpoint_change_callback = callback

    async def _measure_tcp_latency(self, host: str, port: int = 443) -> tuple[float, bool]:
        """
        Measure TCP handshake latency to a host.

        Args:
            host: Hostname to connect to
            port: Port to connect to (443 for HTTPS, 80 for HTTP)

        Returns:
            Tuple of (latency_ms, success)
        """
        try:
            # Resolve hostname
            addr_infos = await asyncio.get_event_loop().getaddrinfo(
                host, port, family=socket.AF_INET, type=socket.SOCK_STREAM
            )

            if not addr_infos:
                return (float('inf'), False)

            # Use first resolved address
            family, socktype, proto, canonname, sockaddr = addr_infos[0]

            # Create socket
            sock = socket.socket(family, socktype, proto)
            sock.setblocking(False)

            # Measure TCP handshake latency
            start_time = time.perf_counter()

            try:
                await asyncio.wait_for(
                    asyncio.get_event_loop().sock_connect(sock, sockaddr),
                    timeout=2.0
                )

                latency_ms = (time.perf_counter() - start_time) * 1000.0

                sock.close()
                return (latency_ms, True)

            except (TimeoutError, OSError):
                sock.close()
                return (float('inf'), False)

        except Exception as e:
            logger.debug(f"Failed to measure TCP latency for {host}:{port}: {e}")
            return (float('inf'), False)

    async def _measure_http_latency(self, rest_url: str) -> tuple[float, bool]:
        """
        Measure HTTP request latency (includes TCP + TLS + HTTP).

        Args:
            rest_url: REST API URL (e.g., "https://api.bybit.com")

        Returns:
            Tuple of (latency_ms, success)
        """
        if not self._http_session:
            return (float('inf'), False)

        try:
            # Simple health check endpoint (if available) or just connect
            # For Bybit, we can use a lightweight endpoint
            endpoint = f"{rest_url}/v5/market/time"  # Public endpoint, no auth needed

            start_time = time.perf_counter()

            async with self._http_session.get(endpoint, allow_redirects=False) as response:
                latency_ms = (time.perf_counter() - start_time) * 1000.0

                # Consider success if status < 500
                success = response.status < 500

                return (latency_ms, success)

        except TimeoutError:
            return (float('inf'), False)
        except Exception as e:
            logger.debug(f"Failed to measure HTTP latency for {rest_url}: {e}")
            return (float('inf'), False)

    async def _check_node_latency(self, node: NodeLatency) -> None:
        """Check latency for a single node.

        Args:
            node: Node to check
        """
        # Measure both TCP handshake and HTTP latency
        tcp_latency, tcp_success = await self._measure_tcp_latency(node.host, port=443)
        http_latency, http_success = await self._measure_http_latency(node.rest_url)

        # Use the minimum of TCP and HTTP latency (TCP is faster, HTTP is more realistic)
        # Weight TCP more heavily since it's closer to "raw" latency
        combined_latency = min(tcp_latency, http_latency * 0.8) if http_success else tcp_latency
        success = tcp_success or http_success

        node.update_latency(combined_latency, success)

        logger.debug(
            f"Node {node.host}: TCP={tcp_latency:.2f}ms, HTTP={http_latency:.2f}ms, "
            f"combined={combined_latency:.2f}ms, success={success}"
        )

    async def _check_all_nodes(self) -> None:
        """Check latency for all nodes in parallel."""
        tasks = [self._check_node_latency(node) for node in self._nodes]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Sort nodes by latency
        self._nodes.sort(key=lambda n: (n.get_avg_latency(), -n.get_reliability()))

    async def _scout_loop(self) -> None:
        """Main scouting loop."""
        while self._running:
            try:
                # Check all nodes
                await self._check_all_nodes()

                # Get top N nodes
                top_nodes = self._nodes[:self._top_n]

                if top_nodes:
                    fastest = top_nodes[0]

                    # Check if we should switch endpoints
                    current_time = time.time()
                    should_switch = (
                        self._current_fastest is None or
                        self._current_fastest.host != fastest.host or
                        (current_time - self._last_switch_time) >= self._switch_interval
                    )

                    if should_switch:
                        logger.info(
                            f"Switching to fastest node: {fastest.host} "
                            f"(latency={fastest.get_avg_latency():.2f}ms, "
                            f"reliability={fastest.get_reliability():.2%})"
                        )

                        self._current_fastest = fastest
                        self._last_switch_time = current_time

                        # Notify Executor to switch endpoints
                        if self._endpoint_change_callback:
                            try:
                                await self._endpoint_change_callback(
                                    fastest.ws_url,
                                    fastest.rest_url
                                )
                            except Exception as e:
                                logger.error(f"Error in endpoint change callback: {e}")

                # Log top nodes
                if len(top_nodes) > 0:
                    logger.debug("Top API nodes:")
                    for i, node in enumerate(top_nodes[:5], 1):
                        logger.debug(
                            f"  {i}. {node.host}: {node.get_avg_latency():.2f}ms "
                            f"(reliability={node.get_reliability():.2%})"
                        )

                # Wait for next check
                await asyncio.sleep(self._check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in scout loop: {e}")
                await asyncio.sleep(self._check_interval)

    def get_fastest_endpoints(self) -> tuple[str | None, str | None]:
        """Get current fastest WebSocket and REST endpoints.

        Returns:
            Tuple of (ws_url, rest_url) or (None, None) if no node available
        """
        if self._current_fastest:
            return (self._current_fastest.ws_url, self._current_fastest.rest_url)

        if self._nodes:
            # Return fastest node even if not switched yet
            self._nodes.sort(key=lambda n: (n.get_avg_latency(), -n.get_reliability()))
            fastest = self._nodes[0]
            return (fastest.ws_url, fastest.rest_url)

        return (None, None)

    def get_top_nodes(self, n: int = 5) -> list[NodeLatency]:
        """Get top N fastest nodes.

        Args:
            n: Number of nodes to return

        Returns:
            List of top N nodes sorted by latency
        """
        sorted_nodes = sorted(
            self._nodes,
            key=lambda node: (node.get_avg_latency(), -node.get_reliability())
        )
        return sorted_nodes[:n]

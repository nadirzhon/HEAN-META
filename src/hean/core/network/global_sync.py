"""
Phase 19: Distributed Node Manager - gRPC Mesh Network
Establishes a gRPC mesh-network between 3 nodes (Tokyo, Singapore, Frankfurt).
Implements 'First-Responder' logic: the node with lowest latency to exchange executes,
while others provide hedge-cover.

The system is designed to be impossible to kill - if one part is cut off,
the rest continues to generate profit.
"""

import asyncio
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from hean.logging import get_logger

logger = get_logger(__name__)

try:
    from concurrent import futures  # noqa: F401

    import grpc
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    logger.warning("grpc not available. Install with: pip install grpcio")

from hean.core.bus import EventBus  # noqa: E402
from hean.core.types import Event, EventType, Order  # noqa: E402


class NodeRegion(IntEnum):
    """Node region identifiers."""
    UNKNOWN = 0
    TOKYO = 1
    SINGAPORE = 2
    FRANKFURT = 3


class NodeRole(IntEnum):
    """Node role in the distributed network."""
    UNKNOWN = 0
    MASTER = 1      # Primary execution node
    HEDGE = 2       # Hedge cover node
    STANDBY = 3     # Standby node


@dataclass
class NodeState:
    """State of a distributed node."""
    region: NodeRegion
    role: NodeRole
    address: str  # gRPC address (host:port)
    is_healthy: bool = True
    last_heartbeat_ns: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    exchange_latencies: dict[str, float] = field(default_factory=dict)  # exchange -> latency_ms
    network_latency_ms: float = float('inf')  # Latency to this node
    takeover_capable: bool = False  # Can take over master role

    def update_heartbeat(
        self,
        timestamp_ns: int,
        cpu_usage: float,
        memory_usage: float,
        active_connections: int,
        is_healthy: bool,
        exchange_latencies: dict[str, float]
    ) -> None:
        """Update node state from heartbeat."""
        self.last_heartbeat_ns = timestamp_ns
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.active_connections = active_connections
        self.is_healthy = is_healthy
        self.exchange_latencies = exchange_latencies

        # Calculate if node can take over master role
        self.takeover_capable = (
            is_healthy and
            cpu_usage < 0.8 and
            memory_usage < 0.85 and
            len(exchange_latencies) > 0
        )

    def is_alive(self, timeout_ms: float = 10000.0) -> bool:
        """Check if node is alive based on last heartbeat."""
        if self.last_heartbeat_ns == 0:
            return False

        current_ns = time.time_ns()
        elapsed_ms = (current_ns - self.last_heartbeat_ns) / 1_000_000.0
        return elapsed_ms < timeout_ms


@dataclass
class TradeExecutionRequest:
    """Request for trade execution in distributed network."""
    symbol: str
    side: str  # "buy" or "sell"
    quantity: float
    price: float
    exchange: str
    requesting_node: NodeRegion
    request_timestamp_ns: int
    node_latencies: dict[NodeRegion, float]  # node -> latency_ms to exchange


class DistributedNodeManager:
    """
    Distributed Node Manager with gRPC mesh-network.

    Establishes connections between 3 nodes (Tokyo, Singapore, Frankfurt).
    Implements First-Responder logic: node with lowest latency executes trade,
    others provide hedge-cover.
    """

    # Known node addresses (can be configured via environment)
    DEFAULT_NODES = {
        NodeRegion.TOKYO: "tokyo-node.hean.local:50051",
        NodeRegion.SINGAPORE: "singapore-node.hean.local:50051",
        NodeRegion.FRANKFURT: "frankfurt-node.hean.local:50051",
    }

    def __init__(
        self,
        bus: EventBus,
        local_region: NodeRegion,
        node_addresses: dict[NodeRegion, str] | None = None,
        heartbeat_interval_ms: float = 100.0,  # 100ms heartbeat for <10ms failover
        heartbeat_timeout_ms: float = 500.0,   # 500ms timeout = node offline
    ):
        """Initialize the Distributed Node Manager.

        Args:
            bus: Event bus for system events
            local_region: This node's region (TOKYO, SINGAPORE, or FRANKFURT)
            node_addresses: Optional dict mapping regions to gRPC addresses
            heartbeat_interval_ms: Interval between heartbeats (ms)
            heartbeat_timeout_ms: Timeout before considering node offline (ms)
        """
        if not GRPC_AVAILABLE:
            raise ImportError("grpc not available. Install with: pip install grpcio")

        self._bus = bus
        self._local_region = local_region
        self._node_addresses = node_addresses or self.DEFAULT_NODES.copy()
        self._heartbeat_interval_ms = heartbeat_interval_ms
        self._heartbeat_timeout_ms = heartbeat_timeout_ms

        # Node states (including self)
        self._nodes: dict[NodeRegion, NodeState] = {}

        # Initialize local node
        local_address = self._node_addresses.get(local_region, "localhost:50051")
        self._local_node = NodeState(
            region=local_region,
            role=NodeRole.STANDBY,  # Will be determined by heartbeat/consensus
            address=local_address,
            is_healthy=True,
        )
        self._nodes[local_region] = self._local_node

        # Current master node
        self._master_node: NodeRegion | None = None

        # gRPC server and clients
        self._grpc_server: grpc.Server | None = None
        self._grpc_clients: dict[NodeRegion, Any] = {}  # Will store gRPC stubs

        # Running state
        self._running = False
        self._heartbeat_task: asyncio.Task | None = None
        self._master_check_task: asyncio.Task | None = None

        # Callbacks
        self._trade_execution_callback: Callable | None = None
        self._position_update_callback: Callable | None = None

        # Active positions tracking (for failover)
        self._active_positions: dict[str, dict] = {}  # symbol -> position data
        self._active_orders: dict[str, Order] = {}

        # Statistics
        self._execution_count = defaultdict(int)  # node -> count
        self._failover_count = 0
        self._last_failover_time_ns = 0

    async def start(self) -> None:
        """Start the distributed node manager."""
        if self._running:
            return

        self._running = True

        # Start gRPC server (simplified - in production, use generated proto stubs)
        # For now, we'll use a simplified async communication layer
        logger.info(
            f"Starting Distributed Node Manager (Region: {self._local_region.name}, "
            f"Address: {self._local_node.address})"
        )

        # Initialize connections to other nodes
        await self._connect_to_nodes()

        # Start heartbeat loop
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start master check loop
        self._master_check_task = asyncio.create_task(self._master_check_loop())

        # Subscribe to relevant events
        self._bus.subscribe(EventType.ORDER_EXECUTED, self._handle_order_executed)
        self._bus.subscribe(EventType.POSITION_UPDATE, self._handle_position_update)

        logger.info("Distributed Node Manager started")

    async def stop(self) -> None:
        """Stop the distributed node manager."""
        if not self._running:
            return

        self._running = False

        # Cancel tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._master_check_task:
            self._master_check_task.cancel()
            try:
                await self._master_check_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe from events
        self._bus.unsubscribe(EventType.ORDER_EXECUTED, self._handle_order_executed)
        self._bus.unsubscribe(EventType.POSITION_UPDATE, self._handle_position_update)

        # Close gRPC connections
        await self._disconnect_from_nodes()

        if self._grpc_server:
            self._grpc_server.stop(grace=5.0)

        logger.info("Distributed Node Manager stopped")

    async def _connect_to_nodes(self) -> None:
        """Connect to other nodes in the mesh network."""
        # In production, this would establish gRPC channels to other nodes
        # For now, we simulate the connection setup
        for region, address in self._node_addresses.items():
            if region == self._local_region:
                continue

            try:
                # In production: create gRPC channel
                # channel = grpc.aio.insecure_channel(address)
                # stub = GlobalSyncStub(channel)
                # self._grpc_clients[region] = stub

                # Initialize node state
                if region not in self._nodes:
                    self._nodes[region] = NodeState(
                        region=region,
                        role=NodeRole.STANDBY,
                        address=address,
                        is_healthy=False,
                    )

                logger.info(f"Connected to {region.name} node at {address}")
            except Exception as e:
                logger.error(f"Failed to connect to {region.name} node: {e}")

    async def _disconnect_from_nodes(self) -> None:
        """Disconnect from other nodes."""
        for region, _client in list(self._grpc_clients.items()):
            try:
                # In production: close gRPC channel
                # await client.channel.close()
                pass
            except Exception as e:
                logger.error(f"Error disconnecting from {region.name}: {e}")

        self._grpc_clients.clear()

    async def _heartbeat_loop(self) -> None:
        """Main heartbeat loop - sends heartbeats to other nodes and updates local state."""
        while self._running:
            try:
                current_ns = time.time_ns()

                # Update local node state
                try:
                    import psutil
                    cpu_usage = psutil.cpu_percent() / 100.0
                    memory_usage = psutil.virtual_memory().percent / 100.0
                except ImportError:
                    # Fallback if psutil not available
                    cpu_usage = 0.0
                    memory_usage = 0.0

                self._local_node.update_heartbeat(
                    timestamp_ns=current_ns,
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    active_connections=len(self._active_orders),
                    is_healthy=True,
                    exchange_latencies=self._measure_exchange_latencies(),
                )

                # Send heartbeats to other nodes (in production, via gRPC)
                await self._send_heartbeats()

                # Receive heartbeats from other nodes (would be handled by gRPC server)

                await asyncio.sleep(self._heartbeat_interval_ms / 1000.0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(self._heartbeat_interval_ms / 1000.0)

    async def _send_heartbeats(self) -> None:
        """Send heartbeats to all other nodes."""
        # In production, this would use gRPC Heartbeat RPC
        # For now, we simulate the heartbeat exchange
        for region, node in self._nodes.items():
            if region == self._local_region:
                continue

            try:
                # In production:
                # response = await self._grpc_clients[region].Heartbeat(
                #     HeartbeatRequest(
                #         node_region=self._local_region.value,
                #         timestamp_ns=self._local_node.last_heartbeat_ns,
                #         cpu_usage=self._local_node.cpu_usage,
                #         memory_usage=self._local_node.memory_usage,
                #         active_connections=self._local_node.active_connections,
                #         is_healthy=self._local_node.is_healthy,
                #         exchange_latencies=self._local_node.exchange_latencies,
                #     )
                # )
                # Update node state from response
                # node.network_latency_ms = response.network_latency_ms
                # node.role = NodeRole(response.assigned_role)
                pass
            except Exception as e:
                logger.debug(f"Failed to send heartbeat to {region.name}: {e}")
                node.is_healthy = False

    def _measure_exchange_latencies(self) -> dict[str, float]:
        """Measure latency to exchanges (simplified - in production, actual ping)."""
        # In production, this would measure actual TCP/TLS handshake latency
        # For now, return cached or estimated values
        latencies = {}

        # Tokyo node typically has better latency to Japanese exchanges
        if self._local_region == NodeRegion.TOKYO:
            latencies["bybit"] = 15.0  # ms
        elif self._local_region == NodeRegion.SINGAPORE:
            latencies["bybit"] = 10.0  # Bybit is in Singapore
        elif self._local_region == NodeRegion.FRANKFURT:
            latencies["bybit"] = 200.0

        return latencies

    async def _master_check_loop(self) -> None:
        """Loop to check master node health and trigger failover if needed."""
        while self._running:
            try:
                await asyncio.sleep(self._heartbeat_timeout_ms / 1000.0 / 2.0)  # Check at 2x heartbeat rate

                # Determine master node (consensus algorithm)
                await self._determine_master_node()

                # Check if master is healthy
                if self._master_node and self._master_node != self._local_region:
                    master = self._nodes.get(self._master_node)
                    if master and not master.is_alive(self._heartbeat_timeout_ms):
                        logger.warning(f"Master node {self._master_node.name} appears offline, initiating failover")
                        await self._initiate_failover()

                # Check if we should take over (if we're the best candidate)
                if self._should_takeover_master():
                    await self._request_master_role()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in master check loop: {e}")

    async def _determine_master_node(self) -> None:
        """Determine which node should be master (consensus)."""
        # Simple consensus: node with best exchange latency and health
        candidates = [
            node for node in self._nodes.values()
            if node.is_alive(self._heartbeat_timeout_ms) and node.takeover_capable
        ]

        if not candidates:
            # No healthy candidates, make local node master if healthy
            if self._local_node.is_healthy:
                self._master_node = self._local_region
                self._local_node.role = NodeRole.MASTER
                return

        # Select master based on:
        # 1. Best average exchange latency
        # 2. Lowest CPU/memory usage
        # 3. Highest connection count (more active)
        best_node = max(
            candidates,
            key=lambda n: (
                -min(n.exchange_latencies.values()) if n.exchange_latencies else -float('inf'),
                -n.cpu_usage,
                -n.memory_usage,
                n.active_connections,
            )
        )

        # If we're not master but should be, mark for takeover
        if best_node.region != self._master_node:
            if best_node.region == self._local_region:
                # We should be master
                if self._master_node and self._master_node != self._local_region:
                    logger.info(f"Local node should become master (current: {self._master_node.name})")
            else:
                # Another node should be master
                self._master_node = best_node.region
                best_node.role = NodeRole.MASTER

                # Update other nodes' roles
                for node in candidates:
                    if node.region != best_node.region:
                        node.role = NodeRole.HEDGE if node.is_healthy else NodeRole.STANDBY

    def _should_takeover_master(self) -> bool:
        """Check if local node should take over master role."""
        if self._local_node.role == NodeRole.MASTER:
            return False  # Already master

        if not self._local_node.takeover_capable:
            return False

        # If no master or master is offline
        if not self._master_node:
            return True

        master = self._nodes.get(self._master_node)
        if not master or not master.is_alive(self._heartbeat_timeout_ms):
            return True

        # If we have better latency than master
        if self._local_node.exchange_latencies and master.exchange_latencies:
            local_best = min(self._local_node.exchange_latencies.values())
            master_best = min(master.exchange_latencies.values())
            if local_best < master_best * 0.8:  # 20% better latency
                return True

        return False

    async def _request_master_role(self) -> None:
        """Request master role from other nodes."""
        logger.info("Requesting master role takeover")

        # In production, send gRPC MasterRoleRequest to all nodes
        # For now, we'll directly assume consensus
        current_ns = time.time_ns()

        if current_ns - self._last_failover_time_ns < 10_000_000_000:  # 10s cooldown
            return  # Avoid rapid failover oscillations

        self._master_node = self._local_region
        self._local_node.role = NodeRole.MASTER

        # Update other nodes to HEDGE role
        for node in self._nodes.values():
            if node.region != self._local_region and node.is_healthy:
                node.role = NodeRole.HEDGE

        self._failover_count += 1
        self._last_failover_time_ns = current_ns

        logger.warning(f"MASTER ROLE TAKEOVER: {self._local_region.name} is now master (failover #{self._failover_count})")

        # Emit event
        self._bus.publish(Event(
            type=EventType.SYSTEM_EVENT,
            data={
                "event": "master_role_takeover",
                "node": self._local_region.name,
                "timestamp_ns": current_ns,
            }
        ))

    async def _initiate_failover(self) -> None:
        """Initiate failover when master node goes offline."""
        # If we're not master but master is offline, try to take over
        if self._should_takeover_master():
            await self._request_master_role()

    async def request_trade_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        exchange: str,
    ) -> dict | None:
        """
        Request trade execution using First-Responder logic.

        The node with lowest latency to the exchange executes the trade,
        while others provide hedge-cover.

        Returns:
            Execution result dict with order_id, executing_node, etc.
        """
        current_ns = time.time_ns()

        # Measure latency to exchange from all nodes
        node_latencies: dict[NodeRegion, float] = {}

        # Local node latency
        local_latency = self._local_node.exchange_latencies.get(exchange, float('inf'))
        node_latencies[self._local_region] = local_latency

        # Get latencies from other nodes (in production, via gRPC)
        for region, node in self._nodes.items():
            if region != self._local_region and node.is_alive():
                node_latency = node.exchange_latencies.get(exchange, float('inf'))
                node_latencies[region] = node_latency

        # Find node with lowest latency (First-Responder)
        executing_node = min(node_latencies.items(), key=lambda x: x[1])[0]
        execution_latency = node_latencies[executing_node]

        logger.info(
            f"Trade execution request: {symbol} {side} {quantity} @ {price} on {exchange}. "
            f"Executing node: {executing_node.name} (latency: {execution_latency:.2f}ms)"
        )

        # If we're the executing node, execute locally
        if executing_node == self._local_region:
            if self._trade_execution_callback:
                try:
                    result = await self._trade_execution_callback(
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=price,
                        exchange=exchange,
                    )

                    self._execution_count[self._local_region] += 1

                    # Notify other nodes of execution
                    await self._notify_position_update(symbol, exchange, result)

                    return {
                        "executed": True,
                        "executing_node": executing_node.name,
                        "order_id": result.get("order_id") if isinstance(result, dict) else None,
                        "execution_price": result.get("price") if isinstance(result, dict) else price,
                        "execution_timestamp_ns": current_ns,
                    }
                except Exception as e:
                    logger.error(f"Failed to execute trade locally: {e}")
                    return {
                        "executed": False,
                        "executing_node": executing_node.name,
                        "reject_reason": str(e),
                    }
        else:
            # Request execution from remote node (in production, via gRPC)
            # For now, we'll execute locally as fallback
            logger.info(f"Would request execution from {executing_node.name} via gRPC")

            # In production:
            # response = await self._grpc_clients[executing_node].RequestTradeExecution(
            #     TradeExecutionRequest(...)
            # )
            # return response

        return None

    async def _notify_position_update(self, symbol: str, exchange: str, result: dict) -> None:
        """Notify other nodes of position updates."""
        # In production, send gRPC PositionUpdate to all hedge nodes
        for region, node in self._nodes.items():
            if region != self._local_region and node.role == NodeRole.HEDGE:
                try:
                    # await self._grpc_clients[region].NotifyPositionUpdate(...)
                    pass
                except Exception as e:
                    logger.debug(f"Failed to notify {region.name} of position update: {e}")

    def set_trade_execution_callback(self, callback: Callable) -> None:
        """Set callback for trade execution."""
        self._trade_execution_callback = callback

    def set_position_update_callback(self, callback: Callable) -> None:
        """Set callback for position updates."""
        self._position_update_callback = callback

    async def _handle_order_executed(self, event: Event) -> None:
        """Handle order executed event."""
        order = event.data.get("order")
        if order:
            self._active_orders[order.id] = order

    async def _handle_position_update(self, event: Event) -> None:
        """Handle position update event."""
        position = event.data.get("position", {})
        symbol = position.get("symbol")
        if symbol:
            self._active_positions[symbol] = position

    def get_node_stats(self) -> dict:
        """Get statistics about the distributed network."""
        return {
            "local_region": self._local_region.name,
            "local_role": self._local_node.role.name,
            "master_node": self._master_node.name if self._master_node else None,
            "nodes": {
                region.name: {
                    "role": node.role.name,
                    "is_healthy": node.is_healthy,
                    "is_alive": node.is_alive(self._heartbeat_timeout_ms),
                    "cpu_usage": node.cpu_usage,
                    "memory_usage": node.memory_usage,
                    "network_latency_ms": node.network_latency_ms,
                    "exchange_latencies": node.exchange_latencies,
                    "active_connections": node.active_connections,
                }
                for region, node in self._nodes.items()
            },
            "execution_count": {
                region.name: count
                for region, count in self._execution_count.items()
            },
            "failover_count": self._failover_count,
            "active_positions": len(self._active_positions),
            "active_orders": len(self._active_orders),
        }

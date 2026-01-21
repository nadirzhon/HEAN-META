"""
Atomic Execution Trees: The Singularity Execution

Instead of single orders, this module sends clusters of orders that create
'artificial support' and 'resistance' to trap slower bots.

The system creates order trees that manipulate market microstructure,
executing before market-makers can react.
"""

import asyncio
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Order, OrderRequest, OrderStatus
from hean.logging import get_logger

logger = get_logger(__name__)


class OrderCluster:
    """Represents a cluster of orders forming artificial support/resistance."""
    
    def __init__(
        self,
        cluster_id: str,
        symbol: str,
        target_price: float,
        side: str,
        total_size: float,
        cluster_type: str,  # "support" or "resistance"
        spread_bps: float = 5.0,  # Spread between orders in cluster
        order_count: int = 5,  # Number of orders in cluster
    ):
        """Initialize an order cluster.
        
        Args:
            cluster_id: Unique cluster identifier
            symbol: Trading symbol
            target_price: Target price for the cluster
            side: "buy" or "sell"
            total_size: Total size to distribute across orders
            cluster_type: "support" or "resistance"
            spread_bps: Spread between orders in basis points
            order_count: Number of orders in the cluster
        """
        self.cluster_id = cluster_id
        self.symbol = symbol
        self.target_price = target_price
        self.side = side
        self.total_size = total_size
        self.cluster_type = cluster_type
        self.spread_bps = spread_bps
        self.order_count = order_count
        self.orders: List[Order] = []
        self.created_at = datetime.utcnow()
        self.is_active = True
        
        # Generate order prices and sizes
        self._generate_order_distribution()
    
    def _generate_order_distribution(self) -> None:
        """Generate price and size distribution for cluster orders."""
        self.order_prices: List[float] = []
        self.order_sizes: List[float] = []
        
        # Calculate price spread
        price_spread = self.target_price * (self.spread_bps / 10000.0)
        
        # Generate prices (distributed around target)
        if self.side == "buy":
            # For support: prices below target (ascending)
            base_price = self.target_price - (price_spread * (self.order_count - 1) / 2)
            for i in range(self.order_count):
                price = base_price + (price_spread * i)
                self.order_prices.append(price)
        else:  # sell
            # For resistance: prices above target (descending)
            base_price = self.target_price + (price_spread * (self.order_count - 1) / 2)
            for i in range(self.order_count):
                price = base_price - (price_spread * i)
                self.order_prices.append(price)
        
        # Generate sizes (pyramid distribution: larger at center)
        # Use normal distribution centered at middle order
        center_idx = self.order_count // 2
        sizes = []
        total_weight = 0.0
        
        for i in range(self.order_count):
            # Gaussian weight centered at middle
            distance = abs(i - center_idx)
            weight = np.exp(-0.5 * (distance / (self.order_count / 4.0)) ** 2)
            sizes.append(weight)
            total_weight += weight
        
        # Normalize to total size
        for i in range(self.order_count):
            self.order_sizes.append((sizes[i] / total_weight) * self.total_size)
    
    def create_order_requests(self, strategy_id: str) -> List[OrderRequest]:
        """Create order requests for all orders in the cluster."""
        requests = []
        
        for i, (price, size) in enumerate(zip(self.order_prices, self.order_sizes)):
            if size < 0.001:  # Skip tiny orders
                continue
            
            request = OrderRequest(
                symbol=self.symbol,
                side=self.side,
                size=size,
                price=price,
                order_type="limit",
                strategy_id=strategy_id,
                post_only=True,  # Maker orders for artificial levels
                time_in_force="GTC",
            )
            requests.append(request)
        
        return requests
    
    def get_cluster_info(self) -> Dict:
        """Get cluster information."""
        return {
            "cluster_id": self.cluster_id,
            "symbol": self.symbol,
            "target_price": self.target_price,
            "side": self.side,
            "cluster_type": self.cluster_type,
            "order_count": self.order_count,
            "total_size": self.total_size,
            "spread_bps": self.spread_bps,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
        }


class AtomicExecutor:
    """
    Atomic Execution Trees: Creates order clusters for artificial support/resistance.
    
    This executor creates "order trees" - clusters of orders placed strategically
    to create artificial market levels that trap slower bots and market-makers.
    """
    
    def __init__(self, bus: EventBus):
        """Initialize the Atomic Executor.
        
        Args:
            bus: Event bus for publishing order requests
        """
        self._bus = bus
        self._running = False
        
        # Active clusters
        self._active_clusters: Dict[str, OrderCluster] = {}
        self._cluster_orders: Dict[str, List[str]] = defaultdict(list)  # cluster_id -> order_ids
        
        # Statistics
        self._total_clusters_created = 0
        self._total_orders_placed = 0
        self._clusters_by_type: Dict[str, int] = defaultdict(int)
        
        logger.info("Atomic Executor initialized")
    
    async def start(self) -> None:
        """Start the atomic executor."""
        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self._bus.subscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        self._running = True
        
        # Start background task for cluster management
        asyncio.create_task(self._manage_clusters())
        
        logger.info("Atomic Executor started")
    
    async def stop(self) -> None:
        """Stop the atomic executor."""
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        self._bus.unsubscribe(EventType.ORDER_CANCELLED, self._handle_order_cancelled)
        self._running = False
        logger.info("Atomic Executor stopped")
    
    async def create_support_cluster(
        self,
        symbol: str,
        support_price: float,
        total_size: float,
        strategy_id: str,
        spread_bps: float = 5.0,
        order_count: int = 5,
    ) -> str:
        """
        Create an artificial support cluster (buy orders).
        
        Args:
            symbol: Trading symbol
            support_price: Target support price
            total_size: Total size to distribute
            strategy_id: Strategy identifier
            spread_bps: Spread between orders in basis points
            order_count: Number of orders in cluster
            
        Returns:
            Cluster ID
        """
        cluster_id = f"support_{uuid.uuid4().hex[:8]}"
        
        cluster = OrderCluster(
            cluster_id=cluster_id,
            symbol=symbol,
            target_price=support_price,
            side="buy",
            total_size=total_size,
            cluster_type="support",
            spread_bps=spread_bps,
            order_count=order_count,
        )
        
        self._active_clusters[cluster_id] = cluster
        self._total_clusters_created += 1
        self._clusters_by_type["support"] += 1
        
        # Create and publish order requests
        order_requests = cluster.create_order_requests(strategy_id)
        
        for request in order_requests:
            await self._bus.publish(
                Event(
                    event_type=EventType.ORDER_REQUEST,
                    data={"order_request": request},
                )
            )
            self._total_orders_placed += 1
            # Track order IDs (will be updated when orders are created)
            self._cluster_orders[cluster_id].append(request.order_id if hasattr(request, 'order_id') else 'pending')
        
        logger.info(
            f"Created support cluster {cluster_id}: {symbol} @ {support_price}, "
            f"{len(order_requests)} orders, total size={total_size:.6f}"
        )
        
        return cluster_id
    
    async def create_resistance_cluster(
        self,
        symbol: str,
        resistance_price: float,
        total_size: float,
        strategy_id: str,
        spread_bps: float = 5.0,
        order_count: int = 5,
    ) -> str:
        """
        Create an artificial resistance cluster (sell orders).
        
        Args:
            symbol: Trading symbol
            resistance_price: Target resistance price
            total_size: Total size to distribute
            strategy_id: Strategy identifier
            spread_bps: Spread between orders in basis points
            order_count: Number of orders in cluster
            
        Returns:
            Cluster ID
        """
        cluster_id = f"resistance_{uuid.uuid4().hex[:8]}"
        
        cluster = OrderCluster(
            cluster_id=cluster_id,
            symbol=symbol,
            target_price=resistance_price,
            side="sell",
            total_size=total_size,
            cluster_type="resistance",
            spread_bps=spread_bps,
            order_count=order_count,
        )
        
        self._active_clusters[cluster_id] = cluster
        self._total_clusters_created += 1
        self._clusters_by_type["resistance"] += 1
        
        # Create and publish order requests
        order_requests = cluster.create_order_requests(strategy_id)
        
        for request in order_requests:
            await self._bus.publish(
                Event(
                    event_type=EventType.ORDER_REQUEST,
                    data={"order_request": request},
                )
            )
            self._total_orders_placed += 1
            # Track order IDs
            self._cluster_orders[cluster_id].append(request.order_id if hasattr(request, 'order_id') else 'pending')
        
        logger.info(
            f"Created resistance cluster {cluster_id}: {symbol} @ {resistance_price}, "
            f"{len(order_requests)} orders, total size={total_size:.6f}"
        )
        
        return cluster_id
    
    async def create_trap_cluster(
        self,
        symbol: str,
        trap_price: float,
        total_size: float,
        strategy_id: str,
        trap_type: str = "both",  # "buy", "sell", or "both"
        spread_bps: float = 3.0,
        order_count: int = 7,
    ) -> Tuple[str, Optional[str]]:
        """
        Create a trap cluster - both support and resistance around a price.
        
        This creates artificial levels that trap slower bots by creating
        false support/resistance that gets removed after execution.
        
        Args:
            symbol: Trading symbol
            trap_price: Center price for the trap
            total_size: Total size per side
            strategy_id: Strategy identifier
            trap_type: "buy", "sell", or "both"
            spread_bps: Spread between orders
            order_count: Number of orders per side
            
        Returns:
            Tuple of (support_cluster_id, resistance_cluster_id)
        """
        support_id = None
        resistance_id = None
        
        if trap_type in ("buy", "both"):
            support_id = await self.create_support_cluster(
                symbol=symbol,
                support_price=trap_price * 0.999,  # Slightly below
                total_size=total_size,
                strategy_id=strategy_id,
                spread_bps=spread_bps,
                order_count=order_count,
            )
        
        if trap_type in ("sell", "both"):
            resistance_id = await self.create_resistance_cluster(
                symbol=symbol,
                resistance_price=trap_price * 1.001,  # Slightly above
                total_size=total_size,
                strategy_id=strategy_id,
                spread_bps=spread_bps,
                order_count=order_count,
            )
        
        logger.info(
            f"Created trap cluster: {symbol} @ {trap_price}, "
            f"support={support_id}, resistance={resistance_id}"
        )
        
        return (support_id, resistance_id)
    
    async def cancel_cluster(self, cluster_id: str) -> None:
        """Cancel all orders in a cluster."""
        if cluster_id not in self._active_clusters:
            logger.warning(f"Cluster {cluster_id} not found")
            return
        
        cluster = self._active_clusters[cluster_id]
        cluster.is_active = False
        
        # Cancel all orders in the cluster
        order_ids = self._cluster_orders.get(cluster_id, [])
        
        for order_id in order_ids:
            if order_id != 'pending':
                await self._bus.publish(
                    Event(
                        event_type=EventType.ORDER_CANCEL_REQUEST,
                        data={"order_id": order_id},
                    )
                )
        
        logger.info(f"Cancelled cluster {cluster_id} ({len(order_ids)} orders)")
    
    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled events."""
        order: Order = event.data["order"]
        
        # Find which cluster this order belongs to
        for cluster_id, order_ids in self._cluster_orders.items():
            if order.order_id in order_ids:
                logger.debug(f"Order {order.order_id} from cluster {cluster_id} filled")
                break
    
    async def _handle_order_cancelled(self, event: Event) -> None:
        """Handle order cancelled events."""
        order: Order = event.data["order"]
        
        # Remove from cluster tracking
        for cluster_id, order_ids in self._cluster_orders.items():
            if order.order_id in order_ids:
                order_ids.remove(order.order_id)
                break
    
    async def _manage_clusters(self) -> None:
        """Background task to manage cluster lifecycle."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self._running:
                    break
                
                # Clean up old inactive clusters
                current_time = datetime.utcnow()
                clusters_to_remove = []
                
                for cluster_id, cluster in self._active_clusters.items():
                    if not cluster.is_active:
                        # Check if cluster is old (more than 1 hour)
                        age = current_time - cluster.created_at
                        if age > timedelta(hours=1):
                            clusters_to_remove.append(cluster_id)
                
                for cluster_id in clusters_to_remove:
                    del self._active_clusters[cluster_id]
                    del self._cluster_orders[cluster_id]
                    logger.debug(f"Removed old cluster {cluster_id}")
                
            except Exception as e:
                logger.error(f"Error managing clusters: {e}", exc_info=True)
    
    def get_statistics(self) -> Dict:
        """Get executor statistics."""
        return {
            "total_clusters_created": self._total_clusters_created,
            "active_clusters": len(self._active_clusters),
            "total_orders_placed": self._total_orders_placed,
            "clusters_by_type": dict(self._clusters_by_type),
        }
    
    def get_active_clusters(self) -> List[Dict]:
        """Get information about all active clusters."""
        return [cluster.get_cluster_info() for cluster in self._active_clusters.values()]

"""Order lifecycle management."""

from collections import defaultdict
from datetime import datetime

from hean.core.types import Order, OrderStatus
from hean.logging import get_logger

logger = get_logger(__name__)


class OrderManager:
    """Manages order lifecycle and state."""

    def __init__(self) -> None:
        """Initialize the order manager."""
        self._orders: dict[str, Order] = {}
        self._orders_by_strategy: dict[str, list[str]] = defaultdict(list)
        self._orders_by_symbol: dict[str, list[str]] = defaultdict(list)

    def register_order(self, order: Order) -> None:
        """Register a new order."""
        self._orders[order.order_id] = order
        self._orders_by_strategy[order.strategy_id].append(order.order_id)
        self._orders_by_symbol[order.symbol].append(order.order_id)
        logger.debug(f"Registered order {order.order_id}")

    def get_order(self, order_id: str) -> Order | None:
        """Get an order by ID."""
        return self._orders.get(order_id)

    def update_order(self, order_id: str, **updates: dict) -> None:
        """Update an order with new values."""
        if order_id not in self._orders:
            logger.warning(f"Order {order_id} not found for update")
            return

        order = self._orders[order_id]
        for key, value in updates.items():
            if hasattr(order, key):
                setattr(order, key, value)

        logger.debug(f"Updated order {order_id}: {updates}")

    def get_orders_by_strategy(self, strategy_id: str) -> list[Order]:
        """Get all orders for a strategy."""
        order_ids = self._orders_by_strategy.get(strategy_id, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def get_orders_by_symbol(self, symbol: str) -> list[Order]:
        """Get all orders for a symbol."""
        order_ids = self._orders_by_symbol.get(symbol, [])
        return [self._orders[oid] for oid in order_ids if oid in self._orders]

    def get_open_orders(self) -> list[Order]:
        """Get all open orders (pending, placed, partially filled)."""
        open_statuses = {OrderStatus.PENDING, OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED}
        return [order for order in self._orders.values() if order.status in open_statuses]

    def get_filled_orders(self, since: datetime | None = None) -> list[Order]:
        """Get all filled orders, optionally filtered by time."""
        logger.info(f"[DEBUG] OrderManager.get_filled_orders: total orders={len(self._orders)}")

        # Вывести информацию о всех ордерах
        for order_id, order in self._orders.items():
            logger.info(
                f"[DEBUG] Order {order_id}: "
                f"status={order.status}, "
                f"status_type={type(order.status)}, "
                f"status_repr={repr(order.status)}, "
                f"OrderStatus.FILLED={OrderStatus.FILLED}, "
                f"status == FILLED={order.status == OrderStatus.FILLED}, "
                f"filled_size={order.filled_size}, "
                f"size={order.size}"
            )

        filled = [order for order in self._orders.values() if order.status == OrderStatus.FILLED]
        logger.info(f"[DEBUG] OrderManager.get_filled_orders: found {len(filled)} filled orders")

        if since:
            filled_before_filter = len(filled)
            filled = [order for order in filled if order.timestamp >= since]
            logger.info(
                f"[DEBUG] OrderManager.get_filled_orders: after time filter: {len(filled)}/{filled_before_filter}"
            )

        return filled

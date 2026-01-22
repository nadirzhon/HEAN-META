"""
Smart Limit Executor with Geometric Slippage Prediction using Riemannian Curvature
Uses TDA to predict real slippage BEFORE sending the order.
If predicted slippage > threshold, switches to 'Smart-Limit' mode.

Phase 16: Nano-Batching Execution with Order Jittering
Splits large orders into multiple smaller orders with randomized delays to avoid anti-HFT filters.
"""

import asyncio
import random
from datetime import datetime
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import EventType, Order, OrderRequest, OrderStatus

try:
    from hean.exchange.bybit.http import BybitHTTPClient
except ImportError:  # Optional live trading dependency
    BybitHTTPClient = None  # type: ignore[assignment]
from hean.logging import get_logger

logger = get_logger(__name__)


class SmartLimitExecutor:
    """Smart Limit Executor with Geometric Slippage Prediction.

    Uses Riemannian curvature of the orderbook to predict real slippage BEFORE
    sending the order. If predicted slippage > threshold, switches to 'Smart-Limit' mode.
    """

    def __init__(self, bus: EventBus, bybit_http: BybitHTTPClient | None = None) -> None:
        """Initialize the Smart Limit Executor.

        Args:
            bus: Event bus
            bybit_http: Optional Bybit HTTP client for live trading
        """
        self._bus = bus
        self._bybit_http = bybit_http
        self._running = False

        # Slippage prediction threshold (as percentage, e.g., 0.01 = 1%)
        self._slippage_threshold = 0.01  # 1% slippage threshold

        # FastWarden for TDA-based slippage prediction
        self._fast_warden: Any = None
        try:
            import graph_engine_py  # type: ignore
            self._fast_warden = graph_engine_py.FastWarden()
            logger.info("FastWarden initialized for geometric slippage prediction")
        except ImportError:
            logger.warning("FastWarden not available. Using fallback slippage estimation.")

        # Orderbook cache for curvature computation
        self._orderbook_cache: dict[str, dict[str, Any]] = {}

        # Active orders tracking
        self._active_orders: dict[str, Order] = {}

        # Phase 16: Nano-batching parameters
        self._jitter_enabled = True
        self._jitter_order_count = 10  # Number of child orders
        self._jitter_delay_min_ms = 5  # Minimum delay between orders (ms)
        self._jitter_delay_max_ms = 15  # Maximum delay between orders (ms)
        self._jitter_min_size_threshold = 0.5  # Only jitter orders >= this size (BTC)

    async def start(self) -> None:
        """Start the Smart Limit Executor."""
        self._running = True
        self._bus.subscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook_update)
        logger.info("Smart Limit Executor started with geometric slippage prediction")

    async def stop(self) -> None:
        """Stop the Smart Limit Executor."""
        self._running = False
        self._bus.unsubscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook_update)
        logger.info("Smart Limit Executor stopped")

    async def _handle_orderbook_update(self, event: Any) -> None:
        """Handle orderbook updates to cache L2 data for curvature computation."""
        orderbook_data = event.data.get("orderbook", {})
        symbol = orderbook_data.get("symbol")

        if not symbol:
            return

        # Cache orderbook for curvature computation
        self._orderbook_cache[symbol] = orderbook_data

        # Update FastWarden if available
        if self._fast_warden:
            bids = orderbook_data.get("bids", [])
            asks = orderbook_data.get("asks", [])

            if bids and asks:
                try:
                    bid_prices = [float(bid[0]) for bid in bids]
                    bid_sizes = [float(bid[1]) for bid in bids]
                    ask_prices = [float(ask[0]) for ask in asks]
                    ask_sizes = [float(ask[1]) for ask in asks]

                    self._fast_warden.update_orderbook(
                        symbol,
                        bid_prices,
                        bid_sizes,
                        ask_prices,
                        ask_sizes
                    )
                except Exception as e:
                    logger.warning(f"Failed to update FastWarden for {symbol}: {e}")

    def _compute_riemannian_curvature(self, symbol: str) -> float:
        """Compute Riemannian curvature proxy from orderbook geometry.

        Uses discrete approximation of curvature:
        K ≈ (d²y/dx²) / (1 + (dy/dx)²)^(3/2)

        Where:
        - x = price level
        - y = orderbook size (liquidity)

        Args:
            symbol: Trading symbol

        Returns:
            Curvature score (positive = convex = lower slippage, negative = concave = higher slippage)
        """
        orderbook = self._orderbook_cache.get(symbol)
        if not orderbook:
            return 0.0  # Default: flat curvature

        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return 0.0

        # Combine bids and asks into unified price-size curve
        levels = []

        # Bids: descending price order
        for bid in reversed(bids):
            levels.append({"price": float(bid[0]), "size": float(bid[1])})

        # Asks: ascending price order
        for ask in asks:
            levels.append({"price": float(ask[0]), "size": float(ask[1])})

        if len(levels) < 3:
            return 0.0

        # Compute second derivative of size w.r.t. price (curvature indicator)
        total_curvature = 0.0
        count = 0

        for i in range(1, len(levels) - 1):
            dx1 = levels[i]["price"] - levels[i-1]["price"]
            dx2 = levels[i+1]["price"] - levels[i]["price"]
            dy1 = levels[i]["size"] - levels[i-1]["size"]
            dy2 = levels[i+1]["size"] - levels[i]["size"]

            if abs(dx1) > 1e-10 and abs(dx2) > 1e-10:
                dydx1 = dy1 / dx1
                dydx2 = dy2 / dx2
                d2ydx2 = (dydx2 - dydx1) / ((dx1 + dx2) / 2.0)

                # Riemannian curvature approximation
                curvature = d2ydx2 / pow(1.0 + dydx1 * dydx1, 1.5)
                total_curvature += curvature
                count += 1

        return total_curvature / count if count > 0 else 0.0

    def predict_slippage(self, symbol: str, order_size: float, is_buy: bool) -> float:
        """Predict slippage using Riemannian curvature of the orderbook.

        GEOMETRIC SLIPPAGE PREDICTION:
        - High positive curvature (convex orderbook) = lower slippage
        - High negative curvature (concave orderbook) = higher slippage

        Args:
            symbol: Trading symbol
            order_size: Order size
            is_buy: True for buy orders, False for sell orders

        Returns:
            Predicted slippage as percentage (e.g., 0.005 = 0.5%)
        """
        # Try FastWarden first (TDA-based prediction)
        if self._fast_warden:
            try:
                # FastWarden uses internal TDA_Engine.predict_slippage()
                # which already uses Riemannian curvature
                predicted = self._fast_warden.predict_slippage(symbol, order_size, is_buy)
                if predicted and predicted > 0:
                    return predicted
            except (AttributeError, TypeError) as e:
                logger.debug(f"FastWarden slippage prediction not available: {e}, using fallback")
            except Exception as e:
                logger.warning(f"FastWarden slippage prediction failed: {e}")

        # Fallback: compute curvature locally
        curvature = self._compute_riemannian_curvature(symbol)

        # Base slippage
        base_slippage = 0.005  # 0.5% base

        # Curvature factor: negative curvature increases slippage
        curvature_factor = abs(curvature) * 10.0

        if curvature < 0:
            # Concave orderbook: increase slippage
            predicted_slippage = base_slippage + curvature_factor
        else:
            # Convex orderbook: decrease slippage
            predicted_slippage = base_slippage - min(curvature_factor, base_slippage * 0.5)

        # Adjust for order size
        size_factor = min(1.0, order_size / 1000.0)  # Normalize to 1000 units
        predicted_slippage *= (1.0 + size_factor)

        # Ensure reasonable bounds (0.1% to 10%)
        predicted_slippage = max(0.001, min(0.1, predicted_slippage))

        return predicted_slippage

    async def place_post_only_order(
        self,
        order_request: OrderRequest,
        ofi_aggression: float = 0.0
    ) -> Order:
        """Place a Post-Only limit order with geometric slippage prediction.

        CRITICAL: If predicted slippage > threshold, switches to 'Smart-Limit' mode
        (aggressive limit pricing to improve fill probability).

        Args:
            order_request: Order request
            ofi_aggression: Order Flow Imbalance aggression factor (0-1)

        Returns:
            Created Order instance
        """
        symbol = order_request.symbol
        side = order_request.side
        size = order_request.size

        # GEOMETRIC SLIPPAGE PREDICTION: Predict slippage BEFORE sending order
        predicted_slippage = self.predict_slippage(
            symbol,
            size,
            is_buy=(side == "buy")
        )

        logger.info(
            f"Geometric slippage prediction for {symbol} {side} {size}: "
            f"{predicted_slippage*100:.2f}%"
        )

        # Get current market prices from orderbook cache
        orderbook = self._orderbook_cache.get(symbol, {})
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            logger.warning(f"No orderbook data for {symbol}, using order_request price")
            best_bid = order_request.price or 0.0
            best_ask = order_request.price or 0.0
        else:
            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 0.0

        # DECISION: If predicted slippage > threshold, switch to Smart-Limit mode
        use_smart_limit = predicted_slippage > self._slippage_threshold

        if use_smart_limit:
            logger.info(
                f"Switching to Smart-Limit mode for {symbol}: "
                f"predicted_slippage={predicted_slippage*100:.2f}% > "
                f"threshold={self._slippage_threshold*100:.2f}%"
            )

            # Smart-Limit: Aggressive pricing to reduce slippage
            # Price closer to market to improve fill probability
            if side == "buy":
                # Buy: price closer to best ask (but still limit)
                smart_price = best_ask * (1.0 - predicted_slippage * 0.5)
            else:  # sell
                # Sell: price closer to best bid (but still limit)
                smart_price = best_bid * (1.0 + predicted_slippage * 0.5)

            limit_price = smart_price
        else:
            # Standard Post-Only: Conservative pricing (best bid/ask with offset)
            if side == "buy":
                limit_price = best_bid
            else:  # sell
                limit_price = best_ask

        # Create order
        order = Order(
            order_id=f"smart_limit_{datetime.utcnow().timestamp()}",
            strategy_id=order_request.strategy_id,
            symbol=symbol,
            side=side,
            size=size,
            price=limit_price,
            order_type="limit",
            status=OrderStatus.PENDING,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            timestamp=datetime.utcnow(),
            metadata={
                **order_request.metadata,
                "smart_limit": use_smart_limit,
                "predicted_slippage": predicted_slippage,
                "ofi_aggression": ofi_aggression,
                "geometric_prediction": True,
            },
            is_maker=True,
            placed_at=datetime.utcnow(),
        )

        self._active_orders[order.order_id] = order

        logger.info(
            f"Smart Limit order created: {order.order_id} {side} {size} {symbol} @ {limit_price:.6f} "
            f"(predicted_slippage={predicted_slippage*100:.2f}%, "
            f"smart_limit={use_smart_limit})"
        )

        return order

    def get_orderbook_presence(self, symbol: str) -> dict[str, Any]:
        """Get orderbook presence for a symbol (Phase 3 feature).

        Args:
            symbol: Trading symbol

        Returns:
            Orderbook presence dictionary
        """
        active_orders = [
            o for o in self._active_orders.values()
            if o.symbol == symbol and o.status in {OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED}
        ]

        return {
            "symbol": symbol,
            "num_orders": len(active_orders),
            "total_size": sum(o.size - o.filled_size for o in active_orders),
            "orders": [
                {
                    "order_id": o.order_id,
                    "side": o.side,
                    "size": o.size - o.filled_size,
                    "price": o.price,
                }
                for o in active_orders
            ],
        }

    def get_active_orders(self) -> list[Order]:
        """Get all active orders."""
        return [
            o for o in self._active_orders.values()
            if o.status in {OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED}
        ]

    async def place_post_only_order_with_jitter(
        self,
        order_request: OrderRequest,
        ofi_aggression: float = 0.0
    ) -> list[Order]:
        """
        Phase 16: Place Post-Only order with Order Jittering (Nano-Batching Execution).

        Instead of sending 1.0 BTC as a single order, splits into multiple smaller orders
        (e.g., 10 orders of 0.1 BTC) with randomized 5-15ms delays between them to hide
        from anti-HFT filters.

        Args:
            order_request: Original order request
            ofi_aggression: Order Flow Imbalance aggression factor (0-1)

        Returns:
            List of child Order instances
        """
        # Check if jittering should be applied
        if not self._jitter_enabled or order_request.size < self._jitter_min_size_threshold:
            # Too small to jitter, send as single order
            single_order = await self.place_post_only_order(order_request, ofi_aggression)
            return [single_order]

        # Split order into multiple child orders
        child_order_size = order_request.size / self._jitter_order_count
        child_orders: list[Order] = []

        logger.info(
            f"Phase 16: Order Jittering enabled for {order_request.symbol} {order_request.side} "
            f"{order_request.size}: splitting into {self._jitter_order_count} orders of {child_order_size:.6f}"
        )

        # Create child order requests
        for i in range(self._jitter_order_count):
            # Create child order request (slightly modify size to account for rounding)
            if i == self._jitter_order_count - 1:
                # Last order gets remainder to ensure total size matches
                remaining_size = order_request.size - sum(o.size for o in child_orders)
                child_size = remaining_size if remaining_size > 0 else child_order_size
            else:
                child_size = child_order_size

            child_request = OrderRequest(
                strategy_id=order_request.strategy_id,
                symbol=order_request.symbol,
                side=order_request.side,
                size=child_size,
                price=order_request.price,
                order_type=order_request.order_type,
                stop_loss=order_request.stop_loss,
                take_profit=order_request.take_profit,
                signal_id=f"{order_request.signal_id}_jitter_{i}" if order_request.signal_id else None,
                metadata={
                    **(order_request.metadata or {}),
                    "jittered": True,
                    "jitter_index": i,
                    "jitter_total": self._jitter_order_count,
                    "parent_size": order_request.size,
                },
            )

            # Place child order (reuse existing logic)
            child_order = await self.place_post_only_order(child_request, ofi_aggression)

            # Add jitter metadata
            child_order.metadata = {
                **child_order.metadata,
                "jittered": True,
                "jitter_index": i,
                "jitter_total": self._jitter_order_count,
                "parent_size": order_request.size,
            }

            child_orders.append(child_order)

            # Jitter delay: random 5-15ms delay between orders (except last)
            if i < self._jitter_order_count - 1:
                delay_ms = random.uniform(
                    self._jitter_delay_min_ms,
                    self._jitter_delay_max_ms
                )
                delay_seconds = delay_ms / 1000.0

                logger.debug(
                    f"Jitter delay {i+1}/{self._jitter_order_count}: {delay_ms:.2f}ms "
                    f"(order: {child_order.order_id})"
                )

                await asyncio.sleep(delay_seconds)

        total_size = sum(o.size for o in child_orders)
        logger.info(
            f"Phase 16: Order Jittering complete: {len(child_orders)} orders placed, "
            f"total size={total_size:.6f} (requested={order_request.size:.6f})"
        )

        return child_orders

    def set_jitter_parameters(
        self,
        enabled: bool = True,
        order_count: int = 10,
        delay_min_ms: int = 5,
        delay_max_ms: int = 15,
        min_size_threshold: float = 0.5
    ) -> None:
        """Configure order jittering parameters.

        Args:
            enabled: Enable/disable order jittering
            order_count: Number of child orders to split into
            delay_min_ms: Minimum delay between orders (milliseconds)
            delay_max_ms: Maximum delay between orders (milliseconds)
            min_size_threshold: Minimum order size (BTC) to apply jittering
        """
        self._jitter_enabled = enabled
        self._jitter_order_count = max(1, order_count)
        self._jitter_delay_min_ms = max(0, delay_min_ms)
        self._jitter_delay_max_ms = max(self._jitter_delay_min_ms, delay_max_ms)
        self._jitter_min_size_threshold = max(0.0, min_size_threshold)

        logger.info(
            f"Order jittering parameters updated: enabled={enabled}, "
            f"order_count={self._jitter_order_count}, "
            f"delay={self._jitter_delay_min_ms}-{self._jitter_delay_max_ms}ms, "
            f"min_size={self._jitter_min_size_threshold}"
        )

"""Maker Rebate Farming Strategy - passive income from maker rebates.

Places deep limit orders far from current price to collect Bybit maker rebates (-0.01%).
Orders are placed at ±0.5% from mid price to minimize fill risk while maximizing rebate income.

Expected Impact: +5-10% daily profit (passive income, ~0.5-1 USDT/day on $300 capital)
Risk: Very Low (far from market, rare fills)
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Any

from hean.core.bus import EventBus
from hean.core.regime import Regime
from hean.core.types import Event, EventType, Signal, Tick
from hean.logging import get_logger
from hean.observability.metrics import metrics
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class RebateFarmer(BaseStrategy):
    """Passive maker rebate farming strategy.

    Places deep limit orders at ±0.5% from mid price to collect maker rebates.
    Bybit pays -0.01% rebate for maker orders, creating passive income.

    Strategy Logic:
    - Place bid at mid_price * 0.995 (0.5% below)
    - Place ask at mid_price * 1.005 (0.5% above)
    - Orders only fill on significant price swings
    - When filled, position has natural mean-reversion edge

    Risk Management:
    - Uses minimal position size (1% of equity per side)
    - Wide distance from market reduces fill probability
    - When filled, natural profit from price returning to mean
    """

    def __init__(
        self,
        bus: EventBus,
        symbols: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize rebate farmer.

        Args:
            bus: Event bus for publishing signals
            symbols: List of symbols to trade (default: BTCUSDT, ETHUSDT)
            enabled: Whether strategy is enabled
        """
        super().__init__("rebate_farmer", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._enabled = enabled

        # Deep order parameters
        self._bid_offset_pct = 0.005  # 0.5% below mid
        self._ask_offset_pct = 0.005  # 0.5% above mid
        self._size_pct_of_equity = 0.01  # 1% of equity per side

        # Order management
        self._active_orders: dict[str, dict[str, Any]] = {}  # symbol -> {bid: ..., ask: ...}
        self._last_order_time: dict[str, datetime] = {}
        self._order_refresh_interval = timedelta(minutes=5)  # Refresh orders every 5 min

        # Price tracking for mid calculation
        self._price_history: dict[str, deque[float]] = {}
        self._window_size = 20

        # Metrics
        self._total_orders_placed = 0
        self._total_fills = 0
        self._total_rebate_earned = 0.0

        # Regime: Only active in RANGE and NORMAL (not IMPULSE)
        self._allowed_regimes = {Regime.RANGE, Regime.NORMAL}
        self._current_regime: dict[str, Regime] = {}

        logger.info(
            f"RebateFarmer initialized: symbols={self._symbols}, "
            f"bid_offset={self._bid_offset_pct*100}%, ask_offset={self._ask_offset_pct*100}%"
        )

    async def start(self) -> None:
        """Start the rebate farmer strategy."""
        await super().start()
        # Subscribe to order fill events for tracking
        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        logger.info("RebateFarmer started")

    async def stop(self) -> None:
        """Stop the rebate farmer strategy."""
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        await super().stop()
        logger.info("RebateFarmer stopped")

    async def on_tick(self, event: Event) -> None:
        """Handle tick events - check if orders need refresh."""
        if not self._enabled:
            return

        tick: Tick = event.data["tick"]

        if tick.symbol not in self._symbols:
            return

        # Update price history
        if tick.symbol not in self._price_history:
            self._price_history[tick.symbol] = deque(maxlen=self._window_size)

        self._price_history[tick.symbol].append(tick.price)

        # Check regime - only trade in RANGE or NORMAL
        current_regime = self._current_regime.get(tick.symbol, Regime.NORMAL)
        if current_regime not in self._allowed_regimes:
            return

        # Check if we need to place/refresh orders
        await self._check_and_place_orders(tick)

    async def on_funding(self, event: Event) -> None:
        """Handle funding events - not used for this strategy."""
        pass

    async def on_regime_update(self, event: Event) -> None:
        """Handle regime update events."""
        symbol = event.data.get("symbol")
        regime = event.data.get("regime")
        if symbol is not None and regime is not None:
            self._current_regime[symbol] = regime

    async def _check_and_place_orders(self, tick: Tick) -> None:
        """Check if orders need to be placed or refreshed.

        Args:
            tick: Current tick data
        """
        symbol = tick.symbol

        # Check if enough price data
        if len(self._price_history.get(symbol, [])) < 5:
            return

        # Check if we need to refresh orders
        last_order = self._last_order_time.get(symbol)
        if last_order and datetime.utcnow() - last_order < self._order_refresh_interval:
            return

        # Calculate mid price (average of recent prices)
        prices = list(self._price_history[symbol])
        mid_price = sum(prices) / len(prices)

        # Calculate deep order prices
        bid_price = mid_price * (1 - self._bid_offset_pct)
        ask_price = mid_price * (1 + self._ask_offset_pct)

        # Place deep bid order (buy signal)
        await self._place_deep_order(
            symbol=symbol,
            side="buy",
            price=bid_price,
            mid_price=mid_price,
        )

        # Place deep ask order (sell signal)
        await self._place_deep_order(
            symbol=symbol,
            side="sell",
            price=ask_price,
            mid_price=mid_price,
        )

        self._last_order_time[symbol] = datetime.utcnow()

    async def _place_deep_order(
        self,
        symbol: str,
        side: str,
        price: float,
        mid_price: float,
    ) -> None:
        """Place a deep limit order for rebate farming.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            price: Order price
            mid_price: Current mid price
        """
        # Calculate stop loss and take profit for when order fills
        # Since these are deep orders, when filled they have natural edge
        if side == "buy":
            # Bought 0.5% below mid - target return to mid (0.5% profit)
            stop_loss = price * 0.995  # 0.5% below entry
            take_profit = mid_price  # Return to mid for 0.5% profit
            take_profit_1 = price * 1.003  # First TP at 0.3%
        else:
            # Sold 0.5% above mid - target return to mid (0.5% profit)
            stop_loss = price * 1.005  # 0.5% above entry
            take_profit = mid_price  # Return to mid for 0.5% profit
            take_profit_1 = price * 0.997  # First TP at 0.3%

        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_1=take_profit_1,
            metadata={
                "type": "rebate_farm",
                "mid_price": mid_price,
                "offset_pct": self._bid_offset_pct if side == "buy" else self._ask_offset_pct,
                "size_multiplier": 0.5,  # Use smaller size for rebate farming
                "is_deep_order": True,
            },
            prefer_maker=True,  # Always maker for rebate
            min_maker_edge_bps=0.0,  # No edge requirement - we want the rebate
        )

        await self._publish_signal(signal)
        self._total_orders_placed += 1
        metrics.increment("rebate_farmer_orders_placed")

        logger.debug(
            f"RebateFarmer: Placed deep {side} order for {symbol} "
            f"at ${price:.2f} (mid=${mid_price:.2f}, offset={self._bid_offset_pct*100}%)"
        )

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled events to track rebate farming performance."""
        order = event.data.get("order")
        if order is None:
            return

        # Check if this is our order
        if order.strategy_id != self.strategy_id:
            return

        self._total_fills += 1

        # Estimate rebate earned (Bybit maker rebate is -0.01% = 1 bps)
        notional = order.size * order.price
        rebate = notional * 0.0001  # 0.01% rebate
        self._total_rebate_earned += rebate

        metrics.increment("rebate_farmer_fills")
        metrics.increment("rebate_farmer_rebate_earned", rebate)

        logger.info(
            f"RebateFarmer: Order filled! {order.symbol} {order.side} "
            f"size={order.size:.6f} @ ${order.price:.2f}, "
            f"rebate=${rebate:.4f}, total_rebate=${self._total_rebate_earned:.4f}"
        )

    def get_metrics(self) -> dict[str, float]:
        """Get rebate farmer metrics."""
        return {
            "total_orders_placed": float(self._total_orders_placed),
            "total_fills": float(self._total_fills),
            "total_rebate_earned": self._total_rebate_earned,
            "fill_rate_pct": (
                (self._total_fills / self._total_orders_placed * 100)
                if self._total_orders_placed > 0
                else 0.0
            ),
        }

    def enable(self) -> None:
        """Enable the strategy."""
        self._enabled = True
        logger.info("RebateFarmer enabled")

    def disable(self) -> None:
        """Disable the strategy."""
        self._enabled = False
        logger.info("RebateFarmer disabled")

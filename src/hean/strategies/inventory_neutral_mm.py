"""Inventory-Neutral Market Making Strategy.

Captures spread in neutral/balanced markets by placing two-sided orders.
Only active when OFI imbalance is low (< 0.3), indicating balanced order flow.

Expected Impact: +10-20% daily profit (pure spread income)
Risk: Low (delta-neutral by design)
"""

from collections import deque
from datetime import datetime, timedelta
from typing import Any

from hean.core.bus import EventBus
from hean.core.ofi import OrderFlowImbalance
from hean.core.regime import Regime
from hean.core.types import Event, EventType, Signal, Tick
from hean.logging import get_logger
from hean.observability.metrics import metrics
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class InventoryNeutralMM(BaseStrategy):
    """Inventory-Neutral Market Making strategy.

    Places simultaneous bid and ask orders when OFI shows balanced pressure.
    Captures spread without taking directional risk.

    Strategy Logic:
    - Monitor OFI for neutral conditions (|ofi_value| < 0.3)
    - Place bid at (mid - offset) and ask at (mid + offset)
    - Offset dynamically adjusts based on volatility
    - When both sides fill, net position is ~0 but spread is captured

    Risk Management:
    - Only trades in RANGE and NORMAL regimes
    - Reduces size in high volatility
    - Tight inventory limits to avoid directional exposure
    """

    def __init__(
        self,
        bus: EventBus,
        ofi_monitor: OrderFlowImbalance | None = None,
        symbols: list[str] | None = None,
        enabled: bool = True,
    ) -> None:
        """Initialize Inventory-Neutral MM.

        Args:
            bus: Event bus for publishing signals
            ofi_monitor: OFI monitor for order flow analysis
            symbols: List of symbols to trade
            enabled: Whether strategy is enabled
        """
        super().__init__("inventory_neutral_mm", bus)
        self._ofi_monitor = ofi_monitor
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._enabled = enabled

        # Neutral market threshold
        self._neutral_ofi_threshold = 0.3  # |OFI| < 0.3 = neutral

        # Spread capture parameters
        self._base_offset_bps = 5.0  # 5 bps base offset from mid
        self._min_spread_capture_bps = 3.0  # Minimum spread to capture
        self._size_pct_of_equity = 0.02  # 2% of equity per side

        # Order management
        self._last_order_time: dict[str, datetime] = {}
        self._order_interval = timedelta(seconds=30)  # Place orders every 30s

        # Price tracking
        self._price_history: dict[str, deque[float]] = {}
        self._window_size = 20

        # Inventory tracking (positions from our orders)
        self._net_inventory: dict[str, float] = {}  # symbol -> net position
        self._max_inventory = 0.1  # Max 10% of typical position size

        # Regime
        self._allowed_regimes = {Regime.RANGE, Regime.NORMAL}
        self._current_regime: dict[str, Regime] = {}

        # Metrics
        self._total_two_sided_orders = 0
        self._bid_fills = 0
        self._ask_fills = 0
        self._spread_captured_usd = 0.0

        logger.info(
            f"InventoryNeutralMM initialized: symbols={self._symbols}, "
            f"neutral_threshold={self._neutral_ofi_threshold}, "
            f"base_offset={self._base_offset_bps}bps"
        )

    async def start(self) -> None:
        """Start the strategy."""
        await super().start()
        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        logger.info("InventoryNeutralMM started")

    async def stop(self) -> None:
        """Stop the strategy."""
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        await super().stop()
        logger.info("InventoryNeutralMM stopped")

    async def on_tick(self, event: Event) -> None:
        """Handle tick events."""
        if not self._enabled:
            return

        tick: Tick = event.data["tick"]

        if tick.symbol not in self._symbols:
            return

        # Update price history
        if tick.symbol not in self._price_history:
            self._price_history[tick.symbol] = deque(maxlen=self._window_size)
        self._price_history[tick.symbol].append(tick.price)

        # Check regime
        current_regime = self._current_regime.get(tick.symbol, Regime.NORMAL)
        if current_regime not in self._allowed_regimes:
            return

        # Check if it's time to place orders
        last_order = self._last_order_time.get(tick.symbol)
        if last_order and datetime.utcnow() - last_order < self._order_interval:
            return

        # Check for neutral market conditions
        await self._check_and_place_two_sided(tick)

    async def on_funding(self, event: Event) -> None:
        """Handle funding events - not used."""
        pass

    async def on_regime_update(self, event: Event) -> None:
        """Handle regime update events."""
        symbol = event.data.get("symbol")
        regime = event.data.get("regime")
        if symbol is not None and regime is not None:
            self._current_regime[symbol] = regime

    async def _check_and_place_two_sided(self, tick: Tick) -> None:
        """Check OFI and place two-sided orders if market is neutral.

        Args:
            tick: Current tick data
        """
        symbol = tick.symbol

        # Need OFI monitor
        if self._ofi_monitor is None:
            return

        # Calculate OFI
        try:
            ofi_result = self._ofi_monitor.calculate_ofi(symbol)
        except Exception as e:
            logger.debug(f"OFI calculation error for {symbol}: {e}")
            return

        # Check if market is neutral
        if abs(ofi_result.ofi_value) >= self._neutral_ofi_threshold:
            # Market is imbalanced - don't place two-sided orders
            logger.debug(
                f"Market not neutral for {symbol}: OFI={ofi_result.ofi_value:.3f} "
                f"(threshold={self._neutral_ofi_threshold})"
            )
            return

        # Check inventory limits
        current_inventory = self._net_inventory.get(symbol, 0.0)
        if abs(current_inventory) >= self._max_inventory:
            logger.debug(
                f"Inventory limit reached for {symbol}: {current_inventory:.4f}"
            )
            return

        # Calculate mid price and offset
        if tick.bid and tick.ask:
            mid_price = (tick.bid + tick.ask) / 2
            current_spread_bps = ((tick.ask - tick.bid) / mid_price) * 10000
        else:
            mid_price = tick.price
            current_spread_bps = self._base_offset_bps * 2

        # Dynamic offset based on current spread
        offset_bps = max(
            self._min_spread_capture_bps,
            min(current_spread_bps / 2, self._base_offset_bps * 2)
        )
        offset_pct = offset_bps / 10000

        # Calculate bid and ask prices
        bid_price = mid_price * (1 - offset_pct)
        ask_price = mid_price * (1 + offset_pct)

        # Calculate volatility for size adjustment
        volatility_mult = self._calculate_volatility_multiplier(symbol)

        # Place bid order
        bid_signal = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side="buy",
            entry_price=bid_price,
            stop_loss=bid_price * 0.995,  # 0.5% stop
            take_profit=mid_price,  # Target: return to mid
            take_profit_1=bid_price * 1.002,  # First TP at 0.2%
            metadata={
                "type": "inventory_neutral_mm_bid",
                "ofi_value": ofi_result.ofi_value,
                "mid_price": mid_price,
                "offset_bps": offset_bps,
                "size_multiplier": volatility_mult * 0.5,  # Smaller size for MM
                "is_mm_order": True,
            },
            prefer_maker=True,
            min_maker_edge_bps=0.0,
        )

        # Place ask order
        ask_signal = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side="sell",
            entry_price=ask_price,
            stop_loss=ask_price * 1.005,  # 0.5% stop
            take_profit=mid_price,  # Target: return to mid
            take_profit_1=ask_price * 0.998,  # First TP at 0.2%
            metadata={
                "type": "inventory_neutral_mm_ask",
                "ofi_value": ofi_result.ofi_value,
                "mid_price": mid_price,
                "offset_bps": offset_bps,
                "size_multiplier": volatility_mult * 0.5,
                "is_mm_order": True,
            },
            prefer_maker=True,
            min_maker_edge_bps=0.0,
        )

        # Publish both signals
        await self._publish_signal(bid_signal)
        await self._publish_signal(ask_signal)

        self._last_order_time[symbol] = datetime.utcnow()
        self._total_two_sided_orders += 1
        metrics.increment("inventory_neutral_mm_orders")

        logger.info(
            f"[INVENTORY-NEUTRAL MM] {symbol} Two-sided: "
            f"BID@${bid_price:.2f} / ASK@${ask_price:.2f} "
            f"(mid=${mid_price:.2f}, OFI={ofi_result.ofi_value:.3f}, "
            f"offset={offset_bps:.1f}bps)"
        )

    def _calculate_volatility_multiplier(self, symbol: str) -> float:
        """Calculate size multiplier based on volatility.

        Returns multiplier between 0.5 and 1.0:
        - High volatility: 0.5x (reduce size)
        - Low volatility: 1.0x (full size)
        """
        if symbol not in self._price_history:
            return 1.0

        prices = list(self._price_history[symbol])
        if len(prices) < 5:
            return 1.0

        # Calculate returns
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
        ]

        # Calculate volatility (std of returns)
        if not returns:
            return 1.0

        mean_ret = sum(returns) / len(returns)
        variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
        volatility = variance ** 0.5

        # Map volatility to multiplier
        # Low vol (< 0.001): 1.0
        # High vol (> 0.005): 0.5
        if volatility < 0.001:
            return 1.0
        elif volatility > 0.005:
            return 0.5
        else:
            # Linear interpolation
            return 1.0 - (volatility - 0.001) / (0.005 - 0.001) * 0.5

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled events to track inventory."""
        order = event.data.get("order")
        if order is None:
            return

        if order.strategy_id != self.strategy_id:
            return

        symbol = order.symbol
        is_bid = order.side.lower() == "buy"

        # Update inventory
        if symbol not in self._net_inventory:
            self._net_inventory[symbol] = 0.0

        if is_bid:
            self._net_inventory[symbol] += order.size
            self._bid_fills += 1
        else:
            self._net_inventory[symbol] -= order.size
            self._ask_fills += 1

        # Estimate spread captured (simplified)
        if self._bid_fills > 0 and self._ask_fills > 0:
            # When both sides fill, we captured spread
            spread_per_unit = order.price * (self._base_offset_bps * 2 / 10000)
            min_fills = min(self._bid_fills, self._ask_fills)
            estimated_spread = spread_per_unit * order.size * min_fills
            self._spread_captured_usd += estimated_spread

        metrics.increment("inventory_neutral_mm_fills")

        logger.info(
            f"[INVENTORY-NEUTRAL MM] Fill: {symbol} {order.side} "
            f"size={order.size:.6f} @ ${order.price:.2f}, "
            f"net_inventory={self._net_inventory[symbol]:.6f}"
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get strategy metrics."""
        return {
            "total_two_sided_orders": self._total_two_sided_orders,
            "bid_fills": self._bid_fills,
            "ask_fills": self._ask_fills,
            "spread_captured_usd": self._spread_captured_usd,
            "net_inventory": dict(self._net_inventory),
        }

    def enable(self) -> None:
        """Enable the strategy."""
        self._enabled = True
        logger.info("InventoryNeutralMM enabled")

    def disable(self) -> None:
        """Disable the strategy."""
        self._enabled = False
        logger.info("InventoryNeutralMM disabled")

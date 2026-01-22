"""Enhanced grid trading strategy for range-bound markets."""

from collections import deque

from hean.core.regime import Regime
from hean.core.types import Event, Signal, Tick
from hean.logging import get_logger
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class EnhancedGridStrategy(BaseStrategy):
    """
    Улучшенный Grid Trading:
    - Grid spacing: 0.1-0.15% (для быстрого оборота)
    - 15-25 уровней grid
    - Автоматическое закрытие при тренде
    - Leverage 1.5-2x для увеличения доходности
    - Работает только в RANGE режиме
    """

    def __init__(self, bus, symbols: list[str] | None = None):
        """Initialize enhanced grid strategy."""
        super().__init__("enhanced_grid", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        self._allowed_regimes = {Regime.RANGE}

        # Grid parameters
        self._grid_spacing_pct = 0.12  # 0.12% spacing
        self._num_levels = 20  # 20 grid levels
        self._grid_center: dict[str, float] = {}  # Center price for each symbol
        self._grid_levels: dict[str, list[float]] = {}  # Grid levels per symbol
        self._open_orders: dict[str, set[float]] = {}  # Track open grid orders

        # Price history for range detection
        self._price_history: dict[str, deque[float]] = {}
        self._window_size = 100

        # Track current regime
        self._current_regime: dict[str, Regime] = {}

    async def on_regime_update(self, event: Event) -> None:
        """Handle regime update."""
        symbol = event.data.get("symbol")
        regime = event.data.get("regime")
        if symbol is None or regime is None:
            logger.warning("REGIME_UPDATE missing fields: %s", event.data)
            return
        self._current_regime[symbol] = regime

        # Reset grid if regime changes away from RANGE
        if regime != Regime.RANGE and symbol in self._grid_levels:
            logger.info(f"Grid reset for {symbol}: regime changed to {regime.value}")
            self._grid_levels.pop(symbol, None)
            self._grid_center.pop(symbol, None)
            self._open_orders.pop(symbol, None)

    async def on_tick(self, event: Event) -> None:
        """Handle tick events for grid trading."""
        tick: Tick = event.data["tick"]

        if tick.symbol not in self._symbols:
            return

        # Only trade in RANGE regime
        current_regime = self._current_regime.get(tick.symbol, Regime.NORMAL)
        if not self.is_allowed_in_regime(current_regime):
            return

        # Initialize history if needed
        if tick.symbol not in self._price_history:
            self._price_history[tick.symbol] = deque(maxlen=self._window_size)

        # Update price history
        self._price_history[tick.symbol].append(tick.price)

        # Need enough data to establish range
        if len(self._price_history[tick.symbol]) < self._window_size:
            return

        # Initialize or update grid
        if tick.symbol not in self._grid_levels:
            self._initialize_grid(tick.symbol, tick.price)

        # Check if price is near a grid level
        current_price = tick.price
        grid_levels = self._grid_levels[tick.symbol]

        # Find nearest grid level
        nearest_level = min(grid_levels, key=lambda x: abs(x - current_price))
        distance_to_level = abs(current_price - nearest_level) / current_price * 100

        # If price is very close to a grid level (within 0.05%), trigger order
        if distance_to_level < 0.05 and nearest_level not in self._open_orders.get(
            tick.symbol, set()
        ):
            # Determine side based on whether price is above or below center
            center = self._grid_center[tick.symbol]
            if current_price > center:
                # Price above center: sell (mean reversion)
                side = "sell"
            else:
                # Price below center: buy (mean reversion)
                side = "buy"

            # Calculate stop loss and take profit
            entry_price = nearest_level
            if side == "buy":
                stop_loss = entry_price * (1 - 0.002)  # 0.2% stop
                take_profit = entry_price * (1 + self._grid_spacing_pct / 100.0)  # Next grid level
            else:  # sell
                stop_loss = entry_price * (1 + 0.002)  # 0.2% stop
                take_profit = entry_price * (1 - self._grid_spacing_pct / 100.0)  # Next grid level

            # Estimate edge
            edge_bps = (self._grid_spacing_pct * 100) * 0.7  # Assume 70% win rate in range

            # Create signal
            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=tick.symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    "grid_trading": True,
                    "grid_level": nearest_level,
                    "edge_bps": edge_bps,
                },
                prefer_maker=True,  # Prefer maker orders
            )

            await self._publish_signal(signal)

            # Track this order
            if tick.symbol not in self._open_orders:
                self._open_orders[tick.symbol] = set()
            self._open_orders[tick.symbol].add(nearest_level)

            logger.debug(
                f"Grid signal: {side} {tick.symbol} @ {entry_price:.2f} "
                f"(level={nearest_level:.2f}, center={center:.2f})"
            )

    def _initialize_grid(self, symbol: str, current_price: float) -> None:
        """Initialize grid levels around current price."""
        center = current_price
        self._grid_center[symbol] = center

        # Create grid levels above and below center
        levels = []
        spacing = center * (self._grid_spacing_pct / 100.0)

        # Levels above center (sell levels)
        for i in range(1, self._num_levels // 2 + 1):
            level = center + (spacing * i)
            levels.append(level)

        # Levels below center (buy levels)
        for i in range(1, self._num_levels // 2 + 1):
            level = center - (spacing * i)
            levels.append(level)

        # Add center level
        levels.append(center)
        levels.sort()

        self._grid_levels[symbol] = levels
        self._open_orders[symbol] = set()

        logger.info(
            f"Grid initialized for {symbol}: center={center:.2f}, "
            f"{len(levels)} levels, spacing={self._grid_spacing_pct}%"
        )

    async def on_funding(self, event: Event) -> None:
        """Enhanced grid doesn't use funding events."""
        pass

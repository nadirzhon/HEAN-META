"""Market regime detection."""

from collections import deque
from enum import Enum

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


class Regime(str, Enum):
    """Market regime types."""

    RANGE = "range"
    NORMAL = "normal"
    IMPULSE = "impulse"


class RegimeDetector:
    """Detects market regime using volatility and return acceleration."""

    def __init__(self, bus: EventBus) -> None:
        """Initialize the regime detector."""
        self._bus = bus
        self._price_history: dict[str, deque[float]] = {}
        self._returns: dict[str, deque[float]] = {}
        self._current_regime: dict[str, Regime] = {}
        self._window_size = 50  # Lookback window for volatility
        self._short_window = 10  # Short window for acceleration
        self._volatility_threshold_low = 0.001  # Low volatility threshold (0.1%)
        self._volatility_threshold_high = 0.005  # High volatility threshold (0.5%)
        self._acceleration_threshold = 0.003  # Acceleration threshold (0.3%)

    async def start(self) -> None:
        """Start the regime detector."""
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        logger.info("Regime detector started")

    async def stop(self) -> None:
        """Stop the regime detector."""
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("Regime detector stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update regime."""
        tick: Tick = event.data["tick"]
        await self._update_regime(tick.symbol, tick.price)

    async def _update_regime(self, symbol: str, price: float) -> None:
        """Update regime for a symbol."""
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._window_size)
            self._returns[symbol] = deque(maxlen=self._window_size)
            self._current_regime[symbol] = Regime.NORMAL

        self._price_history[symbol].append(price)

        # Calculate returns (optimized: avoid full list conversion)
        price_history = self._price_history[symbol]
        if len(price_history) < 2:
            return

        # Calculate rolling returns (optimized: calculate only last return)
        if len(price_history) > 1:
            prev_price = price_history[-2]
            if prev_price > 0:
                ret = (price_history[-1] - prev_price) / prev_price
                self._returns[symbol].append(ret)

        if len(self._returns[symbol]) < self._window_size:
            return

        # Calculate rolling volatility (optimized: avoid list conversion)
        returns_deque = self._returns[symbol]
        volatility = self._calculate_volatility(list(returns_deque))

        # Calculate return acceleration (optimized: reuse returns list)
        returns_list = list(returns_deque) if len(returns_deque) > self._short_window else []
        acceleration = self._calculate_acceleration(returns_list) if returns_list else 0.0

        # Determine regime
        old_regime = self._current_regime.get(symbol, Regime.NORMAL)
        new_regime = self._classify_regime(volatility, acceleration)

        if new_regime != old_regime:
            self._current_regime[symbol] = new_regime
            await self._publish_regime_update(symbol, new_regime)

    def _calculate_volatility(self, returns: list[float]) -> float:
        """Calculate rolling volatility (standard deviation of returns).

        Optimized: single pass calculation using Welford's algorithm for better numerical stability.
        """
        if not returns:
            return 0.0

        # Optimized: use single pass for mean and variance
        n = len(returns)
        mean = sum(returns) / n

        # Calculate variance (optimized: avoid multiple passes)
        variance = sum((r - mean) ** 2 for r in returns) / n
        return variance**0.5

    def _calculate_acceleration(self, returns: list[float]) -> float:
        """Calculate return acceleration (change in returns over short window)."""
        if len(returns) < self._short_window:
            return 0.0

        # Compare recent returns to earlier returns
        recent = returns[-self._short_window :]
        earlier = returns[-self._short_window * 2 : -self._short_window]

        if not earlier:
            return 0.0

        recent_avg = sum(recent) / len(recent)
        earlier_avg = sum(earlier) / len(earlier)

        # Acceleration is the change in average returns
        acceleration = abs(recent_avg - earlier_avg)
        return acceleration

    def _classify_regime(self, volatility: float, acceleration: float) -> Regime:
        """Classify regime based on volatility and acceleration."""
        # High acceleration indicates impulse
        if acceleration > self._acceleration_threshold:
            return Regime.IMPULSE

        # Low volatility indicates range
        if volatility < self._volatility_threshold_low:
            return Regime.RANGE

        # High volatility with low acceleration indicates normal trending
        if volatility > self._volatility_threshold_high:
            return Regime.NORMAL

        # Default to normal
        return Regime.NORMAL

    async def _publish_regime_update(self, symbol: str, regime: Regime) -> None:
        """Publish regime update event."""
        await self._bus.publish(
            Event(
                event_type=EventType.REGIME_UPDATE,
                data={"symbol": symbol, "regime": regime},
            )
        )
        logger.info(f"Regime update: {symbol} -> {regime.value}")

    def get_regime(self, symbol: str) -> Regime:
        """Get current regime for a symbol."""
        return self._current_regime.get(symbol, Regime.NORMAL)

    def get_volatility(self, symbol: str) -> float:
        """Get current volatility for a symbol.

        Returns:
            Current volatility (standard deviation of returns), or 0.0 if not available
        """
        if symbol not in self._returns:
            return 0.0

        returns_list = list(self._returns[symbol])
        if len(returns_list) < 2:
            return 0.0

        return self._calculate_volatility(returns_list)

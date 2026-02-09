"""Momentum trading strategy - simple working version."""

from collections import deque

from hean.core.bus import EventBus
from hean.core.types import Event, Signal, Tick
from hean.logging import get_logger
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class MomentumTrader(BaseStrategy):
    """Momentum trading strategy.

    Detects price momentum and generates trading signals.
    Simple implementation that tracks price history and calculates
    average return (momentum) to generate buy/sell signals.
    """

    def __init__(self, bus: EventBus, symbols: list[str]) -> None:
        """Initialize momentum trader.

        Args:
            bus: Event bus
            symbols: List of symbols to trade
        """
        super().__init__("momentum_trader", bus)
        self._symbols = symbols
        self._price_history: dict[str, deque[float]] = {}
        self._window_size = 10
        self._last_signal_time: dict[str, float] = {}
        self._momentum_threshold = 0.001  # 0.1% momentum threshold

    async def on_tick(self, event: Event) -> None:
        """Process tick - detect momentum.

        Args:
            event: Tick event
        """
        tick: Tick = event.data["tick"]

        if tick.symbol not in self._symbols:
            return

        # Initialize price history
        if tick.symbol not in self._price_history:
            self._price_history[tick.symbol] = deque(maxlen=self._window_size)

        self._price_history[tick.symbol].append(tick.price)

        # Need at least window_size prices to calculate momentum
        if len(self._price_history[tick.symbol]) < self._window_size:
            return

        # Calculate returns
        prices = list(self._price_history[tick.symbol])
        returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]

        # Calculate momentum (average return)
        momentum = sum(returns) / len(returns) if returns else 0.0

        # Generate signal based on momentum
        side = None
        if momentum > self._momentum_threshold:
            side = "buy"
        elif momentum < -self._momentum_threshold:
            side = "sell"
        else:
            return  # No signal

        # Generate signal
        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=tick.symbol,
            side=side,
            entry_price=tick.price,
            stop_loss=tick.price * (0.98 if side == "buy" else 1.02),  # 2% stop
            take_profit=tick.price * (1.02 if side == "buy" else 0.98),  # 2% target
            metadata={"momentum": momentum, "reason": "momentum"},
        )

        await self._publish_signal(signal)
        logger.info(
            f"Momentum signal: {side} {tick.symbol} @ {tick.price:.2f}, momentum={momentum:.4f}"
        )

    async def on_funding(self, event: Event) -> None:
        """Handle funding event - not used by momentum strategy."""
        pass

"""
Adaptive Response Loop: Feedback-Agent for Slippage Monitoring
Monitors real-time slippage and switches to 'Hidden-Liquidity' mode when needed.
"""

from collections import deque
from dataclasses import dataclass
from datetime import datetime

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Order
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SlippageMetrics:
    """Slippage metrics for a symbol."""
    symbol: str
    expected_price: float
    actual_price: float
    slippage_bps: float  # Basis points
    order_size: float
    timestamp: datetime
    order_id: str

    def calculate_slippage_bps(self) -> float:
        """Calculate slippage in basis points."""
        if self.expected_price > 0:
            slippage_pct = abs(self.actual_price - self.expected_price) / self.expected_price
            return slippage_pct * 10000.0  # Convert to basis points
        return 0.0


class HiddenLiquidityConfig:
    """Configuration for hidden liquidity mode."""

    def __init__(
        self,
        min_fragment_size: float = 0.01,  # Minimum size per fragment
        max_fragments: int = 10,  # Maximum number of fragments
        fragment_interval_ms: int = 100,  # Delay between fragments (ms)
        aggressive_fragment_size: float = 0.005,  # Smaller fragments when slippage is high
    ):
        self.min_fragment_size = min_fragment_size
        self.max_fragments = max_fragments
        self.fragment_interval_ms = fragment_interval_ms
        self.aggressive_fragment_size = aggressive_fragment_size


class FeedbackAgent:
    """
    Feedback-Agent that monitors slippage and adapts execution strategy.

    If slippage increases, automatically switches to 'Hidden-Liquidity' mode,
    splitting orders into even smaller fragments.
    """

    def __init__(
        self,
        bus: EventBus,
        slippage_threshold_bps: float = 5.0,  # 5 bps threshold
        high_slippage_threshold_bps: float = 10.0,  # 10 bps for aggressive mode
        lookback_window: int = 50,  # Number of recent trades to analyze
    ):
        """Initialize the feedback agent.

        Args:
            bus: Event bus for publishing events
            slippage_threshold_bps: Slippage threshold in basis points to trigger hidden liquidity mode
            high_slippage_threshold_bps: High slippage threshold for aggressive fragmentation
            lookback_window: Number of recent trades to keep in memory
        """
        self._bus = bus
        self._slippage_threshold_bps = slippage_threshold_bps
        self._high_slippage_threshold_bps = high_slippage_threshold_bps
        self._lookback_window = lookback_window

        # Slippage tracking per symbol
        self._slippage_history: dict[str, deque[SlippageMetrics]] = {}

        # Current execution mode per symbol
        self._execution_modes: dict[str, str] = {}  # "normal" or "hidden_liquidity"

        # Hidden liquidity configuration per symbol
        self._hidden_liquidity_configs: dict[str, HiddenLiquidityConfig] = {}

        # Slippage statistics
        self._avg_slippage: dict[str, float] = {}
        self._max_slippage: dict[str, float] = {}

        self._running = False

    async def start(self) -> None:
        """Start the feedback agent."""
        self._running = True
        self._bus.subscribe(EventType.ORDER_PLACED, self._handle_order_placed)
        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        logger.info("Feedback Agent started - monitoring slippage and adapting execution")

    async def stop(self) -> None:
        """Stop the feedback agent."""
        self._running = False
        self._bus.unsubscribe(EventType.ORDER_PLACED, self._handle_order_placed)
        self._bus.unsubscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        logger.info("Feedback Agent stopped")

    async def _handle_order_placed(self, event: Event) -> None:
        """Handle order placed event to track expected price."""
        order: Order = event.data.get("order")
        if not order:
            return

        symbol = order.symbol

        # Initialize tracking if needed
        if symbol not in self._slippage_history:
            self._slippage_history[symbol] = deque(maxlen=self._lookback_window)
            self._execution_modes[symbol] = "normal"
            self._hidden_liquidity_configs[symbol] = HiddenLiquidityConfig()
            self._avg_slippage[symbol] = 0.0
            self._max_slippage[symbol] = 0.0

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled event to calculate slippage."""
        order: Order = event.data.get("order")
        if not order or not order.filled_at:
            return

        symbol = order.symbol

        # Calculate slippage
        expected_price = order.price if order.price else order.filled_price
        actual_price = order.filled_price

        if expected_price and actual_price and expected_price > 0:
            metrics = SlippageMetrics(
                symbol=symbol,
                expected_price=expected_price,
                actual_price=actual_price,
                slippage_bps=0.0,  # Will be calculated
                order_size=order.filled_size or order.size,
                timestamp=order.filled_at,
                order_id=order.order_id
            )
            metrics.slippage_bps = metrics.calculate_slippage_bps()

            # Update history
            if symbol in self._slippage_history:
                self._slippage_history[symbol].append(metrics)

                # Update statistics
                self._update_slippage_statistics(symbol)

                # Check if we need to switch to hidden liquidity mode
                await self._check_and_adapt(symbol)

                logger.debug(
                    f"Order {order.order_id} filled: slippage={metrics.slippage_bps:.2f} bps, "
                    f"mode={self._execution_modes.get(symbol, 'normal')}"
                )

    def _update_slippage_statistics(self, symbol: str) -> None:
        """Update slippage statistics for a symbol."""
        if symbol not in self._slippage_history:
            return

        history = self._slippage_history[symbol]
        if not history:
            return

        # Calculate average and max slippage
        slippages = [m.slippage_bps for m in history]
        self._avg_slippage[symbol] = sum(slippages) / len(slippages)
        self._max_slippage[symbol] = max(slippages)

    async def _check_and_adapt(self, symbol: str) -> None:
        """Check slippage and adapt execution mode if needed."""
        if symbol not in self._slippage_history:
            return

        history = self._slippage_history[symbol]
        if len(history) < 3:  # Need minimum history
            return

        current_mode = self._execution_modes.get(symbol, "normal")
        avg_slippage = self._avg_slippage.get(symbol, 0.0)
        max_slippage = self._max_slippage.get(symbol, 0.0)

        # Decision logic: switch to hidden liquidity if slippage exceeds threshold
        if current_mode == "normal" and avg_slippage > self._slippage_threshold_bps:
            # Switch to hidden liquidity mode
            self._execution_modes[symbol] = "hidden_liquidity"
            logger.warning(
                f"Switching {symbol} to HIDDEN_LIQUIDITY mode: "
                f"avg_slippage={avg_slippage:.2f} bps > threshold={self._slippage_threshold_bps:.2f} bps"
            )

            # Adjust configuration based on slippage severity
            config = self._hidden_liquidity_configs[symbol]
            if max_slippage > self._high_slippage_threshold_bps:
                # Aggressive fragmentation
                config.min_fragment_size = config.aggressive_fragment_size
                config.max_fragments = 20  # More fragments
                config.fragment_interval_ms = 50  # Faster execution
                logger.warning(
                    f"Aggressive fragmentation enabled for {symbol}: "
                    f"max_slippage={max_slippage:.2f} bps"
                )

            # Publish mode change event
            await self._bus.publish(Event(
                event_type=EventType.CONTEXT_UPDATE,
                data={
                    "symbol": symbol,
                    "execution_mode": "hidden_liquidity",
                    "reason": f"slippage_threshold_exceeded: {avg_slippage:.2f} bps",
                    "config": {
                        "min_fragment_size": config.min_fragment_size,
                        "max_fragments": config.max_fragments,
                        "fragment_interval_ms": config.fragment_interval_ms,
                    }
                }
            ))

        elif current_mode == "hidden_liquidity" and avg_slippage < self._slippage_threshold_bps * 0.5:
            # Switch back to normal mode if slippage improves significantly
            self._execution_modes[symbol] = "normal"
            logger.info(
                f"Switching {symbol} back to NORMAL mode: "
                f"avg_slippage={avg_slippage:.2f} bps < threshold={self._slippage_threshold_bps * 0.5:.2f} bps"
            )

            # Reset configuration
            config = self._hidden_liquidity_configs[symbol]
            config.min_fragment_size = 0.01
            config.max_fragments = 10
            config.fragment_interval_ms = 100

            await self._bus.publish(Event(
                event_type=EventType.CONTEXT_UPDATE,
                data={
                    "symbol": symbol,
                    "execution_mode": "normal",
                    "reason": f"slippage_improved: {avg_slippage:.2f} bps"
                }
            ))

    def get_execution_mode(self, symbol: str) -> str:
        """Get current execution mode for a symbol."""
        return self._execution_modes.get(symbol, "normal")

    def get_hidden_liquidity_config(self, symbol: str) -> HiddenLiquidityConfig | None:
        """Get hidden liquidity configuration for a symbol."""
        return self._hidden_liquidity_configs.get(symbol)

    def get_slippage_statistics(self, symbol: str) -> dict:
        """Get slippage statistics for a symbol."""
        return {
            "avg_slippage_bps": self._avg_slippage.get(symbol, 0.0),
            "max_slippage_bps": self._max_slippage.get(symbol, 0.0),
            "mode": self._execution_modes.get(symbol, "normal"),
            "recent_trades": len(self._slippage_history.get(symbol, deque()))
        }

    def should_use_hidden_liquidity(self, symbol: str) -> bool:
        """Check if hidden liquidity mode should be used for a symbol."""
        return self.get_execution_mode(symbol) == "hidden_liquidity"

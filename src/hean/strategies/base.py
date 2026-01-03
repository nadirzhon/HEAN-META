"""Base strategy class."""

from abc import ABC, abstractmethod

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal
from hean.logging import get_logger
from hean.observability.no_trade_report import no_trade_report

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, strategy_id: str, bus: EventBus) -> None:
        """Initialize the strategy."""
        self.strategy_id = strategy_id
        self._bus = bus
        self._running = False
        self._allowed_regimes: set = set()  # Override in subclasses

    async def start(self) -> None:
        """Start the strategy."""
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._bus.subscribe(EventType.FUNDING, self._handle_funding)
        self._bus.subscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        self._running = True
        logger.info(f"Strategy {self.strategy_id} started")

    async def stop(self) -> None:
        """Stop the strategy."""
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._bus.unsubscribe(EventType.FUNDING, self._handle_funding)
        self._bus.unsubscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        self._running = False
        logger.info(f"Strategy {self.strategy_id} stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events."""
        if not self._running:
            return
        await self.on_tick(event)

    async def _handle_funding(self, event: Event) -> None:
        """Handle funding events."""
        if not self._running:
            return
        await self.on_funding(event)

    async def _handle_regime_update(self, event: Event) -> None:
        """Handle regime update events."""
        if not self._running:
            return
        await self.on_regime_update(event)

    @abstractmethod
    async def on_tick(self, event: Event) -> None:
        """Handle tick event - implement in subclass."""
        ...

    @abstractmethod
    async def on_funding(self, event: Event) -> None:
        """Handle funding event - implement in subclass."""
        ...

    async def on_regime_update(self, event: Event) -> None:
        """Handle regime update event - override in subclass if needed."""
        pass  # Optional method, not abstract

    def is_allowed_in_regime(self, regime) -> bool:
        """Check if strategy is allowed in current regime."""
        if not self._allowed_regimes:
            return True  # No restrictions if not set
        return regime in self._allowed_regimes

    async def _publish_signal(self, signal: Signal) -> None:
        """Publish a trading signal."""
        # Track signal emission
        no_trade_report.increment_pipeline("signals_emitted", self.strategy_id)

        logger.info(
            f"[FORCED_PUBLISH] Publishing signal: {self.strategy_id} {signal.symbol} {signal.side}"
        )
        await self._bus.publish(
            Event(
                event_type=EventType.SIGNAL,
                data={"signal": signal},
            )
        )
        logger.info(
            f"[FORCED_PUBLISH] Signal published: {self.strategy_id} {signal.symbol} {signal.side}"
        )

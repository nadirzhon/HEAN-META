"""Base strategy class."""

from abc import ABC, abstractmethod

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal, Tick
from hean.logging import get_logger
from hean.observability.no_trade_report import no_trade_report
from hean.observability.signal_rejection_telemetry import signal_rejection_telemetry
from hean.risk.price_anomaly_detector import price_anomaly_detector

logger = get_logger(__name__)


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, strategy_id: str, bus: EventBus) -> None:
        """Initialize the strategy."""
        self.strategy_id = strategy_id
        self._bus = bus
        self._running = False
        self._allowed_regimes: set = set()  # Override in subclasses
        self._market_context = None  # Unified context from ContextAggregator

    async def start(self) -> None:
        """Start the strategy."""
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._bus.subscribe(EventType.FUNDING, self._handle_funding)
        self._bus.subscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        self._bus.subscribe(EventType.CONTEXT_READY, self._handle_context_ready)
        self._running = True
        logger.info(f"Strategy {self.strategy_id} started")

    async def stop(self) -> None:
        """Stop the strategy."""
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._bus.unsubscribe(EventType.FUNDING, self._handle_funding)
        self._bus.unsubscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        self._bus.unsubscribe(EventType.CONTEXT_READY, self._handle_context_ready)
        self._running = False
        logger.info(f"Strategy {self.strategy_id} stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events."""
        if not self._running:
            return

        # Check price anomaly before passing to strategy
        tick_data = event.data.get("tick")
        if tick_data and hasattr(tick_data, "symbol") and hasattr(tick_data, "price"):
            symbol = tick_data.symbol
            price = tick_data.price

            # Check for price anomalies (gaps, spikes, flash crashes)
            anomaly = price_anomaly_detector.check_price(symbol, price)
            if anomaly and anomaly.should_block_trading:
                logger.warning(
                    f"[ANOMALY] Trading blocked for {symbol}: {anomaly.anomaly_type.value} "
                    f"({anomaly.change_pct:+.2f}%)"
                )
                no_trade_report.increment("price_anomaly_block", symbol, self.strategy_id)
                signal_rejection_telemetry.record_rejection(
                    reason="price_anomaly_block",
                    symbol=symbol,
                    strategy_id=self.strategy_id,
                    details={
                        "anomaly_type": anomaly.anomaly_type.value,
                        "change_pct": anomaly.change_pct,
                        "severity": anomaly.severity,
                    },
                )
                return  # Block tick processing during critical anomaly

            # Check if symbol is in anomaly cooldown
            if price_anomaly_detector.is_blocked(symbol):
                logger.debug(f"[ANOMALY] Symbol {symbol} in cooldown, skipping tick")
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

    async def _handle_context_ready(self, event: Event) -> None:
        """Handle unified context from ContextAggregator.

        Context includes physics, brain, oracle, OFI, and causal signals.
        Strategies can use this enriched context to adjust signal confidence,
        filter signals, or adapt parameters.
        """
        if not self._running:
            return
        context = event.data.get("context")
        if context:
            self._market_context = context
        await self.on_context_ready(event)

    @abstractmethod
    async def on_tick(self, event: Event) -> None:
        """Handle tick event - implement in subclass."""
        ...

    @abstractmethod
    async def on_funding(self, event: Event) -> None:
        """Handle funding event - implement in subclass."""
        ...

    async def on_regime_update(self, event: Event) -> None:  # noqa: B027
        """Handle regime update event - override in subclass if needed."""
        pass  # Optional method, not abstract

    async def on_context_ready(self, event: Event) -> None:  # noqa: B027
        """Handle unified context from ContextAggregator - override in subclass if needed.

        Event data contains:
            - symbol: Trading symbol
            - timestamp: Context timestamp
            - physics: Physics state (temperature, entropy, phase, etc.)
            - brain: Brain analysis (confidence, reasoning, etc.)
            - oracle: Oracle prediction (reversal probability, etc.)
            - ofi: Order flow imbalance
            - causal: Causal signals

        Strategies can use this to:
        - Apply physics-based filters (e.g., don't trade in extreme entropy)
        - Weight signals by brain confidence
        - React to oracle reversal predictions
        - Adjust position sizing based on OFI
        """
        pass  # Optional method, not abstract

    def is_allowed_in_regime(self, regime) -> bool:
        """Check if strategy is allowed in current regime."""
        if not self._allowed_regimes:
            return True  # No restrictions if not set
        return regime in self._allowed_regimes

    def get_anomaly_size_multiplier(self, symbol: str) -> float:
        """Get position size multiplier based on recent price anomalies.

        Returns a value between 0.0 and 1.0 that should be applied
        to the signal's size_multiplier to reduce exposure during
        periods of detected anomalies.

        Args:
            symbol: Trading symbol

        Returns:
            Size multiplier (0.0 to 1.0)
        """
        return price_anomaly_detector.get_size_multiplier(symbol)

    def reset_anomaly_count(self, symbol: str) -> None:
        """Reset anomaly count for a symbol after successful trade.

        Call this after a successful position close to gradually
        restore normal position sizing.

        Args:
            symbol: Trading symbol
        """
        price_anomaly_detector.reset_anomaly_count(symbol)

    async def _publish_signal(self, signal: Signal) -> None:
        """Publish a trading signal."""
        # Track signal emission
        no_trade_report.increment_pipeline("signals_emitted", self.strategy_id)

        # Record signal for rejection rate calculation
        signal_rejection_telemetry.record_signal()

        # Apply anomaly-based size reduction
        anomaly_multiplier = self.get_anomaly_size_multiplier(signal.symbol)
        if anomaly_multiplier < 1.0:
            # Store original multiplier if present
            current_mult = signal.metadata.get("size_multiplier", 1.0)
            signal.metadata["size_multiplier"] = current_mult * anomaly_multiplier
            signal.metadata["anomaly_size_reduction"] = anomaly_multiplier
            logger.info(
                f"[ANOMALY] Size reduced by {(1 - anomaly_multiplier) * 100:.0f}% for {signal.symbol} "
                f"(multiplier: {signal.metadata['size_multiplier']:.2f})"
            )

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

"""
Oracle Integration: Connect Oracle Engine with Trading Swarm
Integrates Algorithmic Fingerprinting and TCN predictions into trading decisions
"""

from datetime import datetime

from hean.core.bus import EventBus
from hean.core.intelligence.oracle_engine import OracleEngine
from hean.core.types import Event, EventType, Position, Signal, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


class OracleIntegration:
    """
    Oracle Integration: Integrates Oracle Engine with Trading Swarm.
    Provides Predictive Alpha signals that can trigger exits or position flips.
    """

    def __init__(self, bus: EventBus, symbols: list[str] | None = None):
        """
        Initialize Oracle Integration.

        Args:
            bus: Event bus
            symbols: List of symbols to track (if None, tracks all)
        """
        self._bus = bus
        self._symbols = set(symbols) if symbols else None
        self._oracle = OracleEngine(sequence_length=10000)
        self._running = False
        self._active_positions: dict[str, Position] = {}
        self._last_signals: dict[str, dict] = {}

    async def start(self) -> None:
        """Start Oracle Integration."""
        self._running = True

        # Subscribe to ticks for TCN predictions
        self._bus.subscribe(EventType.TICK, self._handle_tick)

        # Subscribe to position events to track open positions
        self._bus.subscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)

        # Subscribe to order book updates for fingerprinting
        self._bus.subscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook_update)

        logger.info("Oracle Integration started")

    async def stop(self) -> None:
        """Stop Oracle Integration."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._bus.unsubscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.unsubscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook_update)
        logger.info("Oracle Integration stopped")

    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events for TCN predictions."""
        if not self._running:
            return

        tick: Tick = event.data.get('tick')
        if not tick:
            return

        # Filter by symbols if specified
        if self._symbols and tick.symbol not in self._symbols:
            return

        # Update Oracle Engine with tick
        self._oracle.update_tick(tick)

        # Get predictive alpha signal
        alpha_signal = self._oracle.get_predictive_alpha(tick.symbol)

        if alpha_signal:
            self._last_signals[tick.symbol] = alpha_signal

            # If should exit or flip, publish signal
            if alpha_signal['should_exit']:
                await self._publish_exit_signal(tick.symbol, alpha_signal)
            elif alpha_signal['should_flip']:
                await self._publish_flip_signal(tick.symbol, alpha_signal)

        # Publish price predictions for UI
        price_predictions = self._oracle.get_price_predictions(tick.symbol)
        if price_predictions:
            await self._bus.publish(Event(
                event_type=EventType.CONTEXT_UPDATE,
                data={
                    'type': 'oracle_predictions',
                    'symbol': tick.symbol,
                    'predictions': price_predictions,
                    'tcn_reversal_prob': alpha_signal['tcn_reversal_prob'] if alpha_signal else 0.0,
                    'timestamp': datetime.utcnow()
                }
            ))

    async def _handle_position_opened(self, event: Event) -> None:
        """Track opened positions."""
        position: Position = event.data.get('position')
        if position:
            self._active_positions[position.symbol] = position

    async def _handle_position_closed(self, event: Event) -> None:
        """Track closed positions."""
        position: Position = event.data.get('position')
        if position and position.symbol in self._active_positions:
            del self._active_positions[position.symbol]

    async def _handle_orderbook_update(self, event: Event) -> None:
        """Handle order book updates for fingerprinting."""
        if not self._running:
            return

        orderbook = event.data.get('orderbook')
        if not orderbook:
            return

        symbol = orderbook.get('symbol')
        if not symbol:
            return

        # Filter by symbols if specified
        if self._symbols and symbol not in self._symbols:
            return

        # Update fingerprinting engine with large limit orders
        bids = orderbook.get('bids', [])
        asks = orderbook.get('asks', [])
        timestamp_ns = int(datetime.utcnow().timestamp() * 1e9)

        # Track large limit orders (only first few levels)
        for i, (price, size) in enumerate(bids[:5]):
            if size * price > 10000:  # $10k minimum
                order_id = f"{symbol}_bid_{i}_{timestamp_ns}"
                self._oracle.update_order_book(
                    symbol=symbol,
                    order_id=order_id,
                    price=price,
                    size=size,
                    timestamp_ns=timestamp_ns,
                    is_limit=True
                )

        for i, (price, size) in enumerate(asks[:5]):
            if size * price > 10000:  # $10k minimum
                order_id = f"{symbol}_ask_{i}_{timestamp_ns}"
                self._oracle.update_order_book(
                    symbol=symbol,
                    order_id=order_id,
                    price=price,
                    size=size,
                    timestamp_ns=timestamp_ns,
                    is_limit=True
                )

    async def _publish_exit_signal(self, symbol: str, alpha_signal: dict) -> None:
        """
        Publish exit signal based on Oracle prediction.

        Args:
            symbol: Trading symbol
            alpha_signal: Alpha signal dictionary
        """
        if symbol not in self._active_positions:
            return  # No position to exit

        position = self._active_positions[symbol]

        logger.warning(
            f"Oracle Exit Signal: {symbol} - Reversal probability: {alpha_signal['tcn_reversal_prob']:.2%}, "
            f"Signal strength: {alpha_signal['signal_strength']:.2%}"
        )

        # Publish position close request
        await self._bus.publish(Event(
            event_type=EventType.POSITION_CLOSE_REQUEST,
            data={
                'position_id': position.position_id,
                'reason': 'oracle_reversal_prediction',
                'tcn_reversal_prob': alpha_signal['tcn_reversal_prob'],
                'signal_strength': alpha_signal['signal_strength']
            }
        ))

    async def _publish_flip_signal(self, symbol: str, alpha_signal: dict) -> None:
        """
        Publish flip signal based on Oracle prediction.

        Args:
            symbol: Trading symbol
            alpha_signal: Alpha signal dictionary
        """
        if symbol not in self._active_positions:
            return  # No position to flip

        position = self._active_positions[symbol]
        fingerprint_alpha = alpha_signal.get('fingerprint_alpha')

        logger.warning(
            f"Oracle Flip Signal: {symbol} - Reversal probability: {alpha_signal['tcn_reversal_prob']:.2%}, "
            f"Fingerprint alpha: {fingerprint_alpha['alpha_signal'] if fingerprint_alpha else 0.0:.2f}, "
            f"Bot ID: {fingerprint_alpha['bot_id'] if fingerprint_alpha else 'N/A'}"
        )

        # First close current position
        await self._bus.publish(Event(
            event_type=EventType.POSITION_CLOSE_REQUEST,
            data={
                'position_id': position.position_id,
                'reason': 'oracle_flip_signal',
                'tcn_reversal_prob': alpha_signal['tcn_reversal_prob'],
                'fingerprint_alpha': fingerprint_alpha
            }
        ))

        # Then publish new signal in opposite direction
        new_side = 'sell' if position.side == 'long' else 'buy'
        new_signal = Signal(
            strategy_id='oracle_engine',
            symbol=symbol,
            side=new_side,
            entry_price=position.current_price,  # Use current price as entry
            stop_loss=position.stop_loss,  # Keep same stop loss
            take_profit=position.take_profit,  # Keep same take profit
            metadata={
                'source': 'oracle_flip',
                'original_position_id': position.position_id,
                'tcn_reversal_prob': alpha_signal['tcn_reversal_prob'],
                'fingerprint_alpha': fingerprint_alpha
            }
        )

        await self._bus.publish(Event(
            event_type=EventType.SIGNAL,
            data={'signal': new_signal}
        ))

    def get_last_predictions(self, symbol: str) -> dict | None:
        """
        Get last predictions for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Last predictions dictionary, or None
        """
        return self._last_signals.get(symbol)

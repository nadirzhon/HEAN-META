"""Impulse engine strategy - aggressive but bounded momentum trading."""

from collections import deque
from datetime import datetime, timedelta

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.density import DensityController
from hean.core.market_context import UnifiedMarketContext
from hean.core.regime import Regime
from hean.core.trade_density import trade_density
from hean.core.types import Event, EventType, Position, Signal, Tick
from hean.execution.edge_estimator import ExecutionEdgeEstimator
from hean.logging import get_logger
from hean.observability.metrics import metrics
from hean.observability.no_trade_report import no_trade_report
from hean.strategies.base import BaseStrategy
from hean.strategies.edge_confirmation import EdgeConfirmationLoop
from hean.strategies.impulse_filters import (
    ImpulseFilterPipeline,
    SpreadFilter,
    TimeWindowFilter,
    VolatilityExpansionFilter,
)
from hean.strategies.multi_factor_confirmation import MultiFactorConfirmation

logger = get_logger(__name__)


class ImpulseEngine(BaseStrategy):
    """Detects price impulses and trades with tight stops.

    Uses short-window returns, volume proxy, and spread gate.
    Respects attempt limits and cooldown.

    Primarily active in IMPULSE regime, but can optionally trade in NORMAL
    regime with reduced size (controlled via config).
    """

    def __init__(self, bus: EventBus, symbols: list[str] | None = None,
                 oracle_engine = None, ofi_monitor = None) -> None:
        """Initialize the impulse engine."""
        super().__init__("impulse_engine", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._oracle_engine = oracle_engine  # Oracle Engine for TCN predictions
        self._ofi_monitor = ofi_monitor  # OFI Monitor for order flow analysis
        self._price_history: dict[str, deque[float]] = {}
        self._volume_history: dict[str, deque[float]] = {}  # Real volume from tick data
        self._atr_history: dict[str, deque[float]] = {}  # Average True Range for adaptive thresholds
        self._volatility_history: dict[str, deque[float]] = {}  # For volatility spike detection
        self._last_trade_time: dict[str, datetime] = {}
        self._open_positions: dict[str, Position] = {}  # Track open positions
        self._window_size = 10  # Lookback window
        self._long_window = 50  # Long window for volatility calculation
        self._atr_window = 14  # Standard ATR period

        # Phase 2: Multi-Timeframe Momentum Cascade
        # Extended price history for 1m/5m/15m analysis (assuming ~1 tick/sec)
        self._mtf_price_history: dict[str, deque[float]] = {}
        self._mtf_window = 900  # 15 minutes of ticks at ~1/sec
        self._mtf_enabled = True
        self._mtf_cascade_count = 0  # Count of MTF cascade signals
        # Impulse detection threshold (0.5% = 50 bps)
        self._impulse_threshold = 0.005
        self._spread_gate = settings.impulse_max_spread_bps / 10000.0  # Convert bps to decimal
        self._volatility_spike_threshold = settings.impulse_max_volatility_spike
        self._max_time_in_trade = timedelta(seconds=settings.impulse_max_time_in_trade_sec)
        # Active in IMPULSE, and optionally in NORMAL with reduced size
        if settings.impulse_allow_normal:
            self._allowed_regimes = {Regime.IMPULSE, Regime.NORMAL}
        else:
            self._allowed_regimes = {Regime.IMPULSE}
        self._current_regime: dict[str, Regime] = {}
        # Execution edge estimator
        self._edge_estimator = ExecutionEdgeEstimator()
        # Trade density controller (anti-starvation) for this strategy
        self._density_controller = DensityController(self.strategy_id)
        # Micro-filters for precision improvement
        self._filter_pipeline = ImpulseFilterPipeline(
            [
                SpreadFilter(),
                VolatilityExpansionFilter(),
                TimeWindowFilter(),
            ]
        )
        # Edge confirmation loop (2-step entries)
        self._edge_confirmation = EdgeConfirmationLoop()
        # Multi-factor confirmation system for improved signal quality
        self._multi_factor = MultiFactorConfirmation()
        self._multi_factor_enabled = True  # Can be disabled via settings
        # Metrics
        self._trade_times: list[float] = []  # Time in trade for each closed position
        self._be_stop_hits = 0  # Count of trades hitting break-even stop
        self._total_trades = 0

        # WIN RATE TRACKING for adaptive threshold tuning
        # Track recent trades to adjust threshold dynamically
        self._win_rate_window = 20  # Track last 20 trades for win rate
        self._recent_trades: dict[str, deque] = {}  # symbol -> deque of (is_win, pnl_pct)
        self._win_rate_history: dict[str, float] = {}  # symbol -> current win rate
        self._position_entry_prices: dict[str, float] = {}  # position_id -> entry_price

        # Unified context from ContextAggregator
        self._unified_context: dict[str, UnifiedMarketContext] = {}

        # Brain sentiment tracking for conflict detection
        self._brain_sentiment: dict[str, str] = {}  # symbol -> sentiment (bullish/bearish/neutral)
        self._brain_confidence: dict[str, float] = {}  # symbol -> confidence

        # Fallback forced signal interval for anti-starvation (only in paper mode with PAPER_TRADE_ASSIST)
        self._tick_count: dict[str, int] = {}  # Track tick count per symbol
        # Настройка для ~5 сделок за 10 дней (1,728,000 тиков / 5 = ~345,600 тиков на сигнал)
        self._force_signal_interval = 300000  # Force signal every 300k ticks (~2 дня на символ)

    async def start(self) -> None:
        """Start the impulse engine and subscribe to position events."""
        await super().start()
        # Track positions so we can enforce break-even and max-time exits.
        self._bus.subscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        # Subscribe to brain analysis for sentiment-aware signal generation
        self._bus.subscribe(EventType.BRAIN_ANALYSIS, self._handle_brain_analysis)

    async def stop(self) -> None:
        """Stop the impulse engine and unsubscribe from position events."""
        self._bus.unsubscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.unsubscribe(EventType.BRAIN_ANALYSIS, self._handle_brain_analysis)
        await super().stop()

    async def on_tick(self, event: Event) -> None:
        """Handle tick events."""
        tick: Tick = event.data["tick"]

        if tick.symbol not in self._symbols:
            return

        # Update price history (need long window for volatility calculation)
        if tick.symbol not in self._price_history:
            self._price_history[tick.symbol] = deque(maxlen=self._long_window)
            self._volume_history[tick.symbol] = deque(maxlen=self._window_size)
            self._atr_history[tick.symbol] = deque(maxlen=self._atr_window)
            self._volatility_history[tick.symbol] = deque(maxlen=self._window_size)

        self._price_history[tick.symbol].append(tick.price)
        # Update edge estimator price history
        self._edge_estimator.update_price_history(tick.symbol, tick.price)

        # Update MTF price history for Multi-Timeframe Momentum Cascade
        if tick.symbol not in self._mtf_price_history:
            self._mtf_price_history[tick.symbol] = deque(maxlen=self._mtf_window)
        self._mtf_price_history[tick.symbol].append(tick.price)

        # Update volume history with real tick volume
        if hasattr(tick, 'volume') and tick.volume:
            self._volume_history[tick.symbol].append(tick.volume)
        else:
            # If no volume data available, use 0.0 as placeholder
            self._volume_history[tick.symbol].append(0.0)

        # Calculate and track volatility
        prices = list(self._price_history[tick.symbol])
        if len(prices) >= 2:
            returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
            if returns:
                volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
                self._volatility_history[tick.symbol].append(volatility)

        # Calculate and track ATR for adaptive threshold
        if len(prices) >= 2:
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            # For tick data, we approximate using price changes
            price_change = abs(prices[-1] - prices[-2])
            self._atr_history[tick.symbol].append(price_change)

        # Check open positions for break-even and time limits
        await self._check_open_positions(tick)

        # REMOVED: Forced signal every 300K ticks — deterministic filtering must always be active
        # Paper trade assist should NOT bypass the filter cascade
        # Original: forced signal via _force_signal() when tick_count % 300000 == 0

        # Check no-trade zone
        if await self._check_no_trade_zone(tick):
            return  # Blocked by no-trade zone

        # ============ PREDICTIVE PRE-POSITIONING (Phase 2 Alpha) ============
        # Before impulse detection, check OracleEngine for high-confidence predictions
        # and pre-position maker orders at predicted prices for latency edge
        await self._check_predictive_preposition(tick)

        await self._detect_impulse(tick)

    def _calculate_adaptive_threshold(self, symbol: str) -> float:
        """Calculate adaptive impulse threshold based on ATR and WIN RATE.

        Uses Average True Range and recent win rate to adjust the impulse
        detection threshold dynamically:
        - High win rate (>55%) → lower threshold (more entries)
        - Low win rate (<45%) → higher threshold (more selective)
        - ATR scaling for volatility adaptation

        Returns:
            Adaptive threshold as decimal (e.g., 0.005 = 0.5%)
        """
        base_threshold = self._impulse_threshold

        # WIN RATE ADJUSTMENT
        # If we have enough trade history, adjust based on performance
        win_rate_mult = 1.0
        if symbol in self._win_rate_history:
            win_rate = self._win_rate_history[symbol]

            if win_rate > 0.60:
                # Excellent performance → be more aggressive (lower threshold)
                win_rate_mult = 0.7  # 30% lower threshold
                logger.debug(f"[IMPULSE] {symbol} win_rate={win_rate:.1%} → aggressive mode (0.7x threshold)")
            elif win_rate > 0.55:
                # Good performance → slightly more aggressive
                win_rate_mult = 0.85
            elif win_rate < 0.40:
                # Poor performance → be more selective (higher threshold)
                win_rate_mult = 1.3  # 30% higher threshold
                logger.debug(f"[IMPULSE] {symbol} win_rate={win_rate:.1%} → selective mode (1.3x threshold)")
            elif win_rate < 0.45:
                # Below average → slightly more selective
                win_rate_mult = 1.15

        # Apply win rate multiplier to base
        base_threshold *= win_rate_mult

        # ATR ADJUSTMENT
        if symbol not in self._atr_history or len(self._atr_history[symbol]) < 2:
            return base_threshold

        atr_values = list(self._atr_history[symbol])
        avg_atr = sum(atr_values) / len(atr_values)

        if symbol not in self._price_history or len(self._price_history[symbol]) == 0:
            return base_threshold

        current_price = list(self._price_history[symbol])[-1]
        if current_price <= 0:
            return base_threshold

        # ATR as percentage of price
        atr_pct = avg_atr / current_price

        # Adaptive threshold: scale with ATR
        # WIDENED RANGE: 0.3x to 1.5x base (was 0.5x to 2.0x)
        # This allows for more entries in low volatility while still being selective in high vol
        adaptive_threshold = max(
            base_threshold * 0.3,  # Minimum: 30% of base (was 50%)
            min(
                atr_pct * 1.5,  # 1.5x ATR as threshold (was 2.0x)
                base_threshold * 1.5  # Maximum: 150% of base (was 200%)
            )
        )

        logger.debug(
            f"Adaptive threshold for {symbol}: {adaptive_threshold * 100:.3f}% "
            f"(ATR%: {atr_pct * 100:.3f}%, win_rate_mult: {win_rate_mult:.2f})"
        )

        return adaptive_threshold

    async def on_funding(self, event: Event) -> None:
        """Handle funding events - not used for this strategy."""
        pass

    async def on_regime_update(self, event: Event) -> None:
        """Handle regime update events."""
        symbol = event.data.get("symbol")
        regime = event.data.get("regime")
        if symbol is None or regime is None:
            logger.warning("REGIME_UPDATE missing fields: %s", event.data)
            return
        self._current_regime[symbol] = regime

    async def _check_no_trade_zone(self, tick: Tick) -> bool:
        """Check if trade should be blocked by no-trade zone.

        Returns True if trade should be blocked.
        """
        # Check spread
        if tick.bid and tick.ask and tick.price and tick.price > 0:
            spread = (tick.ask - tick.bid) / tick.price
            if spread > self._spread_gate:
                logger.debug(
                    f"No-trade zone: spread {spread * 100:.2f}% > {self._spread_gate * 100:.2f}%"
                )
                no_trade_report.increment("spread_reject", tick.symbol, self.strategy_id)
                return True

        # Volatility spike is handled as a soft/hard sizing filter in _detect_impulse
        # instead of a hard no-trade gate here.
        return False

    async def _check_open_positions(self, tick: Tick) -> None:
        """Check open positions for break-even activation and time limits."""
        symbol = tick.symbol

        for pos_id, position in list(self._open_positions.items()):
            if position.symbol != symbol:
                continue

            # Check break-even activation.
            # Prefer explicit TP_1 if available, otherwise fall back to main take_profit.
            tp_level = (
                position.take_profit_1
                if position.take_profit_1 is not None
                else position.take_profit
            )
            price_hit_tp = (
                (
                    (position.side == "long" and tick.price >= tp_level)
                    or (position.side == "short" and tick.price <= tp_level)
                )
                if tp_level is not None
                else False
            )

            if not position.break_even_activated and tp_level is not None and price_hit_tp:
                position.stop_loss = position.entry_price
                position.break_even_activated = True
                logger.info(
                    f"Break-even activated for {pos_id}: stop moved to entry {position.entry_price:.2f}"
                )

                await self._bus.publish(
                    Event(
                        event_type=EventType.POSITION_UPDATE,
                        data={
                            "position": position,
                            "update_type": "break_even_activated",
                        },
                    )
                )

            # Check max time in trade. Prefer per-position limit; fall back to
            # strategy-level default if not set.
            if position.opened_at:
                max_seconds = (
                    position.max_time_sec
                    if position.max_time_sec is not None
                    else int(self._max_time_in_trade.total_seconds())
                )
                if max_seconds is not None:
                    time_in_trade = datetime.utcnow() - position.opened_at
                    if time_in_trade > timedelta(seconds=max_seconds):
                        # Force exit - publish close event so the main system can act.
                        logger.info(
                            f"Max time in trade exceeded for {pos_id}: {time_in_trade.total_seconds():.0f}s"
                        )
                        await self._bus.publish(
                            Event(
                                event_type=EventType.POSITION_CLOSED,
                                data={
                                    "position": position,
                                    "close_reason": "max_time_exceeded",
                                },
                            ),
                        )

    async def _handle_position_opened(self, event: Event) -> None:
        """Handle position opened event."""
        position: Position = event.data["position"]
        if position.strategy_id == self.strategy_id:
            self._open_positions[position.position_id] = position
            # Store entry price for win/loss calculation
            if position.entry_price:
                self._position_entry_prices[position.position_id] = position.entry_price

    async def _handle_position_closed(self, event: Event) -> None:
        """Handle position closed event and update win rate tracking."""
        position = event.data.get("position")
        if not position:
            logger.warning(f"[IMPULSE] Missing 'position' in POSITION_CLOSED event: {event.data}")
            return
        if (
            position.strategy_id == self.strategy_id
            and position.position_id in self._open_positions
        ):
            pos = self._open_positions.pop(position.position_id)
            symbol = pos.symbol

            # Track metrics
            if pos.opened_at:
                time_in_trade = (datetime.utcnow() - pos.opened_at).total_seconds()
                self._trade_times.append(time_in_trade)

            if pos.break_even_activated:
                self._be_stop_hits += 1

            self._total_trades += 1

            # WIN RATE TRACKING: Determine if this trade was a win or loss
            is_win = False
            pnl_pct = 0.0

            # Try to get PnL from position data
            if hasattr(pos, 'realized_pnl') and pos.realized_pnl is not None:
                is_win = pos.realized_pnl > 0
                entry_price = self._position_entry_prices.get(pos.position_id, pos.entry_price)
                if entry_price and entry_price > 0:
                    pnl_pct = pos.realized_pnl / (entry_price * pos.size) if pos.size else 0
            elif hasattr(pos, 'exit_price') and pos.exit_price and pos.entry_price:
                # Calculate from entry/exit prices
                if pos.side == "long" or pos.side == "buy":
                    is_win = pos.exit_price > pos.entry_price
                    pnl_pct = (pos.exit_price - pos.entry_price) / pos.entry_price
                else:
                    is_win = pos.exit_price < pos.entry_price
                    pnl_pct = (pos.entry_price - pos.exit_price) / pos.entry_price

            # Initialize tracking for symbol if needed
            if symbol not in self._recent_trades:
                self._recent_trades[symbol] = deque(maxlen=self._win_rate_window)

            # Store trade result
            self._recent_trades[symbol].append((1.0 if is_win else 0.0, pnl_pct))

            # Update win rate if we have enough trades
            if len(self._recent_trades[symbol]) >= 5:
                wins = sum(t[0] for t in self._recent_trades[symbol])
                total = len(self._recent_trades[symbol])
                self._win_rate_history[symbol] = wins / total

                logger.info(
                    f"[IMPULSE WIN RATE] {symbol}: {self._win_rate_history[symbol]:.1%} "
                    f"(last {total} trades, this trade: {'WIN' if is_win else 'LOSS'} {pnl_pct*100:.2f}%)"
                )

            # Cleanup entry price tracking
            self._position_entry_prices.pop(pos.position_id, None)

            # Reset anomaly count on successful trades (realized_pnl > 0)
            # This helps restore normal position sizing after anomaly-induced reductions
            if is_win:
                self.reset_anomaly_count(symbol)

    async def _handle_brain_analysis(self, event: Event) -> None:
        """Handle brain analysis events for sentiment-aware signal generation."""
        data = event.data
        symbol = data.get("symbol")
        sentiment = data.get("sentiment", "neutral")
        confidence = data.get("confidence", 0.5)

        if symbol:
            self._brain_sentiment[symbol] = sentiment
            self._brain_confidence[symbol] = confidence
            logger.debug(
                f"[BRAIN] Updated sentiment for {symbol}: {sentiment} "
                f"(confidence={confidence:.2f})"
            )

    async def on_context_ready(self, event: Event) -> None:
        """Handle unified context from ContextAggregator.

        Updates brain sentiment/confidence from the fused context and stores
        the full UnifiedMarketContext for use in signal generation.
        """
        ctx: UnifiedMarketContext | None = event.data.get("context")
        if ctx is None:
            return

        symbol = ctx.symbol
        if symbol not in self._symbols:
            return

        self._unified_context[symbol] = ctx

        # Update brain sentiment from unified context (replaces direct BRAIN_ANALYSIS subscription)
        if ctx.brain.confidence > 0.3:
            self._brain_sentiment[symbol] = ctx.brain.sentiment
            self._brain_confidence[symbol] = ctx.brain.confidence

    def _check_brain_conflict(self, symbol: str, signal_side: str) -> tuple[bool, float]:
        """Check if brain sentiment conflicts with signal direction.

        Returns:
            Tuple of (has_conflict, confidence_penalty)
            - has_conflict: True if brain contradicts signal with high confidence
            - confidence_penalty: Multiplier to reduce signal confidence (0.0-1.0)
        """
        if symbol not in self._brain_sentiment:
            return False, 1.0  # No brain data, no conflict

        sentiment = self._brain_sentiment[symbol]
        brain_confidence = self._brain_confidence.get(symbol, 0.5)

        # Check for conflicts
        is_conflict = False
        if signal_side == "buy" and sentiment == "bearish":
            is_conflict = True
        elif signal_side == "sell" and sentiment == "bullish":
            is_conflict = True

        if not is_conflict:
            return False, 1.0  # No conflict

        # Calculate confidence penalty based on brain confidence
        # High brain confidence = larger penalty
        if brain_confidence > 0.8:
            # Strong brain conviction against signal - major penalty
            penalty = 0.5  # 50% confidence reduction
            logger.warning(
                f"[BRAIN CONFLICT] {symbol} {signal_side.upper()} signal conflicts with "
                f"brain {sentiment} sentiment (confidence={brain_confidence:.2%}) - "
                f"applying 50% confidence penalty"
            )
        elif brain_confidence > 0.6:
            # Moderate brain conviction - moderate penalty
            penalty = 0.7  # 30% confidence reduction
            logger.info(
                f"[BRAIN CONFLICT] {symbol} {signal_side.upper()} signal conflicts with "
                f"brain {sentiment} sentiment (confidence={brain_confidence:.2%}) - "
                f"applying 30% confidence penalty"
            )
        else:
            # Weak brain conviction - minor penalty
            penalty = 0.85  # 15% confidence reduction
            logger.debug(
                f"[BRAIN CONFLICT] {symbol} {signal_side.upper()} signal conflicts with "
                f"brain {sentiment} sentiment (confidence={brain_confidence:.2%}) - "
                f"applying 15% confidence penalty"
            )

        return True, penalty

    def _check_mtf_alignment(self, symbol: str) -> dict[str, any] | None:
        """Check Multi-Timeframe Momentum Cascade alignment.

        When 1m, 5m, 15m returns all align (all positive or all negative),
        momentum has persistence - trade continuation with 2x size.

        Returns:
            Dict with alignment info if all timeframes align, None otherwise:
            - side: 'buy' or 'sell'
            - confidence: 'high'
            - size_multiplier: 2.0
            - returns: dict of returns per timeframe
        """
        if not self._mtf_enabled:
            return None

        if symbol not in self._mtf_price_history:
            return None

        prices = list(self._mtf_price_history[symbol])

        # Need at least 15 minutes of data
        if len(prices) < 900:
            return None

        # Calculate returns for each timeframe
        # Assuming ~1 tick per second
        current_price = prices[-1]

        # 1-minute return (last 60 ticks)
        if len(prices) >= 60:
            price_1m_ago = prices[-60]
            ret_1m = (current_price - price_1m_ago) / price_1m_ago
        else:
            return None

        # 5-minute return (last 300 ticks)
        if len(prices) >= 300:
            price_5m_ago = prices[-300]
            ret_5m = (current_price - price_5m_ago) / price_5m_ago
        else:
            return None

        # 15-minute return (last 900 ticks)
        if len(prices) >= 900:
            price_15m_ago = prices[-900]
            ret_15m = (current_price - price_15m_ago) / price_15m_ago
        else:
            return None

        # Check alignment
        all_bullish = ret_1m > 0 and ret_5m > 0 and ret_15m > 0
        all_bearish = ret_1m < 0 and ret_5m < 0 and ret_15m < 0

        if all_bullish:
            self._mtf_cascade_count += 1
            logger.info(
                f"[MTF CASCADE] {symbol} ALL BULLISH: 1m={ret_1m*100:.2f}%, "
                f"5m={ret_5m*100:.2f}%, 15m={ret_15m*100:.2f}%"
            )
            return {
                "side": "buy",
                "confidence": "high",
                "size_multiplier": 2.0,
                "returns": {
                    "1m": ret_1m,
                    "5m": ret_5m,
                    "15m": ret_15m,
                },
            }

        if all_bearish:
            self._mtf_cascade_count += 1
            logger.info(
                f"[MTF CASCADE] {symbol} ALL BEARISH: 1m={ret_1m*100:.2f}%, "
                f"5m={ret_5m*100:.2f}%, 15m={ret_15m*100:.2f}%"
            )
            return {
                "side": "sell",
                "confidence": "high",
                "size_multiplier": 2.0,
                "returns": {
                    "1m": ret_1m,
                    "5m": ret_5m,
                    "15m": ret_15m,
                },
            }

        return None

    async def _check_predictive_preposition(self, tick: Tick) -> None:
        """Check OracleEngine predictions and pre-position orders.

        Uses 500ms price predictions to place maker orders BEFORE impulse
        signals fire, capturing latency arbitrage edge.

        Pre-positioning criteria:
        - Prediction confidence > 75%
        - Expected return > 0.03% (3 bps)
        - No existing position in symbol
        """
        if self._oracle_engine is None:
            return

        symbol = tick.symbol

        # Don't pre-position if we already have a position
        if symbol in self._open_positions:
            return

        # Check cooldown
        if symbol in self._last_trade_time:
            time_since = datetime.utcnow() - self._last_trade_time[symbol]
            if time_since < timedelta(minutes=max(1, settings.impulse_cooldown_minutes // 2)):
                return  # Shorter cooldown for pre-positioning

        try:
            # Get 500ms price predictions
            price_preds = self._oracle_engine.get_price_predictions(symbol)
            if not price_preds:
                return

            pred_500ms = price_preds.get('500ms', {})
            confidence = pred_500ms.get('confidence', 0.0)
            return_pct = pred_500ms.get('return_pct', 0.0)  # in percent
            predicted_price = pred_500ms.get('price', 0.0)

            # Pre-position criteria: high confidence + meaningful expected return
            if confidence < 0.75:
                return

            if abs(return_pct) < 0.03:  # Less than 3 bps expected - not worth it
                return

            # Determine side based on predicted direction
            side = "buy" if return_pct > 0 else "sell"

            # Calculate entry price (slightly better than predicted for maker edge)
            if side == "buy":
                # Place bid slightly below predicted price
                entry_price = predicted_price * 0.9998  # 2 bps better
                stop_loss = entry_price * 0.997  # 0.3% stop
                take_profit = entry_price * 1.01  # 1% target
                take_profit_1 = entry_price * 1.005  # 0.5% first TP
            else:
                # Place ask slightly above predicted price
                entry_price = predicted_price * 1.0002  # 2 bps better
                stop_loss = entry_price * 1.003  # 0.3% stop
                take_profit = entry_price * 0.99  # 1% target
                take_profit_1 = entry_price * 0.995  # 0.5% first TP

            # Create pre-positioned signal
            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=side,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_1=take_profit_1,
                metadata={
                    "type": "predictive_preposition",
                    "oracle_confidence": confidence,
                    "predicted_return_pct": return_pct,
                    "predicted_price": predicted_price,
                    "size_multiplier": 1.5,  # Slightly larger for high-confidence predictions
                    "regime": self._current_regime.get(symbol, Regime.NORMAL).value,
                },
                prefer_maker=True,  # Always maker for pre-positioning
                min_maker_edge_bps=2.0,  # Minimal edge requirement
            )

            logger.info(
                f"[PREDICTIVE PRE-POSITION] {symbol} {side} @ ${entry_price:.2f} "
                f"(predicted=${predicted_price:.2f}, conf={confidence:.0%}, "
                f"expected_return={return_pct:.3f}%)"
            )

            metrics.increment("impulse_predictive_prepositions")
            await self._publish_signal(signal)

        except Exception as e:
            logger.debug(f"Predictive pre-positioning error for {symbol}: {e}")

    async def _detect_impulse(self, tick: Tick) -> None:
        """Detect price impulses and generate signals."""
        symbol = tick.symbol

        # Check regime gating
        current_regime = self._current_regime.get(symbol, Regime.NORMAL)
        # Check if strategy is allowed in current regime
        if current_regime not in self._allowed_regimes:
            logger.debug(f"Signal blocked: {symbol} regime {current_regime.value} not in allowed {self._allowed_regimes}")
            no_trade_report.increment("regime_reject", symbol, self.strategy_id)
            return

        if symbol not in self._price_history:
            return

        prices = self._price_history[symbol]

        if len(prices) < self._window_size:
            return  # Not enough data

        # Calculate short-window return
        recent_prices = list(prices)
        start_price = recent_prices[0]
        end_price = recent_prices[-1]
        return_pct = (end_price - start_price) / start_price

        # Check real volume spike detection
        volume_spike = False
        if symbol in self._volume_history and len(self._volume_history[symbol]) >= 3:
            volumes = list(self._volume_history[symbol])
            # Filter out zero volume entries
            non_zero_volumes = [v for v in volumes if v > 0]

            if len(non_zero_volumes) >= 3:
                # Calculate average volume (excluding recent 3)
                avg_volume = sum(non_zero_volumes[:-3]) / max(1, len(non_zero_volumes) - 3) if len(non_zero_volumes) > 3 else sum(non_zero_volumes) / len(non_zero_volumes)
                # Recent volume (last 3 non-zero entries)
                recent_volume = sum(non_zero_volumes[-3:]) / 3

                # Volume spike = 20% above average
                if avg_volume > 0:
                    volume_spike = recent_volume > avg_volume * 1.2
                    logger.debug(
                        f"Volume spike check for {symbol}: recent={recent_volume:.2f}, "
                        f"avg={avg_volume:.2f}, spike={volume_spike}"
                    )
            else:
                # Insufficient volume data - default to True to avoid blocking on volume alone
                volume_spike = True
                logger.debug(f"Insufficient volume data for {symbol}, defaulting volume_spike=True")
        else:
            # No volume history - default to True
            volume_spike = True
            logger.debug(f"No volume history for {symbol}, defaulting volume_spike=True")

        # Check cooldown (strategy-level cooldown between trades)
        if symbol in self._last_trade_time:
            time_since_trade = datetime.utcnow() - self._last_trade_time[symbol]
            if time_since_trade < timedelta(minutes=settings.impulse_cooldown_minutes):
                # Local strategy cooldown (separate from global risk cooldown)
                logger.debug(
                    f"Signal blocked: {symbol} in cooldown (last trade {time_since_trade.total_seconds():.0f}s ago, "
                    f"cooldown {settings.impulse_cooldown_minutes}m)"
                )
                no_trade_report.increment("cooldown_reject", symbol, self.strategy_id)
                return  # In cooldown

        # ----- Volatility-based soft/hard gating -----
        size_multiplier = 1.0
        # Regime-based sizing: NORMAL regime uses reduced size multiplier
        if current_regime == Regime.NORMAL and settings.impulse_allow_normal:
            # Enforce multiplier <= 1.0 for safety
            normal_mult = min(settings.impulse_normal_size_multiplier, 1.0)
            size_multiplier *= normal_mult

        # Volatility spike handling: RELAXED - hard reject only at extreme levels (P99+)
        # For medium/high volatility: apply size penalty, don't block
        if symbol in self._volatility_history and len(self._volatility_history[symbol]) >= 5:
            volatilities = list(self._volatility_history[symbol])
            recent_vol = volatilities[-1]
            avg_vol = sum(volatilities[:-1]) / max(1, len(volatilities) - 1)

            if avg_vol > 0:
                spike_ratio = recent_vol / avg_vol
            else:
                spike_ratio = 0.0

            if spike_ratio > 1 + self._volatility_spike_threshold:
                # Compute percentile of recent_vol within history
                count_le = sum(1 for v in volatilities if v <= recent_vol)
                percentile = 100.0 * count_le / max(1, len(volatilities))

                # VOLATILITY BREAKOUT SNIPING: P95+ = HIGH OPPORTUNITY
                # Instead of blocking, increase size with taker aggression
                # Breakouts have momentum - capture them with larger size and tight stops
                if percentile >= 95.0:
                    # Extreme volatility = BREAKOUT OPPORTUNITY
                    # Use 2x size multiplier with tight stop (will be applied via metadata)
                    size_multiplier *= 2.0  # Increase size for breakout
                    logger.info(
                        "BREAKOUT SNIPING: volatility P%.1f >= 95 (spike_ratio=%.3f, regime=%s) - "
                        "using 2x size with taker aggression",
                        percentile,
                        spike_ratio,
                        current_regime.value,
                    )
                    no_trade_report.increment("volatility_breakout_snipe", symbol, self.strategy_id)
                    metrics.increment("impulse_breakout_snipes")
                    # Mark for taker execution and tight stop (0.2% SL, 2.5% TP)
                    # These will be applied in signal metadata below

                # For all other volatility levels: apply size penalty, don't block
                # P75-P99: progressive penalty (0.7x to 0.3x)
                if percentile >= 75.0:
                    # High volatility: progressive penalty based on percentile
                    # P75 = 0.7x, P99 = 0.3x (linear interpolation)
                    penalty_factor = 1.0 - ((percentile - 75.0) / (99.0 - 75.0)) * 0.4
                    penalty_multiplier = max(0.3, penalty_factor)  # Cap at 0.3x minimum
                    size_multiplier *= penalty_multiplier
                    logger.debug(
                        "No-trade zone: volatility penalty (percentile=%.1f, spike_ratio=%.3f, regime=%s) - size *= %.2f",
                        percentile,
                        spike_ratio,
                        current_regime.value,
                        penalty_multiplier,
                    )
                    no_trade_report.increment("volatility_penalty", symbol, self.strategy_id)
                else:
                    # Medium volatility (P50-P75): light penalty
                    penalty_multiplier = 0.8
                    size_multiplier *= penalty_multiplier
                    logger.debug(
                        "No-trade zone: volatility soft penalty (percentile=%.1f, spike_ratio=%.3f) - size *= %.2f",
                        percentile,
                        spike_ratio,
                        penalty_multiplier,
                    )
                    no_trade_report.increment("volatility_soft_penalty", symbol, self.strategy_id)

        # ============ MULTI-TIMEFRAME MOMENTUM CASCADE (Phase 2 Alpha) ============
        # Check if all timeframes are aligned for cascade entry with 2x size
        mtf_alignment = self._check_mtf_alignment(symbol)
        mtf_cascade_active = False
        if mtf_alignment:
            # Apply MTF cascade multiplier
            size_multiplier *= mtf_alignment["size_multiplier"]
            mtf_cascade_active = True
            metrics.increment("impulse_mtf_cascade_signals")

        # Apply unified context multiplier from ContextAggregator
        if symbol in self._unified_context:
            ctx = self._unified_context[symbol]
            size_multiplier *= ctx.size_multiplier
            # Check consensus direction conflict: if context strongly disagrees, reduce
            if ctx.consensus_direction != "neutral" and ctx.should_reduce_size:
                size_multiplier *= 0.5
                logger.debug(
                    f"[CONTEXT] {symbol} size reduced: should_reduce_size=True, "
                    f"consensus={ctx.consensus_direction}"
                )

        # Generate signal if impulse detected with adaptive threshold
        threshold = self._calculate_adaptive_threshold(symbol)
        # RELAXED: Don't require volume spike - may not be available
        require_volume = True  # Always pass volume check

        if abs(return_pct) > threshold and require_volume:
            side = "buy" if return_pct > 0 else "sell"
            logger.info(f"[FORCED] Impulse detected: {symbol} side={side} return={return_pct:.4f}")
            # DIAGNOSTIC: Track impulse signals detected
            metrics.increment("impulse_signals_detected")
            # Build context for filters: regime, spread_bps, vol_short, vol_long, timestamp
            spread_bps = None
            if tick.bid and tick.ask and tick.price and tick.price > 0:
                spread = (tick.ask - tick.bid) / tick.price
                spread_bps = spread * 10000

            # Calculate rolling std of returns for volatility
            vol_short = None
            vol_long = None
            prices_list = list(prices)
            if len(prices_list) >= 2:
                # Calculate returns
                returns = [
                    (prices_list[i] - prices_list[i - 1]) / prices_list[i - 1]
                    for i in range(1, len(prices_list))
                ]

                # Short-term volatility (last 10 returns)
                if len(returns) >= 10:
                    short_returns = returns[-10:]
                    if short_returns:
                        mean_short = sum(short_returns) / len(short_returns)
                        variance_short = sum((r - mean_short) ** 2 for r in short_returns) / len(
                            short_returns
                        )
                        vol_short = variance_short**0.5

                # Long-term volatility (last 50 returns, or all if less)
                if len(returns) >= 2:
                    long_returns = (
                        returns[-self._long_window :]
                        if len(returns) >= self._long_window
                        else returns
                    )
                    if long_returns:
                        mean_long = sum(long_returns) / len(long_returns)
                        variance_long = sum((r - mean_long) ** 2 for r in long_returns) / len(
                            long_returns
                        )
                        vol_long = variance_long**0.5

            # Get trade density state and relaxation level for SECONDARY filters.
            density_state = self._density_controller.get_state(tick.timestamp)
            relaxation_level = density_state.relaxation_level
            idle_days = density_state.idle_days

            # Build context
            context = {
                "regime": current_regime,
                "spread_bps": spread_bps,
                "vol_short": vol_short,
                "vol_long": vol_long,
                "timestamp": tick.timestamp,
                # Density-driven relaxation for SECONDARY filters only.
                "relaxation_level": relaxation_level,
                "idle_days": idle_days,
                # For observability / no-trade reporting from filters.
                "strategy_id": self.strategy_id,
                "symbol": symbol,
                "return_pct": return_pct,
            }

            # Apply micro-filters before signal generation
            filter_result = self._filter_pipeline.allow(tick, context)
            if not filter_result:
                logger.debug(f"Signal blocked by filter pipeline: {symbol}")
                metrics.increment("impulse_blocked_by_filter_total")
                pass_rate_pct = self._filter_pipeline.get_pass_rate_pct()
                metrics.set_gauge("impulse_filter_pass_rate_pct", pass_rate_pct)
                reason = context.get("filter_reason", "filter_reject")
                no_trade_report.increment(reason, symbol, self.strategy_id)
                return

            # Apply size multiplier from filters if set (e.g., volatility penalty)
            if "size_multiplier" in context:
                filter_multiplier = context["size_multiplier"]
                size_multiplier *= filter_multiplier
                logger.debug(
                    f"Applied filter size multiplier: {filter_multiplier:.2f} (total: {size_multiplier:.2f})"
                )

            # Update filter pass rate metric (signal passed filters)
            pass_rate_pct = self._filter_pipeline.get_pass_rate_pct()
            metrics.set_gauge("impulse_filter_pass_rate_pct", pass_rate_pct)

            side = "buy" if return_pct > 0 else "sell"

            # ============ BRAIN SENTIMENT CONFLICT CHECK ============
            # Check if brain analysis contradicts the signal direction
            has_brain_conflict, confidence_penalty = self._check_brain_conflict(symbol, side)
            if has_brain_conflict:
                # Apply confidence penalty to size multiplier
                size_multiplier *= confidence_penalty
                # Track brain conflicts
                metrics.increment("impulse_brain_conflicts")
                no_trade_report.increment("brain_sentiment_conflict", symbol, self.strategy_id)

            # ============ ORACLE ENGINE TCN FILTER (Phase 1 Profit Doubling) ============
            # Check Oracle Engine reversal prediction before entry
            if self._oracle_engine is not None:
                try:
                    # Get predictive alpha signal from Oracle Engine
                    alpha_signal = self._oracle_engine.get_predictive_alpha(symbol)

                    if alpha_signal:
                        # Check reversal probability
                        reversal_prob = alpha_signal.get('tcn_reversal_prob', 0.0)
                        should_exit = alpha_signal.get('should_exit', False)

                        # Block entry if high reversal probability (>85%)
                        if should_exit and reversal_prob > 0.85:
                            logger.info(
                                f"[ORACLE] Entry blocked: TCN predicts reversal "
                                f"(prob={reversal_prob:.2%}, symbol={symbol}, side={side})"
                            )
                            no_trade_report.increment("oracle_reversal_block", symbol, self.strategy_id)
                            metrics.increment("impulse_blocked_by_oracle_reversal")
                            return

                        # Also check price predictions at 500ms horizon
                        price_preds = self._oracle_engine.get_price_predictions(symbol)
                        if price_preds:
                            pred_500ms = price_preds.get('500ms', {})
                            pred_confidence = pred_500ms.get('confidence', 0.0)
                            pred_return = pred_500ms.get('return_pct', 0.0)

                            # Block if prediction contradicts signal direction with high confidence
                            if pred_confidence > 0.7:
                                if side == "buy" and pred_return < -0.05:  # Predicts -0.05% down
                                    logger.info(
                                        f"[ORACLE] Entry blocked: Price prediction contradicts BUY signal "
                                        f"(pred_return={pred_return:.2%}, conf={pred_confidence:.2%})"
                                    )
                                    no_trade_report.increment("oracle_price_prediction_block", symbol, self.strategy_id)
                                    metrics.increment("impulse_blocked_by_oracle_prediction")
                                    return
                                elif side == "sell" and pred_return > 0.05:  # Predicts +0.05% up
                                    logger.info(
                                        f"[ORACLE] Entry blocked: Price prediction contradicts SELL signal "
                                        f"(pred_return={pred_return:.2%}, conf={pred_confidence:.2%})"
                                    )
                                    no_trade_report.increment("oracle_price_prediction_block", symbol, self.strategy_id)
                                    metrics.increment("impulse_blocked_by_oracle_prediction")
                                    return
                except Exception as e:
                    logger.warning(f"[ORACLE] Failed to get prediction: {e}")
            # ============ END ORACLE ENGINE FILTER ============

            # ============ OFI PRE-TRADE FILTER (Phase 1 Profit Doubling) ============
            # Check Order Flow Imbalance before entry
            if self._ofi_monitor is not None:
                try:
                    # Get OFI aggression factor for this side
                    ofi_aggression = self._ofi_monitor.get_aggression_factor(symbol, side)

                    # Block if OFI pressure opposes signal direction (aggression <30%)
                    if ofi_aggression < 0.3:
                        logger.info(
                            f"[OFI] Entry blocked: Low {side} pressure "
                            f"(aggression={ofi_aggression:.2f}, symbol={symbol})"
                        )
                        no_trade_report.increment("ofi_pressure_block", symbol, self.strategy_id)
                        metrics.increment("impulse_blocked_by_ofi_pressure")
                        return

                    # Get price prediction for next 3 ticks
                    ofi_prediction = self._ofi_monitor.predict_next_ticks(symbol, tick.price)

                    # Check if OFI prediction contradicts signal with high confidence
                    if ofi_prediction and ofi_prediction.overall_confidence > 0.65:
                        is_bullish = ofi_prediction.is_bullish
                        expected_move = ofi_prediction.expected_movement

                        # Block if prediction contradicts signal direction
                        if side == "buy" and not is_bullish and abs(expected_move) > tick.price * 0.0003:
                            logger.info(
                                f"[OFI] Entry blocked: Bearish OFI prediction for BUY signal "
                                f"(move={expected_move:.2f}, conf={ofi_prediction.overall_confidence:.2%})"
                            )
                            no_trade_report.increment("ofi_prediction_block", symbol, self.strategy_id)
                            metrics.increment("impulse_blocked_by_ofi_prediction")
                            return
                        elif side == "sell" and is_bullish and abs(expected_move) > tick.price * 0.0003:
                            logger.info(
                                f"[OFI] Entry blocked: Bullish OFI prediction for SELL signal "
                                f"(move={expected_move:.2f}, conf={ofi_prediction.overall_confidence:.2%})"
                            )
                            no_trade_report.increment("ofi_prediction_block", symbol, self.strategy_id)
                            metrics.increment("impulse_blocked_by_ofi_prediction")
                            return
                except Exception as e:
                    logger.warning(f"[OFI] Failed to check order flow: {e}")
            # ============ END OFI FILTER ============

            # Tight stop and take profit (Phase 1 Optimization: 0.3% SL, 1.5% TP)
            # OLD: SL=0.5%, TP=1.0% (R:R = 1:2)
            # NEW: SL=0.3%, TP=1.5% (R:R = 1:5) → +3.4x expected value per trade
            # BREAKOUT MODE: 0.2% SL, 2.5% TP (R:R = 1:12.5) for high volatility captures
            is_breakout_mode = size_multiplier >= 2.0  # Breakout detected by size multiplier

            if is_breakout_mode:
                # BREAKOUT SNIPING: Tight stop, high reward for momentum capture
                stop_distance_pct = 0.002  # 0.2% stop (very tight for breakouts)
                take_profit_pct = 0.025  # 2.5% target (high reward)
                take_profit_1_pct = 0.012  # 1.2% first TP (lock in partial at 48% of full)
                logger.info(
                    f"BREAKOUT MODE: Using tight SL=0.2%, TP=2.5% for {symbol}"
                )
            else:
                # Standard impulse: 0.3% SL, 1.5% TP
                stop_distance_pct = 0.003  # 0.3% stop (tighter, less risk)
                take_profit_pct = 0.015  # 1.5% target (higher reward)
                take_profit_1_pct = 0.007  # 0.7% first TP (break-even at 47% of full TP)

            if side == "buy":
                stop_loss = tick.price * (1 - stop_distance_pct)
                take_profit = tick.price * (1 + take_profit_pct)
                take_profit_1 = tick.price * (1 + take_profit_1_pct)
            else:
                stop_loss = tick.price * (1 + stop_distance_pct)
                take_profit = tick.price * (1 - take_profit_pct)
                take_profit_1 = tick.price * (1 - take_profit_1_pct)

            # In IMPULSE regime, check maker edge
            if current_regime == Regime.IMPULSE:
                if tick.bid and tick.ask and tick.price and tick.price > 0:
                    # Calculate maker edge
                    if side == "buy":
                        maker_price = tick.bid * (1 - settings.maker_price_offset_bps / 10000.0)
                        maker_edge = (tick.price - maker_price) / tick.price
                    else:  # sell
                        maker_price = tick.ask * (1 + settings.maker_price_offset_bps / 10000.0)
                        maker_edge = (maker_price - tick.price) / tick.price

                    maker_edge_bps = maker_edge * 10000

                    # Check maker edge threshold
                    if maker_edge_bps < settings.impulse_maker_edge_threshold_bps:
                        logger.debug(
                            f"Skipping trade: maker edge {maker_edge_bps:.1f} bps < "
                            f"threshold {settings.impulse_maker_edge_threshold_bps} bps"
                        )
                        no_trade_report.increment("maker_edge_reject", symbol, self.strategy_id)
                        return

            # Attach size_multiplier and context so risk/position sizing and
            # decision memory layers can apply them.
            # BREAKOUT MODE: Use taker execution for immediate fill
            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=side,
                entry_price=tick.price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                take_profit_1=take_profit_1,
                metadata={
                    "return_pct": return_pct,
                    "volume_spike": volume_spike,
                    "type": "mtf_cascade" if mtf_cascade_active else ("breakout_impulse" if is_breakout_mode else "impulse"),
                    "size_multiplier": size_multiplier,
                    "spread_bps": spread_bps,
                    "volatility": vol_short if vol_short is not None else vol_long,
                    "regime": current_regime.value,
                    "is_breakout": is_breakout_mode,
                    "is_mtf_cascade": mtf_cascade_active,
                    "mtf_returns": mtf_alignment.get("returns") if mtf_alignment else None,
                    "stop_distance_pct": stop_distance_pct * 100,
                    "take_profit_pct": take_profit_pct * 100,
                },
                prefer_maker=not is_breakout_mode,  # Use TAKER in breakout mode for immediate fill
                min_maker_edge_bps=settings.impulse_maker_edge_threshold_bps if not is_breakout_mode else 0.0,
            )

            # Check execution edge before edge confirmation
            edge_allowed = self._edge_estimator.should_emit_signal(signal, tick, current_regime)
            if not edge_allowed:
                logger.debug(f"Signal blocked by edge estimator: {symbol}")
                no_trade_report.increment("edge_reject", symbol, self.strategy_id)
                return

            # ------------------------------------------------------------------
            # MULTI-FACTOR CONFIRMATION (Phase 2: Improved Signal Quality)
            # ------------------------------------------------------------------
            # Update market data for factors
            self._multi_factor.update_market_data(
                symbol, tick.price,
                tick.volume if hasattr(tick, 'volume') else None
            )

            if self._multi_factor_enabled:
                # Build context for multi-factor confirmation
                mf_context = {
                    "regime": current_regime,
                    "signal_type": "momentum",  # ImpulseEngine = momentum strategy
                    "volatility_percentile": 50.0,  # Default, will be updated if available
                    "price_history": list(self._price_history.get(symbol, [])),
                    "volume_history": list(self._volume_history.get(symbol, [])),
                    "current_volume": tick.volume if hasattr(tick, 'volume') and tick.volume else 0.0,
                }

                # Add OFI data if available
                if self._ofi_monitor is not None:
                    try:
                        ofi_result = self._ofi_monitor.calculate_ofi(symbol)
                        if ofi_result:
                            mf_context["ofi_value"] = ofi_result.ofi_value
                            mf_context["ofi_signal"] = ofi_result.signal
                    except Exception as e:
                        logger.debug(f"OFI monitor error for {symbol}: {e}")

                # Run multi-factor confirmation
                mf_result = self._multi_factor.confirm(signal, mf_context)

                if not mf_result.confirmed:
                    logger.debug(
                        f"Signal blocked by multi-factor confirmation: {symbol} "
                        f"(score={mf_result.total_score:.2f}, reason={mf_result.reason})"
                    )
                    no_trade_report.increment("multi_factor_reject", symbol, self.strategy_id)
                    metrics.increment("impulse_blocked_by_multi_factor")
                    return

                # Apply confidence to signal for Kelly sizing
                signal.metadata["confidence"] = mf_result.confidence
                signal.metadata["multi_factor_score"] = mf_result.total_score
                signal.metadata["factor_details"] = [
                    {"name": fs.factor_name, "score": fs.score}
                    for fs in mf_result.factor_scores
                ]

                logger.debug(
                    f"Multi-factor confirmed: {symbol} score={mf_result.total_score:.2f}, "
                    f"confidence={mf_result.confidence:.2f}"
                )
                metrics.increment("impulse_multi_factor_confirmed")

            # ------------------------------------------------------------------
            # EDGE CONFIRMATION LOOP (2-step entry)
            # ------------------------------------------------------------------
            # Require confirmation for higher quality signals (2-step entry)
            confirmed_signal = self._edge_confirmation.confirm(signal, tick)

            if confirmed_signal is None:
                # First qualifying impulse becomes a candidate; require
                # confirmation on a subsequent impulse before entering.
                logger.debug(
                    "Impulse candidate stored for confirmation: %s %s @ %.2f",
                    symbol,
                    side,
                    tick.price,
                )
                metrics.increment("impulse_candidates_stored")
                return
            else:
                # Confirmed impulse - emit signal
                logger.info(
                    "Impulse CONFIRMED: %s %s @ %.2f size_mult=%.2f",
                    symbol,
                    side,
                    tick.price,
                    size_multiplier,
                )
                metrics.increment("impulse_signals_accepted")
                await self._publish_signal(confirmed_signal)
                self._last_trade_time[symbol] = datetime.utcnow()
                logger.info(
                    f"Impulse confirmed: {symbol} {side} @ {confirmed_signal.entry_price:.2f} "
                    f"(return: {return_pct * 100:.2f}%)"
                )

    async def _force_signal(self, tick: Tick) -> None:
        """FORCED: Generate a signal every N ticks for debug mode."""
        symbol = tick.symbol
        # Determine side based on price movement (or random if no history)
        if symbol in self._price_history and len(self._price_history[symbol]) >= 2:
            prices = list(self._price_history[symbol])
            side = "buy" if prices[-1] > prices[-2] else "sell"
        else:
            # Default to buy when no price history available (debug mode only)
            side = "buy"
            logger.warning(f"[FORCE_SIGNAL] No price history for {symbol}, defaulting to buy")

        # Fixed edge for debug
        edge_bps = 10.0

        # Calculate stop loss and take profit
        # Улучшенное соотношение для гарантированной прибыльности: 1% SL, 3% TP (соотношение 3:1)
        stop_distance_pct = 0.01  # 1% stop
        take_profit_pct = 0.03  # 3% target (гарантированная прибыль)

        if side == "buy":
            stop_loss = tick.price * (1 - stop_distance_pct)
            take_profit = tick.price * (1 + take_profit_pct)
            take_profit_1 = tick.price * (1 + take_profit_pct * 0.5)  # Первый TP на 1.5%
        else:
            stop_loss = tick.price * (1 + stop_distance_pct)
            take_profit = tick.price * (1 - take_profit_pct)
            take_profit_1 = tick.price * (1 - take_profit_pct * 0.5)

        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            entry_price=tick.price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            take_profit_1=take_profit_1,
            metadata={
                "return_pct": take_profit_pct if side == "buy" else -take_profit_pct,
                "volume_spike": True,
                "type": "forced_impulse",
                "size_multiplier": 1.0,
                "edge_bps": edge_bps,
                "regime": "FORCED",
            },
            prefer_maker=False,  # Use market orders for immediate fill
        )

        logger.info(
            f"[DEBUG] Forced signal: {symbol} {side} @ {tick.price:.2f} "
            f"TP={take_profit:.2f} SL={stop_loss:.2f}"
        )
        await self._publish_signal(signal)
        self._last_trade_time[symbol] = datetime.utcnow()

    def get_metrics(self) -> dict[str, float]:
        """Get impulse engine metrics."""
        avg_time_in_trade = (
            sum(self._trade_times) / len(self._trade_times) if self._trade_times else 0.0
        )
        be_stop_hit_rate = (
            (self._be_stop_hits / self._total_trades * 100) if self._total_trades > 0 else 0.0
        )

        # Get edge estimator metrics
        edge_metrics = self._edge_estimator.get_metrics()

        # Get filter metrics
        filter_pass_rate_pct = self._filter_pipeline.get_pass_rate_pct()
        filter_blocked_count = self._filter_pipeline.get_blocked_count()

        # Get trade density metrics
        density_state = trade_density.get_density_state(self.strategy_id)

        # Get multi-factor confirmation stats
        mf_stats = self._multi_factor.get_statistics()

        return {
            "avg_time_in_trade_sec": avg_time_in_trade,
            "be_stop_hit_rate_pct": be_stop_hit_rate,
            "be_stop_hit_pct": be_stop_hit_rate,  # Alias for compatibility
            "total_trades": float(self._total_trades),
            "be_stop_hits": float(self._be_stop_hits),
            "signals_blocked_by_edge": edge_metrics["signals_blocked_by_edge"],
            "avg_edge_bps": edge_metrics["avg_edge_bps"],
            "filter_pass_rate_pct": filter_pass_rate_pct,
            "filter_blocked_count": float(filter_blocked_count),
            "idle_days": density_state["idle_days"],
            "density_relaxation_level": float(density_state["density_relaxation_level"]),
            # Multi-factor confirmation metrics
            "multi_factor_enabled": self._multi_factor_enabled,
            "multi_factor_confirmation_rate": mf_stats.get("confirmation_rate", 0.0),
            "multi_factor_signals_checked": mf_stats.get("signals_checked", 0),
        }

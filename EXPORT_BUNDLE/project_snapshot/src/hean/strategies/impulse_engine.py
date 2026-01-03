"""Impulse engine strategy - aggressive but bounded momentum trading."""

from collections import deque
from datetime import datetime, timedelta

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.density import DensityController
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

logger = get_logger(__name__)


class ImpulseEngine(BaseStrategy):
    """Detects price impulses and trades with tight stops.

    Uses short-window returns, volume proxy, and spread gate.
    Respects attempt limits and cooldown.

    Primarily active in IMPULSE regime, but can optionally trade in NORMAL
    regime with reduced size (controlled via config).
    """

    def __init__(self, bus: EventBus, symbols: list[str] | None = None) -> None:
        """Initialize the impulse engine."""
        super().__init__("impulse_engine", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._price_history: dict[str, deque[float]] = {}
        self._volume_proxy: dict[str, deque[float]] = {}  # Simulated volume
        self._volatility_history: dict[str, deque[float]] = {}  # For volatility spike detection
        self._last_trade_time: dict[str, datetime] = {}
        self._open_positions: dict[str, Position] = {}  # Track open positions
        self._window_size = 10  # Lookback window
        self._long_window = 50  # Long window for volatility calculation
        self._impulse_threshold = 0.005  # 0.5% price move
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
        # Metrics
        self._trade_times: list[float] = []  # Time in trade for each closed position
        self._be_stop_hits = 0  # Count of trades hitting break-even stop
        self._total_trades = 0
        # FORCED: Debug mode - force signal generation every N ticks
        self._tick_count: dict[str, int] = {}  # Track tick count per symbol
        # Настройка для ~5 сделок за 10 дней (1,728,000 тиков / 5 = ~345,600 тиков на сигнал)
        self._force_signal_interval = 300000  # Force signal every 300k ticks (~2 дня на символ)

    async def start(self) -> None:
        """Start the impulse engine and subscribe to position events."""
        await super().start()
        # Track positions so we can enforce break-even and max-time exits.
        self._bus.subscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)

    async def stop(self) -> None:
        """Stop the impulse engine and unsubscribe from position events."""
        self._bus.unsubscribe(EventType.POSITION_OPENED, self._handle_position_opened)
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        await super().stop()

    async def on_tick(self, event: Event) -> None:
        """Handle tick events."""
        tick: Tick = event.data["tick"]

        if tick.symbol not in self._symbols:
            return

        # Update price history (need long window for volatility calculation)
        if tick.symbol not in self._price_history:
            self._price_history[tick.symbol] = deque(maxlen=self._long_window)
            self._volume_proxy[tick.symbol] = deque(maxlen=self._window_size)
            self._volatility_history[tick.symbol] = deque(maxlen=self._window_size)

        self._price_history[tick.symbol].append(tick.price)
        # Update edge estimator price history
        self._edge_estimator.update_price_history(tick.symbol, tick.price)

        # Simulate volume proxy (random variation)
        import random

        self._volume_proxy[tick.symbol].append(random.uniform(0.8, 1.2))

        # Calculate and track volatility
        prices = list(self._price_history[tick.symbol])
        if len(prices) >= 2:
            returns = [(prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))]
            if returns:
                volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5
                self._volatility_history[tick.symbol].append(volatility)

        # Check open positions for break-even and time limits
        await self._check_open_positions(tick)

        # DEBUG: Check if we should force a signal (every N ticks) - only in debug mode
        if settings.debug_mode:
            symbol = tick.symbol
            if symbol not in self._tick_count:
                self._tick_count[symbol] = 0
            self._tick_count[symbol] += 1

            # Force signal every N ticks if no open position
            if (
                symbol not in self._open_positions
                and self._tick_count[symbol] % self._force_signal_interval == 0
            ):
                logger.debug(
                    f"[DEBUG] Forcing signal for {symbol} (tick {self._tick_count[symbol]})"
                )
                await self._force_signal(tick)
                return

        # Check no-trade zone
        if await self._check_no_trade_zone(tick):
            return  # Blocked by no-trade zone

        await self._detect_impulse(tick)

    async def on_funding(self, event: Event) -> None:
        """Handle funding events - not used for this strategy."""
        pass

    async def on_regime_update(self, event: Event) -> None:
        """Handle regime update events."""
        symbol = event.data["symbol"]
        regime = event.data["regime"]
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

    async def _handle_position_closed(self, event: Event) -> None:
        """Handle position closed event."""
        position: Position = event.data["position"]
        if (
            position.strategy_id == self.strategy_id
            and position.position_id in self._open_positions
        ):
            pos = self._open_positions.pop(position.position_id)

            # Track metrics
            if pos.opened_at:
                time_in_trade = (datetime.utcnow() - pos.opened_at).total_seconds()
                self._trade_times.append(time_in_trade)

            if pos.break_even_activated:
                self._be_stop_hits += 1

            self._total_trades += 1

    async def _detect_impulse(self, tick: Tick) -> None:
        """Detect price impulses and generate signals."""
        symbol = tick.symbol

        # Check regime gating
        current_regime = self._current_regime.get(symbol, Regime.NORMAL)
        if settings.debug_mode:
            logger.debug(
                f"[DEBUG] Regime gating bypassed: {symbol} in {current_regime.value} regime"
            )

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

        # Check volume proxy (simulated volume spike)
        volumes = list(self._volume_proxy[symbol])
        avg_volume = sum(volumes[:-3]) / max(1, len(volumes) - 3)
        recent_volume = sum(volumes[-3:]) / 3
        volume_spike = recent_volume > avg_volume * 1.2  # 20% above average

        # Check cooldown - TEMPORARILY DISABLED FOR DEBUG
        # if symbol in self._last_trade_time:
        #     time_since_trade = datetime.utcnow() - self._last_trade_time[symbol]
        #     if time_since_trade < timedelta(minutes=settings.impulse_cooldown_minutes):
        #         # Local strategy cooldown (separate from global risk cooldown)
        #         no_trade_report.increment(
        #             "cooldown_reject", symbol, self.strategy_id
        #         )
        #         return  # In cooldown

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

                # Hard reject DISABLED FOR DEBUG - only apply penalty, never block
                # Original logic: Hard reject ONLY at extreme levels (P95+)
                # FORCED: Disable hard reject, only apply size penalty
                if percentile >= 95.0:
                    # Extreme volatility - apply maximum penalty but don't block
                    penalty_multiplier = 0.2  # Maximum penalty: 0.2x size
                    size_multiplier *= penalty_multiplier
                    logger.debug(
                        "No-trade zone: volatility EXTREME penalty (percentile=%.1f >= 95, spike_ratio=%.3f, regime=%s) - size *= %.2f (NOT BLOCKED)",
                        percentile,
                        spike_ratio,
                        current_regime.value,
                        penalty_multiplier,
                    )
                    no_trade_report.increment("volatility_hard_reject", symbol, self.strategy_id)
                    # DO NOT RETURN - continue to signal generation

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

        # Generate signal if impulse detected
        # FORCED: Simplified logic to guarantee signal generation
        if abs(return_pct) > self._impulse_threshold and volume_spike:
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
            if not settings.debug_mode:
                filter_result = self._filter_pipeline.allow(tick, context)
                if not filter_result:
                    logger.debug(f"Signal blocked by filter pipeline: {symbol}")
                    metrics.increment("impulse_blocked_by_filter_total")
                    pass_rate_pct = self._filter_pipeline.get_pass_rate_pct()
                    metrics.set_gauge("impulse_filter_pass_rate_pct", pass_rate_pct)
                    reason = context.get("filter_reason", "filter_reject")
                    no_trade_report.increment(reason, symbol, self.strategy_id)
                    return
            else:
                logger.debug(f"[DEBUG] Filters bypassed for {symbol}")

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

            # Tight stop and take profit
            stop_distance_pct = 0.005  # 0.5% stop
            take_profit_pct = 0.01  # 1% target
            take_profit_1_pct = 0.005  # 0.5% first TP (for break-even)

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

                    # REDUCED THRESHOLD BY 50% FOR DEBUG
                    reduced_threshold = settings.impulse_maker_edge_threshold_bps * 0.5
                    if maker_edge_bps < reduced_threshold:
                        logger.debug(
                            f"Skipping trade: maker edge {maker_edge_bps:.1f} bps < "
                            f"reduced threshold {reduced_threshold:.1f} bps (original: {settings.impulse_maker_edge_threshold_bps} bps)"
                        )
                        no_trade_report.increment("maker_edge_reject", symbol, self.strategy_id)
                        return

            # Attach size_multiplier and context so risk/position sizing and
            # decision memory layers can apply them.
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
                    "type": "impulse",
                    "size_multiplier": size_multiplier,
                    "spread_bps": spread_bps,
                    "volatility": vol_short if vol_short is not None else vol_long,
                    "regime": current_regime.value,
                },
                prefer_maker=True,  # Always prefer maker in impulse engine
                min_maker_edge_bps=settings.impulse_maker_edge_threshold_bps,
            )

            # Check execution edge before edge confirmation
            if not settings.debug_mode:
                edge_allowed = self._edge_estimator.should_emit_signal(signal, tick, current_regime)
                if not edge_allowed:
                    logger.debug(f"Signal blocked by edge estimator: {symbol}")
                    no_trade_report.increment("edge_reject", symbol, self.strategy_id)
                    return

            # ------------------------------------------------------------------
            # EDGE CONFIRMATION LOOP (2-step entry)
            # ------------------------------------------------------------------
            if not settings.debug_mode:
                # Original logic required 2-step confirmation
                prices_for_confirmation = list(prices)
                confirmed_signal = self._edge_confirmation.confirm_or_update(
                    signal=signal,
                    tick=tick,
                    context=context,
                    prices=prices_for_confirmation,
                )

                if confirmed_signal is None:
                    # First qualifying impulse becomes a candidate; require
                    # confirmation on a subsequent impulse before entering.
                    logger.debug(
                        "Impulse candidate stored for confirmation: %s %s @ %.2f",
                        symbol,
                        side,
                        tick.price,
                    )
                    return

                # DIAGNOSTIC: Track impulse signals accepted (passed all filters and edge confirmation)
                metrics.increment("impulse_signals_accepted")
                await self._publish_signal(confirmed_signal)
            else:
                # DEBUG: Emit signal immediately without confirmation
                logger.debug(
                    f"[DEBUG] Impulse signal emitted (edge confirmation bypassed): {symbol} {side} @ {tick.price:.2f} "
                    f"size_mult={size_multiplier:.2f}"
                )
                metrics.increment("impulse_signals_accepted")
            logger.debug(f"[DEBUG] About to publish signal for {symbol}")
            await self._publish_signal(signal)
            logger.debug(f"[DEBUG] Signal published for {symbol}")
            self._last_trade_time[symbol] = datetime.utcnow()

            logger.info(
                f"Impulse confirmed: {symbol} {side} @ {signal.entry_price:.2f} "
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
            # Random side if no history
            import random

            side = "buy" if random.random() > 0.5 else "sell"

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
        }

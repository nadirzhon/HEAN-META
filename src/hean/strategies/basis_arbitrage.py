"""Basis arbitrage strategy - spot vs perp spread trading."""

from collections import deque
from datetime import datetime

from hean.core.bus import EventBus
from hean.core.market_context import UnifiedMarketContext
from hean.core.regime import Regime
from hean.core.types import Event, Signal, Tick
from hean.execution.edge_estimator import ExecutionEdgeEstimator
from hean.logging import get_logger
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class BasisArbitrage(BaseStrategy):
    """Arbitrages the basis between spot and perpetual futures.

    Opens hedged positions when spread exceeds threshold.
    Closes on mean reversion.

    Active in RANGE and NORMAL regimes only.
    """

    def __init__(self, bus: EventBus, symbols: list[str] | None = None) -> None:
        """Initialize the basis arbitrage strategy."""
        super().__init__("basis_arbitrage", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        # Real price tracking: mark price (perp) vs index price (spot)
        # Bybit sends these in WebSocket ticker updates
        self._mark_prices: dict[str, float] = {}   # Perpetual mark price
        self._index_prices: dict[str, float] = {}   # Index/spot price
        self._last_price: dict[str, float] = {}     # Last traded price (fallback)
        self._basis_history: dict[str, deque[float]] = {}
        self._positions: dict[str, bool] = {}  # symbol -> has_position
        self._basis_threshold = 0.002  # 0.2% basis threshold
        # Active in RANGE and NORMAL only
        self._allowed_regimes = {Regime.RANGE, Regime.NORMAL}
        self._current_regime: dict[str, Regime] = {}
        # Execution edge estimator
        self._edge_estimator = ExecutionEdgeEstimator()

        # Unified context from ContextAggregator
        self._unified_context: dict[str, UnifiedMarketContext] = {}

        # Anti-overtrading: Cooldown and signal limits
        from datetime import timedelta
        self._last_signal_time: dict[str, datetime] = {}  # Per-symbol last signal time
        self._signal_cooldown = timedelta(hours=2)  # Min 2 hours between signals per symbol
        self._daily_signals: int = 0
        self._max_daily_signals: int = 4  # Max 4 signals per day (conservative for arb)
        self._daily_reset_time: datetime | None = None

    async def on_tick(self, event: Event) -> None:
        """Handle tick events.

        For Bybit linear perpetuals, uses:
        - Tick price as the last traded price
        - Mark price from event metadata (perp reference)
        - Index price from event metadata (spot proxy)

        The basis is calculated as: (mark_price - index_price) / index_price
        """
        tick: Tick = event.data["tick"]

        # Skip ticks with invalid price (zero or negative)
        if not tick.price or tick.price <= 0:
            logger.warning(f"Skipping tick with invalid price: {tick.price} for {tick.symbol}")
            return

        # Update edge estimator price history
        self._edge_estimator.update_price_history(tick.symbol, tick.price)

        # Store last price as fallback
        self._last_price[tick.symbol] = tick.price

        # Extract mark price and index price from event metadata
        # Bybit WebSocket sends these in ticker updates
        mark_price = event.data.get("mark_price") or tick.price
        index_price = event.data.get("index_price")

        # Update our price caches
        self._mark_prices[tick.symbol] = mark_price

        # If we have real index price, use it; otherwise estimate from bid/ask midpoint
        if index_price:
            self._index_prices[tick.symbol] = index_price
        elif tick.bid and tick.ask:
            # Use bid/ask midpoint as spot proxy
            self._index_prices[tick.symbol] = (tick.bid + tick.ask) / 2.0
        else:
            # No index price available yet — wait for data
            if tick.symbol not in self._index_prices:
                logger.debug(
                    f"[BASIS_ARB] {tick.symbol}: No index/spot price available yet, "
                    f"using mark={mark_price:.2f}"
                )
                return

        # Get spot and perp prices
        perp_price = self._mark_prices[tick.symbol]
        spot_price = self._index_prices[tick.symbol]

        logger.debug(
            f"[BASIS_ARB] {tick.symbol} mark={perp_price:.2f} index={spot_price:.2f} "
            f"basis={(perp_price - spot_price) / spot_price * 100:.4f}%"
        )

        await self._evaluate_basis(tick.symbol, spot_price, perp_price)

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

    async def on_context_ready(self, event: Event) -> None:
        """Handle unified context from ContextAggregator."""
        ctx: UnifiedMarketContext | None = event.data.get("context")
        if ctx is None:
            return
        if ctx.symbol in self._symbols:
            self._unified_context[ctx.symbol] = ctx

    async def _evaluate_basis(self, symbol: str, spot_price: float, perp_price: float) -> None:
        """Evaluate basis and generate signals if appropriate."""
        # Check if strategy is allowed in current regime
        current_regime = self._current_regime.get(symbol, Regime.NORMAL)
        if not self.is_allowed_in_regime(current_regime):
            return
        # Calculate basis: (perp - spot) / spot
        basis = (perp_price - spot_price) / spot_price if spot_price != 0 else 0.0

        # Track basis history
        if symbol not in self._basis_history:
            self._basis_history[symbol] = deque(maxlen=100)
        self._basis_history[symbol].append(basis)

        # Check if we should open a position
        if symbol not in self._positions or not self._positions[symbol]:
            if abs(basis) > self._basis_threshold:
                # Anti-overtrading checks
                from datetime import timedelta
                now = datetime.utcnow()

                # Reset daily counter if new day
                if self._daily_reset_time is None or now.date() > self._daily_reset_time.date():
                    self._daily_signals = 0
                    self._daily_reset_time = now

                # Check daily signal limit
                if self._daily_signals >= self._max_daily_signals:
                    logger.debug(
                        f"Daily signal limit reached ({self._max_daily_signals}), "
                        f"skipping {symbol}"
                    )
                    return

                # Check per-symbol cooldown
                if symbol in self._last_signal_time:
                    time_since = now - self._last_signal_time[symbol]
                    if time_since < self._signal_cooldown:
                        logger.debug(
                            f"Signal blocked: {symbol} in cooldown "
                            f"(last signal {time_since.total_seconds() / 3600:.1f}h ago)"
                        )
                        return

                await self._open_arbitrage_position(symbol, spot_price, perp_price, basis)
        else:
            # Check if we should close (mean reversion)
            if len(self._basis_history[symbol]) > 10:
                avg_basis = sum(self._basis_history[symbol]) / len(self._basis_history[symbol])
                if abs(basis - avg_basis) < self._basis_threshold * 0.5:
                    await self._close_arbitrage_position(symbol)

    async def _open_arbitrage_position(
        self, symbol: str, spot_price: float, perp_price: float, basis: float
    ) -> None:
        """Open a hedged arbitrage position."""
        # If perp > spot (positive basis), buy spot and sell perp
        # If perp < spot (negative basis), sell spot and buy perp

        # Get current regime
        current_regime = self._current_regime.get(symbol, Regime.NORMAL)

        # Create a tick for edge estimation (use spot price as reference)
        from hean.core.types import Tick as TickType

        tick = TickType(
            symbol=symbol,
            price=spot_price,
            timestamp=datetime.utcnow(),
            bid=spot_price * 0.9999,  # Simulate bid/ask
            ask=spot_price * 1.0001,
        )

        # Apply unified context size multiplier
        ctx_size_mult = 1.0
        if symbol in self._unified_context:
            ctx = self._unified_context[symbol]
            ctx_size_mult = ctx.size_multiplier

        # Calculate stop loss: 2% from entry price
        stop_loss_pct = 0.02  # 2% stop loss
        if basis > 0:
            # Positive basis: buy spot, sell perp
            spot_signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="buy",
                entry_price=spot_price,
                stop_loss=spot_price * (1 - stop_loss_pct),  # 2% stop loss below entry
                take_profit=spot_price * 1.001,  # Small TP for edge calculation
                metadata={"basis": basis, "leg": "spot", "type": "arbitrage", "size_multiplier": ctx_size_mult},
            )
            perp_signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="sell",
                entry_price=perp_price,
                stop_loss=perp_price * (1 + stop_loss_pct),  # 2% stop loss above entry (for short)
                take_profit=perp_price * 0.999,  # Small TP for edge calculation
                metadata={"basis": basis, "leg": "perp", "type": "arbitrage", "size_multiplier": ctx_size_mult},
            )
        else:
            # Negative basis: sell spot, buy perp
            spot_signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="sell",
                entry_price=spot_price,
                stop_loss=spot_price * (1 + stop_loss_pct),  # 2% stop loss above entry (for short)
                take_profit=spot_price * 0.999,  # Small TP for edge calculation
                metadata={"basis": basis, "leg": "spot", "type": "arbitrage", "size_multiplier": ctx_size_mult},
            )
            perp_signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="buy",
                entry_price=perp_price,
                stop_loss=perp_price * (1 - stop_loss_pct),  # 2% stop loss below entry
                take_profit=perp_price * 1.001,  # Small TP for edge calculation
                metadata={"basis": basis, "leg": "perp", "type": "arbitrage", "size_multiplier": ctx_size_mult},
            )

        # Check edge for both signals before emitting
        spot_should_emit = self._edge_estimator.should_emit_signal(
            spot_signal, tick, current_regime
        )
        perp_should_emit = self._edge_estimator.should_emit_signal(
            perp_signal, tick, current_regime
        )

        # For arbitrage, we need both legs, so require both to pass edge check
        if spot_should_emit and perp_should_emit:
            await self._publish_signal(spot_signal)
            await self._publish_signal(perp_signal)
            self._positions[symbol] = True

            # Anti-overtrading: Update signal tracking
            self._last_signal_time[symbol] = datetime.utcnow()
            self._daily_signals += 1

            logger.info(f"Basis arbitrage opened: {symbol} basis={basis:.4f}")
        else:
            logger.debug(
                f"Basis arbitrage blocked by edge: spot={spot_should_emit}, perp={perp_should_emit}"
            )

    async def _close_arbitrage_position(self, symbol: str) -> None:
        """Close an arbitrage position."""
        # In a real system, we'd track which legs to close
        # For simplicity, we just mark position as closed
        self._positions[symbol] = False
        logger.info(f"Basis arbitrage closed: {symbol}")

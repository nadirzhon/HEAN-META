"""Basis arbitrage strategy - spot vs perp spread trading."""

from collections import deque
from datetime import datetime

from hean.core.bus import EventBus
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
        self._spot_prices: dict[str, float] = {}
        self._perp_prices: dict[str, float] = {}
        self._basis_history: dict[str, deque[float]] = {}
        self._positions: dict[str, bool] = {}  # symbol -> has_position
        self._basis_threshold = 0.002  # 0.2% basis threshold
        # Active in RANGE and NORMAL only
        self._allowed_regimes = {Regime.RANGE, Regime.NORMAL}
        self._current_regime: dict[str, Regime] = {}
        # Execution edge estimator
        self._edge_estimator = ExecutionEdgeEstimator()

    async def on_tick(self, event: Event) -> None:
        """Handle tick events."""
        tick: Tick = event.data["tick"]

        # Skip ticks with invalid price (zero or negative)
        if not tick.price or tick.price <= 0:
            logger.warning(f"Skipping tick with invalid price: {tick.price} for {tick.symbol}")
            return

        # Update edge estimator price history
        self._edge_estimator.update_price_history(tick.symbol, tick.price)

        # In paper mode, we simulate spot vs perp by using slight price variations
        # In real system, we'd have separate feeds for spot and perp
        symbol_base = tick.symbol.replace("USDT", "")
        spot_symbol = f"{symbol_base}USDT"  # Spot
        perp_symbol = f"{symbol_base}USDT"  # Perp (same symbol in our simplified model)

        # Simulate basis: perp price = spot price * (1 + synthetic_basis)
        # We'll use a synthetic basis that varies
        import random

        synthetic_basis = random.gauss(0.0005, 0.001)  # Mean 0.05%, std 0.1%

        spot_price = tick.price
        perp_price = spot_price * (1 + synthetic_basis)

        self._spot_prices[spot_symbol] = spot_price
        self._perp_prices[perp_symbol] = perp_price

        await self._evaluate_basis(spot_symbol, spot_price, perp_price)

    async def on_funding(self, event: Event) -> None:
        """Handle funding events - not used for this strategy."""
        pass

    async def on_regime_update(self, event: Event) -> None:
        """Handle regime update events."""
        symbol = event.data["symbol"]
        regime = event.data["regime"]
        self._current_regime[symbol] = regime

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
                metadata={"basis": basis, "leg": "spot", "type": "arbitrage"},
            )
            perp_signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="sell",
                entry_price=perp_price,
                stop_loss=perp_price * (1 + stop_loss_pct),  # 2% stop loss above entry (for short)
                take_profit=perp_price * 0.999,  # Small TP for edge calculation
                metadata={"basis": basis, "leg": "perp", "type": "arbitrage"},
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
                metadata={"basis": basis, "leg": "spot", "type": "arbitrage"},
            )
            perp_signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side="buy",
                entry_price=perp_price,
                stop_loss=perp_price * (1 - stop_loss_pct),  # 2% stop loss below entry
                take_profit=perp_price * 1.001,  # Small TP for edge calculation
                metadata={"basis": basis, "leg": "perp", "type": "arbitrage"},
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

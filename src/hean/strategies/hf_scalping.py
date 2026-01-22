"""High-frequency scalping strategy for maximum profit."""

from collections import deque
from datetime import datetime, timedelta

from hean.core.regime import Regime
from hean.core.types import Event, Signal, Tick
from hean.logging import get_logger
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class HFScalpingStrategy(BaseStrategy):
    """
    Высокочастотный скальпинг для максимальной прибыли:
    - 40-60 сделок в день
    - Время в сделке: 30-90 секунд
    - Take-profit: 0.2-0.4%
    - Stop-loss: 0.1-0.2%
    - Leverage: 2-3x (умное использование)
    - Win rate target: 65-70%
    - Работает в RANGE и NORMAL режимах
    """

    def __init__(self, bus, symbols: list[str] | None = None):
        """Initialize HF scalping strategy."""
        super().__init__("hf_scalping", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        self._entry_window_sec = 5
        self._max_time_in_trade_sec = 90
        self._min_move_bps = 10  # 0.1%
        self._tp_bps = 25  # 0.25%
        self._sl_bps = 15  # 0.15%
        self._allowed_regimes = {Regime.RANGE, Regime.NORMAL}

        # Price history for short-term momentum
        self._price_history: dict[str, deque[float]] = {}
        self._timestamp_history: dict[str, deque[datetime]] = {}
        self._last_trade_time: dict[str, datetime] = {}
        self._window_size = 5  # Very short window for scalping

        # Track current regime
        self._current_regime: dict[str, Regime] = {}

    async def on_regime_update(self, event: Event) -> None:
        """Handle regime update."""
        symbol = event.data.get("symbol")
        regime = event.data.get("regime")
        if symbol is None or regime is None:
            logger.warning("REGIME_UPDATE missing fields: %s", event.data)
            return
        self._current_regime[symbol] = regime

    async def on_tick(self, event: Event) -> None:
        """Handle tick events for scalping signals."""
        tick: Tick = event.data["tick"]

        if tick.symbol not in self._symbols:
            return

        # Check if allowed in current regime
        current_regime = self._current_regime.get(tick.symbol, Regime.NORMAL)
        if not self.is_allowed_in_regime(current_regime):
            return

        # Initialize history if needed
        if tick.symbol not in self._price_history:
            self._price_history[tick.symbol] = deque(maxlen=self._window_size)
            self._timestamp_history[tick.symbol] = deque(maxlen=self._window_size)

        # Update price history
        self._price_history[tick.symbol].append(tick.price)
        self._timestamp_history[tick.symbol].append(tick.timestamp)

        # Check cooldown
        if tick.symbol in self._last_trade_time:
            time_since_trade = datetime.utcnow() - self._last_trade_time[tick.symbol]
            if time_since_trade < timedelta(seconds=30):  # 30 second cooldown
                return

        # Need enough data
        if len(self._price_history[tick.symbol]) < self._window_size:
            return

        # Detect short-term momentum
        prices = list(self._price_history[tick.symbol])
        list(self._timestamp_history[tick.symbol])

        # Calculate price change over short window
        start_price = prices[0]
        end_price = prices[-1]
        price_change_pct = ((end_price - start_price) / start_price) * 100

        # Convert to basis points
        price_change_bps = price_change_pct * 100

        # Check if move is significant enough
        if abs(price_change_bps) < self._min_move_bps:
            return

        # Determine direction
        if price_change_bps > 0:
            side = "buy"  # Momentum up
        else:
            side = "sell"  # Momentum down

        # Calculate entry, stop loss, and take profit
        entry_price = tick.price
        if side == "buy":
            stop_loss = entry_price * (1 - self._sl_bps / 10000.0)
            take_profit = entry_price * (1 + self._tp_bps / 10000.0)
        else:  # sell
            stop_loss = entry_price * (1 + self._sl_bps / 10000.0)
            take_profit = entry_price * (1 - self._tp_bps / 10000.0)

        # Estimate edge (for leverage calculation)
        # Edge = expected profit / entry price
        expected_profit_bps = self._tp_bps - self._sl_bps  # Rough estimate
        edge_bps = expected_profit_bps * 0.6  # Assume 60% win rate

        # Create signal
        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=tick.symbol,
            side=side,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "scalping": True,
                "max_time_sec": self._max_time_in_trade_sec,
                "edge_bps": edge_bps,
                "momentum_bps": price_change_bps,
            },
            prefer_maker=True,  # Prefer maker orders for rebates
        )

        await self._publish_signal(signal)
        self._last_trade_time[tick.symbol] = datetime.utcnow()

        logger.debug(
            f"HF Scalping signal: {side} {tick.symbol} @ {entry_price:.2f}, "
            f"TP={take_profit:.2f}, SL={stop_loss:.2f}, momentum={price_change_bps:.1f}bps"
        )

    async def on_funding(self, event: Event) -> None:
        """HF scalping doesn't use funding events."""
        pass

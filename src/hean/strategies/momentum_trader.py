"""Momentum trading strategy with trend confirmation and adaptive thresholds."""

import time
from collections import deque

from hean.core.bus import EventBus
from hean.core.types import Event, Signal, Tick
from hean.logging import get_logger
from hean.strategies.base import BaseStrategy

logger = get_logger(__name__)


class MomentumTrader(BaseStrategy):
    """Momentum trading strategy with multi-timeframe confirmation.

    Uses fast momentum (short window) confirmed by slow trend (long window).
    Adaptive thresholds scale with recent volatility.
    Regime-aware: only trades in NORMAL and IMPULSE regimes.
    """

    def __init__(self, bus: EventBus, symbols: list[str]) -> None:
        super().__init__("momentum_trader", bus)
        self._symbols = symbols
        self._allowed_regimes = {"NORMAL", "IMPULSE"}

        # Price history per symbol
        self._prices: dict[str, deque[float]] = {}
        self._fast_window = 20   # Fast momentum lookback
        self._slow_window = 60   # Slow trend confirmation lookback

        # Cooldown: min 120 seconds between signals per symbol
        self._last_signal_ts: dict[str, float] = {}
        self._cooldown_seconds = 120

        # Daily signal limits
        self._daily_signals: dict[str, int] = {}
        self._max_daily_signals = 8
        self._last_daily_reset = 0.0

        # Current regime
        self._current_regime: str = "NORMAL"

    async def on_tick(self, event: Event) -> None:
        tick: Tick = event.data["tick"]
        if tick.symbol not in self._symbols:
            return

        # Initialize history
        if tick.symbol not in self._prices:
            self._prices[tick.symbol] = deque(maxlen=self._slow_window)

        self._prices[tick.symbol].append(tick.price)

        # Need full slow window for trend confirmation
        if len(self._prices[tick.symbol]) < self._slow_window:
            return

        # Regime check
        if self._current_regime not in self._allowed_regimes:
            return

        # Cooldown check
        now = time.monotonic()
        last_ts = self._last_signal_ts.get(tick.symbol, 0)
        if now - last_ts < self._cooldown_seconds:
            return

        # Daily limit reset (rough 24h check)
        if now - self._last_daily_reset > 86400:
            self._daily_signals.clear()
            self._last_daily_reset = now

        if self._daily_signals.get(tick.symbol, 0) >= self._max_daily_signals:
            return

        prices = list(self._prices[tick.symbol])

        # Fast momentum: average return over fast window
        fast_prices = prices[-self._fast_window:]
        fast_returns = [
            (fast_prices[i] - fast_prices[i - 1]) / fast_prices[i - 1]
            for i in range(1, len(fast_prices))
        ]
        fast_momentum = sum(fast_returns) / len(fast_returns)

        # Slow trend: direction over slow window
        slow_return = (prices[-1] - prices[0]) / prices[0]

        # Volatility: standard deviation of fast returns (for adaptive threshold)
        mean_ret = sum(fast_returns) / len(fast_returns)
        variance = sum((r - mean_ret) ** 2 for r in fast_returns) / len(fast_returns)
        volatility = variance ** 0.5

        # Adaptive threshold: 1.5x volatility, minimum 0.05%
        threshold = max(volatility * 1.5, 0.0005)

        # Signal logic: fast momentum must exceed threshold AND align with slow trend
        side = None
        if fast_momentum > threshold and slow_return > 0:
            side = "buy"
        elif fast_momentum < -threshold and slow_return < 0:
            side = "sell"
        else:
            return

        # Confidence: scaled by momentum strength relative to threshold (0.3 to 0.9)
        raw_confidence = min(abs(fast_momentum) / threshold, 3.0) / 3.0
        confidence = 0.3 + raw_confidence * 0.6  # Range: 0.3 – 0.9

        # ATR-based stop/take-profit (approximate ATR from volatility)
        atr_estimate = volatility * tick.price * (self._fast_window ** 0.5)
        atr_estimate = max(atr_estimate, tick.price * 0.005)  # Minimum 0.5%

        if side == "buy":
            stop_loss = tick.price - atr_estimate * 2.0
            take_profit = tick.price + atr_estimate * 3.0
        else:
            stop_loss = tick.price + atr_estimate * 2.0
            take_profit = tick.price - atr_estimate * 3.0

        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=tick.symbol,
            side=side,
            entry_price=tick.price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            metadata={
                "fast_momentum": round(fast_momentum, 6),
                "slow_trend": round(slow_return, 6),
                "volatility": round(volatility, 6),
                "threshold": round(threshold, 6),
                "confidence": round(confidence, 3),
                "atr_estimate": round(atr_estimate, 2),
                "regime": self._current_regime,
                "reason": "momentum_trend_aligned",
            },
        )

        await self._publish_signal(signal)
        self._last_signal_ts[tick.symbol] = now
        self._daily_signals[tick.symbol] = self._daily_signals.get(tick.symbol, 0) + 1
        logger.info(
            f"[MOMENTUM] {side.upper()} {tick.symbol} @ {tick.price:.2f} | "
            f"momentum={fast_momentum:.4f} trend={slow_return:.4f} "
            f"conf={confidence:.2f} ATR≈{atr_estimate:.2f}"
        )

    async def on_funding(self, event: Event) -> None:
        pass

    async def on_regime_update(self, event: Event) -> None:
        regime = event.data.get("regime", "NORMAL")
        if isinstance(regime, str):
            self._current_regime = regime
        else:
            self._current_regime = getattr(regime, "name", str(regime))

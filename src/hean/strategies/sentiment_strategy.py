"""
Sentiment-based trading strategy.

Trades based on sentiment analysis from social media and news sources.
"""

import logging
from datetime import datetime, timedelta

from hean.core.bus import EventBus
from hean.core.types import Event, Signal
from hean.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class SentimentStrategy(BaseStrategy):
    """Strategy that trades based on sentiment analysis.

    Combines Twitter, Reddit, and News sentiment.

    Entry conditions:
    - Strong bullish/bearish sentiment (score > 0.6)
    - High confidence (> 0.75)
    - Agreement across sources
    """

    def __init__(
        self,
        bus: EventBus,
        symbols: list[str] | None = None,
        enabled: bool = True,
        min_confidence: float = 0.75,
        min_score: float = 0.6,
    ) -> None:
        super().__init__("sentiment_strategy", bus)
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._enabled = enabled
        self.min_confidence = min_confidence
        self.min_score = min_score

        self._last_signal_time: dict[str, datetime] = {}
        self._cooldown = timedelta(minutes=30)

        # Try to import sentiment aggregator (optional dependency)
        self._aggregator = None
        try:
            from hean.sentiment import SentimentAggregator
            self._aggregator = SentimentAggregator()
            logger.info("Sentiment aggregator initialized successfully")
        except (ImportError, Exception) as e:
            logger.warning(f"Sentiment aggregator not available ({e}) — using simple price momentum fallback")

        # Fallback: Simple price momentum when sentiment API unavailable
        self._price_history: dict[str, list[float]] = {}
        self._momentum_window = 50
        self._fallback_cooldown: dict[str, datetime] = {}
        self._fallback_cooldown_minutes = 30

    async def on_tick(self, event: Event) -> None:
        """Process tick events — check sentiment on each tick."""
        if not self._enabled:
            return

        tick = event.data.get("tick")
        if tick is None:
            return

        symbol = tick.symbol if hasattr(tick, "symbol") else event.data.get("symbol", "")
        if symbol not in self._symbols:
            return

        # FALLBACK: Use simple momentum when sentiment aggregator unavailable
        if self._aggregator is None:
            await self._momentum_fallback_signal(tick, symbol)
            return

        # Cooldown check
        last = self._last_signal_time.get(symbol)
        if last and (datetime.utcnow() - last) < self._cooldown:
            return

        try:
            crypto = symbol.replace("USDT", "")
            sentiment = await self._aggregator.get_signal(crypto)

            if not sentiment:
                return

            # Check thresholds
            if sentiment.confidence < self.min_confidence:
                return
            if abs(sentiment.overall_score) < self.min_score:
                return
            if sentiment.action == "HOLD":
                return

            price = tick.price if hasattr(tick, "price") else event.data.get("price", 0)
            if price <= 0:
                return

            side = "buy" if sentiment.action == "BUY" else "sell"

            # Stop loss at 1%, take profit at 2% (1:2 risk:reward)
            sl_pct = 0.01
            tp_pct = 0.02
            if side == "buy":
                stop_loss = price * (1 - sl_pct)
                take_profit = price * (1 + tp_pct)
            else:
                stop_loss = price * (1 + sl_pct)
                take_profit = price * (1 - tp_pct)

            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=side,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                confidence=sentiment.confidence,
                urgency=0.4,
                metadata={
                    "sentiment_score": sentiment.overall_score,
                    "reason": getattr(sentiment, "reason", "sentiment_signal"),
                },
            )

            await self._publish_signal(signal)
            self._last_signal_time[symbol] = datetime.utcnow()

        except Exception as e:
            logger.debug(f"Sentiment check error for {symbol}: {e}")

    async def _momentum_fallback_signal(self, tick, symbol: str) -> None:
        """Generate signals based on simple price momentum when sentiment API unavailable."""
        price = tick.price if hasattr(tick, "price") else 0
        if price <= 0:
            return

        # Initialize price history
        if symbol not in self._price_history:
            self._price_history[symbol] = []

        self._price_history[symbol].append(price)
        if len(self._price_history[symbol]) > self._momentum_window:
            self._price_history[symbol] = self._price_history[symbol][-self._momentum_window:]

        # Need enough history
        if len(self._price_history[symbol]) < 20:
            return

        # Check fallback cooldown
        if symbol in self._fallback_cooldown:
            time_since = (datetime.utcnow() - self._fallback_cooldown[symbol]).total_seconds() / 60
            if time_since < self._fallback_cooldown_minutes:
                return

        # Calculate momentum (recent vs older average)
        prices = self._price_history[symbol]
        recent_avg = sum(prices[-5:]) / 5
        older_avg = sum(prices[-20:-5]) / 15
        momentum_pct = (recent_avg - older_avg) / older_avg

        # Trade on significant momentum (>0.3%)
        if abs(momentum_pct) < 0.003:
            return

        side = "buy" if momentum_pct > 0 else "sell"
        confidence = min(0.7, abs(momentum_pct) * 50)  # Scale to 0.5-0.7

        # Stop loss at 0.8%, take profit at 1.6% (1:2 risk:reward)
        sl_pct = 0.008
        tp_pct = 0.016
        if side == "buy":
            stop_loss = price * (1 - sl_pct)
            take_profit = price * (1 + tp_pct)
        else:
            stop_loss = price * (1 + sl_pct)
            take_profit = price * (1 - tp_pct)

        signal = Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=side,
            entry_price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            urgency=0.3,
            metadata={
                "momentum_pct": momentum_pct,
                "reason": "momentum_fallback_no_sentiment_api",
            },
        )

        await self._publish_signal(signal)
        self._fallback_cooldown[symbol] = datetime.utcnow()
        logger.info(
            f"[SENTIMENT FALLBACK] {symbol} {side.upper()} @ ${price:.2f} "
            f"(momentum={momentum_pct:.2%}, no sentiment API)"
        )

    async def on_funding(self, event: Event) -> None:
        """Not used by sentiment strategy."""
        pass

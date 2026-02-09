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
        except ImportError:
            logger.warning("Sentiment aggregator not available — strategy will be passive")

    async def on_tick(self, event: Event) -> None:
        """Process tick events — check sentiment on each tick."""
        if not self._enabled or self._aggregator is None:
            return

        tick = event.data.get("tick")
        if tick is None:
            return

        symbol = tick.symbol if hasattr(tick, "symbol") else event.data.get("symbol", "")
        if symbol not in self._symbols:
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

            signal = Signal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                side=side,
                entry_price=price,
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

    async def on_funding(self, event: Event) -> None:
        """Not used by sentiment strategy."""
        pass

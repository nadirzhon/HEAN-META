"""
Sentiment aggregator

Combines sentiment from multiple sources into trading signals
"""

import asyncio
import logging
from datetime import datetime

from .models import SentimentLabel, SentimentScore, SentimentSignal, SentimentSource
from .news_client import NewsSentiment
from .reddit_client import RedditSentiment
from .twitter_client import TwitterSentiment

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """
    Aggregates sentiment from multiple sources

    Usage:
        aggregator = SentimentAggregator()
        signal = await aggregator.get_signal("BTC")

        if signal.should_trade:
            if signal.action == "BUY":
                execute_buy()
    """

    def __init__(
        self,
        weights: dict[SentimentSource, float] | None = None
    ):
        """
        Initialize aggregator

        Args:
            weights: source weights (default: news=0.5, twitter=0.3, reddit=0.2)
        """
        # Default weights (news most reliable, twitter/reddit less)
        self.weights = weights or {
            SentimentSource.NEWS: 0.5,
            SentimentSource.TWITTER: 0.3,
            SentimentSource.REDDIT: 0.2,
        }

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Initialize clients
        self.twitter = TwitterSentiment()
        self.reddit = RedditSentiment()
        self.news = NewsSentiment()

        self._initialized = False

    async def initialize(self):
        """Initialize all clients"""
        if self._initialized:
            return

        await asyncio.gather(
            self.twitter.initialize(),
            self.reddit.initialize(),
            self.news.initialize(),
            return_exceptions=True
        )

        self._initialized = True
        logger.info("Sentiment aggregator initialized")

    async def get_signal(
        self,
        symbol: str,
        twitter_hours: int = 1,
        reddit_hours: int = 24,
        news_hours: int = 24
    ) -> SentimentSignal | None:
        """
        Get aggregated sentiment signal

        Args:
            symbol: trading symbol
            twitter_hours: Twitter lookback period
            reddit_hours: Reddit lookback period
            news_hours: News lookback period

        Returns:
            SentimentSignal with action recommendation
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Fetch from all sources in parallel
            scores = await asyncio.gather(
                self.twitter.get_sentiment(symbol, hours=twitter_hours),
                self.reddit.get_sentiment(symbol, hours=reddit_hours),
                self.news.get_sentiment(symbol, hours=news_hours),
                return_exceptions=True
            )

            # Filter out errors and None
            source_scores = {}
            for i, source in enumerate([
                SentimentSource.TWITTER,
                SentimentSource.REDDIT,
                SentimentSource.NEWS
            ]):
                if isinstance(scores[i], SentimentScore):
                    source_scores[source] = scores[i]

            if not source_scores:
                logger.debug(f"No sentiment data available for {symbol}")
                return None

            # Aggregate
            return self._aggregate(symbol, source_scores)

        except Exception as e:
            logger.error(f"Error getting sentiment signal: {e}")
            return None

    def _aggregate(
        self,
        symbol: str,
        sources: dict[SentimentSource, SentimentScore]
    ) -> SentimentSignal:
        """
        Aggregate scores from multiple sources

        Uses weighted average based on source reliability
        """
        if not sources:
            return None

        # Calculate weighted score
        total_weight = 0.0
        weighted_score = 0.0

        for source, score in sources.items():
            weight = self.weights.get(source, 0.1)
            weighted_score += score.score * weight
            total_weight += weight

        if total_weight == 0:
            return None

        overall_score = weighted_score / total_weight

        # Calculate confidence
        confidence = self._calculate_confidence(sources, overall_score)

        # Determine action
        action = self._determine_action(overall_score, confidence)

        # Generate reason
        reason = self._generate_reason(sources, overall_score)

        return SentimentSignal(
            symbol=symbol,
            overall_score=overall_score,
            confidence=confidence,
            action=action,
            sources=sources,
            timestamp=datetime.utcnow(),
            reason=reason
        )

    def _calculate_confidence(
        self,
        sources: dict[SentimentSource, SentimentScore],
        overall_score: float
    ) -> float:
        """
        Calculate confidence in signal

        Higher confidence when:
        - Multiple sources agree
        - High volume of data
        - Strong sentiment (not neutral)
        """
        if not sources:
            return 0.0

        # 1. Agreement factor (do sources agree?)
        agreements = []
        for score in sources.values():
            # Check if source agrees with overall direction
            if overall_score > 0 and score.score > 0:
                agreements.append(1.0)
            elif overall_score < 0 and score.score < 0:
                agreements.append(1.0)
            elif abs(overall_score) < 0.1 and abs(score.score) < 0.1:
                agreements.append(0.5)  # Both neutral
            else:
                agreements.append(0.0)  # Disagree

        agreement_factor = sum(agreements) / len(agreements)

        # 2. Volume factor (more data = more confidence)
        total_volume = sum(s.volume for s in sources.values())
        volume_factor = min(1.0, total_volume / 100)  # Cap at 100 items

        # 3. Strength factor (strong sentiment = more confident)
        strength_factor = min(1.0, abs(overall_score) / 0.8)

        # 4. Source diversity factor (more sources = better)
        diversity_factor = len(sources) / 3  # 3 sources max

        # Weighted combination
        confidence = (
            agreement_factor * 0.4 +
            volume_factor * 0.2 +
            strength_factor * 0.3 +
            diversity_factor * 0.1
        )

        return min(1.0, max(0.0, confidence))

    def _determine_action(self, score: float, confidence: float) -> str:
        """Determine trading action"""
        # Need both strong signal AND high confidence
        if confidence < 0.5:
            return "HOLD"

        if score > 0.5 and confidence > 0.7:
            return "BUY"
        elif score < -0.5 and confidence > 0.7:
            return "SELL"
        else:
            return "HOLD"

    def _generate_reason(
        self,
        sources: dict[SentimentSource, SentimentScore],
        overall_score: float
    ) -> str:
        """Generate human-readable reason"""
        parts = []

        # Overall direction
        if overall_score > 0.6:
            parts.append("Strong bullish sentiment")
        elif overall_score > 0.2:
            parts.append("Bullish sentiment")
        elif overall_score < -0.6:
            parts.append("Strong bearish sentiment")
        elif overall_score < -0.2:
            parts.append("Bearish sentiment")
        else:
            parts.append("Neutral sentiment")

        # Source breakdown
        source_parts = []
        for source, score in sources.items():
            if score.label != SentimentLabel.NEUTRAL:
                source_parts.append(
                    f"{source.value}:{score.label.value}({score.volume})"
                )

        if source_parts:
            parts.append(f"[{', '.join(source_parts)}]")

        return " - ".join(parts)

    async def monitor_continuous(
        self,
        symbol: str,
        callback: callable,
        interval_seconds: int = 300  # 5 minutes
    ):
        """
        Continuously monitor sentiment

        Args:
            symbol: symbol to monitor
            callback: async function called with SentimentSignal
            interval_seconds: check interval
        """
        logger.info(f"Starting continuous sentiment monitoring for {symbol}")

        previous_action = None

        while True:
            try:
                signal = await self.get_signal(symbol)

                if signal:
                    # Only callback if action changed or strong signal
                    if (signal.action != previous_action or
                        signal.confidence > 0.85):

                        await callback(signal)
                        previous_action = signal.action

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(interval_seconds)


# Example usage
async def main():
    """Example usage"""
    aggregator = SentimentAggregator()
    await aggregator.initialize()

    # Get signal
    signal = await aggregator.get_signal("BTC")

    if signal:
        print("\nSentiment Signal for BTC:")
        print(f"  Action: {signal.action}")
        print(f"  Score: {signal.overall_score:.2f}")
        print(f"  Confidence: {signal.confidence:.2f}")
        print(f"  Should Trade: {signal.should_trade}")
        print(f"  Reason: {signal.reason}")
        print("\n  Sources:")
        for source, score in signal.sources.items():
            print(f"    {source.value}: {score.label.value} "
                  f"({score.score:.2f}, {score.volume} items)")
    else:
        print("No sentiment signal available")

    # Cleanup
    await aggregator.news.close()


if __name__ == "__main__":
    asyncio.run(main())

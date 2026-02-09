"""
Twitter sentiment monitoring

Monitors Twitter for crypto-related tweets and analyzes sentiment
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

from .analyzer import get_sentiment_analyzer
from .models import SentimentLabel, SentimentScore, SentimentSource, TextSentiment

logger = logging.getLogger(__name__)


class TwitterSentiment:
    """
    Twitter sentiment analyzer

    Usage:
        twitter = TwitterSentiment(api_key="...", api_secret="...")
        score = await twitter.get_sentiment("BTC")
        # Returns: SentimentScore(label="bullish", score=0.65, volume=150)
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        bearer_token: str | None = None
    ):
        """
        Initialize Twitter client

        Args:
            api_key: Twitter API key (or from env TWITTER_API_KEY)
            api_secret: Twitter API secret (or from env TWITTER_API_SECRET)
            bearer_token: Twitter Bearer token (or from env TWITTER_BEARER_TOKEN)
        """
        self.api_key = api_key or os.getenv("TWITTER_API_KEY")
        self.api_secret = api_secret or os.getenv("TWITTER_API_SECRET")
        self.bearer_token = bearer_token or os.getenv("TWITTER_BEARER_TOKEN")

        self.client = None
        self._initialized = False

        if not TWEEPY_AVAILABLE:
            logger.warning(
                "tweepy not installed. Install with: "
                "pip install tweepy --break-system-packages"
            )

        # Symbol mappings
        self.symbol_keywords = {
            "BTC": ["bitcoin", "btc", "$btc"],
            "ETH": ["ethereum", "eth", "$eth"],
            "SOL": ["solana", "sol", "$sol"],
            "BNB": ["binance", "bnb", "$bnb"],
        }

    async def initialize(self):
        """Initialize Twitter API client"""
        if self._initialized:
            return

        if not TWEEPY_AVAILABLE:
            logger.error("Cannot initialize: tweepy not available")
            return

        if not self.bearer_token:
            logger.warning(
                "Twitter credentials not provided. "
                "Set TWITTER_BEARER_TOKEN environment variable"
            )
            return

        try:
            self.client = tweepy.Client(
                bearer_token=self.bearer_token,
                wait_on_rate_limit=True
            )

            self._initialized = True
            logger.info("Twitter client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Twitter client: {e}")

    async def get_sentiment(
        self,
        symbol: str,
        hours: int = 1,
        max_tweets: int = 100
    ) -> SentimentScore | None:
        """
        Get aggregated sentiment for symbol

        Args:
            symbol: trading symbol (e.g., "BTC")
            hours: hours to look back
            max_tweets: maximum tweets to analyze

        Returns:
            SentimentScore or None
        """
        if not self._initialized:
            await self.initialize()

        if not self.client:
            logger.warning("Twitter client not available")
            return None

        try:
            # Get tweets
            tweets = await self._fetch_tweets(symbol, hours, max_tweets)

            if not tweets:
                logger.debug(f"No tweets found for {symbol}")
                return None

            # Analyze sentiment
            analyzer = await get_sentiment_analyzer()
            sentiments = await analyzer.analyze_batch(
                tweets,
                source=SentimentSource.TWITTER
            )

            if not sentiments:
                return None

            # Aggregate
            return self._aggregate_sentiments(sentiments)

        except Exception as e:
            logger.error(f"Error getting Twitter sentiment: {e}")
            return None

    async def _fetch_tweets(
        self,
        symbol: str,
        hours: int,
        max_tweets: int
    ) -> list[str]:
        """Fetch recent tweets about symbol"""
        keywords = self.symbol_keywords.get(symbol, [symbol.lower()])
        query = " OR ".join(keywords)

        # Add filters
        query += " -is:retweet lang:en"  # No retweets, English only

        # Time window
        start_time = datetime.utcnow() - timedelta(hours=hours)

        try:
            # Run in executor to not block
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.search_recent_tweets(
                    query=query,
                    start_time=start_time,
                    max_results=min(max_tweets, 100),
                    tweet_fields=['created_at', 'public_metrics']
                )
            )

            if not response.data:
                return []

            # Extract tweet texts
            tweets = [tweet.text for tweet in response.data]
            logger.info(f"Fetched {len(tweets)} tweets for {symbol}")

            return tweets

        except Exception as e:
            logger.error(f"Error fetching tweets: {e}")
            return []

    def _aggregate_sentiments(
        self,
        sentiments: list[TextSentiment]
    ) -> SentimentScore:
        """
        Aggregate individual sentiments into overall score

        Uses weighted average based on confidence
        """
        if not sentiments:
            return None

        # Calculate weighted average
        total_weight = sum(s.confidence for s in sentiments)
        if total_weight == 0:
            return None

        weighted_score = sum(
            s.score * s.confidence
            for s in sentiments
        ) / total_weight

        # Determine overall label
        if weighted_score > 0.2:
            label = SentimentLabel.BULLISH
        elif weighted_score < -0.2:
            label = SentimentLabel.BEARISH
        else:
            label = SentimentLabel.NEUTRAL

        return SentimentScore(
            label=label,
            score=weighted_score,
            volume=len(sentiments),
            source=SentimentSource.TWITTER,
            timestamp=datetime.utcnow()
        )

    async def monitor_realtime(
        self,
        symbol: str,
        callback: callable,
        interval_seconds: int = 60
    ):
        """
        Monitor Twitter in real-time

        Args:
            symbol: symbol to monitor
            callback: async function called with SentimentScore
            interval_seconds: how often to check
        """
        logger.info(f"Starting real-time Twitter monitoring for {symbol}")

        while True:
            try:
                score = await self.get_sentiment(symbol, hours=1)

                if score:
                    await callback(score)

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in Twitter monitoring: {e}")
                await asyncio.sleep(interval_seconds)


# Example usage
async def main():
    """Example usage"""
    twitter = TwitterSentiment()
    await twitter.initialize()

    # Get current sentiment
    score = await twitter.get_sentiment("BTC", hours=1)

    if score:
        print("Twitter Sentiment for BTC:")
        print(f"  Label: {score.label}")
        print(f"  Score: {score.score:.2f}")
        print(f"  Volume: {score.volume} tweets")
    else:
        print("No Twitter data available")


if __name__ == "__main__":
    asyncio.run(main())

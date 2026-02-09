"""
Reddit sentiment monitoring

Monitors crypto subreddits for sentiment analysis
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta

try:
    import praw
    PRAW_AVAILABLE = True
except ImportError:
    PRAW_AVAILABLE = False

from .analyzer import get_sentiment_analyzer
from .models import SentimentLabel, SentimentScore, SentimentSource, TextSentiment

logger = logging.getLogger(__name__)


class RedditSentiment:
    """
    Reddit sentiment analyzer

    Monitors:
    - r/cryptocurrency
    - r/bitcoin
    - r/ethereum
    - r/CryptoMarkets
    - r/wallstreetbets (for overall market sentiment)

    Usage:
        reddit = RedditSentiment(client_id="...", client_secret="...")
        score = await reddit.get_sentiment("BTC")
    """

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str = "HEAN Trading Bot"
    ):
        """
        Initialize Reddit client

        Get credentials at: https://www.reddit.com/prefs/apps

        Args:
            client_id: Reddit client ID (or from env REDDIT_CLIENT_ID)
            client_secret: Reddit client secret (or from env REDDIT_CLIENT_SECRET)
            user_agent: User agent string
        """
        self.client_id = client_id or os.getenv("REDDIT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("REDDIT_CLIENT_SECRET")
        self.user_agent = user_agent

        self.reddit = None
        self._initialized = False

        if not PRAW_AVAILABLE:
            logger.warning(
                "praw not installed. Install with: "
                "pip install praw --break-system-packages"
            )

        # Subreddits to monitor
        self.subreddits = [
            "cryptocurrency",
            "bitcoin",
            "ethereum",
            "CryptoMarkets",
            "CryptoCurrency",
        ]

        # Symbol keywords
        self.symbol_keywords = {
            "BTC": ["bitcoin", "btc"],
            "ETH": ["ethereum", "eth", "ether"],
            "SOL": ["solana", "sol"],
            "BNB": ["binance", "bnb"],
        }

    async def initialize(self):
        """Initialize Reddit API client"""
        if self._initialized:
            return

        if not PRAW_AVAILABLE:
            logger.error("Cannot initialize: praw not available")
            return

        if not self.client_id or not self.client_secret:
            logger.warning(
                "Reddit credentials not provided. "
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET"
            )
            return

        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )

            # Test connection
            _ = self.reddit.user.me()

            self._initialized = True
            logger.info("Reddit client initialized")

        except Exception as e:
            logger.error(f"Failed to initialize Reddit client: {e}")

    async def get_sentiment(
        self,
        symbol: str,
        hours: int = 24,
        max_posts: int = 50
    ) -> SentimentScore | None:
        """
        Get aggregated sentiment from Reddit

        Args:
            symbol: trading symbol
            hours: hours to look back
            max_posts: maximum posts/comments to analyze

        Returns:
            SentimentScore or None
        """
        if not self._initialized:
            await self.initialize()

        if not self.reddit:
            logger.warning("Reddit client not available")
            return None

        try:
            # Fetch posts and comments
            texts = await self._fetch_content(symbol, hours, max_posts)

            if not texts:
                logger.debug(f"No Reddit content found for {symbol}")
                return None

            # Analyze sentiment
            analyzer = await get_sentiment_analyzer()
            sentiments = await analyzer.analyze_batch(
                texts,
                source=SentimentSource.REDDIT
            )

            if not sentiments:
                return None

            # Aggregate
            return self._aggregate_sentiments(sentiments)

        except Exception as e:
            logger.error(f"Error getting Reddit sentiment: {e}")
            return None

    async def _fetch_content(
        self,
        symbol: str,
        hours: int,
        max_posts: int
    ) -> list[str]:
        """Fetch posts and comments about symbol"""
        keywords = self.symbol_keywords.get(symbol, [symbol.lower()])

        texts = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        try:
            loop = asyncio.get_event_loop()

            # Search across multiple subreddits
            for subreddit_name in self.subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    # Fetch hot posts in executor (capture subreddit to avoid late binding)
                    _subreddit = subreddit
                    posts = await loop.run_in_executor(
                        None,
                        lambda sr=_subreddit: list(sr.hot(limit=max_posts // len(self.subreddits)))
                    )

                    for post in posts:
                        # Check if post is recent
                        post_time = datetime.fromtimestamp(post.created_utc)
                        if post_time < cutoff_time:
                            continue

                        # Check if post mentions symbol
                        title_lower = post.title.lower()
                        if any(keyword in title_lower for keyword in keywords):
                            # Add title
                            texts.append(post.title)

                            # Add selftext if exists
                            if post.selftext:
                                texts.append(post.selftext[:500])

                            # Add top comments (capture post to avoid late binding)
                            _post = post
                            await loop.run_in_executor(
                                None,
                                lambda p=_post: p.comments.replace_more(limit=0)
                            )

                            for comment in list(post.comments)[:5]:
                                if hasattr(comment, 'body'):
                                    texts.append(comment.body[:500])

                except Exception as e:
                    logger.warning(f"Error fetching from r/{subreddit_name}: {e}")
                    continue

            logger.info(f"Fetched {len(texts)} Reddit texts for {symbol}")
            return texts[:max_posts]  # Limit total

        except Exception as e:
            logger.error(f"Error fetching Reddit content: {e}")
            return []

    def _aggregate_sentiments(
        self,
        sentiments: list[TextSentiment]
    ) -> SentimentScore:
        """Aggregate sentiments with weighted average"""
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

        # Determine label
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
            source=SentimentSource.REDDIT,
            timestamp=datetime.utcnow()
        )

    async def get_trending_topics(self, limit: int = 10) -> list[str]:
        """
        Get trending topics from crypto subreddits

        Returns:
            List of trending keywords
        """
        if not self._initialized:
            await self.initialize()

        if not self.reddit:
            return []

        try:
            topics = []
            loop = asyncio.get_event_loop()

            for subreddit_name in self.subreddits[:3]:  # Top 3 subreddits
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)
                    _subreddit = subreddit  # Capture to avoid late binding
                    posts = await loop.run_in_executor(
                        None,
                        lambda sr=_subreddit, lim=limit: list(sr.hot(limit=lim))
                    )

                    for post in posts:
                        topics.append(post.title)

                except Exception as e:
                    logger.warning(f"Error getting trends from r/{subreddit_name}: {e}")

            return topics

        except Exception as e:
            logger.error(f"Error getting trending topics: {e}")
            return []


# Example usage
async def main():
    """Example usage"""
    reddit = RedditSentiment()
    await reddit.initialize()

    # Get sentiment
    score = await reddit.get_sentiment("BTC", hours=24)

    if score:
        print("Reddit Sentiment for BTC:")
        print(f"  Label: {score.label}")
        print(f"  Score: {score.score:.2f}")
        print(f"  Volume: {score.volume} posts/comments")
    else:
        print("No Reddit data available")

    # Get trending
    trends = await reddit.get_trending_topics()
    print(f"\nTrending topics: {trends[:5]}")


if __name__ == "__main__":
    asyncio.run(main())

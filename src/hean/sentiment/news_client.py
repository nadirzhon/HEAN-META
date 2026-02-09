"""
News sentiment monitoring

Monitors crypto news sources for breaking news and sentiment
"""

import asyncio
import logging
import os
from datetime import datetime, timedelta

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(
        "feedparser not installed. RSS parsing will use fallback. "
        "Install with: pip install feedparser --break-system-packages"
    )

from .analyzer import get_sentiment_analyzer
from .models import SentimentLabel, SentimentScore, SentimentSource, TextSentiment

logger = logging.getLogger(__name__)


class NewsSentiment:
    """
    News sentiment analyzer

    Monitors:
    - CoinDesk
    - CoinTelegraph
    - Decrypt
    - The Block
    - Bitcoin.com

    Usage:
        news = NewsSentiment()
        score = await news.get_sentiment("BTC")
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize news client

        Args:
            api_key: NewsAPI key (optional, for more sources)
        """
        self.api_key = api_key or os.getenv("NEWS_API_KEY")
        self._session = None

        if not AIOHTTP_AVAILABLE:
            logger.warning(
                "aiohttp not installed. Install with: "
                "pip install aiohttp --break-system-packages"
            )

        # RSS feeds for crypto news
        self.news_sources = [
            {
                "name": "CoinDesk",
                "rss": "https://www.coindesk.com/arc/outboundfeeds/rss/",
                "url": "https://www.coindesk.com/"
            },
            {
                "name": "CoinTelegraph",
                "rss": "https://cointelegraph.com/rss",
                "url": "https://cointelegraph.com/"
            },
            {
                "name": "Decrypt",
                "rss": "https://decrypt.co/feed",
                "url": "https://decrypt.co/"
            }
        ]

        # Symbol keywords
        self.symbol_keywords = {
            "BTC": ["bitcoin", "btc"],
            "ETH": ["ethereum", "eth", "ether"],
            "SOL": ["solana", "sol"],
            "BNB": ["binance", "bnb"],
        }

    async def initialize(self):
        """Initialize HTTP session"""
        if not AIOHTTP_AVAILABLE:
            logger.error("Cannot initialize: aiohttp not available")
            return

        if not self._session:
            self._session = aiohttp.ClientSession()
            logger.info("News client initialized")

    async def close(self):
        """Close HTTP session"""
        if self._session:
            await self._session.close()

    async def get_sentiment(
        self,
        symbol: str,
        hours: int = 24,
        max_articles: int = 20
    ) -> SentimentScore | None:
        """
        Get sentiment from news articles

        Args:
            symbol: trading symbol
            hours: hours to look back
            max_articles: maximum articles to analyze

        Returns:
            SentimentScore or None
        """
        if not self._session:
            await self.initialize()

        try:
            # Fetch articles
            articles = await self._fetch_articles(symbol, hours, max_articles)

            if not articles:
                logger.debug(f"No news articles found for {symbol}")
                return None

            # Analyze sentiment
            analyzer = await get_sentiment_analyzer()
            sentiments = await analyzer.analyze_batch(
                articles,
                source=SentimentSource.NEWS
            )

            if not sentiments:
                return None

            # Aggregate
            return self._aggregate_sentiments(sentiments)

        except Exception as e:
            logger.error(f"Error getting news sentiment: {e}")
            return None

    async def _fetch_articles(
        self,
        symbol: str,
        hours: int,
        max_articles: int
    ) -> list[str]:
        """
        Fetch news articles (simplified version)

        In production, use feedparser for RSS or NewsAPI
        """
        keywords = self.symbol_keywords.get(symbol, [symbol.lower()])
        articles = []

        try:
            # Simpler approach: use NewsAPI if available
            if self.api_key:
                articles = await self._fetch_from_newsapi(
                    keywords,
                    hours,
                    max_articles
                )
            else:
                # Fallback: fetch from public APIs
                articles = await self._fetch_from_public_sources(
                    keywords,
                    hours,
                    max_articles
                )

            logger.info(f"Fetched {len(articles)} news articles for {symbol}")
            return articles

        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            return []

    async def _fetch_from_newsapi(
        self,
        keywords: list[str],
        hours: int,
        max_articles: int
    ) -> list[str]:
        """Fetch from NewsAPI (requires API key)"""
        if not self._session or not self.api_key:
            return []

        url = "https://newsapi.org/v2/everything"
        query = " OR ".join(keywords)

        params = {
            "q": query + " AND crypto",
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.api_key,
            "pageSize": max_articles
        }

        # Time range
        from_time = datetime.utcnow() - timedelta(hours=hours)
        params["from"] = from_time.isoformat()

        try:
            async with self._session.get(url, params=params) as response:
                if response.status != 200:
                    logger.warning(f"NewsAPI returned {response.status}")
                    return []

                data = await response.json()
                articles = [
                    article.get('title', '') + ". " + article.get('description', '')
                    for article in data.get('articles', [])
                ]

                return articles

        except Exception as e:
            logger.error(f"Error fetching from NewsAPI: {e}")
            return []

    async def _fetch_from_public_sources(
        self,
        keywords: list[str],
        hours: int,
        max_articles: int
    ) -> list[str]:
        """
        Fetch from public crypto news sites using RSS feeds

        Uses feedparser to parse RSS feeds from configured sources
        """
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not available, using fallback empty list")
            return self._get_fallback_news()

        articles = []
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)

        try:
            for source in self.news_sources:
                try:
                    # Parse RSS feed
                    feed = await asyncio.get_event_loop().run_in_executor(
                        None, feedparser.parse, source["rss"]
                    )

                    if not feed.entries:
                        logger.debug(f"No entries in feed from {source['name']}")
                        continue

                    # Process entries
                    for entry in feed.entries:
                        if len(articles) >= max_articles:
                            break

                        # Check if article is relevant to keywords
                        title = entry.get('title', '').lower()
                        description = entry.get('description', '').lower()
                        content = f"{title} {description}"

                        if not any(kw in content for kw in keywords):
                            continue

                        # Check time filter
                        published = entry.get('published_parsed')
                        if published:
                            pub_time = datetime(*published[:6])
                            if pub_time < cutoff_time:
                                continue

                        # Add article
                        article_text = f"{entry.get('title', '')}. {entry.get('description', '')}"
                        articles.append(article_text)

                except Exception as e:
                    logger.error(f"Error parsing RSS from {source['name']}: {e}")
                    continue

            if not articles:
                logger.debug(f"No articles found matching keywords {keywords}")
                return self._get_fallback_news()

            logger.info(f"Fetched {len(articles)} articles from RSS feeds")
            return articles

        except Exception as e:
            logger.error(f"Error fetching from RSS sources: {e}")
            return self._get_fallback_news(max_articles)

    def _get_fallback_news(self) -> list[str]:
        """Fallback when feedparser unavailable — returns empty list."""
        logger.warning(
            "[NEWS_CLIENT] feedparser not available. No news data will be used. "
            "Install feedparser: pip install feedparser"
        )
        return []

    def _aggregate_sentiments(
        self,
        sentiments: list[TextSentiment]
    ) -> SentimentScore:
        """Aggregate news sentiments"""
        if not sentiments:
            return None

        # News gets higher weight than social media (more reliable)
        total_weight = sum(s.confidence * 1.5 for s in sentiments)
        if total_weight == 0:
            return None

        weighted_score = sum(
            s.score * s.confidence * 1.5
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
            source=SentimentSource.NEWS,
            timestamp=datetime.utcnow()
        )

    async def monitor_breaking_news(
        self,
        callback: callable,
        interval_seconds: int = 300  # 5 minutes
    ):
        """
        Monitor for breaking news in real-time

        Args:
            callback: async function called with important news
            interval_seconds: check interval
        """
        logger.info("Starting breaking news monitoring")

        seen_titles = set()

        while True:
            try:
                # Fetch latest news
                articles = await self._fetch_articles("BTC", hours=1, max_articles=10)

                # Analyze sentiment
                analyzer = await get_sentiment_analyzer()

                for article in articles:
                    # Skip if seen
                    if article in seen_titles:
                        continue

                    seen_titles.add(article)

                    # Analyze
                    sentiment = await analyzer.analyze(article, SentimentSource.NEWS)

                    if sentiment and abs(sentiment.score) > 0.8:
                        # Strong sentiment = important news
                        await callback(sentiment)

                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in news monitoring: {e}")
                await asyncio.sleep(interval_seconds)


# Example usage
async def main():
    """Example usage"""
    news = NewsSentiment()
    await news.initialize()

    try:
        # Get sentiment
        score = await news.get_sentiment("BTC", hours=24)

        if score:
            print("News Sentiment for BTC:")
            print(f"  Label: {score.label}")
            print(f"  Score: {score.score:.2f}")
            print(f"  Volume: {score.volume} articles")
        else:
            print("No news data available")

    finally:
        await news.close()


if __name__ == "__main__":
    asyncio.run(main())

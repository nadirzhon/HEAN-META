"""News sentiment analyzer for crypto news sources.

Aggregates and analyzes sentiment from crypto news APIs.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import redis
import requests
from transformers import pipeline

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """News article data."""

    title: str
    description: str
    source: str
    url: str
    published_at: datetime
    sentiment: Optional[float] = None  # -1 to +1


class Newssentiment:
    """Crypto news sentiment analyzer."""

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        cryptopanic_key: Optional[str] = None,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 600,  # 10 minutes
    ):
        self.newsapi_key = newsapi_key
        self.cryptopanic_key = cryptopanic_key
        self.cache_ttl = cache_ttl

        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )

        # Initialize sentiment analyzer
        logger.info("Loading news sentiment model")
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            device=0 if requests.get("http://localhost") else -1,
        )
        logger.info("News sentiment model loaded")

        self.newsapi_url = "https://newsapi.org/v2/everything"
        self.cryptopanic_url = "https://cryptopanic.com/api/v1/posts/"

    def _get_cache_key(self, symbol: str) -> str:
        """Generate Redis cache key."""
        return f"news_sentiment:{symbol}"

    def _get_cached_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get cached news sentiment."""
        cache_key = self._get_cache_key(symbol)
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                return eval(cached)
        except Exception as e:
            logger.warning(f"Cache retrieval error for {cache_key}: {e}")
        return None

    def _cache_sentiment(self, symbol: str, data: Dict):
        """Cache news sentiment in Redis."""
        cache_key = self._get_cache_key(symbol)
        try:
            self.redis_client.setex(cache_key, self.cache_ttl, str(data))
        except Exception as e:
            logger.warning(f"Cache write error for {cache_key}: {e}")

    def fetch_newsapi_articles(
        self, symbol: str, hours: int = 24
    ) -> List[NewsArticle]:
        """Fetch articles from NewsAPI.

        Args:
            symbol: Crypto symbol (e.g., "Bitcoin", "Ethereum")
            hours: How many hours back to fetch

        Returns:
            List of NewsArticle objects
        """
        if not self.newsapi_key:
            logger.warning("NewsAPI key not configured")
            return []

        # Map symbols to search terms
        search_terms = {
            "BTC": "Bitcoin",
            "ETH": "Ethereum",
            "SOL": "Solana",
            "DOGE": "Dogecoin",
        }
        
        query = search_terms.get(symbol.upper(), symbol)
        from_date = datetime.utcnow() - timedelta(hours=hours)

        try:
            response = requests.get(
                self.newsapi_url,
                params={
                    "q": query,
                    "from": from_date.isoformat(),
                    "sortBy": "publishedAt",
                    "language": "en",
                    "apiKey": self.newsapi_key,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            for article in data.get("articles", []):
                articles.append(NewsArticle(
                    title=article["title"],
                    description=article.get("description", ""),
                    source=article["source"]["name"],
                    url=article["url"],
                    published_at=datetime.fromisoformat(
                        article["publishedAt"].replace("Z", "+00:00")
                    ),
                ))

            logger.info(f"Fetched {len(articles)} articles from NewsAPI for {query}")
            return articles

        except Exception as e:
            logger.error(f"NewsAPI fetch error for {symbol}: {e}")
            return []

    def fetch_cryptopanic_news(
        self, symbol: str, hours: int = 24
    ) -> List[NewsArticle]:
        """Fetch news from CryptoPanic.

        Args:
            symbol: Crypto symbol
            hours: How many hours back to fetch

        Returns:
            List of NewsArticle objects
        """
        if not self.cryptopanic_key:
            logger.warning("CryptoPanic key not configured")
            return []

        try:
            response = requests.get(
                self.cryptopanic_url,
                params={
                    "auth_token": self.cryptopanic_key,
                    "currencies": symbol.upper(),
                    "filter": "hot",
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            articles = []
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)

            for post in data.get("results", []):
                published_at = datetime.fromisoformat(
                    post["published_at"].replace("Z", "+00:00")
                )
                
                if published_at < cutoff_time:
                    continue

                articles.append(NewsArticle(
                    title=post["title"],
                    description=post.get("title", ""),  # CryptoPanic doesn't have description
                    source=post["source"]["title"],
                    url=post["url"],
                    published_at=published_at,
                ))

            logger.info(f"Fetched {len(articles)} articles from CryptoPanic for {symbol}")
            return articles

        except Exception as e:
            logger.error(f"CryptoPanic fetch error for {symbol}: {e}")
            return []

    def analyze_article_sentiment(self, article: NewsArticle) -> float:
        """Analyze sentiment of a single article.

        Args:
            article: NewsArticle object

        Returns:
            Sentiment score (-1 to +1)
        """
        text = f"{article.title}. {article.description}"
        
        try:
            result = self.sentiment_pipeline(text[:512])[0]  # Limit to 512 chars
            
            # Convert FinBERT output to -1 to +1 scale
            label = result["label"].lower()
            score = result["score"]
            
            if label == "positive":
                return score
            elif label == "negative":
                return -score
            else:  # neutral
                return 0.0

        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return 0.0

    def get_news_sentiment(
        self, symbol: str, hours: int = 24, use_cache: bool = True
    ) -> Dict:
        """Get aggregated news sentiment for a symbol.

        Args:
            symbol: Crypto symbol
            hours: How many hours back to analyze
            use_cache: Whether to use cached data

        Returns:
            Dict with sentiment analysis results
        """
        if use_cache:
            cached = self._get_cached_sentiment(symbol)
            if cached:
                logger.debug(f"Cache hit for news sentiment {symbol}")
                return cached

        # Fetch articles from all sources
        articles = []
        articles.extend(self.fetch_newsapi_articles(symbol, hours))
        articles.extend(self.fetch_cryptopanic_news(symbol, hours))

        if not articles:
            logger.warning(f"No news articles found for {symbol}")
            result = {
                "symbol": symbol,
                "sentiment": 0.0,
                "article_count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "timestamp": datetime.utcnow().isoformat(),
            }
            return result

        # Analyze sentiment for each article
        logger.info(f"Analyzing sentiment for {len(articles)} articles about {symbol}")
        
        sentiments = []
        for article in articles:
            sentiment = self.analyze_article_sentiment(article)
            article.sentiment = sentiment
            sentiments.append(sentiment)

        # Aggregate results
        avg_sentiment = sum(sentiments) / len(sentiments)
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count

        result = {
            "symbol": symbol,
            "sentiment": avg_sentiment,
            "article_count": len(articles),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self._cache_sentiment(symbol, result)

        logger.info(
            f"{symbol} news sentiment: {avg_sentiment:.3f} "
            f"({positive_count} pos, {negative_count} neg, {neutral_count} neutral)"
        )

        return result

    def health_check(self) -> Dict[str, bool]:
        """Check if all components are healthy."""
        health = {
            "redis_connected": False,
            "model_loaded": self.sentiment_pipeline is not None,
            "newsapi_configured": self.newsapi_key is not None,
            "cryptopanic_configured": self.cryptopanic_key is not None,
        }

        try:
            self.redis_client.ping()
            health["redis_connected"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        return health

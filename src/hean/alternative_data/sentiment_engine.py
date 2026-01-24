"""Sentiment analysis engine using FinBERT for Twitter and Reddit data.

Real-time sentiment analysis for crypto markets with caching and logging.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import redis
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class SentimentScore:
    """Sentiment analysis result."""

    symbol: str
    source: str
    sentiment: float  # -1 (bearish) to +1 (bullish)
    confidence: float
    volume: int  # number of mentions
    timestamp: datetime
    raw_scores: Dict[str, float]  # positive, negative, neutral


class SentimentEngine:
    """Twitter and Reddit sentiment analyzer using FinBERT."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        cache_ttl: int = 300,  # 5 minutes
        model_name: str = "ProsusAI/finbert",
    ):
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            decode_responses=True,
        )
        self.cache_ttl = cache_ttl

        logger.info(f"Loading FinBERT model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"FinBERT loaded on device: {self.device}")

    def _get_cache_key(self, symbol: str, source: str) -> str:
        """Generate Redis cache key."""
        return f"sentiment:{source}:{symbol}"

    def _get_cached_sentiment(self, symbol: str, source: str) -> Optional[SentimentScore]:
        """Get cached sentiment score."""
        cache_key = self._get_cache_key(symbol, source)
        try:
            cached = self.redis_client.get(cache_key)
            if cached:
                data = eval(cached)  # Safe as we control the data
                return SentimentScore(**data)
        except Exception as e:
            logger.warning(f"Cache retrieval error for {cache_key}: {e}")
        return None

    def _cache_sentiment(self, score: SentimentScore):
        """Cache sentiment score in Redis."""
        cache_key = self._get_cache_key(score.symbol, score.source)
        try:
            data = {
                "symbol": score.symbol,
                "source": score.source,
                "sentiment": score.sentiment,
                "confidence": score.confidence,
                "volume": score.volume,
                "timestamp": score.timestamp.isoformat(),
                "raw_scores": score.raw_scores,
            }
            self.redis_client.setex(
                cache_key,
                self.cache_ttl,
                str(data),
            )
        except Exception as e:
            logger.warning(f"Cache write error for {cache_key}: {e}")

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        # Remove URLs
        text = re.sub(r"http\S+|www.\S+", "", text)
        # Remove mentions
        text = re.sub(r"@\w+", "", text)
        # Remove hashtags but keep the word
        text = re.sub(r"#(\w+)", r"\1", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text using FinBERT.

        Returns:
            Dict with positive, negative, neutral scores
        """
        cleaned_text = self._preprocess_text(text)

        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]

        # FinBERT outputs: [positive, negative, neutral]
        return {
            "positive": probs[0].item(),
            "negative": probs[1].item(),
            "neutral": probs[2].item(),
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Analyze sentiment for multiple texts in batch."""
        cleaned_texts = [self._preprocess_text(t) for t in texts]

        inputs = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        results = []
        for prob in probs:
            results.append({
                "positive": prob[0].item(),
                "negative": prob[1].item(),
                "neutral": prob[2].item(),
            })

        return results

    def get_sentiment_score(
        self,
        symbol: str,
        texts: List[str],
        source: str = "twitter",
        use_cache: bool = True,
    ) -> SentimentScore:
        """Get aggregated sentiment score for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            texts: List of texts to analyze
            source: Data source ("twitter" or "reddit")
            use_cache: Whether to use cached results

        Returns:
            SentimentScore with aggregated sentiment
        """
        # Check cache first
        if use_cache:
            cached = self._get_cached_sentiment(symbol, source)
            if cached:
                logger.debug(f"Cache hit for {symbol} from {source}")
                return cached

        if not texts:
            logger.warning(f"No texts provided for {symbol} from {source}")
            return SentimentScore(
                symbol=symbol,
                source=source,
                sentiment=0.0,
                confidence=0.0,
                volume=0,
                timestamp=datetime.utcnow(),
                raw_scores={"positive": 0.0, "negative": 0.0, "neutral": 1.0},
            )

        # Analyze texts in batch
        logger.info(f"Analyzing {len(texts)} texts for {symbol} from {source}")
        sentiment_results = self.analyze_batch(texts)

        # Aggregate scores
        avg_positive = np.mean([r["positive"] for r in sentiment_results])
        avg_negative = np.mean([r["negative"] for r in sentiment_results])
        avg_neutral = np.mean([r["neutral"] for r in sentiment_results])

        # Calculate overall sentiment: positive - negative (range: -1 to +1)
        sentiment = float(avg_positive - avg_negative)

        # Confidence based on how polarized the sentiment is
        confidence = float(1.0 - avg_neutral)

        score = SentimentScore(
            symbol=symbol,
            source=source,
            sentiment=sentiment,
            confidence=confidence,
            volume=len(texts),
            timestamp=datetime.utcnow(),
            raw_scores={
                "positive": float(avg_positive),
                "negative": float(avg_negative),
                "neutral": float(avg_neutral),
            },
        )

        # Cache the result
        self._cache_sentiment(score)

        logger.info(
            f"{symbol} {source} sentiment: {sentiment:.3f} "
            f"(confidence: {confidence:.3f}, volume: {len(texts)})"
        )

        return score

    def get_twitter_sentiment(
        self, symbol: str, tweets: List[str], use_cache: bool = True
    ) -> SentimentScore:
        """Get Twitter sentiment for a symbol."""
        return self.get_sentiment_score(symbol, tweets, "twitter", use_cache)

    def get_reddit_sentiment(
        self, symbol: str, posts: List[str], use_cache: bool = True
    ) -> SentimentScore:
        """Get Reddit sentiment for a symbol."""
        return self.get_sentiment_score(symbol, posts, "reddit", use_cache)

    def get_combined_sentiment(
        self,
        symbol: str,
        twitter_texts: List[str],
        reddit_texts: List[str],
        twitter_weight: float = 0.6,
        reddit_weight: float = 0.4,
    ) -> Dict[str, any]:
        """Get combined sentiment from Twitter and Reddit.

        Args:
            symbol: Trading symbol
            twitter_texts: List of tweets
            reddit_texts: List of reddit posts/comments
            twitter_weight: Weight for Twitter sentiment
            reddit_weight: Weight for Reddit sentiment

        Returns:
            Combined sentiment data
        """
        twitter_sentiment = self.get_twitter_sentiment(symbol, twitter_texts)
        reddit_sentiment = self.get_reddit_sentiment(symbol, reddit_texts)

        # Weighted average
        combined_score = (
            twitter_sentiment.sentiment * twitter_weight
            + reddit_sentiment.sentiment * reddit_weight
        )

        combined_confidence = (
            twitter_sentiment.confidence * twitter_weight
            + reddit_sentiment.confidence * reddit_weight
        )

        total_volume = twitter_sentiment.volume + reddit_sentiment.volume

        logger.info(
            f"{symbol} combined sentiment: {combined_score:.3f} "
            f"(Twitter: {twitter_sentiment.sentiment:.3f}, "
            f"Reddit: {reddit_sentiment.sentiment:.3f}, "
            f"Total volume: {total_volume})"
        )

        return {
            "symbol": symbol,
            "combined_sentiment": combined_score,
            "combined_confidence": combined_confidence,
            "total_volume": total_volume,
            "twitter": twitter_sentiment,
            "reddit": reddit_sentiment,
            "timestamp": datetime.utcnow(),
        }

    def health_check(self) -> Dict[str, bool]:
        """Check if all components are healthy."""
        health = {
            "model_loaded": self.model is not None,
            "redis_connected": False,
        }

        try:
            self.redis_client.ping()
            health["redis_connected"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        return health

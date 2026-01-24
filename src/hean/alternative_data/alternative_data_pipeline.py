"""Alternative Data Pipeline - Unified interface for all alternative data sources.

Aggregates:
- Sentiment (Twitter/Reddit via FinBERT)
- On-chain metrics (Glassnode/CryptoQuant)
- Funding rates (Bybit, Binance, OKX)
- News sentiment
- Fear & Greed Index

All data is cached in Redis and logged.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from hean.alternative_data.fear_greed_index import FearGreedIndexCollector
from hean.alternative_data.funding_rates_aggregator import FundingRatesAggregator
from hean.alternative_data.news_sentiment import Newssentiment
from hean.alternative_data.onchain_collector import OnChainDataCollector
from hean.alternative_data.sentiment_engine import SentimentEngine

logger = logging.getLogger(__name__)


@dataclass
class AlternativeDataSnapshot:
    """Complete alternative data snapshot for a symbol."""

    symbol: str
    timestamp: datetime

    # Sentiment data
    twitter_sentiment: float = 0.0
    reddit_sentiment: float = 0.0
    combined_sentiment: float = 0.0
    sentiment_volume: int = 0

    # On-chain metrics
    exchange_netflow: float = 0.0
    mvrv_ratio: float = 1.0
    active_addresses: int = 0

    # Funding rates
    avg_funding_rate: float = 0.0
    funding_spread: float = 0.0
    funding_signal: str = "neutral"

    # News sentiment
    news_sentiment: float = 0.0
    news_article_count: int = 0

    # Fear & Greed Index
    fear_greed_value: int = 50
    fear_greed_classification: str = "Neutral"
    fear_greed_signal: str = "hold"

    # Composite scores
    overall_sentiment: float = 0.0
    confidence_score: float = 0.0

    def __post_init__(self):
        """Calculate composite scores."""
        self._calculate_composite_scores()

    def _calculate_composite_scores(self):
        """Calculate overall sentiment and confidence from all sources."""
        # Weight different sources
        weights = {
            "social_sentiment": 0.25,
            "news_sentiment": 0.20,
            "funding_rates": 0.20,
            "onchain": 0.20,
            "fear_greed": 0.15,
        }

        # Normalize funding rate to -1 to +1 scale
        funding_normalized = max(min(self.avg_funding_rate / 0.01, 1.0), -1.0)

        # Normalize MVRV: >3.7 = -1 (overvalued), <1.0 = +1 (undervalued)
        if self.mvrv_ratio > 3.7:
            mvrv_signal = -1.0
        elif self.mvrv_ratio < 1.0:
            mvrv_signal = 1.0
        else:
            mvrv_signal = (2.35 - self.mvrv_ratio) / 1.35  # Linear interpolation

        # Normalize exchange netflow: positive flow (into exchanges) = bearish
        netflow_normalized = -max(min(self.exchange_netflow / 10000, 1.0), -1.0)

        # Combine on-chain signals
        onchain_signal = (mvrv_signal + netflow_normalized) / 2.0

        # Normalize Fear & Greed (contrarian): high value = bearish
        fg_normalized = (50 - self.fear_greed_value) / 50.0

        # Calculate weighted overall sentiment
        self.overall_sentiment = (
            weights["social_sentiment"] * self.combined_sentiment
            + weights["news_sentiment"] * self.news_sentiment
            + weights["funding_rates"] * funding_normalized
            + weights["onchain"] * onchain_signal
            + weights["fear_greed"] * fg_normalized
        )

        # Calculate confidence based on data availability and agreement
        confidence_factors = []
        
        if self.sentiment_volume > 0:
            confidence_factors.append(min(self.sentiment_volume / 100, 1.0))
        if self.news_article_count > 0:
            confidence_factors.append(min(self.news_article_count / 20, 1.0))
        if abs(self.avg_funding_rate) > 0:
            confidence_factors.append(0.8)

        self.confidence_score = (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors
            else 0.0
        )


class AlternativeDataPipeline:
    """Unified pipeline for all alternative data sources."""

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        glassnode_api_key: Optional[str] = None,
        cryptoquant_api_key: Optional[str] = None,
        newsapi_key: Optional[str] = None,
        cryptopanic_key: Optional[str] = None,
    ):
        """Initialize all data collectors.

        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            glassnode_api_key: Glassnode API key
            cryptoquant_api_key: CryptoQuant API key
            newsapi_key: NewsAPI key
            cryptopanic_key: CryptoPanic API key
        """
        logger.info("Initializing Alternative Data Pipeline")

        # Initialize sentiment engine
        self.sentiment_engine = SentimentEngine(
            redis_host=redis_host,
            redis_port=redis_port,
        )

        # Initialize on-chain collector
        self.onchain_collector = OnChainDataCollector(
            glassnode_api_key=glassnode_api_key,
            cryptoquant_api_key=cryptoquant_api_key,
            redis_host=redis_host,
            redis_port=redis_port,
        )

        # Initialize funding rates aggregator
        self.funding_aggregator = FundingRatesAggregator(
            redis_host=redis_host,
            redis_port=redis_port,
        )

        # Initialize news sentiment
        self.news_sentiment = Newssentiment(
            newsapi_key=newsapi_key,
            cryptopanic_key=cryptopanic_key,
            redis_host=redis_host,
            redis_port=redis_port,
        )

        # Initialize Fear & Greed Index
        self.fear_greed = FearGreedIndexCollector(
            redis_host=redis_host,
            redis_port=redis_port,
        )

        logger.info("Alternative Data Pipeline initialized successfully")

    def get_complete_snapshot(
        self,
        symbol: str,
        twitter_texts: Optional[List[str]] = None,
        reddit_texts: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> AlternativeDataSnapshot:
        """Get complete alternative data snapshot for a symbol.

        Args:
            symbol: Trading symbol (e.g., "BTC", "ETH")
            twitter_texts: Optional list of recent tweets
            reddit_texts: Optional list of recent reddit posts
            use_cache: Whether to use cached data

        Returns:
            AlternativeDataSnapshot with all metrics
        """
        logger.info(f"Collecting alternative data snapshot for {symbol}")

        snapshot = AlternativeDataSnapshot(
            symbol=symbol,
            timestamp=datetime.utcnow(),
        )

        # 1. Social sentiment (Twitter + Reddit)
        try:
            if twitter_texts or reddit_texts:
                sentiment_data = self.sentiment_engine.get_combined_sentiment(
                    symbol,
                    twitter_texts or [],
                    reddit_texts or [],
                )
                snapshot.twitter_sentiment = sentiment_data["twitter"].sentiment
                snapshot.reddit_sentiment = sentiment_data["reddit"].sentiment
                snapshot.combined_sentiment = sentiment_data["combined_sentiment"]
                snapshot.sentiment_volume = sentiment_data["total_volume"]
                logger.info(f"✓ Social sentiment collected for {symbol}")
        except Exception as e:
            logger.error(f"✗ Social sentiment error for {symbol}: {e}")

        # 2. On-chain metrics
        try:
            onchain_metrics = self.onchain_collector.get_metrics(symbol, use_cache)
            snapshot.exchange_netflow = onchain_metrics.exchange_netflow
            snapshot.mvrv_ratio = onchain_metrics.mvrv_ratio
            snapshot.active_addresses = onchain_metrics.active_addresses
            logger.info(f"✓ On-chain metrics collected for {symbol}")
        except Exception as e:
            logger.error(f"✗ On-chain metrics error for {symbol}: {e}")

        # 3. Funding rates
        try:
            funding_analysis = self.funding_aggregator.analyze_funding_signal(
                symbol, use_cache
            )
            snapshot.avg_funding_rate = funding_analysis["avg_funding_rate"]
            snapshot.funding_spread = funding_analysis["spread_analysis"]["spread"]
            snapshot.funding_signal = funding_analysis["signal"]
            logger.info(f"✓ Funding rates collected for {symbol}")
        except Exception as e:
            logger.error(f"✗ Funding rates error for {symbol}: {e}")

        # 4. News sentiment
        try:
            news_data = self.news_sentiment.get_news_sentiment(symbol, use_cache=use_cache)
            snapshot.news_sentiment = news_data["sentiment"]
            snapshot.news_article_count = news_data["article_count"]
            logger.info(f"✓ News sentiment collected for {symbol}")
        except Exception as e:
            logger.error(f"✗ News sentiment error for {symbol}: {e}")

        # 5. Fear & Greed Index
        try:
            fg_signal = self.fear_greed.get_trading_signal()
            snapshot.fear_greed_value = fg_signal["value"]
            snapshot.fear_greed_classification = fg_signal["classification"]
            snapshot.fear_greed_signal = fg_signal["signal"]
            logger.info(f"✓ Fear & Greed Index collected")
        except Exception as e:
            logger.error(f"✗ Fear & Greed Index error: {e}")

        # Composite scores are auto-calculated in __post_init__
        logger.info(
            f"Alternative data snapshot complete for {symbol}: "
            f"overall_sentiment={snapshot.overall_sentiment:.3f}, "
            f"confidence={snapshot.confidence_score:.3f}"
        )

        return snapshot

    def get_features_dict(
        self,
        symbol: str,
        twitter_texts: Optional[List[str]] = None,
        reddit_texts: Optional[List[str]] = None,
        use_cache: bool = True,
    ) -> Dict[str, float]:
        """Get alternative data as feature dictionary for ML models.

        Args:
            symbol: Trading symbol
            twitter_texts: Optional list of tweets
            reddit_texts: Optional list of reddit posts
            use_cache: Whether to use cached data

        Returns:
            Dict of features ready for ML pipeline
        """
        snapshot = self.get_complete_snapshot(
            symbol, twitter_texts, reddit_texts, use_cache
        )

        features = {
            # Sentiment features
            "alt_twitter_sentiment": snapshot.twitter_sentiment,
            "alt_reddit_sentiment": snapshot.reddit_sentiment,
            "alt_combined_sentiment": snapshot.combined_sentiment,
            "alt_sentiment_volume": float(snapshot.sentiment_volume),
            
            # On-chain features
            "alt_exchange_netflow": snapshot.exchange_netflow,
            "alt_mvrv_ratio": snapshot.mvrv_ratio,
            "alt_active_addresses": float(snapshot.active_addresses),
            
            # Funding features
            "alt_avg_funding_rate": snapshot.avg_funding_rate,
            "alt_funding_spread": snapshot.funding_spread,
            
            # News features
            "alt_news_sentiment": snapshot.news_sentiment,
            "alt_news_volume": float(snapshot.news_article_count),
            
            # Fear & Greed
            "alt_fear_greed_value": float(snapshot.fear_greed_value),
            
            # Composite
            "alt_overall_sentiment": snapshot.overall_sentiment,
            "alt_confidence": snapshot.confidence_score,
        }

        logger.info(f"Generated {len(features)} alternative data features for {symbol}")
        return features

    def health_check(self) -> Dict[str, Dict[str, bool]]:
        """Check health of all data sources.

        Returns:
            Dict with health status of each component
        """
        logger.info("Running health check on all alternative data sources")

        health = {
            "sentiment_engine": self.sentiment_engine.health_check(),
            "onchain_collector": self.onchain_collector.health_check(),
            "funding_aggregator": self.funding_aggregator.health_check(),
            "news_sentiment": self.news_sentiment.health_check(),
            "fear_greed": self.fear_greed.health_check(),
        }

        # Overall health
        all_healthy = all(
            all(status.values()) if isinstance(status, dict) else status
            for component in health.values()
            for status in [component]
        )

        health["overall_healthy"] = all_healthy

        logger.info(f"Health check complete: {'✓ All systems operational' if all_healthy else '✗ Some systems down'}")

        return health

    def get_summary_report(
        self,
        symbol: str,
        twitter_texts: Optional[List[str]] = None,
        reddit_texts: Optional[List[str]] = None,
    ) -> str:
        """Generate human-readable summary report.

        Args:
            symbol: Trading symbol
            twitter_texts: Optional tweets
            reddit_texts: Optional reddit posts

        Returns:
            Formatted summary string
        """
        snapshot = self.get_complete_snapshot(symbol, twitter_texts, reddit_texts)

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║  ALTERNATIVE DATA REPORT - {symbol}
║  {snapshot.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
╚══════════════════════════════════════════════════════════════╝

📊 OVERALL SENTIMENT: {snapshot.overall_sentiment:+.3f}
   Confidence: {snapshot.confidence_score:.1%}

🐦 SOCIAL SENTIMENT
   Twitter:  {snapshot.twitter_sentiment:+.3f}
   Reddit:   {snapshot.reddit_sentiment:+.3f}
   Combined: {snapshot.combined_sentiment:+.3f} ({snapshot.sentiment_volume} mentions)

⛓️  ON-CHAIN METRICS
   Exchange Netflow: {snapshot.exchange_netflow:+,.2f} {symbol}
   MVRV Ratio:       {snapshot.mvrv_ratio:.2f}
   Active Addresses: {snapshot.active_addresses:,}

💰 FUNDING RATES
   Average:  {snapshot.avg_funding_rate:.6f}
   Spread:   {snapshot.funding_spread:.6f}
   Signal:   {snapshot.funding_signal}

📰 NEWS SENTIMENT
   Sentiment: {snapshot.news_sentiment:+.3f}
   Articles:  {snapshot.news_article_count}

😱 FEAR & GREED INDEX
   Value:          {snapshot.fear_greed_value}/100
   Classification: {snapshot.fear_greed_classification}
   Signal:         {snapshot.fear_greed_signal}

═══════════════════════════════════════════════════════════════
"""
        return report

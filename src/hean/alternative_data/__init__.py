"""Alternative data sources for trading edge.

This module provides access to:
- Sentiment analysis (Twitter, Reddit)
- On-chain metrics
- Funding rates aggregation
- News sentiment
- Fear & Greed Index
"""

from hean.alternative_data.alternative_data_pipeline import AlternativeDataPipeline
from hean.alternative_data.fear_greed_index import FearGreedIndexCollector
from hean.alternative_data.funding_rates_aggregator import FundingRatesAggregator
from hean.alternative_data.news_sentiment import Newssentiment
from hean.alternative_data.onchain_collector import OnChainDataCollector
from hean.alternative_data.sentiment_engine import SentimentEngine

__all__ = [
    "SentimentEngine",
    "OnChainDataCollector",
    "FundingRatesAggregator",
    "Newssentiment",
    "FearGreedIndexCollector",
    "AlternativeDataPipeline",
]

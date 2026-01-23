"""
Alternative Data module for HEAN trading system.

Provides:
- Sentiment Analysis (Twitter, Reddit, News)
- On-Chain Metrics (Exchange flows, MVRV)
- Social signals
- Fear & Greed Index
"""

from hean.alternative_data.sentiment_engine import (
    SentimentEngine,
    SentimentConfig,
    SentimentSignal,
    SentimentScore,
)

__all__ = [
    "SentimentEngine",
    "SentimentConfig",
    "SentimentSignal",
    "SentimentScore",
]

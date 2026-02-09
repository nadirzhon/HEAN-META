"""
HEAN Sentiment Analysis Module

Анализирует настроения из:
- Twitter
- Reddit
- News sources
- Telegram (будущее)

Использует FinBERT для анализа финансовых текстов
"""

from .aggregator import SentimentAggregator
from .analyzer import SentimentAnalyzer
from .models import SentimentScore, SentimentSignal
from .news_client import NewsSentiment
from .reddit_client import RedditSentiment
from .twitter_client import TwitterSentiment

__all__ = [
    "SentimentAnalyzer",
    "TwitterSentiment",
    "RedditSentiment",
    "NewsSentiment",
    "SentimentAggregator",
    "SentimentSignal",
    "SentimentScore",
]

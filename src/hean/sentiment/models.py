"""
Data models for sentiment analysis
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SentimentLabel(str, Enum):
    """Sentiment labels"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class SentimentSource(str, Enum):
    """Data sources"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    NEWS = "news"
    TELEGRAM = "telegram"


@dataclass
class SentimentScore:
    """
    Sentiment score from a single source

    Attributes:
        label: bullish/bearish/neutral
        score: confidence score (-1 to +1)
        volume: number of mentions
        source: data source
    """
    label: SentimentLabel
    score: float  # -1 (bearish) to +1 (bullish)
    volume: int
    source: SentimentSource
    timestamp: datetime

    def __post_init__(self):
        """Validate score range"""
        if not -1 <= self.score <= 1:
            raise ValueError(f"Score must be between -1 and 1, got {self.score}")


@dataclass
class SentimentSignal:
    """
    Aggregated sentiment signal for trading

    Attributes:
        symbol: trading symbol (e.g., "BTCUSDT")
        overall_score: aggregated score (-1 to +1)
        confidence: confidence in signal (0 to 1)
        action: recommended action
        sources: breakdown by source
    """
    symbol: str
    overall_score: float
    confidence: float
    action: str  # "BUY", "SELL", "HOLD"
    sources: dict[SentimentSource, SentimentScore]
    timestamp: datetime
    reason: str

    @property
    def is_strong_bullish(self) -> bool:
        """Check if strongly bullish"""
        return self.overall_score > 0.6 and self.confidence > 0.75

    @property
    def is_strong_bearish(self) -> bool:
        """Check if strongly bearish"""
        return self.overall_score < -0.6 and self.confidence > 0.75

    @property
    def should_trade(self) -> bool:
        """Should we trade on this signal?"""
        return self.confidence > 0.7 and abs(self.overall_score) > 0.5


@dataclass
class TextSentiment:
    """
    Sentiment from a single text (tweet, post, article)
    """
    text: str
    label: SentimentLabel
    score: float
    confidence: float
    source: SentimentSource
    timestamp: datetime
    author: str | None = None
    url: str | None = None
    engagement: int | None = None  # likes, retweets, etc.

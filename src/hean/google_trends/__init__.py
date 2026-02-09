"""
Google Trends Integration for Crypto Trading

Analyzes search interest as a leading indicator for price movements.

Research shows Google Trends data can predict crypto price movements:
- Increased search volume → Price increase (24-48h lead time)
- Sudden spikes → Volatility incoming
- Declining interest → Bearish signal

Usage:
    from hean.google_trends import GoogleTrendsAnalyzer

    analyzer = GoogleTrendsAnalyzer()
    await analyzer.initialize()

    signal = await analyzer.get_signal("BTC")
    if signal.should_trade:
        print(f"Action: {signal.action}")
        print(f"Search interest: {signal.interest_score}")
"""

from .analyzer import GoogleTrendsAnalyzer
from .client import GoogleTrendsClient
from .models import InterestLevel, TrendDirection, TrendsData, TrendsSignal
from .strategy import GoogleTrendsStrategy

__all__ = [
    "TrendsData",
    "TrendsSignal",
    "TrendDirection",
    "InterestLevel",
    "GoogleTrendsClient",
    "GoogleTrendsAnalyzer",
    "GoogleTrendsStrategy",
]

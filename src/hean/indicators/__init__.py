"""Technical indicators module"""

try:
    from .fast_indicators import FastIndicators, ema, macd, rsi
    __all__ = ["FastIndicators", "rsi", "macd", "ema"]
except ImportError:
    # C++ modules not built yet
    __all__ = []

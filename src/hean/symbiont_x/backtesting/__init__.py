"""
Backtesting module for SYMBIONT X

Provides framework for testing strategies on historical data
"""

from .backtest_engine import BacktestConfig, BacktestEngine, BacktestResult

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'BacktestConfig'
]

"""
Backtesting module for HEAN trading system.

Provides:
- Vectorized backtesting (vectorbt)
- Walk-forward analysis
- Parameter optimization
- Performance metrics
"""

from hean.backtesting.vectorbt_engine import (
    VectorBTBacktester,
    BacktestConfig,
    BacktestResult,
)

__all__ = [
    "VectorBTBacktester",
    "BacktestConfig",
    "BacktestResult",
]

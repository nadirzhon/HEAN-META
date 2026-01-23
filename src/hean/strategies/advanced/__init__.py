"""
Advanced trading strategies for HEAN system.

Provides:
- Statistical Arbitrage (pairs trading)
- Lead-Lag Arbitrage
- Volatility Trading
- Model Stacking
"""

from hean.strategies.advanced.stat_arb import (
    StatisticalArbitrage,
    PairConfig,
    ArbSignal,
)

__all__ = [
    "StatisticalArbitrage",
    "PairConfig",
    "ArbSignal",
]

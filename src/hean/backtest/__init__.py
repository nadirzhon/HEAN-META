"""Backtesting module -- event-driven simulation and performance metrics."""

from .event_sim import EventSimulator
from .metrics import BacktestMetrics

__all__ = [
    "BacktestMetrics",
    "EventSimulator",
]

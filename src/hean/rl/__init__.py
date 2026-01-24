"""
Reinforcement Learning module for Bitcoin trading.

Provides PPO-based RL agent for learning to trade Bitcoin.
"""

from .trading_environment import BitcoinTradingEnv, TradingConfig, TradingState, Action

__all__ = [
    'BitcoinTradingEnv',
    'TradingConfig',
    'TradingState',
    'Action',
]

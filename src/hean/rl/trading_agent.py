"""
Reinforcement Learning Trading Agent

PPO (Proximal Policy Optimization) agent that learns to trade through simulations.

The agent:
- Observes: price, volume, indicators, position, PnL
- Actions: BUY, SELL, HOLD, position size
- Reward: profit - fees - drawdown penalty
- Learns: optimal entry/exit/sizing through millions of episodes

Expected Performance:
- Discovers non-obvious patterns
- Adaptive to market regimes
- Sharpe: +0.5-1.0 (vs manual strategies)
- Win Rate: +5-10%

Author: HEAN Team
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger

try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    logger.warning("Gymnasium not installed. Install with: pip install gymnasium")

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.warning("Stable-Baselines3 not installed. Install with: pip install stable-baselines3")


class Action(IntEnum):
    """Trading actions."""
    HOLD = 0
    BUY_SMALL = 1  # 25% position
    BUY_MEDIUM = 2  # 50% position
    BUY_LARGE = 3  # 100% position
    SELL_SMALL = 4
    SELL_MEDIUM = 5
    SELL_LARGE = 6
    CLOSE = 7  # Close position


@dataclass
class RLConfig:
    """Configuration for RL trading agent."""

    # Environment
    initial_balance: float = 10000.0
    commission: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%

    # Reward function
    reward_scaling: float = 100.0  # Scale rewards for training
    drawdown_penalty: float = 2.0  # Penalty multiplier for drawdown
    holding_penalty: float = 0.0001  # Small penalty for holding (encourage action)

    # PPO parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01  # Entropy coefficient (exploration)

    # Training
    total_timesteps: int = 1_000_000
    eval_freq: int = 10_000
    save_freq: int = 50_000

    # Model
    policy: str = "MlpPolicy"
    model_dir: str = "models/rl"


class TradingEnv(gym.Env):
    """
    Custom Gymnasium environment for crypto trading.

    Observation space:
    - Price features (normalized)
    - Technical indicators
    - Position state
    - Account state

    Action space:
    - Discrete: HOLD, BUY (small/medium/large), SELL, CLOSE

    Reward:
    - PnL - commission - drawdown penalty
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[RLConfig] = None,
        features: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize trading environment.

        Args:
            df: OHLCV DataFrame with optional technical indicators
            config: RL configuration
            features: List of feature columns to use
        """
        super().__init__()

        if not GYMNASIUM_AVAILABLE:
            raise ImportError("Gymnasium required")

        self.config = config or RLConfig()
        self.df = df.reset_index(drop=True)

        # Features
        if features is None:
            # Use all numeric columns except OHLCV
            self.features = [
                col for col in df.columns
                if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']
                and pd.api.types.is_numeric_dtype(df[col])
            ]
        else:
            self.features = features

        # Always include price
        if 'close' not in self.features:
            self.features = ['close'] + self.features

        # State tracking
        self.current_step = 0
        self.balance = self.config.initial_balance
        self.position = 0.0  # Position size (BTC, contracts, etc.)
        self.entry_price = 0.0
        self.max_balance = self.balance
        self.total_trades = 0

        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(Action))

        # Observation: features + account state
        n_features = len(self.features)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_features + 4,),  # features + position + balance + pnl + max_dd
            dtype=np.float32,
        )

        logger.info(
            f"TradingEnv initialized: {len(df)} steps, {n_features} features"
        )

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.current_step = 0
        self.balance = self.config.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.max_balance = self.balance
        self.total_trades = 0

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute action and return next state.

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get current price
        current_price = self.df.loc[self.current_step, 'close']

        # Execute action
        reward = self._execute_action(action, current_price)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= len(self.df) - 1
        truncated = self.balance <= 0  # Bankruptcy

        # Get next observation
        obs = self._get_observation()

        # Info
        info = {
            'balance': self.balance,
            'position': self.position,
            'total_trades': self.total_trades,
        }

        return obs, reward, terminated, truncated, info

    def _execute_action(self, action: int, price: float) -> float:
        """Execute trading action and return reward."""
        action_enum = Action(action)
        reward = 0.0

        # Close position
        if action_enum == Action.CLOSE and self.position != 0:
            pnl = self._close_position(price)
            reward = pnl * self.config.reward_scaling
            self.total_trades += 1

        # Buy actions
        elif action_enum in [Action.BUY_SMALL, Action.BUY_MEDIUM, Action.BUY_LARGE]:
            if self.position <= 0:  # Only buy if not long
                size_pct = {
                    Action.BUY_SMALL: 0.25,
                    Action.BUY_MEDIUM: 0.50,
                    Action.BUY_LARGE: 1.00,
                }[action_enum]

                self._open_position(price, size_pct)
                # Small negative reward for opening (encourage profitable closes)
                reward = -self.config.holding_penalty

        # Sell actions
        elif action_enum in [Action.SELL_SMALL, Action.SELL_MEDIUM, Action.SELL_LARGE]:
            if self.position >= 0:  # Only sell if not short
                size_pct = {
                    Action.SELL_SMALL: 0.25,
                    Action.SELL_MEDIUM: 0.50,
                    Action.SELL_LARGE: 1.00,
                }[action_enum]

                self._open_position(price, -size_pct)  # Negative for short
                reward = -self.config.holding_penalty

        # HOLD
        else:
            # Small penalty for holding (encourage action)
            reward = -self.config.holding_penalty

        # Add unrealized PnL to reward (if position open)
        if self.position != 0:
            unrealized_pnl = self._calculate_unrealized_pnl(price)
            reward += unrealized_pnl * self.config.reward_scaling * 0.1

        # Drawdown penalty
        drawdown = (self.max_balance - self.balance) / self.max_balance
        if drawdown > 0:
            reward -= drawdown * self.config.drawdown_penalty * self.config.reward_scaling

        return reward

    def _open_position(self, price: float, size_pct: float) -> None:
        """Open a position."""
        # Calculate position size
        position_value = self.balance * abs(size_pct)
        commission = position_value * self.config.commission
        slippage = position_value * self.config.slippage

        # Update balance
        self.balance -= (commission + slippage)

        # Update position
        self.position = (position_value / price) * np.sign(size_pct)
        self.entry_price = price

    def _close_position(self, price: float) -> float:
        """Close position and return PnL."""
        if self.position == 0:
            return 0.0

        # Calculate PnL
        position_value = abs(self.position * price)
        entry_value = abs(self.position * self.entry_price)

        if self.position > 0:  # Long
            pnl_value = position_value - entry_value
        else:  # Short
            pnl_value = entry_value - position_value

        # Commission and slippage
        commission = position_value * self.config.commission
        slippage = position_value * self.config.slippage

        total_pnl = pnl_value - commission - slippage

        # Update balance
        self.balance += position_value + total_pnl
        self.max_balance = max(self.max_balance, self.balance)

        # Reset position
        self.position = 0.0
        self.entry_price = 0.0

        # Return PnL as fraction of initial balance
        return total_pnl / self.config.initial_balance

    def _calculate_unrealized_pnl(self, price: float) -> float:
        """Calculate unrealized PnL."""
        if self.position == 0:
            return 0.0

        position_value = abs(self.position * price)
        entry_value = abs(self.position * self.entry_price)

        if self.position > 0:
            pnl_value = position_value - entry_value
        else:
            pnl_value = entry_value - position_value

        return pnl_value / self.config.initial_balance

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        if self.current_step >= len(self.df):
            self.current_step = len(self.df) - 1

        # Features
        feature_values = self.df.loc[self.current_step, self.features].values

        # Normalize features (simple z-score)
        feature_values = (feature_values - np.mean(feature_values)) / (np.std(feature_values) + 1e-8)

        # Account state
        position_norm = self.position / 10.0  # Normalize position
        balance_norm = self.balance / self.config.initial_balance - 1.0
        pnl_norm = (self.balance - self.config.initial_balance) / self.config.initial_balance
        max_dd_norm = (self.max_balance - self.balance) / self.max_balance

        # Combine
        obs = np.concatenate([
            feature_values,
            [position_norm, balance_norm, pnl_norm, max_dd_norm]
        ]).astype(np.float32)

        return obs

    def render(self, mode: str = 'human') -> None:
        """Render environment (optional)."""
        pass


class TradingAgent:
    """
    RL Trading Agent using PPO.

    Usage:
        # Train
        config = RLConfig(total_timesteps=1_000_000)
        agent = TradingAgent(config)
        agent.train(train_df)

        # Predict
        action = agent.predict(observation)

        # Backtest
        results = agent.backtest(test_df)
    """

    def __init__(self, config: Optional[RLConfig] = None) -> None:
        """Initialize RL trading agent."""
        if not SB3_AVAILABLE or not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "Stable-Baselines3 and Gymnasium required. "
                "Install with: pip install stable-baselines3 gymnasium"
            )

        self.config = config or RLConfig()
        self.model: Optional[PPO] = None
        self.env: Optional[TradingEnv] = None

        Path(self.config.model_dir).mkdir(parents=True, exist_ok=True)

        logger.info("TradingAgent initialized", config=self.config)

    def train(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
        eval_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Train the RL agent.

        Args:
            df: Training data (OHLCV + features)
            features: Feature columns to use
            eval_df: Evaluation data (optional)
        """
        logger.info(f"Training RL agent on {len(df)} steps...")

        # Create environment
        self.env = TradingEnv(df, self.config, features)

        # Wrap in vectorized environment
        vec_env = DummyVecEnv([lambda: self.env])

        # Create PPO model
        self.model = PPO(
            self.config.policy,
            vec_env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.n_steps,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            ent_coef=self.config.ent_coef,
            verbose=1,
            tensorboard_log=f"{self.config.model_dir}/tensorboard/",
        )

        # Train
        self.model.learn(
            total_timesteps=self.config.total_timesteps,
            progress_bar=True,
        )

        logger.info("Training complete!")

        # Save model
        model_path = f"{self.config.model_dir}/ppo_trading_agent.zip"
        self.model.save(model_path)
        logger.info(f"Model saved to {model_path}")

    def predict(self, observation: np.ndarray) -> int:
        """Predict action from observation."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        action, _states = self.model.predict(observation, deterministic=True)
        return int(action)

    def backtest(
        self,
        df: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Backtest trained agent.

        Returns:
            Backtest results
        """
        if self.model is None:
            raise ValueError("Model not trained")

        logger.info(f"Backtesting on {len(df)} steps...")

        # Create environment
        env = TradingEnv(df, self.config, features)

        # Run episode
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = self.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        # Results
        final_balance = info['balance']
        total_return = (final_balance - self.config.initial_balance) / self.config.initial_balance

        results = {
            'final_balance': final_balance,
            'total_return': total_return,
            'total_reward': total_reward,
            'total_trades': info['total_trades'],
        }

        logger.info(f"Backtest complete: {total_return:.1%} return, {info['total_trades']} trades")

        return results

    def save(self, path: str) -> None:
        """Save trained model."""
        if self.model is None:
            raise ValueError("No model to save")

        self.model.save(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, config: Optional[RLConfig] = None) -> TradingAgent:
        """Load trained model."""
        agent = cls(config)
        agent.model = PPO.load(path)
        logger.info(f"Model loaded from {path}")
        return agent

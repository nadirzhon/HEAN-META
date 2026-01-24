"""
Bitcoin Trading Environment for Reinforcement Learning

Gymnasium-compatible environment for training RL agents to trade Bitcoin.
Implements PPO-compatible action/observation spaces with realistic trading simulation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger(__name__)


class Action(IntEnum):
    """Trading actions available to the agent."""
    HOLD = 0
    BUY_SMALL = 1
    BUY_MEDIUM = 2
    BUY_LARGE = 3
    SELL_SMALL = 4
    SELL_MEDIUM = 5
    SELL_LARGE = 6


@dataclass
class TradingConfig:
    """Configuration for trading environment."""

    # Capital settings
    initial_capital: float = 10000.0
    max_position_size: float = 1.0  # BTC

    # Action sizes (as % of available capital/position)
    small_size: float = 0.25
    medium_size: float = 0.50
    large_size: float = 1.0

    # Fees and costs
    maker_fee: float = 0.0002  # 0.02%
    taker_fee: float = 0.0006  # 0.06%
    slippage: float = 0.0005   # 0.05%

    # Risk parameters
    max_drawdown_pct: float = 0.20  # 20%
    drawdown_penalty_scale: float = 10.0

    # Episode settings
    max_steps: int = 1000

    # Reward shaping
    profit_scale: float = 1.0
    fee_penalty_scale: float = 1.0
    hold_penalty: float = 0.0001  # Small penalty to encourage action


@dataclass
class TradingState:
    """Current state of the trading simulation."""

    # Position and capital
    position: float = 0.0  # BTC held
    cash: float = 10000.0

    # Performance tracking
    total_pnl: float = 0.0
    total_fees: float = 0.0
    total_trades: int = 0

    # Drawdown tracking
    peak_equity: float = 10000.0
    max_drawdown: float = 0.0

    # Episode tracking
    step: int = 0

    @property
    def equity(self) -> float:
        """Total equity including unrealized PnL."""
        return self.cash + self.position  # Will be multiplied by current price

    def update_peak_and_drawdown(self, equity: float) -> None:
        """Update peak equity and max drawdown."""
        if equity > self.peak_equity:
            self.peak_equity = equity

        if self.peak_equity > 0:
            drawdown = (self.peak_equity - equity) / self.peak_equity
            self.max_drawdown = max(self.max_drawdown, drawdown)


class BitcoinTradingEnv(gym.Env):
    """
    Gymnasium environment for Bitcoin trading using PPO.

    Observation Space (25 features):
        - Price features (5): current, returns, volatility, high/low
        - Volume features (3): current, MA, volume change
        - Technical indicators (8): RSI, MACD, BB, ATR, etc.
        - Position features (4): position size, avg entry, unrealized PnL, position duration
        - Portfolio features (5): cash, equity, drawdown, total PnL, win rate

    Action Space (7 discrete actions):
        - HOLD, BUY_SMALL, BUY_MEDIUM, BUY_LARGE, SELL_SMALL, SELL_MEDIUM, SELL_LARGE

    Reward:
        reward = profit - fees - drawdown_penalty - hold_penalty
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data: np.ndarray,
        config: Optional[TradingConfig] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize trading environment.

        Args:
            data: Historical OHLCV data [N x 5] (open, high, low, close, volume)
            config: Trading configuration
            render_mode: Rendering mode (currently only 'human')
        """
        super().__init__()

        self.config = config or TradingConfig()
        self.render_mode = render_mode

        # Data validation
        if data.shape[1] != 5:
            raise ValueError(f"Data must have 5 columns (OHLCV), got {data.shape[1]}")

        self.data = data
        self.n_steps = len(data)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(7)

        # Observation space: 25 features (normalized)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(25,),
            dtype=np.float32
        )

        # Initialize state
        self.state = TradingState(cash=self.config.initial_capital)
        self.current_step = 0

        # Precompute features for efficiency
        self._precompute_features()

        # Episode statistics
        self.episode_trades: list[Dict[str, Any]] = []

    def _precompute_features(self) -> None:
        """Precompute technical indicators for all timesteps."""
        close = self.data[:, 3]
        high = self.data[:, 1]
        low = self.data[:, 2]
        volume = self.data[:, 4]

        # Returns
        self.returns = np.zeros(len(close))
        self.returns[1:] = np.diff(close) / close[:-1]

        # Volatility (rolling std of returns, 20 periods)
        self.volatility = self._rolling_std(self.returns, window=20)

        # Volume features
        self.volume_ma = self._moving_average(volume, window=20)
        self.volume_change = np.zeros(len(volume))
        self.volume_change[1:] = np.diff(volume) / (volume[:-1] + 1e-8)

        # RSI (14 periods)
        self.rsi = self._compute_rsi(close, period=14)

        # MACD
        self.macd, self.macd_signal = self._compute_macd(close)

        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self._compute_bollinger_bands(close)

        # ATR (14 periods)
        self.atr = self._compute_atr(high, low, close, period=14)

        # Moving averages
        self.sma_20 = self._moving_average(close, window=20)
        self.sma_50 = self._moving_average(close, window=50)
        self.ema_12 = self._exponential_moving_average(close, span=12)

        # Position tracking
        self.position_duration = np.zeros(len(close))

    @staticmethod
    def _moving_average(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate simple moving average."""
        result = np.zeros(len(data))
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.mean(data[start:i+1])
        return result

    @staticmethod
    def _exponential_moving_average(data: np.ndarray, span: int) -> np.ndarray:
        """Calculate exponential moving average."""
        alpha = 2.0 / (span + 1)
        result = np.zeros(len(data))
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result

    @staticmethod
    def _rolling_std(data: np.ndarray, window: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        result = np.zeros(len(data))
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result[i] = np.std(data[start:i+1]) if i >= window - 1 else 0.0
        return result

    @staticmethod
    def _compute_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index."""
        deltas = np.diff(prices)
        seed = deltas[:period]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period

        rs = up / (down + 1e-8)
        rsi = np.zeros(len(prices))
        rsi[:period] = 100.0 - 100.0 / (1.0 + rs)

        for i in range(period, len(deltas)):
            delta = deltas[i]
            if delta > 0:
                upval = delta
                downval = 0.0
            else:
                upval = 0.0
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period

            rs = up / (down + 1e-8)
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

        return rsi

    @staticmethod
    def _compute_macd(
        prices: np.ndarray,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MACD and signal line."""
        ema_fast = BitcoinTradingEnv._exponential_moving_average(prices, fast)
        ema_slow = BitcoinTradingEnv._exponential_moving_average(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = BitcoinTradingEnv._exponential_moving_average(macd, signal)
        return macd, macd_signal

    @staticmethod
    def _compute_bollinger_bands(
        prices: np.ndarray,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate Bollinger Bands."""
        middle = BitcoinTradingEnv._moving_average(prices, window)
        std = BitcoinTradingEnv._rolling_std(prices, window)
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower

    @staticmethod
    def _compute_atr(
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        period: int = 14
    ) -> np.ndarray:
        """Calculate Average True Range."""
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)

        return BitcoinTradingEnv._moving_average(tr, period)

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (25 features).

        Features:
            1-5: Price (current, return, volatility, high, low)
            6-8: Volume (current, MA, change)
            9-16: Technical indicators (RSI, MACD, MACD signal, BB upper/mid/low, ATR, SMA ratio)
            17-20: Position (size, avg entry price, unrealized PnL %, duration)
            21-25: Portfolio (cash %, equity, drawdown, total PnL %, win rate)
        """
        i = self.current_step

        # Current price
        current_price = self.data[i, 3]  # Close price

        # Price features (normalized)
        price_features = np.array([
            current_price / 10000.0,  # Normalized price
            self.returns[i],
            self.volatility[i],
            (self.data[i, 1] - current_price) / current_price,  # High relative
            (self.data[i, 2] - current_price) / current_price,  # Low relative
        ], dtype=np.float32)

        # Volume features (normalized)
        volume = self.data[i, 4]
        volume_features = np.array([
            volume / 1000.0,  # Normalized volume
            self.volume_ma[i] / 1000.0,
            self.volume_change[i],
        ], dtype=np.float32)

        # Technical indicators
        technical_features = np.array([
            self.rsi[i] / 100.0,  # Normalized RSI
            self.macd[i] / current_price,
            self.macd_signal[i] / current_price,
            (self.bb_upper[i] - current_price) / current_price,
            (self.bb_middle[i] - current_price) / current_price,
            (self.bb_lower[i] - current_price) / current_price,
            self.atr[i] / current_price,
            (current_price - self.sma_50[i]) / self.sma_50[i] if self.sma_50[i] > 0 else 0.0,
        ], dtype=np.float32)

        # Position features
        position_value = self.state.position * current_price
        avg_entry_price = position_value / self.state.position if self.state.position != 0 else current_price
        unrealized_pnl_pct = (current_price - avg_entry_price) / avg_entry_price if self.state.position != 0 else 0.0

        position_features = np.array([
            self.state.position / self.config.max_position_size,  # Normalized position
            avg_entry_price / 10000.0,
            unrealized_pnl_pct,
            self.position_duration[i] / 100.0,  # Normalized duration
        ], dtype=np.float32)

        # Portfolio features
        equity = self.state.cash + position_value
        win_rate = 0.0
        if self.state.total_trades > 0:
            winning_trades = sum(1 for t in self.episode_trades if t.get('pnl', 0) > 0)
            win_rate = winning_trades / self.state.total_trades

        portfolio_features = np.array([
            self.state.cash / equity if equity > 0 else 0.0,
            equity / self.config.initial_capital,
            self.state.max_drawdown,
            self.state.total_pnl / self.config.initial_capital,
            win_rate,
        ], dtype=np.float32)

        # Concatenate all features
        observation = np.concatenate([
            price_features,
            volume_features,
            technical_features,
            position_features,
            portfolio_features,
        ])

        # Replace NaN/Inf with 0
        observation = np.nan_to_num(observation, nan=0.0, posinf=1e6, neginf=-1e6)

        return observation.astype(np.float32)

    def _execute_action(self, action: int) -> Tuple[float, float]:
        """
        Execute trading action and return (pnl, fees).

        Args:
            action: Action to execute

        Returns:
            Tuple of (pnl, fees)
        """
        current_price = self.data[self.current_step, 3]

        if action == Action.HOLD:
            return 0.0, 0.0

        # Determine trade size and direction
        if action in [Action.BUY_SMALL, Action.BUY_MEDIUM, Action.BUY_LARGE]:
            direction = 1  # Buy
            if action == Action.BUY_SMALL:
                size_pct = self.config.small_size
            elif action == Action.BUY_MEDIUM:
                size_pct = self.config.medium_size
            else:
                size_pct = self.config.large_size

            # Calculate BTC amount we can buy
            available_cash = self.state.cash * size_pct
            effective_price = current_price * (1 + self.config.slippage)
            btc_amount = available_cash / effective_price

            # Check if we have enough cash
            cost = btc_amount * effective_price
            fee = cost * self.config.taker_fee
            total_cost = cost + fee

            if total_cost > self.state.cash:
                # Not enough cash
                return 0.0, 0.0

            # Execute buy
            self.state.position += btc_amount
            self.state.cash -= total_cost
            self.state.total_fees += fee
            self.state.total_trades += 1

            self.episode_trades.append({
                'step': self.current_step,
                'action': 'BUY',
                'size': btc_amount,
                'price': effective_price,
                'cost': total_cost,
                'fee': fee,
            })

            return 0.0, fee  # No immediate PnL from buying

        else:  # SELL actions
            direction = -1
            if action == Action.SELL_SMALL:
                size_pct = self.config.small_size
            elif action == Action.SELL_MEDIUM:
                size_pct = self.config.medium_size
            else:
                size_pct = self.config.large_size

            # Calculate BTC amount to sell
            btc_amount = self.state.position * size_pct

            if btc_amount <= 0 or self.state.position <= 0:
                # No position to sell
                return 0.0, 0.0

            # Execute sell
            effective_price = current_price * (1 - self.config.slippage)
            proceeds = btc_amount * effective_price
            fee = proceeds * self.config.taker_fee
            net_proceeds = proceeds - fee

            # Calculate PnL (simplified: assume FIFO)
            pnl = net_proceeds - btc_amount * (self.state.cash / max(self.state.position, 1e-8))

            self.state.position -= btc_amount
            self.state.cash += net_proceeds
            self.state.total_fees += fee
            self.state.total_pnl += pnl
            self.state.total_trades += 1

            self.episode_trades.append({
                'step': self.current_step,
                'action': 'SELL',
                'size': btc_amount,
                'price': effective_price,
                'proceeds': net_proceeds,
                'fee': fee,
                'pnl': pnl,
            })

            return pnl, fee

    def _calculate_reward(self, pnl: float, fees: float, action: int) -> float:
        """
        Calculate reward for the current step.

        Reward components:
            1. Profit/loss from trade
            2. Fee penalty
            3. Drawdown penalty (exponential)
            4. Small hold penalty (encourage action)
        """
        # Base reward from PnL
        reward = pnl * self.config.profit_scale

        # Fee penalty
        reward -= fees * self.config.fee_penalty_scale

        # Drawdown penalty (exponential to heavily penalize large drawdowns)
        if self.state.max_drawdown > 0:
            drawdown_penalty = (self.state.max_drawdown ** 2) * self.config.drawdown_penalty_scale
            reward -= drawdown_penalty

        # Small penalty for holding (encourage action)
        if action == Action.HOLD:
            reward -= self.config.hold_penalty

        return reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Execute action
        pnl, fees = self._execute_action(action)

        # Calculate current equity
        current_price = self.data[self.current_step, 3]
        equity = self.state.cash + self.state.position * current_price

        # Update drawdown tracking
        self.state.update_peak_and_drawdown(equity)

        # Calculate reward
        reward = self._calculate_reward(pnl, fees, action)

        # Move to next step
        self.current_step += 1
        self.state.step += 1

        # Check termination conditions
        terminated = False
        truncated = False

        # Episode ends if max steps reached
        if self.current_step >= self.n_steps - 1:
            truncated = True

        # Episode ends if max steps in config reached
        if self.state.step >= self.config.max_steps:
            truncated = True

        # Episode ends if drawdown exceeds limit
        if self.state.max_drawdown >= self.config.max_drawdown_pct:
            terminated = True
            reward -= 100.0  # Heavy penalty for blowing up

        # Episode ends if equity falls too low
        if equity < self.config.initial_capital * 0.1:
            terminated = True
            reward -= 100.0

        # Get next observation
        observation = self._get_observation()

        # Info dict
        info = {
            'step': self.state.step,
            'equity': equity,
            'position': self.state.position,
            'cash': self.state.cash,
            'total_pnl': self.state.total_pnl,
            'total_fees': self.state.total_fees,
            'max_drawdown': self.state.max_drawdown,
            'total_trades': self.state.total_trades,
            'action': Action(action).name,
            'pnl': pnl,
            'fees': fees,
        }

        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # Reset state
        self.state = TradingState(cash=self.config.initial_capital)

        # Random start position within data (leave room for episode)
        if options and 'start_step' in options:
            self.current_step = options['start_step']
        else:
            max_start = max(0, self.n_steps - self.config.max_steps - 1)
            self.current_step = self.np_random.integers(50, max_start) if max_start > 50 else 50

        self.episode_trades = []

        # Get initial observation
        observation = self._get_observation()

        info = {
            'start_step': self.current_step,
            'initial_capital': self.config.initial_capital,
        }

        return observation, info

    def render(self) -> None:
        """Render environment state."""
        if self.render_mode == "human":
            current_price = self.data[self.current_step, 3]
            equity = self.state.cash + self.state.position * current_price

            print(f"\n{'='*60}")
            print(f"Step: {self.state.step} | Price: ${current_price:.2f}")
            print(f"Position: {self.state.position:.6f} BTC | Cash: ${self.state.cash:.2f}")
            print(f"Equity: ${equity:.2f} | PnL: ${self.state.total_pnl:.2f}")
            print(f"Drawdown: {self.state.max_drawdown*100:.2f}% | Trades: {self.state.total_trades}")
            print(f"{'='*60}\n")

    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the completed episode."""
        current_price = self.data[self.current_step, 3]
        final_equity = self.state.cash + self.state.position * current_price

        winning_trades = sum(1 for t in self.episode_trades if t.get('pnl', 0) > 0)
        losing_trades = sum(1 for t in self.episode_trades if t.get('pnl', 0) < 0)

        total_profit = sum(t.get('pnl', 0) for t in self.episode_trades if t.get('pnl', 0) > 0)
        total_loss = sum(t.get('pnl', 0) for t in self.episode_trades if t.get('pnl', 0) < 0)

        return {
            'initial_capital': self.config.initial_capital,
            'final_equity': final_equity,
            'total_return': (final_equity - self.config.initial_capital) / self.config.initial_capital,
            'total_pnl': self.state.total_pnl,
            'total_fees': self.state.total_fees,
            'total_trades': self.state.total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': winning_trades / max(self.state.total_trades, 1),
            'max_drawdown': self.state.max_drawdown,
            'profit_factor': abs(total_profit / total_loss) if total_loss != 0 else float('inf'),
            'avg_trade_pnl': self.state.total_pnl / max(self.state.total_trades, 1),
        }

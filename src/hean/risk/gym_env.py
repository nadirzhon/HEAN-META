"""Gymnasium environment for RL-based risk management training.

This environment simulates trading scenarios for training a PPO agent to:
- Adjust leverage dynamically based on market conditions
- Optimize position sizing relative to risk
- Set optimal stop-loss and take-profit levels
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from hean.logging import get_logger

logger = get_logger(__name__)


class TradingRiskEnv(gym.Env):
    """Gym environment for training RL risk manager.

    Observation Space:
        - Current drawdown (%)
        - Win rate (last N trades)
        - Profit factor (last N trades)
        - Current volatility (normalized)
        - Market phase (one-hot: accumulation, markup, distribution, markdown)
        - Temperature (normalized 0-1)
        - Entropy (normalized 0-1)
        - Current leverage (1-10x)
        - Open position count
        - Current equity ratio (current / initial)

    Action Space:
        - Leverage adjustment: continuous [-1, 1] → maps to [1.0, 10.0]x
        - Position size multiplier: continuous [0.5, 2.0]
        - Stop loss distance (% from entry): continuous [0.5, 10.0]%

    Reward:
        - Primary: Sharpe ratio improvement
        - Penalties: Drawdown exceeding thresholds, consecutive losses
        - Bonuses: Recovery from drawdown, maintaining low variance
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_drawdown_pct: float = 20.0,
        lookback_trades: int = 30,
        max_steps: int = 1000,
    ):
        super().__init__()

        self.initial_capital = initial_capital
        self.max_drawdown_pct = max_drawdown_pct
        self.lookback_trades = lookback_trades
        self.max_steps = max_steps

        # Observation space: 15 features
        self.observation_space = spaces.Box(
            low=np.array([
                -100.0,  # drawdown %
                0.0,     # win rate
                0.0,     # profit factor
                0.0,     # volatility
                0.0, 0.0, 0.0, 0.0,  # phase one-hot (4)
                0.0,     # temperature
                0.0,     # entropy
                1.0,     # current leverage
                0.0,     # open positions
                0.0,     # equity ratio
                0.0,     # consecutive losses
                0.0,     # hours since last trade
            ], dtype=np.float32),
            high=np.array([
                0.0,     # drawdown %
                1.0,     # win rate
                10.0,    # profit factor
                1.0,     # volatility
                1.0, 1.0, 1.0, 1.0,  # phase one-hot
                1.0,     # temperature
                1.0,     # entropy
                10.0,    # current leverage
                10.0,    # open positions
                10.0,    # equity ratio
                10.0,    # consecutive losses
                48.0,    # hours since last trade
            ], dtype=np.float32),
            dtype=np.float32,
        )

        # Action space: 3 continuous values
        # [leverage_adj, size_mult, stop_loss_pct]
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.5, 0.5], dtype=np.float32),
            high=np.array([1.0, 2.0, 10.0], dtype=np.float32),
            dtype=np.float32,
        )

        # State tracking
        self.equity = initial_capital
        self.peak_equity = initial_capital
        self.trades: list[dict[str, float]] = []
        self.current_step = 0
        self.current_leverage = 3.0
        self.consecutive_losses = 0

        # Simulated market state (will be updated from real data during training)
        self.market_volatility = 0.02
        self.market_phase = 0  # 0=accumulation, 1=markup, 2=distribution, 3=markdown
        self.market_temperature = 0.5
        self.market_entropy = 0.5

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.equity = self.initial_capital
        self.peak_equity = self.initial_capital
        self.trades = []
        self.current_step = 0
        self.current_leverage = 3.0
        self.consecutive_losses = 0

        # Randomize initial market conditions
        if seed is not None:
            np.random.seed(seed)
        self.market_volatility = np.random.uniform(0.01, 0.05)
        self.market_phase = np.random.randint(0, 4)
        self.market_temperature = np.random.uniform(0.2, 0.8)
        self.market_entropy = np.random.uniform(0.2, 0.8)

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1

        # Parse action
        leverage_adj, size_mult, stop_loss_pct = action

        # Map leverage adjustment from [-1, 1] to [1.0, 10.0]
        new_leverage = 1.0 + (leverage_adj + 1.0) * 4.5  # Range: 1.0 to 10.0
        new_leverage = np.clip(new_leverage, 1.0, 10.0)
        self.current_leverage = new_leverage

        # Simulate a trade outcome based on current market conditions
        # This is a simplified simulation; in real training, we'd use historical data
        trade_outcome = self._simulate_trade(size_mult, stop_loss_pct)

        # Update equity
        pnl = trade_outcome["pnl"]
        self.equity += pnl

        # Track peak for drawdown calculation
        if self.equity > self.peak_equity:
            self.peak_equity = self.equity

        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # Store trade
        self.trades.append(trade_outcome)
        if len(self.trades) > self.lookback_trades:
            self.trades.pop(0)

        # Calculate reward
        reward = self._calculate_reward(trade_outcome)

        # Check termination conditions
        drawdown_pct = ((self.peak_equity - self.equity) / self.peak_equity) * 100
        terminated = (
            drawdown_pct >= self.max_drawdown_pct
            or self.equity <= self.initial_capital * 0.5  # 50% capital loss
            or self.current_step >= self.max_steps
        )

        # Update market state randomly (simulate regime changes)
        self._update_market_state()

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, False, info

    def _simulate_trade(
        self, size_mult: float, stop_loss_pct: float
    ) -> dict[str, float]:
        """Simulate a trade outcome based on current market conditions.

        Factors:
        - Higher leverage in volatile markets → higher risk
        - Tighter stops in choppy markets → more stop-outs
        - Phase alignment → better outcomes
        """
        # Base win probability depends on market phase
        phase_win_probs = [0.45, 0.65, 0.35, 0.55]  # acc, markup, dist, markdown
        base_win_prob = phase_win_probs[self.market_phase]

        # Adjust for volatility vs leverage mismatch
        leverage_risk = self.current_leverage * self.market_volatility
        if leverage_risk > 0.20:  # High risk zone
            base_win_prob -= 0.15

        # Adjust for stop loss placement
        if stop_loss_pct < self.market_volatility * 100:  # Stop too tight
            base_win_prob -= 0.10

        # Determine win/loss
        is_win = np.random.random() < base_win_prob

        # Calculate PnL
        position_size = (self.equity * size_mult * 0.01)  # 1% base risk

        if is_win:
            # Win: 1.5-3.0x risk-reward
            rr_ratio = np.random.uniform(1.5, 3.0)
            pnl = position_size * rr_ratio
        else:
            # Loss: capped at stop loss
            pnl = -position_size * (stop_loss_pct / 100.0) * self.current_leverage

        return {
            "pnl": pnl,
            "is_win": is_win,
            "leverage": self.current_leverage,
            "size_mult": size_mult,
            "stop_loss_pct": stop_loss_pct,
        }

    def _calculate_reward(self, trade_outcome: dict[str, float]) -> float:
        """Calculate reward for this step.

        Components:
        1. Normalized PnL (primary signal)
        2. Sharpe ratio improvement
        3. Drawdown penalty
        4. Risk-adjusted return bonus
        """
        pnl = trade_outcome["pnl"]

        # Component 1: Normalized PnL
        pnl_reward = pnl / self.initial_capital * 100  # As percentage

        # Component 2: Sharpe ratio (if enough trades)
        sharpe_bonus = 0.0
        if len(self.trades) >= 10:
            returns = [t["pnl"] / self.initial_capital for t in self.trades]
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            if std_return > 0:
                sharpe = mean_return / std_return
                sharpe_bonus = sharpe * 0.5  # Weighted bonus

        # Component 3: Drawdown penalty
        drawdown_pct = ((self.peak_equity - self.equity) / self.peak_equity) * 100
        drawdown_penalty = 0.0
        if drawdown_pct > 10:
            drawdown_penalty = -(drawdown_pct - 10) * 0.2  # Escalating penalty

        # Component 4: Consecutive loss penalty
        loss_penalty = 0.0
        if self.consecutive_losses >= 3:
            loss_penalty = -(self.consecutive_losses - 2) * 0.5

        # Component 5: Risk-adjustment bonus
        # Reward conservative leverage in high volatility
        risk_bonus = 0.0
        if self.market_volatility > 0.03 and self.current_leverage < 3.0:
            risk_bonus = 0.3

        total_reward = (
            pnl_reward
            + sharpe_bonus
            + drawdown_penalty
            + loss_penalty
            + risk_bonus
        )

        return float(total_reward)

    def _update_market_state(self) -> None:
        """Simulate market regime changes."""
        # Gradually evolve volatility
        self.market_volatility += np.random.normal(0, 0.002)
        self.market_volatility = np.clip(self.market_volatility, 0.005, 0.10)

        # Occasionally shift phase (10% chance per step)
        if np.random.random() < 0.10:
            self.market_phase = (self.market_phase + 1) % 4

        # Evolve temperature and entropy
        self.market_temperature += np.random.normal(0, 0.05)
        self.market_temperature = np.clip(self.market_temperature, 0.0, 1.0)

        self.market_entropy += np.random.normal(0, 0.05)
        self.market_entropy = np.clip(self.market_entropy, 0.0, 1.0)

    def _get_observation(self) -> np.ndarray:
        """Get current observation vector."""
        # Calculate metrics from recent trades
        win_rate = 0.5
        profit_factor = 1.0
        if len(self.trades) > 0:
            wins = sum(1 for t in self.trades if t["is_win"])
            win_rate = wins / len(self.trades)

            gross_wins = sum(t["pnl"] for t in self.trades if t["pnl"] > 0)
            gross_losses = abs(sum(t["pnl"] for t in self.trades if t["pnl"] < 0))
            if gross_losses > 0:
                profit_factor = gross_wins / gross_losses

        # Drawdown
        drawdown_pct = -((self.peak_equity - self.equity) / self.peak_equity) * 100

        # Phase one-hot encoding
        phase_one_hot = [0.0, 0.0, 0.0, 0.0]
        phase_one_hot[self.market_phase] = 1.0

        # Open positions (simulated)
        open_positions = float(np.random.randint(0, 3))

        # Equity ratio
        equity_ratio = self.equity / self.initial_capital

        # Hours since last trade (simulated)
        hours_since_trade = float(self.current_step % 24)

        obs = np.array([
            drawdown_pct,
            win_rate,
            profit_factor,
            self.market_volatility,
            *phase_one_hot,
            self.market_temperature,
            self.market_entropy,
            self.current_leverage,
            open_positions,
            equity_ratio,
            float(self.consecutive_losses),
            hours_since_trade,
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> dict[str, Any]:
        """Get additional info dict."""
        return {
            "equity": self.equity,
            "peak_equity": self.peak_equity,
            "num_trades": len(self.trades),
            "current_leverage": self.current_leverage,
            "market_phase": self.market_phase,
            "market_volatility": self.market_volatility,
        }

    def render(self) -> None:
        """Render environment state (human-readable)."""
        drawdown = ((self.peak_equity - self.equity) / self.peak_equity) * 100
        logger.info(
            f"Step {self.current_step}: Equity=${self.equity:.2f} "
            f"Drawdown={drawdown:.2f}% Leverage={self.current_leverage:.1f}x "
            f"Phase={self.market_phase} Vol={self.market_volatility:.3f}"
        )

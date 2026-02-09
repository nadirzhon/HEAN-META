"""
Reinforcement Learning Portfolio Allocator.

Uses PPO (Proximal Policy Optimization) for dynamic capital allocation
between trading strategies based on:
- Historical performance (PF, Sharpe, win rate)
- Drawdown levels
- Correlation between strategies
- Market regime

Learns optimal allocation weights that maximize risk-adjusted returns.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)

# Try to import torch for PPO, fallback to simple allocation
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Dirichlet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using simple allocation")


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy."""

    strategy_id: str
    profit_factor: float
    sharpe_ratio: float
    win_rate: float
    drawdown_pct: float
    volatility: float
    recent_returns: list[float]
    correlation_with_others: float


@dataclass
class AllocationResult:
    """Result of allocation decision."""

    weights: dict[str, float]  # strategy_id -> weight
    confidence: float
    method: str  # "rl", "risk_parity", "equal", etc.
    features_used: dict[str, float]
    timestamp: datetime


class PPOPolicy(nn.Module):
    """PPO Actor-Critic Network for portfolio allocation."""

    def __init__(
        self,
        state_dim: int,
        num_strategies: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.num_strategies = num_strategies

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (outputs Dirichlet concentration parameters)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_strategies),
            nn.Softplus(),  # Ensure positive concentrations
        )

        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: State tensor (batch, state_dim)

        Returns:
            concentrations: Dirichlet concentration params (batch, num_strategies)
            value: State value estimate (batch, 1)
        """
        features = self.feature_net(state)
        concentrations = self.actor(features) + 1.0  # Ensure > 1 for peaked distribution
        value = self.critic(features)
        return concentrations, value

    def get_action(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: State tensor

        Returns:
            action: Sampled weights (sum to 1)
            log_prob: Log probability of action
            value: Value estimate
        """
        concentrations, value = self.forward(state)
        distribution = Dirichlet(concentrations)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob, value


class RLPortfolioAllocator:
    """
    RL-based portfolio allocator using PPO.

    State space (per strategy):
    - Profit factor (normalized)
    - Sharpe ratio (normalized)
    - Win rate
    - Current drawdown
    - Recent volatility
    - Correlation with portfolio

    Action space:
    - Allocation weights for each strategy (Dirichlet distribution)

    Reward:
    - Risk-adjusted portfolio return (Sharpe ratio of combined returns)
    """

    FEATURES_PER_STRATEGY = 6

    def __init__(
        self,
        strategy_ids: list[str],
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ):
        """
        Initialize the RL allocator.

        Args:
            strategy_ids: List of strategy IDs to allocate between
            learning_rate: PPO learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping epsilon
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            device: Device to run on
        """
        self._strategy_ids = strategy_ids
        self._num_strategies = len(strategy_ids)
        self._state_dim = self._num_strategies * self.FEATURES_PER_STRATEGY

        self._gamma = gamma
        self._gae_lambda = gae_lambda
        self._clip_epsilon = clip_epsilon
        self._value_loss_coef = value_loss_coef
        self._entropy_coef = entropy_coef

        self._device = torch.device(device if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")

        # Initialize PPO policy
        if TORCH_AVAILABLE:
            self._policy = PPOPolicy(
                state_dim=self._state_dim,
                num_strategies=self._num_strategies,
            ).to(self._device)
            self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=learning_rate)
        else:
            self._policy = None
            self._optimizer = None

        # Experience buffer for training
        self._experience_buffer: deque = deque(maxlen=10000)

        # Current allocation (equal weights initially)
        equal_weight = 1.0 / self._num_strategies
        self._current_allocation: dict[str, float] = dict.fromkeys(
            self._strategy_ids, equal_weight
        )

        # Performance tracking
        self._allocation_history: deque = deque(maxlen=1000)
        self._return_history: deque = deque(maxlen=1000)

        # Metrics
        self._allocations_made = 0
        self._training_steps = 0
        self._avg_reward = 0.0

        logger.info(
            f"RLPortfolioAllocator initialized for {self._num_strategies} strategies "
            f"(RL {'enabled' if TORCH_AVAILABLE else 'disabled'})"
        )

    def get_allocation(
        self,
        strategy_metrics: dict[str, StrategyMetrics],
        min_allocation: float = 0.05,
        max_allocation: float = 0.5,
    ) -> AllocationResult:
        """
        Get optimal allocation weights for each strategy.

        Args:
            strategy_metrics: Current metrics for each strategy
            min_allocation: Minimum allocation per strategy
            max_allocation: Maximum allocation per strategy

        Returns:
            AllocationResult with weights and metadata
        """
        self._allocations_made += 1
        timestamp = datetime.utcnow()

        # Build state vector
        state = self._build_state(strategy_metrics)
        features_dict = self._state_to_dict(state)

        if TORCH_AVAILABLE and self._policy is not None:
            weights = self._get_rl_allocation(state)
            method = "rl"
            confidence = self._calculate_confidence(state)
        else:
            weights = self._get_risk_parity_allocation(strategy_metrics)
            method = "risk_parity"
            confidence = 0.5

        # Apply allocation bounds
        weights = self._apply_bounds(weights, min_allocation, max_allocation)

        # Update current allocation
        self._current_allocation = weights

        result = AllocationResult(
            weights=weights,
            confidence=confidence,
            method=method,
            features_used=features_dict,
            timestamp=timestamp,
        )

        self._allocation_history.append(result)

        logger.info(
            f"Portfolio allocation ({method}): "
            + ", ".join(f"{sid}={w:.1%}" for sid, w in weights.items())
        )

        return result

    def _build_state(self, strategy_metrics: dict[str, StrategyMetrics]) -> np.ndarray:
        """Build state vector from strategy metrics."""
        state = []

        for sid in self._strategy_ids:
            metrics = strategy_metrics.get(sid)

            if metrics is None:
                # Default values if metrics not available
                state.extend([1.0, 0.0, 0.5, 0.0, 0.01, 0.0])
            else:
                # Normalize features
                pf_normalized = min(metrics.profit_factor / 2.0, 2.0)  # Cap at 2
                sharpe_normalized = np.clip(metrics.sharpe_ratio / 2.0, -1.0, 1.0)
                win_rate = metrics.win_rate
                dd_normalized = min(metrics.drawdown_pct / 20.0, 1.0)  # Normalize to 20% max
                vol_normalized = min(metrics.volatility / 0.05, 1.0)  # Normalize to 5% max
                correlation = metrics.correlation_with_others

                state.extend([
                    pf_normalized,
                    sharpe_normalized,
                    win_rate,
                    dd_normalized,
                    vol_normalized,
                    correlation,
                ])

        return np.array(state, dtype=np.float32)

    def _state_to_dict(self, state: np.ndarray) -> dict[str, float]:
        """Convert state array to named dictionary."""
        feature_names = ["pf", "sharpe", "win_rate", "dd", "vol", "corr"]
        result = {}

        for i, sid in enumerate(self._strategy_ids):
            offset = i * self.FEATURES_PER_STRATEGY
            for j, name in enumerate(feature_names):
                result[f"{sid}_{name}"] = float(state[offset + j])

        return result

    def _get_rl_allocation(self, state: np.ndarray) -> dict[str, float]:
        """Get allocation using RL policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self._device)

        with torch.no_grad():
            self._policy.eval()
            action, _, _ = self._policy.get_action(state_tensor)

        weights_array = action.cpu().numpy()[0]

        return {
            sid: float(weights_array[i])
            for i, sid in enumerate(self._strategy_ids)
        }

    def _get_risk_parity_allocation(
        self,
        strategy_metrics: dict[str, StrategyMetrics],
    ) -> dict[str, float]:
        """Get allocation using risk parity (inverse volatility)."""
        weights = {}
        total_inv_vol = 0.0

        for sid in self._strategy_ids:
            metrics = strategy_metrics.get(sid)
            if metrics is None:
                vol = 0.02  # Default volatility
            else:
                vol = max(metrics.volatility, 0.001)

            inv_vol = 1.0 / vol
            weights[sid] = inv_vol
            total_inv_vol += inv_vol

        # Normalize to sum to 1
        for sid in weights:
            weights[sid] /= total_inv_vol

        return weights

    def _apply_bounds(
        self,
        weights: dict[str, float],
        min_allocation: float,
        max_allocation: float,
    ) -> dict[str, float]:
        """Apply min/max allocation bounds."""
        # First pass: apply bounds
        bounded = {}
        for sid, w in weights.items():
            bounded[sid] = np.clip(w, min_allocation, max_allocation)

        # Second pass: renormalize to sum to 1
        total = sum(bounded.values())
        for sid in bounded:
            bounded[sid] /= total

        return bounded

    def _calculate_confidence(self, state: np.ndarray) -> float:
        """Calculate confidence in allocation decision."""
        # Higher confidence when strategies have diverse, positive metrics
        pf_values = [state[i * self.FEATURES_PER_STRATEGY] for i in range(self._num_strategies)]
        sharpe_values = [state[i * self.FEATURES_PER_STRATEGY + 1] for i in range(self._num_strategies)]

        # Confidence based on:
        # 1. At least one strategy has good PF
        # 2. Strategies have diverse Sharpe ratios (not all correlated)
        max_pf = max(pf_values)
        sharpe_diversity = np.std(sharpe_values)

        confidence = min(1.0, (max_pf / 2.0) * 0.5 + sharpe_diversity * 0.5)
        return max(0.0, min(1.0, confidence))

    def update_from_returns(
        self,
        realized_returns: dict[str, float],
        previous_allocation: dict[str, float],
    ) -> float | None:
        """
        Update policy from realized returns.

        Args:
            realized_returns: Realized return per strategy
            previous_allocation: Allocation that was used

        Returns:
            Training loss if training occurred, None otherwise
        """
        # Calculate portfolio return
        portfolio_return = sum(
            realized_returns.get(sid, 0.0) * previous_allocation.get(sid, 0.0)
            for sid in self._strategy_ids
        )

        self._return_history.append(portfolio_return)

        # Calculate reward (Sharpe-like)
        if len(self._return_history) >= 10:
            returns = list(self._return_history)[-50:]
            mean_return = np.mean(returns)
            std_return = np.std(returns) + 1e-8
            reward = mean_return / std_return  # Sharpe ratio proxy
        else:
            reward = portfolio_return

        # Store experience
        if len(self._allocation_history) > 0:
            last_alloc = self._allocation_history[-1]
            self._experience_buffer.append({
                "state": last_alloc.features_used,
                "action": previous_allocation,
                "reward": reward,
            })

        # Update running average reward
        self._avg_reward = 0.95 * self._avg_reward + 0.05 * reward

        # Train if enough experience
        if TORCH_AVAILABLE and len(self._experience_buffer) >= 64:
            return self._train_step()

        return None

    def _train_step(self, batch_size: int = 32) -> float:
        """Perform one PPO training step."""
        if not TORCH_AVAILABLE or self._policy is None:
            return 0.0

        self._training_steps += 1

        # Sample batch
        indices = np.random.choice(len(self._experience_buffer), batch_size, replace=False)
        batch = [self._experience_buffer[i] for i in indices]

        # Prepare data
        states = []
        actions = []
        rewards = []

        for exp in batch:
            state = self._dict_to_state_array(exp["state"])
            action = [exp["action"].get(sid, 1.0 / self._num_strategies) for sid in self._strategy_ids]
            states.append(state)
            actions.append(action)
            rewards.append(exp["reward"])

        states = torch.FloatTensor(np.array(states)).to(self._device)
        actions = torch.FloatTensor(np.array(actions)).to(self._device)
        rewards = torch.FloatTensor(np.array(rewards)).to(self._device)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Forward pass
        self._policy.train()
        concentrations, values = self._policy(states)
        distribution = Dirichlet(concentrations)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()

        # Policy loss (simplified - using rewards as advantages)
        policy_loss = -(log_probs * rewards).mean()

        # Value loss (simplified)
        value_loss = F.mse_loss(values.squeeze(), rewards)

        # Total loss
        loss = (
            policy_loss
            + self._value_loss_coef * value_loss
            - self._entropy_coef * entropy.mean()
        )

        # Backward pass
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._policy.parameters(), 0.5)
        self._optimizer.step()

        self._policy.eval()

        return float(loss)

    def _dict_to_state_array(self, features_dict: dict) -> np.ndarray:
        """Convert features dictionary back to state array."""
        feature_names = ["pf", "sharpe", "win_rate", "dd", "vol", "corr"]
        state = []

        for sid in self._strategy_ids:
            for name in feature_names:
                key = f"{sid}_{name}"
                state.append(features_dict.get(key, 0.0))

        return np.array(state, dtype=np.float32)

    def get_current_allocation(self) -> dict[str, float]:
        """Get current allocation weights."""
        return self._current_allocation.copy()

    def get_metrics(self) -> dict:
        """Get allocator metrics."""
        return {
            "allocations_made": self._allocations_made,
            "training_steps": self._training_steps,
            "avg_reward": self._avg_reward,
            "experience_buffer_size": len(self._experience_buffer),
            "num_strategies": self._num_strategies,
            "rl_enabled": TORCH_AVAILABLE,
            "current_allocation": self._current_allocation,
        }

    def save_model(self, path: str) -> bool:
        """Save model to disk."""
        if not TORCH_AVAILABLE or self._policy is None:
            return False

        try:
            torch.save({
                "policy_state_dict": self._policy.state_dict(),
                "optimizer_state_dict": self._optimizer.state_dict(),
                "strategy_ids": self._strategy_ids,
                "metrics": self.get_metrics(),
            }, path)
            logger.info(f"RL allocator model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, path: str) -> bool:
        """Load model from disk."""
        if not TORCH_AVAILABLE:
            return False

        try:
            checkpoint = torch.load(path, map_location=self._device)

            # Verify strategy IDs match
            if checkpoint.get("strategy_ids") != self._strategy_ids:
                logger.warning("Strategy IDs mismatch, model may not work correctly")

            self._policy.load_state_dict(checkpoint["policy_state_dict"])
            self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._policy.eval()

            logger.info(f"RL allocator model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

"""
PPO Agent for Bitcoin trading using Ray RLlib.

Provides configurable PPO agent with customizable network architecture,
learning parameters, and training callbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CustomTradingNetwork(TorchModelV2, nn.Module):
    """
    Custom neural network for trading policy.

    Architecture:
        - Input layer (25 features)
        - Hidden layers (256 -> 256 -> 128)
        - Separate policy and value heads
        - Dropout and LayerNorm for regularization
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config: ModelConfigDict,
        name: str,
        **kwargs
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Network configuration
        hidden_sizes = model_config.get("fcnet_hiddens", [256, 256, 128])
        activation = model_config.get("fcnet_activation", "relu")
        dropout_rate = model_config.get("dropout", 0.1)

        # Activation function
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "elu":
            act_fn = nn.ELU
        else:
            act_fn = nn.ReLU

        # Shared feature extractor
        layers = []
        input_size = obs_space.shape[0]

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.LayerNorm(hidden_size),
                act_fn(),
                nn.Dropout(dropout_rate),
            ])
            input_size = hidden_size

        self.feature_extractor = nn.Sequential(*layers)

        # Policy head
        self.policy_head = nn.Linear(input_size, num_outputs)

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(input_size, 64),
            act_fn(),
            nn.Linear(64, 1),
        )

        # Store last features for value function
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        """Forward pass through the network."""
        obs = input_dict["obs"].float()

        # Extract features
        self._features = self.feature_extractor(obs)

        # Policy logits
        logits = self.policy_head(self._features)

        return logits, state

    def value_function(self):
        """Return value function estimate."""
        assert self._features is not None, "Must call forward() first"
        return self.value_head(self._features).squeeze(1)


class RLTradingAgent:
    """
    Wrapper for Ray RLlib PPO agent for Bitcoin trading.

    Provides high-level interface for training, evaluation, and inference.
    """

    def __init__(
        self,
        env_class,
        env_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        use_custom_model: bool = True,
    ):
        """
        Initialize RL trading agent.

        Args:
            env_class: Environment class (e.g., BitcoinTradingEnv)
            env_config: Environment configuration dict
            model_config: Model configuration dict
            training_config: Training configuration dict
            use_custom_model: Whether to use custom network architecture
        """
        self.env_class = env_class
        self.env_config = env_config or {}
        self.use_custom_model = use_custom_model

        # Register custom model if needed
        if use_custom_model:
            ModelCatalog.register_custom_model("trading_network", CustomTradingNetwork)

        # Build PPO config
        self.config = self._build_config(model_config, training_config)

        # Algorithm instance
        self.algo: Optional[PPO] = None

    def _build_config(
        self,
        model_config: Optional[Dict[str, Any]] = None,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> PPOConfig:
        """Build PPO configuration."""
        config = PPOConfig()

        # Environment settings
        config.environment(env=self.env_class, env_config=self.env_config)

        # Model settings
        if self.use_custom_model:
            config.training(
                model={
                    "custom_model": "trading_network",
                    "fcnet_hiddens": [256, 256, 128],
                    "fcnet_activation": "relu",
                    "dropout": 0.1,
                    **(model_config or {}),
                }
            )
        else:
            config.training(
                model={
                    "fcnet_hiddens": [256, 256, 128],
                    "fcnet_activation": "relu",
                    **(model_config or {}),
                }
            )

        # PPO hyperparameters (optimized for trading)
        training_defaults = {
            "lr": 3e-4,
            "gamma": 0.99,  # Discount factor
            "lambda": 0.95,  # GAE lambda
            "clip_param": 0.2,  # PPO clip parameter
            "kl_coeff": 0.0,  # KL penalty coefficient
            "num_sgd_iter": 10,  # SGD iterations per update
            "sgd_minibatch_size": 256,
            "train_batch_size": 4096,
            "vf_loss_coeff": 0.5,  # Value function loss coefficient
            "entropy_coeff": 0.01,  # Entropy bonus
            "grad_clip": 0.5,  # Gradient clipping
        }

        if training_config:
            training_defaults.update(training_config)

        config.training(**training_defaults)

        # Rollout settings
        config.rollouts(
            num_rollout_workers=4,  # Parallel workers for data collection
            num_envs_per_worker=1,
            rollout_fragment_length="auto",
        )

        # Resources
        config.resources(
            num_gpus=1 if torch.cuda.is_available() else 0,
            num_cpus_for_local_worker=1,
        )

        # Debugging
        config.debugging(
            log_level="INFO",
        )

        # Evaluation
        config.evaluation(
            evaluation_interval=10,  # Evaluate every 10 iterations
            evaluation_duration=10,  # 10 episodes per evaluation
            evaluation_num_workers=1,
        )

        return config

    def build(self) -> None:
        """Build the algorithm instance."""
        logger.info("Building PPO algorithm...")
        self.algo = self.config.build()
        logger.info("Algorithm built successfully")

    def train(self, num_iterations: int = 100, checkpoint_freq: int = 10) -> Dict[str, Any]:
        """
        Train the agent.

        Args:
            num_iterations: Number of training iterations
            checkpoint_freq: Save checkpoint every N iterations

        Returns:
            Training statistics
        """
        if self.algo is None:
            self.build()

        logger.info(f"Starting training for {num_iterations} iterations...")

        results = []
        for i in range(num_iterations):
            result = self.algo.train()

            # Log progress
            episode_reward_mean = result.get("episode_reward_mean", 0)
            episode_len_mean = result.get("episode_len_mean", 0)

            logger.info(
                f"Iteration {i+1}/{num_iterations} | "
                f"Reward: {episode_reward_mean:.2f} | "
                f"Episode Length: {episode_len_mean:.1f}"
            )

            results.append(result)

            # Checkpoint
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_dir = self.algo.save()
                logger.info(f"Checkpoint saved to {checkpoint_dir}")

        logger.info("Training completed!")
        return {
            'iterations': num_iterations,
            'final_reward': results[-1].get("episode_reward_mean", 0),
            'results': results,
        }

    def evaluate(self, num_episodes: int = 10) -> Dict[str, Any]:
        """
        Evaluate the agent.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Evaluation statistics
        """
        if self.algo is None:
            raise ValueError("Algorithm not built. Call build() or train() first.")

        logger.info(f"Evaluating agent for {num_episodes} episodes...")

        # Create evaluation environment
        env = self.env_class(self.env_config)

        episode_rewards = []
        episode_lengths = []
        episode_stats = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_length = 0

            while not (done or truncated):
                # Get action from policy
                action = self.algo.compute_single_action(obs, explore=False)

                # Step environment
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                episode_length += 1

            # Collect episode stats
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            episode_stats.append(env.get_episode_stats())

            logger.info(
                f"Episode {ep+1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Length: {episode_length}"
            )

        # Aggregate statistics
        stats = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'episode_stats': episode_stats,
        }

        # Average trading metrics
        if episode_stats:
            stats['avg_return'] = np.mean([s['total_return'] for s in episode_stats])
            stats['avg_trades'] = np.mean([s['total_trades'] for s in episode_stats])
            stats['avg_win_rate'] = np.mean([s['win_rate'] for s in episode_stats])
            stats['avg_max_drawdown'] = np.mean([s['max_drawdown'] for s in episode_stats])
            stats['avg_profit_factor'] = np.mean([
                s['profit_factor'] for s in episode_stats
                if s['profit_factor'] != float('inf')
            ])

        logger.info(f"Evaluation complete: Mean Reward = {stats['mean_reward']:.2f}")
        return stats

    def save(self, path: str | Path) -> None:
        """
        Save agent checkpoint.

        Args:
            path: Path to save checkpoint
        """
        if self.algo is None:
            raise ValueError("Algorithm not built. Call build() or train() first.")

        checkpoint_dir = self.algo.save(str(path))
        logger.info(f"Agent saved to {checkpoint_dir}")

    def load(self, path: str | Path) -> None:
        """
        Load agent checkpoint.

        Args:
            path: Path to checkpoint
        """
        if self.algo is None:
            self.build()

        self.algo.restore(str(path))
        logger.info(f"Agent loaded from {path}")

    def predict(self, observation: np.ndarray, explore: bool = False) -> int:
        """
        Get action prediction for a single observation.

        Args:
            observation: Environment observation
            explore: Whether to explore (use stochastic policy)

        Returns:
            Action to take
        """
        if self.algo is None:
            raise ValueError("Algorithm not built. Call build() or train() first.")

        action = self.algo.compute_single_action(observation, explore=explore)
        return int(action)

    def get_policy_weights(self) -> Dict[str, Any]:
        """Get policy network weights."""
        if self.algo is None:
            raise ValueError("Algorithm not built.")

        return self.algo.get_policy().get_weights()

    def set_policy_weights(self, weights: Dict[str, Any]) -> None:
        """Set policy network weights."""
        if self.algo is None:
            raise ValueError("Algorithm not built.")

        self.algo.get_policy().set_weights(weights)

    def stop(self) -> None:
        """Stop the algorithm and cleanup resources."""
        if self.algo is not None:
            self.algo.stop()
            logger.info("Algorithm stopped")


def create_default_agent(
    env_class,
    env_config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> RLTradingAgent:
    """
    Create agent with default configuration.

    Args:
        env_class: Environment class
        env_config: Environment config
        **kwargs: Additional agent config

    Returns:
        Configured RLTradingAgent
    """
    return RLTradingAgent(
        env_class=env_class,
        env_config=env_config,
        use_custom_model=True,
        **kwargs
    )

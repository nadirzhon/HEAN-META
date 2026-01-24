"""
Training script for Bitcoin trading RL agent.

Supports:
- Training from scratch or from checkpoint
- Automatic checkpointing and logging
- TensorBoard integration
- Hyperparameter tuning
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

from .agent import RLTradingAgent
from .data_loader import load_sample_data, DataLoader
from .trading_environment import BitcoinTradingEnv, TradingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingSession:
    """Manages a complete training session for the RL agent."""

    def __init__(
        self,
        data: np.ndarray,
        output_dir: str | Path = "outputs/rl_training",
        config: Optional[TradingConfig] = None,
        training_config: Optional[Dict[str, Any]] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize training session.

        Args:
            data: Historical OHLCV data
            output_dir: Directory for outputs (checkpoints, logs)
            config: Trading environment config
            training_config: PPO training config
            model_config: Model architecture config
        """
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.trading_config = config or TradingConfig()
        self.training_config = training_config or {}
        self.model_config = model_config or {}

        # Split data
        self.train_data, self.val_data, self.test_data = DataLoader.split_data(
            data,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        logger.info(f"Data split: train={len(self.train_data)}, "
                   f"val={len(self.val_data)}, test={len(self.test_data)}")

        # Create agent
        self.agent = self._create_agent()

        # Training history
        self.history: list[Dict[str, Any]] = []

    def _create_agent(self) -> RLTradingAgent:
        """Create RL agent with current configuration."""
        env_config = {
            'data': self.train_data,
            'config': self.trading_config,
        }

        agent = RLTradingAgent(
            env_class=BitcoinTradingEnv,
            env_config=env_config,
            model_config=self.model_config,
            training_config=self.training_config,
            use_custom_model=True,
        )

        return agent

    def train(
        self,
        num_iterations: int = 1000,
        checkpoint_freq: int = 50,
        eval_freq: int = 10,
    ) -> Dict[str, Any]:
        """
        Train the agent.

        Args:
            num_iterations: Number of training iterations
            checkpoint_freq: Save checkpoint every N iterations
            eval_freq: Evaluate on validation set every N iterations

        Returns:
            Training results
        """
        logger.info("="*60)
        logger.info("Starting Training Session")
        logger.info("="*60)
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Iterations: {num_iterations}")
        logger.info(f"Training samples: {len(self.train_data)}")
        logger.info("="*60)

        # Build agent
        self.agent.build()

        # Training loop
        for i in range(num_iterations):
            result = self.agent.algo.train()

            # Extract metrics
            metrics = {
                'iteration': i + 1,
                'episode_reward_mean': result.get('episode_reward_mean', 0),
                'episode_reward_max': result.get('episode_reward_max', 0),
                'episode_reward_min': result.get('episode_reward_min', 0),
                'episode_len_mean': result.get('episode_len_mean', 0),
                'policy_loss': result.get('info', {}).get('learner', {}).get('default_policy', {}).get('learner_stats', {}).get('policy_loss', 0),
                'vf_loss': result.get('info', {}).get('learner', {}).get('default_policy', {}).get('learner_stats', {}).get('vf_loss', 0),
                'entropy': result.get('info', {}).get('learner', {}).get('default_policy', {}).get('learner_stats', {}).get('entropy', 0),
            }

            self.history.append(metrics)

            # Log progress
            logger.info(
                f"Iter {i+1}/{num_iterations} | "
                f"Reward: {metrics['episode_reward_mean']:.2f} "
                f"(min={metrics['episode_reward_min']:.2f}, max={metrics['episode_reward_max']:.2f}) | "
                f"Len: {metrics['episode_len_mean']:.1f} | "
                f"Policy Loss: {metrics['policy_loss']:.4f}"
            )

            # Validation evaluation
            if (i + 1) % eval_freq == 0:
                val_stats = self._evaluate_on_validation()
                logger.info(
                    f"Validation | "
                    f"Return: {val_stats['avg_return']*100:.2f}% | "
                    f"Win Rate: {val_stats['avg_win_rate']*100:.1f}% | "
                    f"Drawdown: {val_stats['avg_max_drawdown']*100:.1f}%"
                )

            # Checkpoint
            if (i + 1) % checkpoint_freq == 0:
                checkpoint_path = self.output_dir / f"checkpoint_{i+1}"
                self.agent.save(checkpoint_path)
                self._save_history()
                logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Final checkpoint
        final_checkpoint = self.output_dir / "checkpoint_final"
        self.agent.save(final_checkpoint)
        self._save_history()

        logger.info("="*60)
        logger.info("Training Complete!")
        logger.info("="*60)

        return {
            'num_iterations': num_iterations,
            'final_reward': self.history[-1]['episode_reward_mean'],
            'best_reward': max(m['episode_reward_mean'] for m in self.history),
            'history': self.history,
        }

    def _evaluate_on_validation(self) -> Dict[str, Any]:
        """Evaluate agent on validation data."""
        # Temporarily update environment config to use validation data
        original_data = self.agent.env_config['data']
        self.agent.env_config['data'] = self.val_data

        # Evaluate
        stats = self.agent.evaluate(num_episodes=5)

        # Restore original data
        self.agent.env_config['data'] = original_data

        return stats

    def evaluate_on_test(self, num_episodes: int = 20) -> Dict[str, Any]:
        """
        Evaluate trained agent on test data.

        Args:
            num_episodes: Number of test episodes

        Returns:
            Test statistics
        """
        logger.info("Evaluating on test set...")

        # Update environment config
        self.agent.env_config['data'] = self.test_data

        # Evaluate
        stats = self.agent.evaluate(num_episodes=num_episodes)

        logger.info(f"Test Results: Return={stats['avg_return']*100:.2f}%, "
                   f"Win Rate={stats['avg_win_rate']*100:.1f}%")

        # Save test results
        test_results_path = self.output_dir / "test_results.json"
        with open(test_results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            serializable_stats = {}
            for k, v in stats.items():
                if isinstance(v, (np.integer, np.floating)):
                    serializable_stats[k] = float(v)
                elif isinstance(v, list):
                    serializable_stats[k] = v
                else:
                    serializable_stats[k] = v

            json.dump(serializable_stats, f, indent=2)

        logger.info(f"Test results saved to {test_results_path}")

        return stats

    def _save_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def hyperparameter_tuning(
    data: np.ndarray,
    num_samples: int = 10,
    max_iterations: int = 100,
) -> Dict[str, Any]:
    """
    Run hyperparameter tuning using Ray Tune.

    Args:
        data: Training data
        num_samples: Number of hyperparameter configurations to try
        max_iterations: Max training iterations per trial

    Returns:
        Best configuration and results
    """
    logger.info("Starting hyperparameter tuning...")

    ray.init(ignore_reinit_error=True)

    # Define search space
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-3),
        "gamma": tune.uniform(0.95, 0.999),
        "lambda": tune.uniform(0.9, 0.99),
        "clip_param": tune.uniform(0.1, 0.3),
        "entropy_coeff": tune.loguniform(0.001, 0.1),
        "vf_loss_coeff": tune.uniform(0.3, 1.0),
        "num_sgd_iter": tune.choice([5, 10, 20]),
        "sgd_minibatch_size": tune.choice([128, 256, 512]),
    }

    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        max_t=max_iterations,
        grace_period=10,
        reduction_factor=2,
    )

    def train_function(config):
        """Training function for Ray Tune."""
        # Create training session
        session = TrainingSession(
            data=data,
            training_config=config,
        )

        # Train
        session.train(num_iterations=max_iterations, checkpoint_freq=max_iterations)

        # Report final reward
        final_reward = session.history[-1]['episode_reward_mean']
        tune.report(reward=final_reward)

    # Run tuning
    analysis = tune.run(
        train_function,
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
        metric="reward",
        mode="max",
    )

    # Get best config
    best_config = analysis.get_best_config(metric="reward", mode="max")
    best_reward = analysis.best_result["reward"]

    logger.info(f"Best config: {best_config}")
    logger.info(f"Best reward: {best_reward}")

    ray.shutdown()

    return {
        'best_config': best_config,
        'best_reward': best_reward,
        'all_results': analysis.results_df.to_dict(),
    }


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train Bitcoin RL trading agent")

    # Data arguments
    parser.add_argument('--data-source', type=str, default='synthetic',
                       choices=['synthetic', 'csv', 'binance', 'bybit'],
                       help='Data source')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data file (for CSV source)')
    parser.add_argument('--num-candles', type=int, default=50000,
                       help='Number of candles (for synthetic data)')

    # Training arguments
    parser.add_argument('--num-iterations', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--checkpoint-freq', type=int, default=50,
                       help='Checkpoint frequency')
    parser.add_argument('--eval-freq', type=int, default=10,
                       help='Evaluation frequency')

    # Model arguments
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--hidden-sizes', type=int, nargs='+',
                       default=[256, 256, 128],
                       help='Hidden layer sizes')

    # Output arguments
    parser.add_argument('--output-dir', type=str,
                       default=f'outputs/rl_training_{datetime.now():%Y%m%d_%H%M%S}',
                       help='Output directory')

    # Hyperparameter tuning
    parser.add_argument('--tune', action='store_true',
                       help='Run hyperparameter tuning')
    parser.add_argument('--tune-samples', type=int, default=10,
                       help='Number of tuning samples')

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from source: {args.data_source}")
    if args.data_source == 'csv':
        data = load_sample_data('csv', file_path=args.data_path)
    elif args.data_source == 'synthetic':
        data = load_sample_data('synthetic', n_candles=args.num_candles, seed=42)
    elif args.data_source == 'binance':
        data = load_sample_data('binance', symbol='BTCUSDT', limit=args.num_candles)
    else:
        data = load_sample_data('bybit', symbol='BTCUSDT', limit=args.num_candles)

    logger.info(f"Loaded {len(data)} candles")

    # Hyperparameter tuning
    if args.tune:
        results = hyperparameter_tuning(
            data=data,
            num_samples=args.tune_samples,
            max_iterations=args.num_iterations,
        )

        # Save tuning results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "tuning_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Hyperparameter tuning complete!")
        return

    # Regular training
    training_config = {
        'lr': args.lr,
        'gamma': args.gamma,
    }

    model_config = {
        'fcnet_hiddens': args.hidden_sizes,
    }

    session = TrainingSession(
        data=data,
        output_dir=args.output_dir,
        training_config=training_config,
        model_config=model_config,
    )

    # Train
    results = session.train(
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
    )

    # Test evaluation
    test_stats = session.evaluate_on_test(num_episodes=20)

    logger.info("Training session complete!")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

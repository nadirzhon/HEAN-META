"""
Evaluation script for trained Bitcoin trading RL agent.

Provides comprehensive evaluation metrics, visualization, and analysis.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .agent import RLTradingAgent
from .data_loader import load_sample_data
from .trading_environment import BitcoinTradingEnv, TradingConfig, Action

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Comprehensive evaluator for trained RL trading agents."""

    def __init__(
        self,
        agent: RLTradingAgent,
        data: np.ndarray,
        config: Optional[TradingConfig] = None,
    ):
        """
        Initialize evaluator.

        Args:
            agent: Trained RL agent
            data: Evaluation data
            config: Trading config
        """
        self.agent = agent
        self.data = data
        self.config = config or TradingConfig()

        self.episode_histories: List[Dict[str, Any]] = []

    def evaluate(
        self,
        num_episodes: int = 20,
        render: bool = False,
        save_history: bool = True,
    ) -> Dict[str, Any]:
        """
        Run evaluation episodes and collect statistics.

        Args:
            num_episodes: Number of episodes to run
            render: Whether to render episodes
            save_history: Whether to save episode histories

        Returns:
            Evaluation statistics
        """
        logger.info(f"Running {num_episodes} evaluation episodes...")

        # Create environment
        env = BitcoinTradingEnv(
            data=self.data,
            config=self.config,
            render_mode="human" if render else None,
        )

        episode_rewards = []
        episode_returns = []
        episode_trades = []
        episode_win_rates = []
        episode_drawdowns = []
        episode_profit_factors = []

        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            episode_history = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'infos': [],
            }

            while not (done or truncated):
                # Get action
                action = self.agent.predict(obs, explore=False)

                # Step
                obs, reward, done, truncated, step_info = env.step(action)

                episode_reward += reward

                # Record history
                if save_history:
                    episode_history['observations'].append(obs)
                    episode_history['actions'].append(action)
                    episode_history['rewards'].append(reward)
                    episode_history['infos'].append(step_info)

                if render:
                    env.render()

            # Get episode stats
            stats = env.get_episode_stats()

            episode_rewards.append(episode_reward)
            episode_returns.append(stats['total_return'])
            episode_trades.append(stats['total_trades'])
            episode_win_rates.append(stats['win_rate'])
            episode_drawdowns.append(stats['max_drawdown'])

            if stats['profit_factor'] != float('inf'):
                episode_profit_factors.append(stats['profit_factor'])

            if save_history:
                episode_history['stats'] = stats
                self.episode_histories.append(episode_history)

            logger.info(
                f"Episode {ep+1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Return: {stats['total_return']*100:.2f}% | "
                f"Trades: {stats['total_trades']} | "
                f"Win Rate: {stats['win_rate']*100:.1f}%"
            )

        # Aggregate statistics
        results = {
            'num_episodes': num_episodes,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'median_reward': np.median(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_return': np.mean(episode_returns),
            'std_return': np.std(episode_returns),
            'median_return': np.median(episode_returns),
            'sharpe_ratio': np.mean(episode_returns) / (np.std(episode_returns) + 1e-8),
            'mean_trades': np.mean(episode_trades),
            'mean_win_rate': np.mean(episode_win_rates),
            'mean_drawdown': np.mean(episode_drawdowns),
            'max_drawdown': np.max(episode_drawdowns),
            'mean_profit_factor': np.mean(episode_profit_factors) if episode_profit_factors else 0,
            'profitable_episodes': sum(1 for r in episode_returns if r > 0),
            'profitable_rate': sum(1 for r in episode_returns if r > 0) / num_episodes,
        }

        logger.info("="*60)
        logger.info("Evaluation Results:")
        logger.info(f"  Mean Return: {results['mean_return']*100:.2f}% ± {results['std_return']*100:.2f}%")
        logger.info(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"  Win Rate: {results['mean_win_rate']*100:.1f}%")
        logger.info(f"  Avg Trades: {results['mean_trades']:.0f}")
        logger.info(f"  Max Drawdown: {results['max_drawdown']*100:.1f}%")
        logger.info(f"  Profitable Episodes: {results['profitable_episodes']}/{num_episodes}")
        logger.info("="*60)

        return results

    def analyze_actions(self) -> Dict[str, Any]:
        """
        Analyze action distribution across episodes.

        Returns:
            Action statistics
        """
        if not self.episode_histories:
            raise ValueError("No episode histories available. Run evaluate() with save_history=True first.")

        all_actions = []
        for episode in self.episode_histories:
            all_actions.extend(episode['actions'])

        # Count actions
        action_counts = {action.name: 0 for action in Action}
        for action in all_actions:
            action_counts[Action(action).name] += 1

        total_actions = len(all_actions)
        action_percentages = {
            name: (count / total_actions * 100) for name, count in action_counts.items()
        }

        logger.info("Action Distribution:")
        for name, pct in action_percentages.items():
            logger.info(f"  {name}: {pct:.1f}%")

        return {
            'action_counts': action_counts,
            'action_percentages': action_percentages,
            'total_actions': total_actions,
        }

    def plot_episode(
        self,
        episode_idx: int = 0,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """
        Plot detailed visualization of a single episode.

        Args:
            episode_idx: Episode index to plot
            save_path: Path to save figure (if None, will display)
        """
        if not self.episode_histories:
            raise ValueError("No episode histories available.")

        if episode_idx >= len(self.episode_histories):
            raise ValueError(f"Episode {episode_idx} not found. Only {len(self.episode_histories)} episodes available.")

        episode = self.episode_histories[episode_idx]
        stats = episode['stats']

        # Extract data
        steps = list(range(len(episode['rewards'])))
        rewards = episode['rewards']
        actions = episode['actions']

        # Calculate cumulative metrics
        cumulative_rewards = np.cumsum(rewards)

        # Extract equity from infos
        equity = [info['equity'] for info in episode['infos']]
        positions = [info['position'] for info in episode['infos']]
        cash = [info['cash'] for info in episode['infos']]

        # Create figure
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

        # Plot 1: Equity curve
        axes[0].plot(steps, equity, linewidth=2, color='blue', label='Equity')
        axes[0].axhline(y=self.config.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        axes[0].set_ylabel('Equity ($)')
        axes[0].set_title(f'Episode {episode_idx} - Total Return: {stats["total_return"]*100:.2f}%')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Position size
        axes[1].fill_between(steps, 0, positions, alpha=0.5, color='green', label='Position (BTC)')
        axes[1].set_ylabel('Position (BTC)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Cumulative rewards
        axes[2].plot(steps, cumulative_rewards, linewidth=2, color='purple', label='Cumulative Reward')
        axes[2].set_ylabel('Cumulative Reward')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        # Plot 4: Actions
        action_colors = {
            Action.HOLD: 'gray',
            Action.BUY_SMALL: 'lightgreen',
            Action.BUY_MEDIUM: 'green',
            Action.BUY_LARGE: 'darkgreen',
            Action.SELL_SMALL: 'lightcoral',
            Action.SELL_MEDIUM: 'red',
            Action.SELL_LARGE: 'darkred',
        }

        for step, action in zip(steps, actions):
            action_enum = Action(action)
            axes[3].scatter(step, 0, color=action_colors[action_enum], s=50, alpha=0.6)

        axes[3].set_yticks([])
        axes[3].set_xlabel('Step')
        axes[3].set_title('Actions (Green=Buy, Red=Sell, Gray=Hold)')

        # Add legend for actions
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color, label=action.name)
            for action, color in action_colors.items()
        ]
        axes[3].legend(handles=legend_elements, loc='upper right', ncol=3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Episode plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_training_curve(
        self,
        history_path: str | Path,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """
        Plot training curve from training history.

        Args:
            history_path: Path to training_history.json
            save_path: Path to save figure
        """
        with open(history_path, 'r') as f:
            history = json.load(f)

        iterations = [h['iteration'] for h in history]
        rewards = [h['episode_reward_mean'] for h in history]
        policy_loss = [h.get('policy_loss', 0) for h in history]
        vf_loss = [h.get('vf_loss', 0) for h in history]

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot rewards
        axes[0].plot(iterations, rewards, linewidth=2, color='blue')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Training Progress')
        axes[0].grid(True, alpha=0.3)

        # Plot losses
        axes[1].plot(iterations, policy_loss, linewidth=2, color='red', label='Policy Loss')
        axes[1].plot(iterations, vf_loss, linewidth=2, color='green', label='Value Loss')
        axes[1].set_xlabel('Iteration')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Training curve saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def save_results(self, results: Dict[str, Any], output_path: str | Path) -> None:
        """
        Save evaluation results to JSON.

        Args:
            results: Evaluation results
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (np.integer, np.floating)):
                serializable_results[k] = float(v)
            else:
                serializable_results[k] = v

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {output_path}")


def main():
    """Main evaluation script entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Bitcoin RL trading agent")

    # Model arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')

    # Data arguments
    parser.add_argument('--data-source', type=str, default='synthetic',
                       choices=['synthetic', 'csv', 'binance', 'bybit'],
                       help='Data source')
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to data file (for CSV source)')
    parser.add_argument('--num-candles', type=int, default=10000,
                       help='Number of candles (for synthetic data)')

    # Evaluation arguments
    parser.add_argument('--num-episodes', type=int, default=20,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')

    # Output arguments
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation',
                       help='Output directory')
    parser.add_argument('--plot-episodes', action='store_true',
                       help='Plot episode visualizations')
    parser.add_argument('--training-history', type=str, default=None,
                       help='Path to training_history.json for plotting training curve')

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from source: {args.data_source}")
    if args.data_source == 'csv':
        data = load_sample_data('csv', file_path=args.data_path)
    elif args.data_source == 'synthetic':
        data = load_sample_data('synthetic', n_candles=args.num_candles, seed=123)
    elif args.data_source == 'binance':
        data = load_sample_data('binance', symbol='BTCUSDT', limit=args.num_candles)
    else:
        data = load_sample_data('bybit', symbol='BTCUSDT', limit=args.num_candles)

    logger.info(f"Loaded {len(data)} candles")

    # Create agent and load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    env_config = {'data': data, 'config': TradingConfig()}
    agent = RLTradingAgent(
        env_class=BitcoinTradingEnv,
        env_config=env_config,
        use_custom_model=True,
    )
    agent.load(args.checkpoint)

    # Create evaluator
    evaluator = AgentEvaluator(agent=agent, data=data)

    # Run evaluation
    results = evaluator.evaluate(
        num_episodes=args.num_episodes,
        render=args.render,
        save_history=True,
    )

    # Analyze actions
    action_stats = evaluator.analyze_actions()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save results
    evaluator.save_results(results, output_dir / 'evaluation_results.json')

    with open(output_dir / 'action_stats.json', 'w') as f:
        json.dump(action_stats, f, indent=2)

    # Plot episodes
    if args.plot_episodes:
        logger.info("Plotting episodes...")
        for i in range(min(5, len(evaluator.episode_histories))):
            evaluator.plot_episode(
                episode_idx=i,
                save_path=output_dir / f'episode_{i}.png'
            )

    # Plot training curve
    if args.training_history:
        evaluator.plot_training_curve(
            history_path=args.training_history,
            save_path=output_dir / 'training_curve.png'
        )

    logger.info(f"Evaluation complete! Results saved to {output_dir}")


if __name__ == '__main__':
    main()

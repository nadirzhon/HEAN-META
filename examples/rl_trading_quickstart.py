"""
Quickstart example for training and evaluating RL Bitcoin trading agent.

This script demonstrates:
1. Loading data
2. Creating and training RL agent
3. Evaluating performance
4. Deploying in HEAN system
"""

import asyncio
import logging
from pathlib import Path

import numpy as np

from hean.core.bus import EventBus
from hean.rl.agent import RLTradingAgent
from hean.rl.config import get_quick_test_config, RLAgentConfig
from hean.rl.data_loader import load_sample_data, DataLoader
from hean.rl.evaluation import AgentEvaluator
from hean.rl.trading_environment import BitcoinTradingEnv, TradingConfig
from hean.rl.training import TrainingSession
from hean.strategies.rl_strategy import RLStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_1_quick_training():
    """Example 1: Quick training on synthetic data."""
    logger.info("="*60)
    logger.info("Example 1: Quick Training on Synthetic Data")
    logger.info("="*60)

    # Load synthetic data
    data = load_sample_data('synthetic', n_candles=10000, seed=42)
    logger.info(f"Loaded {len(data)} candles")

    # Create training session with quick test config
    config = get_quick_test_config()

    session = TrainingSession(
        data=data,
        output_dir="outputs/quickstart_training",
        config=config.environment,
        training_config={
            'lr': config.training.lr,
            'gamma': config.training.gamma,
            'num_sgd_iter': config.training.num_sgd_iter,
        },
        model_config={
            'fcnet_hiddens': config.model.hidden_layers,
        }
    )

    # Train for 100 iterations
    results = session.train(
        num_iterations=100,
        checkpoint_freq=50,
        eval_freq=10,
    )

    logger.info(f"Training complete! Final reward: {results['final_reward']:.2f}")

    # Evaluate on test set
    test_stats = session.evaluate_on_test(num_episodes=10)
    logger.info(f"Test performance: {test_stats['avg_return']*100:.2f}% return")

    return session.agent


def example_2_custom_training():
    """Example 2: Custom training configuration."""
    logger.info("="*60)
    logger.info("Example 2: Custom Training Configuration")
    logger.info("="*60)

    # Load data
    data = load_sample_data('synthetic', n_candles=20000, seed=123)

    # Custom trading config
    trading_config = TradingConfig(
        initial_capital=10000.0,
        max_position_size=1.0,
        maker_fee=0.0002,
        taker_fee=0.0006,
        max_drawdown_pct=0.15,  # Stricter drawdown limit
        max_steps=2000,
    )

    # Custom training config
    training_config = {
        'lr': 5e-4,
        'gamma': 0.995,
        'entropy_coeff': 0.02,  # More exploration
        'num_sgd_iter': 15,
    }

    # Custom model config
    model_config = {
        'fcnet_hiddens': [256, 256, 128],
        'dropout': 0.15,
    }

    session = TrainingSession(
        data=data,
        output_dir="outputs/custom_training",
        config=trading_config,
        training_config=training_config,
        model_config=model_config,
    )

    # Train
    results = session.train(num_iterations=200)

    logger.info("Custom training complete!")
    return session.agent


def example_3_evaluation():
    """Example 3: Comprehensive evaluation of trained agent."""
    logger.info("="*60)
    logger.info("Example 3: Agent Evaluation")
    logger.info("="*60)

    # Load test data
    data = load_sample_data('synthetic', n_candles=5000, seed=999)

    # Create a simple agent for demo (normally you'd load a checkpoint)
    env_config = {'data': data, 'config': TradingConfig()}
    agent = RLTradingAgent(
        env_class=BitcoinTradingEnv,
        env_config=env_config,
        use_custom_model=True,
    )
    agent.build()

    # Create evaluator
    evaluator = AgentEvaluator(agent=agent, data=data)

    # Run evaluation
    results = evaluator.evaluate(num_episodes=20, save_history=True)

    # Analyze actions
    action_stats = evaluator.analyze_actions()

    # Plot episode
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator.plot_episode(episode_idx=0, save_path=output_dir / "episode_0.png")

    logger.info("Evaluation complete!")
    return results


def example_4_load_and_deploy():
    """Example 4: Load trained agent and deploy in HEAN."""
    logger.info("="*60)
    logger.info("Example 4: Load and Deploy Agent")
    logger.info("="*60)

    # Simulate loading a checkpoint
    # In real usage: agent.load("path/to/checkpoint")

    data = load_sample_data('synthetic', n_candles=5000, seed=456)
    env_config = {'data': data, 'config': TradingConfig()}

    agent = RLTradingAgent(
        env_class=BitcoinTradingEnv,
        env_config=env_config,
        use_custom_model=True,
    )
    agent.build()

    # Create event bus
    bus = EventBus()

    # Create RL strategy
    rl_strategy = RLStrategy(
        strategy_id="rl_btc_trader",
        bus=bus,
        agent=agent,
        symbol="BTCUSDT",
        trading_config=TradingConfig(),
    )

    logger.info("RL Strategy created and ready to deploy!")
    logger.info(f"Strategy stats: {rl_strategy.get_stats()}")

    return rl_strategy


def example_5_full_pipeline():
    """Example 5: Complete pipeline from training to deployment."""
    logger.info("="*60)
    logger.info("Example 5: Full Pipeline")
    logger.info("="*60)

    # 1. Load and split data
    logger.info("Step 1: Loading data...")
    full_data = load_sample_data('synthetic', n_candles=30000, seed=42)
    train_data, val_data, test_data = DataLoader.split_data(full_data)

    # 2. Train agent
    logger.info("Step 2: Training agent...")
    session = TrainingSession(
        data=full_data,  # Session will split internally
        output_dir="outputs/full_pipeline",
        config=TradingConfig(),
    )
    train_results = session.train(num_iterations=150, checkpoint_freq=50)
    logger.info(f"Training complete: {train_results['final_reward']:.2f}")

    # 3. Evaluate on test
    logger.info("Step 3: Evaluating on test set...")
    test_results = session.evaluate_on_test(num_episodes=20)
    logger.info(f"Test return: {test_results['avg_return']*100:.2f}%")

    # 4. Save agent
    logger.info("Step 4: Saving agent...")
    checkpoint_path = Path("outputs/full_pipeline/final_agent")
    session.agent.save(checkpoint_path)

    # 5. Load agent and create strategy
    logger.info("Step 5: Creating deployment strategy...")
    bus = EventBus()
    rl_strategy = RLStrategy(
        strategy_id="rl_production",
        bus=bus,
        agent=session.agent,
        symbol="BTCUSDT",
    )

    logger.info("Full pipeline complete!")
    logger.info(f"Agent checkpoint: {checkpoint_path}")
    logger.info(f"Test performance: {test_results['avg_return']*100:.2f}% return, "
               f"{test_results['avg_win_rate']*100:.1f}% win rate")

    return rl_strategy


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("RL Bitcoin Trading Agent - Quickstart Examples")
    print("="*60 + "\n")

    # Example 1: Quick training
    agent_1 = example_1_quick_training()
    print("\n")

    # Example 2: Custom training
    # agent_2 = example_2_custom_training()
    # print("\n")

    # Example 3: Evaluation
    # results_3 = example_3_evaluation()
    # print("\n")

    # Example 4: Deployment
    strategy_4 = example_4_load_and_deploy()
    print("\n")

    # Example 5: Full pipeline
    # strategy_5 = example_5_full_pipeline()
    # print("\n")

    print("="*60)
    print("All examples complete!")
    print("Check outputs/ directory for results")
    print("="*60)


if __name__ == '__main__':
    main()

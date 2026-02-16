#!/usr/bin/env python3
"""Train RL Risk Manager using PPO.

Usage:
    python3 scripts/train_rl_risk.py --timesteps 50000 --output models/rl_risk_ppo.zip

Requires: stable-baselines3
    pip install stable-baselines3[extra]
"""

import argparse
from pathlib import Path

from hean.risk.gym_env import TradingRiskEnv
from hean.logging import get_logger, setup_logging

logger = get_logger(__name__)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    logger.error("stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")


def train_rl_risk_manager(
    timesteps: int = 50000,
    output_path: str = "models/rl_risk_ppo.zip",
    checkpoint_freq: int = 5000,
):
    """Train PPO agent for risk management.

    Args:
        timesteps: Total training timesteps
        output_path: Path to save final model
        checkpoint_freq: Frequency for checkpoint saves
    """
    if not SB3_AVAILABLE:
        raise ImportError("stable-baselines3 required for training")

    setup_logging("INFO")
    logger.info("Starting RL Risk Manager training")

    # Create environment
    env = TradingRiskEnv(
        initial_capital=10000.0,
        max_drawdown_pct=20.0,
        lookback_trades=30,
        max_steps=1000,
    )

    # Wrap in vectorized environment
    vec_env = DummyVecEnv([lambda: env])

    # Create evaluation environment
    eval_env = TradingRiskEnv(
        initial_capital=10000.0,
        max_drawdown_pct=20.0,
        lookback_trades=30,
        max_steps=1000,
    )
    eval_vec_env = DummyVecEnv([lambda: eval_env])

    # Create output directory
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="rl_risk",
    )

    eval_callback = EvalCallback(
        eval_vec_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "logs"),
        eval_freq=2000,
        deterministic=True,
        render=False,
    )

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=str(output_dir / "tensorboard"),
        policy_kwargs=dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        ),
        verbose=1,
    )

    logger.info(f"Training PPO model for {timesteps} timesteps")
    logger.info(f"Checkpoint frequency: {checkpoint_freq}")
    logger.info(f"Output path: {output_path}")

    # Train the model
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(output_path)
    logger.info(f"Training complete. Model saved to {output_path}")

    # Test trained model
    logger.info("Testing trained model...")
    obs, _ = vec_env.reset()
    total_reward = 0.0
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = vec_env.step(action)
        total_reward += reward[0]
        if done or truncated:
            break

    logger.info(f"Test episode total reward: {total_reward:.2f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train RL Risk Manager")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="Total training timesteps (default: 50000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/rl_risk_ppo.zip",
        help="Output path for trained model (default: models/rl_risk_ppo.zip)"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=5000,
        help="Checkpoint save frequency (default: 5000)"
    )

    args = parser.parse_args()

    train_rl_risk_manager(
        timesteps=args.timesteps,
        output_path=args.output,
        checkpoint_freq=args.checkpoint_freq,
    )


if __name__ == "__main__":
    main()

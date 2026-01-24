"""
Configuration for RL trading components.

Extends main HEAN config with RL-specific settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RLModelConfig:
    """Configuration for RL model architecture."""

    # Network architecture
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256, 128])
    activation: str = "relu"  # relu, tanh, elu
    dropout: float = 0.1
    use_layer_norm: bool = True

    # Custom model
    use_custom_model: bool = True
    model_name: str = "trading_network"


@dataclass
class RLTrainingConfig:
    """Configuration for RL training."""

    # PPO hyperparameters
    lr: float = 3e-4
    gamma: float = 0.99  # Discount factor
    lambda_: float = 0.95  # GAE lambda
    clip_param: float = 0.2  # PPO clip
    kl_coeff: float = 0.0
    num_sgd_iter: int = 10
    sgd_minibatch_size: int = 256
    train_batch_size: int = 4096
    vf_loss_coeff: float = 0.5
    entropy_coeff: float = 0.01
    grad_clip: float = 0.5

    # Training settings
    num_iterations: int = 1000
    checkpoint_freq: int = 50
    eval_freq: int = 10

    # Rollout settings
    num_rollout_workers: int = 4
    num_envs_per_worker: int = 1

    # Evaluation
    evaluation_interval: int = 10
    evaluation_duration: int = 10  # episodes
    evaluation_num_workers: int = 1

    # Resources
    num_gpus: int = 0  # Will be auto-detected
    num_cpus_per_worker: int = 1


@dataclass
class RLEnvironmentConfig:
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
    hold_penalty: float = 0.0001


@dataclass
class RLDataConfig:
    """Configuration for data loading."""

    # Data source
    source: str = "synthetic"  # synthetic, csv, binance, bybit
    symbol: str = "BTCUSDT"
    interval: str = "1h"

    # File paths
    csv_path: Optional[str] = None

    # Data size
    num_candles: int = 50000

    # Data splitting
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Normalization
    normalize_method: str = "none"  # none, minmax, standard

    # Synthetic data params
    synthetic_initial_price: float = 30000.0
    synthetic_trend: float = 0.0001
    synthetic_volatility: float = 0.02
    synthetic_seed: Optional[int] = 42


@dataclass
class RLAgentConfig:
    """Complete configuration for RL agent."""

    model: RLModelConfig = field(default_factory=RLModelConfig)
    training: RLTrainingConfig = field(default_factory=RLTrainingConfig)
    environment: RLEnvironmentConfig = field(default_factory=RLEnvironmentConfig)
    data: RLDataConfig = field(default_factory=RLDataConfig)

    # Checkpointing
    checkpoint_dir: str = "outputs/rl_checkpoints"
    restore_from_checkpoint: Optional[str] = None

    # Logging
    log_level: str = "INFO"
    tensorboard_dir: str = "outputs/tensorboard"

    # Mode
    mode: str = "train"  # train, eval, deploy

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model': {
                'hidden_layers': self.model.hidden_layers,
                'activation': self.model.activation,
                'dropout': self.model.dropout,
                'use_layer_norm': self.model.use_layer_norm,
                'use_custom_model': self.model.use_custom_model,
            },
            'training': {
                'lr': self.training.lr,
                'gamma': self.training.gamma,
                'lambda_': self.training.lambda_,
                'clip_param': self.training.clip_param,
                'entropy_coeff': self.training.entropy_coeff,
                'num_iterations': self.training.num_iterations,
            },
            'environment': {
                'initial_capital': self.environment.initial_capital,
                'max_position_size': self.environment.max_position_size,
                'maker_fee': self.environment.maker_fee,
                'taker_fee': self.environment.taker_fee,
            },
            'data': {
                'source': self.data.source,
                'symbol': self.data.symbol,
                'num_candles': self.data.num_candles,
            }
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> RLAgentConfig:
        """Create from dictionary."""
        model_config = RLModelConfig(**config_dict.get('model', {}))
        training_config = RLTrainingConfig(**config_dict.get('training', {}))
        env_config = RLEnvironmentConfig(**config_dict.get('environment', {}))
        data_config = RLDataConfig(**config_dict.get('data', {}))

        return cls(
            model=model_config,
            training=training_config,
            environment=env_config,
            data=data_config,
        )


# Default configurations for different use cases

def get_quick_test_config() -> RLAgentConfig:
    """Get config for quick testing (fast, small model)."""
    config = RLAgentConfig()
    config.model.hidden_layers = [128, 128]
    config.training.num_iterations = 100
    config.training.train_batch_size = 2048
    config.training.num_rollout_workers = 2
    config.environment.max_steps = 500
    config.data.num_candles = 5000
    return config


def get_production_config() -> RLAgentConfig:
    """Get config for production training (large model, long training)."""
    config = RLAgentConfig()
    config.model.hidden_layers = [512, 512, 256, 128]
    config.model.dropout = 0.2
    config.training.num_iterations = 5000
    config.training.train_batch_size = 8192
    config.training.num_rollout_workers = 8
    config.training.num_gpus = 1
    config.environment.max_steps = 2000
    config.data.num_candles = 100000
    config.data.source = "binance"  # Use real data
    return config


def get_hyperparameter_tuning_config() -> RLAgentConfig:
    """Get config for hyperparameter tuning."""
    config = RLAgentConfig()
    config.training.num_iterations = 200
    config.training.num_rollout_workers = 4
    config.data.num_candles = 20000
    return config

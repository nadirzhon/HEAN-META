# RL Trading Agent for Bitcoin

Reinforcement Learning agent for Bitcoin trading using Proximal Policy Optimization (PPO).

## Overview

This module provides a complete RL framework for training agents to trade Bitcoin:

- **Environment**: Gymnasium-compatible trading environment with 25 features
- **Agent**: PPO-based agent with custom neural network architecture
- **Training**: Scalable training with Ray RLlib (1M+ steps)
- **Evaluation**: Comprehensive evaluation metrics and visualization
- **Deployment**: Integration with HEAN trading system

## Features

### Trading Environment (`BitcoinTradingEnv`)

- **State Space** (25 features):
  - Price features (5): current price, returns, volatility, high/low
  - Volume features (3): current volume, MA, volume change
  - Technical indicators (8): RSI, MACD, Bollinger Bands, ATR, SMAs
  - Position features (4): position size, avg entry, unrealized PnL, duration
  - Portfolio features (5): cash ratio, equity, drawdown, total PnL, win rate

- **Action Space** (7 discrete actions):
  - `HOLD`: No action
  - `BUY_SMALL`: Buy 25% of available capital
  - `BUY_MEDIUM`: Buy 50% of available capital
  - `BUY_LARGE`: Buy 100% of available capital
  - `SELL_SMALL`: Sell 25% of position
  - `SELL_MEDIUM`: Sell 50% of position
  - `SELL_LARGE`: Sell 100% of position

- **Reward Function**:
  ```
  reward = profit - fees - drawdown_penalty - hold_penalty
  ```
  - Profit: Realized PnL from trades
  - Fees: Trading fees (maker/taker)
  - Drawdown penalty: Exponential penalty for large drawdowns
  - Hold penalty: Small penalty to encourage action

### PPO Agent (`RLTradingAgent`)

- **Architecture**:
  - Custom neural network: 256 → 256 → 128 with LayerNorm and Dropout
  - Separate policy and value heads
  - GPU support

- **Hyperparameters** (optimized for trading):
  - Learning rate: 3e-4
  - Discount factor (γ): 0.99
  - GAE lambda (λ): 0.95
  - PPO clip: 0.2
  - Entropy coefficient: 0.01
  - Batch size: 4096
  - SGD iterations: 10

- **Training**:
  - Parallel rollout workers for data collection
  - Automatic checkpointing
  - Periodic validation
  - TensorBoard logging

## Installation

```bash
# Install RL dependencies
pip install -r src/hean/rl/requirements.txt

# Or install specific packages
pip install ray[rllib] gymnasium torch numpy pandas matplotlib
```

## Quick Start

### 1. Train Agent (Synthetic Data)

```python
from hean.rl.data_loader import load_sample_data
from hean.rl.training import TrainingSession
from hean.rl.config import get_quick_test_config

# Load data
data = load_sample_data('synthetic', n_candles=10000)

# Create training session
config = get_quick_test_config()
session = TrainingSession(
    data=data,
    output_dir="outputs/rl_training",
    config=config.environment,
)

# Train
results = session.train(num_iterations=100)

# Evaluate
test_stats = session.evaluate_on_test(num_episodes=20)
print(f"Return: {test_stats['avg_return']*100:.2f}%")
```

### 2. Train with Real Data

```python
from hean.rl.data_loader import DataLoader

# Load from Binance
data = DataLoader.load_binance(
    symbol='BTCUSDT',
    interval='1h',
    limit=10000
)

# Train as above
session = TrainingSession(data=data, ...)
```

### 3. Evaluate Trained Agent

```python
from hean.rl.agent import RLTradingAgent
from hean.rl.evaluation import AgentEvaluator
from hean.rl.trading_environment import BitcoinTradingEnv

# Load agent
env_config = {'data': data, 'config': TradingConfig()}
agent = RLTradingAgent(
    env_class=BitcoinTradingEnv,
    env_config=env_config,
)
agent.load("outputs/rl_training/checkpoint_final")

# Evaluate
evaluator = AgentEvaluator(agent=agent, data=test_data)
results = evaluator.evaluate(num_episodes=20)

# Analyze
action_stats = evaluator.analyze_actions()
evaluator.plot_episode(episode_idx=0, save_path="episode.png")
```

### 4. Deploy in HEAN System

```python
from hean.core.bus import EventBus
from hean.strategies.rl_strategy import RLStrategy

# Create event bus
bus = EventBus()

# Create RL strategy
rl_strategy = RLStrategy(
    strategy_id="rl_btc_trader",
    bus=bus,
    agent=agent,  # Trained agent
    symbol="BTCUSDT",
)

# Start strategy
await rl_strategy.start()
```

## CLI Usage

### Training

```bash
# Train on synthetic data
python -m hean.rl.training \
    --data-source synthetic \
    --num-candles 50000 \
    --num-iterations 1000 \
    --output-dir outputs/rl_training

# Train on real data (Binance)
python -m hean.rl.training \
    --data-source binance \
    --num-candles 10000 \
    --num-iterations 1000 \
    --lr 3e-4 \
    --gamma 0.99

# Hyperparameter tuning
python -m hean.rl.training \
    --tune \
    --tune-samples 20 \
    --num-iterations 200
```

### Evaluation

```bash
# Evaluate checkpoint
python -m hean.rl.evaluation \
    --checkpoint outputs/rl_training/checkpoint_final \
    --data-source synthetic \
    --num-episodes 50 \
    --plot-episodes \
    --output-dir outputs/evaluation

# With training history
python -m hean.rl.evaluation \
    --checkpoint outputs/rl_training/checkpoint_final \
    --training-history outputs/rl_training/training_history.json \
    --plot-episodes
```

## Configuration

### Quick Test Config (Fast Training)

```python
from hean.rl.config import get_quick_test_config

config = get_quick_test_config()
# Small model, 100 iterations, 5K candles
```

### Production Config (Full Training)

```python
from hean.rl.config import get_production_config

config = get_production_config()
# Large model, 5K iterations, 100K candles, real data
```

### Custom Config

```python
from hean.rl.config import RLAgentConfig, RLModelConfig, RLTrainingConfig

config = RLAgentConfig(
    model=RLModelConfig(
        hidden_layers=[512, 512, 256],
        dropout=0.2,
    ),
    training=RLTrainingConfig(
        lr=5e-4,
        gamma=0.995,
        num_iterations=2000,
    ),
    environment=RLEnvironmentConfig(
        initial_capital=10000.0,
        max_drawdown_pct=0.15,
    ),
)
```

## Data Sources

### Synthetic Data (Testing)

```python
from hean.rl.data_loader import load_sample_data

data = load_sample_data(
    'synthetic',
    n_candles=10000,
    initial_price=30000.0,
    trend=0.0001,
    volatility=0.02,
    seed=42
)
```

### CSV Files

```python
data = load_sample_data(
    'csv',
    file_path='btc_data.csv',
    columns=['open', 'high', 'low', 'close', 'volume']
)
```

### Binance API

```python
data = load_sample_data(
    'binance',
    symbol='BTCUSDT',
    interval='1h',
    limit=10000
)
```

### Bybit API

```python
data = load_sample_data(
    'bybit',
    symbol='BTCUSDT',
    interval='60',
    limit=10000
)
```

## Performance Metrics

The evaluator provides comprehensive metrics:

- **Return Metrics**: mean/std/median return, Sharpe ratio
- **Trading Metrics**: total trades, win rate, profit factor
- **Risk Metrics**: max drawdown, drawdown distribution
- **Action Analysis**: action distribution, buy/sell ratios
- **Episode Visualization**: equity curves, position tracking, action timeline

## Advanced Usage

### Hyperparameter Tuning

```python
from hean.rl.training import hyperparameter_tuning

results = hyperparameter_tuning(
    data=data,
    num_samples=20,
    max_iterations=200,
)

best_config = results['best_config']
best_reward = results['best_reward']
```

### Custom Reward Function

Modify `BitcoinTradingEnv._calculate_reward()`:

```python
def _calculate_reward(self, pnl, fees, action):
    # Custom reward logic
    reward = pnl * 2.0  # Higher weight on profit
    reward -= fees * 0.5  # Lower penalty on fees

    # Add custom components
    if self.state.max_drawdown > 0.1:
        reward -= 50.0  # Heavy penalty

    return reward
```

### Multi-Symbol Trading

Extend `RLStrategy` to support multiple symbols:

```python
class MultiSymbolRLStrategy(BaseStrategy):
    def __init__(self, symbols: List[str], ...):
        self.strategies = {
            symbol: RLStrategy(..., symbol=symbol)
            for symbol in symbols
        }
```

## Integration with HEAN

### 1. Add to Strategy Portfolio

```python
from hean.portfolio.allocator import CapitalAllocator

allocator = CapitalAllocator(...)
allocator.add_strategy(rl_strategy, allocation=0.2)  # 20% capital
```

### 2. Risk Management Integration

```python
from hean.risk.risk_governor import RiskGovernor

risk_governor = RiskGovernor(...)
# RL signals will pass through risk checks
```

### 3. Event-Driven Integration

The RL strategy automatically:
- Subscribes to `TICK` events
- Publishes `SIGNAL` events
- Integrates with execution pipeline
- Reports metrics to observability

## Troubleshooting

### Low Performance

1. **Increase training iterations**: Try 2000+ iterations
2. **Tune hyperparameters**: Run hyperparameter search
3. **Check reward function**: Ensure rewards align with objectives
4. **Increase data quality**: Use real market data instead of synthetic

### Training Instability

1. **Reduce learning rate**: Try 1e-4 instead of 3e-4
2. **Increase batch size**: Try 8192 instead of 4096
3. **Add gradient clipping**: Ensure `grad_clip=0.5`
4. **Reduce entropy**: Lower `entropy_coeff` to 0.005

### Memory Issues

1. **Reduce workers**: Set `num_rollout_workers=2`
2. **Reduce batch size**: Set `train_batch_size=2048`
3. **Reduce episode length**: Set `max_steps=500`

## Examples

See `examples/rl_trading_quickstart.py` for complete examples:

1. Quick training on synthetic data
2. Custom training configuration
3. Agent evaluation and analysis
4. Deployment in HEAN system
5. Full pipeline from training to production

## References

- [Ray RLlib Documentation](https://docs.ray.io/en/latest/rllib/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

## License

Part of the HEAN-META trading system.

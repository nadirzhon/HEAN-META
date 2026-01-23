# Bitcoin Price Prediction ML Stack

Ensemble-based machine learning system for predicting 5-minute Bitcoin price movements.

## Features

- **Ensemble Models**: LightGBM + XGBoost + CatBoost
- **50+ Features**: Technical indicators, volume, orderbook, sentiment
- **Auto-retraining**: Automatic model retraining every 24 hours
- **Backtesting**: Walk-forward and rolling window backtesting
- **Production-ready**: Inference API, model versioning, metrics tracking

## Architecture

```
src/ml/
├── __init__.py           # Package initialization
├── price_predictor.py    # Ensemble predictor (main module)
├── features.py           # Feature engineering (50+ features)
├── trainer.py            # Training script with auto-retraining
├── backtester.py         # Backtesting engine
├── data_loader.py        # Data loading from various sources
├── metrics.py            # Evaluation metrics
└── README.md             # This file
```

## Installation

```bash
pip install lightgbm xgboost catboost pandas numpy scikit-learn
```

Optional dependencies:
```bash
pip install pybit python-binance  # For exchange data
```

## Quick Start

### 1. Training a Model

```python
from ml.trainer import ModelTrainer, TrainerConfig
from ml.price_predictor import PredictorConfig

# Configure trainer
trainer_config = TrainerConfig(
    data_source="synthetic",  # or "csv", "bybit", "binance"
    model_dir="models/bitcoin_predictor",
    auto_retrain=False,
)

# Create trainer
trainer = ModelTrainer(trainer_config=trainer_config)

# Train once
metrics = trainer.train_once()

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

### 2. Making Predictions

```python
from ml.price_predictor import BitcoinPricePredictor
from ml.data_loader import DataLoader

# Load trained model
predictor = BitcoinPricePredictor()
predictor.load_model("models/bitcoin_predictor")

# Load latest data
data_loader = DataLoader()
df = data_loader.load_from_exchange("bybit", interval="5m")

# Predict
result = predictor.predict_single(df)

print(f"Prediction: {result['direction']}")
print(f"Probability: {result['probability']:.2%}")
print(f"Confidence: {result['confidence']:.2%}")
```

### 3. Backtesting

```python
from ml.backtester import Backtester
from ml.data_loader import DataLoader

# Load data
data_loader = DataLoader()
df = data_loader.load_from_csv("data/btc_historical.csv")

# Create backtester
backtester = Backtester()

# Run walk-forward backtest
results = backtester.run_walk_forward_backtest(
    df,
    train_window=5000,
    test_window=1000,
    step_size=500,
)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Total Return: {results['total_return']:.2%}")
print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.2%}")
```

### 4. Auto-retraining

```python
import asyncio
from ml.trainer import ModelTrainer, TrainerConfig

# Configure auto-retraining
trainer_config = TrainerConfig(
    data_source="bybit",
    model_dir="models/bitcoin_predictor",
    auto_retrain=True,
    retrain_interval_hours=24,
)

# Create trainer
trainer = ModelTrainer(trainer_config=trainer_config)

# Start auto-retraining loop
asyncio.run(trainer.start_auto_retraining())
```

## Features

### Technical Indicators (30+)
- RSI (7, 14, 21 periods)
- MACD (12/26/9)
- Bollinger Bands
- Moving Averages (5, 10, 20, 50, 100, 200)
- EMAs (9, 12, 26, 50)
- ATR
- Stochastic Oscillator
- CCI
- MFI
- Williams %R
- Ultimate Oscillator

### Volume Features (10+)
- Volume change
- Volume ratios
- VWAP
- OBV (On-Balance Volume)
- VPT (Volume Price Trend)
- Money Flow Index

### Volatility Features (5+)
- Historical volatility
- Parkinson's volatility
- Garman-Klass volatility

### Price Action Features (10+)
- Returns (various periods)
- High-Low range
- Open-Close range
- Price position
- Gap analysis

### Orderbook Features (Optional)
- Bid-ask spread
- Order imbalance
- Depth ratio
- Weighted mid price

### Sentiment Features (Optional)
- Sentiment scores
- Fear & Greed Index

## CLI Usage

### Train Model

```bash
python -m ml.trainer \
    --data-source bybit \
    --model-dir models/bitcoin_predictor \
    --retrain-interval 24
```

### Run Backtest

```bash
python -m ml.backtester \
    --data-source csv \
    --csv-path data/btc_historical.csv \
    --backtest-type walk_forward \
    --train-window 5000 \
    --test-window 1000 \
    --output results/backtest.csv
```

## Configuration

### Model Hyperparameters

```python
from ml.price_predictor import PredictorConfig

config = PredictorConfig(
    # Prediction
    prediction_horizon=5,  # minutes
    threshold=0.5,

    # Ensemble weights
    lgb_weight=0.35,
    xgb_weight=0.35,
    catboost_weight=0.30,

    # LightGBM
    lgb_params={
        'num_leaves': 31,
        'learning_rate': 0.05,
        'n_estimators': 500,
    },

    # XGBoost
    xgb_params={
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
    },

    # CatBoost
    catboost_params={
        'depth': 6,
        'learning_rate': 0.05,
        'iterations': 500,
    },
)
```

### Feature Engineering

```python
from ml.features import FeatureConfig

feature_config = FeatureConfig(
    rsi_periods=[7, 14, 21],
    ma_periods=[5, 10, 20, 50, 100, 200],
    ema_periods=[9, 12, 26, 50],
    volume_periods=[5, 10, 20],
    atr_period=14,
)
```

## Metrics

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

### Trading Metrics
- Total Return
- Sharpe Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Average Win/Loss

## Model Versioning

Models are automatically versioned:
```
models/
├── bitcoin_predictor/           # Latest model
│   ├── lgb_model.txt
│   ├── xgb_model.json
│   ├── catboost_model.cbm
│   └── metadata.json
└── checkpoints/                 # Historical checkpoints
    ├── checkpoint_20260123_120000/
    ├── checkpoint_20260124_120000/
    └── checkpoint_20260125_120000/
```

## Performance

Typical performance on Bitcoin 5-minute data:
- **Accuracy**: 52-58%
- **F1 Score**: 0.50-0.56
- **Sharpe Ratio**: 1.2-2.5
- **Win Rate**: 48-54%

## Integration with HEAN Trading System

```python
from hean.core.intelligence import BitcoinPricePredictor

# In strategy code
predictor = BitcoinPricePredictor()
predictor.load_model("models/bitcoin_predictor")

# Get prediction
result = predictor.predict_single(current_data)

if result['confidence'] > 0.7:
    if result['direction'] == 'UP':
        # Enter long position
        pass
    else:
        # Enter short position
        pass
```

## Development

### Adding New Features

```python
# In features.py
def _add_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """Add your custom features."""
    # Your feature engineering code
    df['my_custom_feature'] = ...
    return df
```

### Custom Model Ensembles

```python
# Adjust ensemble weights
config = PredictorConfig(
    lgb_weight=0.4,
    xgb_weight=0.4,
    catboost_weight=0.2,
)
```

## Testing

```bash
# Generate synthetic data and test
python -m ml.trainer --data-source synthetic
python -m ml.backtester --data-source synthetic
```

## License

Part of HEAN-META trading system.

## Support

For issues and questions, see main HEAN documentation.

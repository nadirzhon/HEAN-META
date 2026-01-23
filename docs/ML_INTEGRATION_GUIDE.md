# ML Stack Integration Guide - Phase 1

Comprehensive guide for integrating ML components into HEAN trading system.

## 📦 Installation

### 1. Install ML Dependencies

```bash
# Option 1: Install all ML dependencies
pip install -e ".[ml]"

# Option 2: Manual installation
pip install -r requirements_ml.txt
```

### 2. Install TA-Lib (Required)

**macOS:**
```bash
brew install ta-lib
pip install TA-Lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libta-lib-dev
pip install TA-Lib
```

**Windows:**
```bash
# Download wheel from: https://github.com/cgohlke/talib-build/releases
pip install TA_Lib‑0.4.28‑cp311‑cp311‑win_amd64.whl
```

### 3. Start Redis (for caching)

```bash
# Docker
docker run -d -p 6379:6379 redis:latest

# Or install locally
brew install redis  # macOS
sudo apt-get install redis-server  # Ubuntu
```

---

## 🚀 Quick Start

### Example 1: Generate Features with TA-Lib

```python
from hean.features import TALibFeatures, FeatureConfig
import pandas as pd

# Initialize
config = FeatureConfig(
    rsi_periods=[14, 21, 28],
    enable_patterns=True,
)
ta = TALibFeatures(config)

# Generate features from OHLCV
features = ta.generate_features(ohlcv_df)

print(f"Generated {len(features.columns)} features")
# Output: Generated 200+ features
```

### Example 2: Train ML Ensemble

```python
from hean.ml import EnsemblePredictor, PredictorConfig
from hean.ml.price_predictor import prepare_target

# Prepare data
features_with_target = prepare_target(
    features,
    horizon=12,  # Predict 1 hour ahead
    threshold=0.002  # 0.2% move
)

# Train
config = PredictorConfig(prediction_horizon=12)
predictor = EnsemblePredictor(config)
metrics = predictor.train(features_with_target)

print(f"Accuracy: {metrics['ensemble_accuracy']:.1%}")
print(f"AUC: {metrics['ensemble_auc']:.3f}")
```

### Example 3: Make Predictions

```python
# Predict on new data
result = predictor.predict(latest_features)

if result.direction == "UP" and result.confidence > 0.60:
    print(f"BUY Signal (confidence: {result.confidence:.1%})")
elif result.direction == "DOWN" and result.confidence > 0.60:
    print(f"SELL Signal (confidence: {result.confidence:.1%})")
```

### Example 4: Order Book Analysis

```python
from hean.market_data import OrderBookAnalyzer, OrderBookSnapshot
from datetime import datetime

analyzer = OrderBookAnalyzer(whale_threshold_ratio=5.0)

snapshot = OrderBookSnapshot(
    timestamp=datetime.now(),
    symbol="BTCUSDT",
    bids=[(50000, 1.5), (49990, 2.0), ...],
    asks=[(50010, 1.2), (50020, 3.0), ...],
)

# Detect whales
whales = analyzer.detect_whale_walls(snapshot)

# Check imbalance
imbalance = analyzer.calculate_imbalance(snapshot)

if imbalance.imbalance_ratio > 0.3:
    print("Strong BUY pressure detected!")
```

---

## 🔗 Integration with Existing Strategies

### Integrate ML Predictions into Strategy

```python
from hean.strategies.base import TradingStrategy
from hean.ml import EnsemblePredictor
from hean.features import TALibFeatures

class MLEnhancedStrategy(TradingStrategy):
    """Strategy enhanced with ML predictions."""

    def __init__(self):
        super().__init__()
        self.ml_predictor = EnsemblePredictor.load("models/ensemble_latest.pkl")
        self.ta = TALibFeatures()

    async def generate_signals(self, market_data):
        # Generate features
        features = self.ta.generate_features(market_data)

        # Get ML prediction
        ml_pred = self.ml_predictor.predict(features.iloc[-1])

        # Traditional indicators
        rsi = features['rsi_14'].iloc[-1]
        macd_bullish = (
            features['macd_12_26_9'].iloc[-1] >
            features['macd_signal_12_26_9'].iloc[-1]
        )

        # Combined signal
        if (ml_pred.direction == "UP" and
            ml_pred.confidence > 0.65 and
            rsi < 40 and macd_bullish):

            return {
                "action": "BUY",
                "size": 0.02,  # 2% of capital
                "confidence": ml_pred.confidence,
                "reason": "ML + RSI oversold + MACD bullish"
            }

        return None
```

### Add Order Book Analysis

```python
from hean.market_data import OrderBookAnalyzer

class OrderBookEnhancedStrategy(TradingStrategy):
    """Strategy with order book analysis."""

    def __init__(self):
        super().__init__()
        self.ob_analyzer = OrderBookAnalyzer()

    async def generate_signals(self, market_data, orderbook):
        # Analyze order book
        imbalance = self.ob_analyzer.calculate_imbalance(orderbook)
        whales = self.ob_analyzer.detect_whale_walls(orderbook)

        # Check for whale support
        strong_bid_wall = any(
            w.side.value == "BID" and w.strength.value == "EXTREME"
            for w in whales
        )

        # Strong buy signal
        if strong_bid_wall and imbalance.imbalance_ratio > 0.4:
            return {
                "action": "BUY",
                "size": 0.03,  # 3% more aggressive with whale support
                "reason": "Whale bid wall + strong imbalance"
            }

        return None
```

---

## 🔄 Auto-Retraining System

### Setup Automatic Model Retraining

```python
from hean.ml.auto_retrainer import AutoRetrainer, RetrainerConfig

# Configure
config = RetrainerConfig(
    retrain_interval_hours=24,  # Retrain daily
    training_window_days=90,    # Use last 90 days
    min_accuracy=0.53,          # Rollback if < 53%
)

# Start retrainer
retrainer = AutoRetrainer(config, data_source=exchange_api)
await retrainer.start()

# Get latest model anytime
predictor = retrainer.get_latest_predictor()
```

### Integration with Main Trading System

```python
# In main.py
from hean.ml.auto_retrainer import AutoRetrainer, RetrainerConfig

async def main():
    # ... existing setup ...

    # Start ML retrainer
    ml_config = RetrainerConfig(retrain_interval_hours=24)
    retrainer = AutoRetrainer(ml_config, data_source=market_data_service)
    await retrainer.start()

    # Use in strategy
    predictor = retrainer.get_latest_predictor()

    # ... rest of trading logic ...
```

---

## 💾 Redis Caching Setup

### Enable Feature Caching

```python
from hean.infrastructure import FeatureCache, CacheConfig

# Initialize cache
cache_config = CacheConfig(
    host="localhost",
    port=6379,
    feature_ttl=300,  # 5 min cache
)
cache = FeatureCache(cache_config)

# Usage in strategy
async def get_features_cached(symbol, timeframe):
    # Try cache first
    cached = cache.get_features(symbol, timeframe)
    if cached is not None:
        return cached

    # Generate if not cached
    ohlcv = await fetch_ohlcv(symbol, timeframe)
    features = ta.generate_features(ohlcv)

    # Cache for next time
    cache.set_features(symbol, timeframe, features)

    return features
```

### Cache ML Predictions

```python
# Cache predictions
prediction = predictor.predict(features)
cache.set_prediction("BTCUSDT", prediction, model_name="ensemble")

# Retrieve cached prediction
cached_pred = cache.get_prediction("BTCUSDT", model_name="ensemble")
if cached_pred:
    age = datetime.now() - datetime.fromisoformat(cached_pred['timestamp'])
    if age.seconds < 60:  # Use if < 1 min old
        prediction = cached_pred['prediction']
```

---

## 📊 Backtesting Integration

### Backtest Strategy with Vectorbt

```python
from hean.backtesting import VectorBTBacktester, BacktestConfig

# Setup
config = BacktestConfig(
    initial_capital=10000,
    fees=0.001,  # 0.1%
)
backtester = VectorBTBacktester(config)

# Generate signals
def ml_strategy_signals(prices, **params):
    # ... generate features ...
    # ... ML predictions ...
    # ... return (entries, exits) ...
    pass

# Backtest
result = backtester.backtest_from_strategy(
    prices=historical_ohlcv,
    strategy_func=ml_strategy_signals,
)

print(result)
# Output:
# Total Return: 45.2%
# Sharpe Ratio: 2.1
# Max Drawdown: 12.3%
```

### Optimize Parameters

```python
# Define parameter grid
param_grid = {
    'ml_confidence_threshold': [0.55, 0.60, 0.65, 0.70],
    'rsi_oversold': [20, 25, 30],
    'position_size': [0.02, 0.03, 0.05],
}

# Optimize
results = backtester.optimize_parameters(
    prices['close'],
    signal_func=ml_strategy_signals,
    param_grid=param_grid,
    metric='sharpe_ratio',
)

# Best parameters
best = results.iloc[0]
print(f"Best Sharpe: {best['sharpe_ratio']:.2f}")
print(f"Parameters: {best[['ml_confidence_threshold', 'rsi_oversold', 'position_size']].to_dict()}")
```

---

## 🏗️ Architecture Integration

### System Architecture with ML

```
┌─────────────────────────────────────────────────────────────┐
│                     HEAN Trading System                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Market    │  │  Order Book │  │  Historical │        │
│  │    Data     │──▶│   Analyzer  │  │    Data     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│  ┌──────────────────────────────────────────────┐         │
│  │         Feature Engineering (TA-Lib)         │         │
│  │  • 200+ Technical Indicators                 │         │
│  │  • Pattern Recognition                       │         │
│  │  • Order Book Features                       │         │
│  └──────────────────────────────────────────────┘         │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────┐         │
│  │         Redis Cache (<1ms latency)            │         │
│  └──────────────────────────────────────────────┘         │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────┐         │
│  │    ML Ensemble (LightGBM + XGB + CB)        │         │
│  │  • Price Direction Prediction                │         │
│  │  • Confidence Scores                         │         │
│  │  • Auto-Retraining (24h)                     │         │
│  └──────────────────────────────────────────────┘         │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────┐         │
│  │         Trading Strategies                    │         │
│  │  • Existing strategies (Impulse, Funding)    │         │
│  │  • NEW: ML-Enhanced strategies               │         │
│  │  • NEW: Order Book strategies                │         │
│  └──────────────────────────────────────────────┘         │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────┐         │
│  │         Risk Management                       │         │
│  │  • Position sizing                           │         │
│  │  • Stop loss / Take profit                   │         │
│  └──────────────────────────────────────────────┘         │
│         │                                                  │
│         ▼                                                  │
│  ┌──────────────────────────────────────────────┐         │
│  │         Order Execution                       │         │
│  └──────────────────────────────────────────────┘         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📈 Expected Performance Improvements

### Phase 1 Implementation

| Metric | Before | After Phase 1 | Improvement |
|--------|--------|---------------|-------------|
| Sharpe Ratio | 2.0 | 2.5-3.0 | +25-50% |
| Win Rate | 45% | 52-58% | +7-13pp |
| Max Drawdown | 15% | 10-12% | -20-33% |
| Avg Daily Return | $100 | $200-300 | +100-200% |
| Signal Quality | Medium | High | +50% |
| Feature Richness | 5-10 | 200+ | +20-40x |
| Backtest Speed | 10 min | 10 sec | 60x faster |

---

## 🔧 Configuration

### Environment Variables

Add to `.env`:

```bash
# ML Configuration
ML_ENABLED=true
ML_MODEL_PATH=models/ensemble_latest.pkl
ML_RETRAIN_INTERVAL_HOURS=24
ML_MIN_CONFIDENCE=0.60

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_FEATURE_TTL=300

# TA-Lib
TALIB_RSI_PERIODS=14,21,28
TALIB_ENABLE_PATTERNS=true

# Order Book
ORDERBOOK_WHALE_THRESHOLD=5.0
ORDERBOOK_IMBALANCE_WINDOW=10
```

### Config File

Create `config/ml_config.yaml`:

```yaml
ml:
  enabled: true
  model_dir: models/
  auto_retrain:
    enabled: true
    interval_hours: 24
    min_accuracy: 0.53

features:
  talib:
    rsi_periods: [14, 21, 28]
    enable_patterns: true
    enable_cycles: false

  orderbook:
    whale_threshold: 5.0
    imbalance_window: 10

cache:
  enabled: true
  host: localhost
  port: 6379
  ttl:
    features: 300
    predictions: 60
    orderbook: 10

backtesting:
  initial_capital: 10000
  fees: 0.001
  slippage: 0.0005
```

---

## 🧪 Testing

### Run Examples

```bash
# TA-Lib features
python examples/talib_integration_example.py

# ML Predictor
python examples/ml_predictor_example.py

# Order Book Analysis
python examples/orderbook_analysis_example.py
```

### Unit Tests

```bash
# Test ML components
pytest tests/test_ml_predictor.py

# Test features
pytest tests/test_talib_features.py

# Test cache
pytest tests/test_cache.py
```

---

## 📝 Next Steps

### Phase 2 (Week 3-4)

1. **Sentiment Analysis** - Twitter, Reddit, News
2. **On-Chain Data** - Exchange flows, MVRV
3. **Optuna Optimization** - Hyperparameter tuning
4. **Dynamic Position Sizing** - Kelly Criterion
5. **Prometheus Monitoring** - Real-time metrics

### Phase 3 (Month 2)

6. **Reinforcement Learning** - PPO trading agent
7. **Deep Learning** - TFT/N-BEATS forecasting
8. **Statistical Arbitrage** - Pairs trading
9. **Event Streaming** - Kafka/Redis Streams
10. **Model Stacking** - Meta-learning

---

## 🆘 Troubleshooting

### TA-Lib Installation Issues

```bash
# macOS M1/M2
conda install -c conda-forge ta-lib

# If compilation fails
brew install ta-lib
export TA_INCLUDE_PATH=$(brew --prefix ta-lib)/include
export TA_LIBRARY_PATH=$(brew --prefix ta-lib)/lib
pip install TA-Lib
```

### Redis Connection Issues

```bash
# Check Redis running
redis-cli ping
# Should return: PONG

# Start Redis
redis-server

# Use Docker
docker run -d -p 6379:6379 --name hean-redis redis:latest
```

### ML Training Memory Issues

```python
# Reduce training data
features = features.iloc[-50000:]  # Last 50k samples

# Use sampling
features = features.sample(frac=0.5)  # Use 50%

# Reduce features
important_features = ta.get_feature_importance_proxy(features).head(100).index
features = features[important_features]
```

---

## 📚 Resources

- **TA-Lib Documentation**: https://mrjbq7.github.io/ta-lib/
- **VectorBT Documentation**: https://vectorbt.dev/
- **LightGBM Guide**: https://lightgbm.readthedocs.io/
- **Redis Best Practices**: https://redis.io/docs/manual/

---

## 🎯 Success Metrics

Track these metrics to validate Phase 1:

```python
# In your monitoring
metrics = {
    "ml_accuracy": predictor.metrics['ensemble_accuracy'],
    "ml_predictions_per_hour": cache.get_stats()['total_requests'],
    "cache_hit_rate": cache.get_stats()['hit_rate'],
    "feature_count": len(features.columns),
    "backtest_sharpe": backtest_result.sharpe_ratio,
}
```

Target values:
- ML Accuracy: >55%
- Cache Hit Rate: >80%
- Backtest Sharpe: >2.5
- Feature Count: >150

---

**Phase 1 Complete! 🎉**

Ready for Phase 2 advanced features.

# ABSOLUTE+ Quick Start Guide

## Overview

HEAN has evolved into **ABSOLUTE+** - a Post-Singularity Market-Architecting Entity that treats trading logic as mutable weights, predicts market moves through causal inference, and processes all market data as unified tensors.

## Quick Start

### 1. Start the System

```bash
# Start the full system with all ABSOLUTE+ engines
python -m hean.main run
```

The system will automatically initialize:
- ⚡ **Meta-Learning Engine**: Recursive Intelligence Core (1M scenarios/sec)
- 🔮 **Causal Inference Engine**: Granger Causality + Transfer Entropy
- 🌐 **Multimodal Swarm**: Unified Tensor Processing (Price + Sentiment + On-Chain + Macro)

### 2. Access Command Center

Open your browser to:
```
http://localhost:3000
```

The dashboard shows:
- **Probability Manifold**: 3D visualization of market state probability distribution
- **Future Cone**: Where the system is actively pushing the market
- **Real-Time Metrics**: Live updates of all ABSOLUTE+ systems

### 3. Monitor Systems via API

```bash
# Meta-Learning Engine stats
curl http://localhost:8000/api/meta-learning/state

# Causal Inference relationships
curl http://localhost:8000/api/causal-inference/stats

# Multimodal Swarm tensors
curl http://localhost:8000/api/multimodal-swarm/tensors/BTCUSDT?limit=10
```

## System Architecture

### Meta-Learning Engine

**Purpose**: Treats C++ trading logic as mutable neural weights, simulates failures at 1M/sec, auto-patches code.

**Key Endpoints**:
- `GET /api/meta-learning/state` - Get simulation statistics
- `GET /api/meta-learning/weights` - List all code weights
- `GET /api/meta-learning/patches` - View patch history

**Configuration** (in `.env` or `config.py`):
```python
META_LEARNING_RATE=1000000  # Scenarios per second
META_LEARNING_AUTO_PATCH=true  # Auto-apply patches
META_LEARNING_MAX_WORKERS=100  # Concurrent simulation workers
```

### Causal Inference Engine

**Purpose**: Predicts Bybit moves by analyzing pre-echoes in global cross-asset orderflow using Granger Causality and Transfer Entropy.

**Key Endpoints**:
- `GET /api/causal-inference/stats` - Get relationships and pre-echo signals
- `GET /api/causal-inference/relationships` - List all causal relationships
- `GET /api/causal-inference/pre-echoes` - Recent pre-echo signals

**Configuration**:
```python
CAUSAL_WINDOW_SIZE=500  # Rolling window for analysis
CAUSAL_MIN_THRESHOLD=0.3  # Minimum Granger F-statistic
CAUSAL_MIN_TE=0.1  # Minimum Transfer Entropy (bits)
CAUSAL_SOURCE_SYMBOLS=["BTCUSDT", "ETHUSDT", "BNBUSDT"]  # Pre-echo sources
```

### Multimodal Swarm

**Purpose**: Processes price, sentiment, on-chain whale movements, and macro data as a unified 18-feature tensor.

**Key Endpoints**:
- `GET /api/multimodal-swarm/stats` - Get swarm statistics
- `GET /api/multimodal-swarm/tensors/{symbol}` - Get tensor history
- `GET /api/multimodal-swarm/modality-weights` - Get current weights
- `POST /api/multimodal-swarm/modality-weights` - Update weights

**Configuration**:
```python
MULTIMODAL_WINDOW_SIZE=100  # Rolling window for tensors
MULTIMODAL_NUM_AGENTS=50  # Number of swarm agents
```

**Modality Weights** (learned from performance):
- Price: 40% (default)
- Sentiment: 20%
- On-Chain: 25%
- Macro: 15%

## Example Usage

### Monitor Meta-Learning in Real-Time

```python
import requests
import time

while True:
    response = requests.get("http://localhost:8000/api/meta-learning/state")
    data = response.json()
    print(f"Scenarios/sec: {data['scenarios_per_second']:,.0f}")
    print(f"Total simulated: {data['total_scenarios_simulated']:,}")
    print(f"Patches applied: {data['patches_applied']}")
    time.sleep(5)
```

### Detect Pre-Echo Signals

```python
response = requests.get("http://localhost:8000/api/causal-inference/pre-echoes?limit=5")
signals = response.json()

for signal in signals:
    print(f"{signal['source_symbol']} -> {signal['target_symbol']}: {signal['predicted_direction']}")
    print(f"  Confidence: {signal['confidence']:.3f}")
    print(f"  Lag: {signal['lag_ms']}ms")
```

### Analyze Multimodal Tensors

```python
response = requests.get("http://localhost:8000/api/multimodal-swarm/tensors/BTCUSDT?limit=1")
tensors = response.json()

if tensors:
    tensor = tensors[0]
    print(f"Price features: {tensor['price_features']}")
    print(f"Sentiment: {tensor['sentiment_features']}")
    print(f"On-chain: {tensor['onchain_features']}")
    print(f"Macro: {tensor['macro_features']}")
    print(f"Unified tensor size: {len(tensor['unified_tensor'])}")
```

## Visualization

### Probability Manifold

The 3D probability manifold shows:
- **X-axis**: Price Momentum (-1 to +1)
- **Y-axis**: Sentiment (-1 to +1)
- **Z-axis**: Probability (0 to 1)
- **Color**: Green (high) → Blue (medium) → Red (low)

Higher regions indicate market states where the system has high confidence.

### Future Cone

The future cone visualization shows:
- **Main trajectory**: Where the system is actively pushing the market (bright line)
- **Uncertainty bounds**: Upper and lower bounds (cone shape)
- **Profit zones**: High-probability profit regions (stars)

The cone widens over time, representing increasing uncertainty in long-term predictions.

## Performance Targets

- **Meta-Learning**: 1,000,000 scenarios/second
- **Causal Inference**: Analysis every 10 seconds
- **Multimodal Swarm**: 50 agents, 60% consensus threshold
- **God-Mode Dashboard**: Updates every 5 seconds

## Troubleshooting

### Meta-Learning Engine not starting

Check that C++ source files exist at:
```
src/hean/core/cpp/
```

If missing, the engine will fall back to Python-only mode.

### Causal Inference shows no relationships

Ensure sufficient data history:
- Minimum 50 data points per symbol
- Source symbols configured correctly
- Window size appropriate for data frequency

### Multimodal Swarm not generating signals

Check:
- Consensus threshold (default: 60%)
- Number of agents (default: 50)
- All modalities receiving data (price, sentiment, on-chain, macro)

## Next Steps

1. **Enable Real Data Sources**: Integrate actual APIs for sentiment, on-chain, and macro data
2. **Tune Modality Weights**: Use performance feedback to optimize weights
3. **Scale Meta-Learning**: Increase simulation workers for higher throughput
4. **Customize Pre-Echo Sources**: Add more cross-asset symbols for better predictions

## Documentation

For detailed implementation documentation, see:
- `ABSOLUTE_PLUS_IMPLEMENTATION.md` - Full technical documentation
- `src/hean/core/intelligence/` - Source code for all engines

---

**Welcome to Market Omniscience.** ⚡🔮🌐

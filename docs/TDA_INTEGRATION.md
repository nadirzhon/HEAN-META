# Topological Data Analysis (TDA) Integration

## Overview

Topological Data Analysis (TDA) has been deeply integrated into the HEAN core execution flow. The topology is now the primary sense through which the system perceives market reality.

## Architecture

### 1. C++ Layer (Core Engine)

#### TDA_Engine (`src/hean/core/cpp/TDA_Engine.h`, `.cpp`)
- **Persistent Homology Computation**: Computes Vietoris-Rips complex persistence on L2 orderbook point clouds
- **Topology Score Metrics**:
  - `stability_score`: Market structural stability (0-1)
  - `connectivity_score`: Manifold connectivity (0-1)
  - `curvature_score`: Riemannian curvature proxy
  - `is_disconnected`: Boolean flag for disconnected manifold
  - `num_holes`: Number of topological holes (1D homology)
- **Background Thread**: Continuously updates topology scores for all symbols

#### FastWarden (`src/hean/core/cpp/FastWarden.cpp`)
- **L2 Orderbook Integration**: Continuously updates Persistent Homology map of L2 Orderbook
- **Background Processing**: Dedicated thread for topology updates
- **Python Interface**: Exposed via pybind11 bindings

### 2. Python Layer (Integration)

#### Swarm Intelligence (`src/hean/core/intelligence/swarm.py`)
- **Mandatory Topology Score**: All agents receive `topology_score` as a mandatory input feature
- **Structural Collapse Detection**: If `topology_score < 0.3`, signal confidence is decreased by 50% regardless of momentum
- **Topology-Driven Logic**: Topology is the primary filter for all trading signals

#### Smart Limit Executor (`src/hean/exchange/executor.py`)
- **Geometric Slippage Prediction**: Uses Riemannian curvature of the orderbook to predict slippage BEFORE sending orders
- **Smart-Limit Mode**: If predicted slippage > threshold (1%), automatically switches to aggressive limit pricing
- **Real-time Adaptation**: Continuously updates predictions based on orderbook geometry

#### Topological Watchdog (`src/hean/core/intelligence/topological_watchdog.py`)
- **Disconnection Detection**: Monitors market manifold connectivity continuously
- **Automatic Trading Halt**: If manifold becomes disconnected (flash-crash or API lag), immediately halts all trading
- **Threshold-Based**: Halt triggers if disconnection persists > 2 seconds

### 3. Shared Memory Optimization

#### TDA Point Cloud Support (`src/hean/hft/shared_memory.py`)
- **Lock-Free Access**: Optimized structure for TDA point-cloud processing
- **Zero-Copy**: Direct memory mapping for ultra-low latency
- **Point Cloud Entry**: Specialized structure for orderbook manifold data

### 4. API & Visualization

#### REST Endpoints (`src/hean/api/routers/graph_engine.py`)
- `GET /api/graph-engine/topology/score`: Market topology score
- `GET /api/graph-engine/topology/manifold/{symbol}`: 3D manifold data for visualization
- `GET /api/graph-engine/topology/watchdog`: Watchdog status

#### 3D Visualization (`web/topological_manifold_visualization.html`)
- **Three.js Integration**: Real-time 3D manifold visualization
- **Topological Holes Overlay**: Visual representation of persistence barcodes
- **Orderbook Heatmap Overlay**: Combined with orderbook heatmap for comprehensive view
- **Interactive Controls**: Camera rotation, wireframe toggle, symbol selection

## Usage

### Building the C++ Components

```bash
cd src/hean/core/cpp
mkdir -p build
cd build
cmake ..
make
```

### Python Usage

```python
from hean.core.intelligence.swarm import SwarmIntelligence
from hean.core.intelligence.topological_watchdog import TopologicalWatchdog
from hean.exchange.executor import SmartLimitExecutor

# Initialize swarm with topology integration
swarm = SwarmIntelligence(bus)
await swarm.start()

# Register agent - topology_score will be automatically included
agent = swarm.register_agent("agent_1", "strategy_1")
signal = swarm.evaluate_signal(
    agent_id="agent_1",
    symbol="BTCUSDT",
    side="buy",
    momentum_score=0.8,
    entry_price=50000.0
)
# Signal confidence automatically adjusted by topology_score

# Initialize watchdog
watchdog = TopologicalWatchdog(bus)
await watchdog.start()
# Trading will be automatically halted if manifold disconnects

# Use Smart Limit Executor for geometric slippage prediction
executor = SmartLimitExecutor(bus, bybit_http)
await executor.start()
# Executor automatically predicts slippage and switches to Smart-Limit mode if needed
```

## Key Features

### 1. Structural Collapse Detection
- **Threshold**: `topology_score < 0.3` indicates structural collapse
- **Action**: Signal confidence reduced by 50% regardless of momentum
- **Purpose**: Prevent trading during market instability

### 2. Geometric Slippage Prediction
- **Method**: Riemannian curvature computation from orderbook geometry
- **Formula**: `K ≈ (d²y/dx²) / (1 + (dy/dx)²)^(3/2)`
- **Decision**: If predicted slippage > 1%, switch to Smart-Limit mode

### 3. Topological Watchdog
- **Monitoring**: Continuous manifold connectivity checks
- **Trigger**: Disconnection detected for > 2 seconds
- **Action**: Immediate trading halt via `STOP_TRADING` event

### 4. Real-time Visualization
- **3D Manifold**: Point cloud visualization of orderbook structure
- **Topological Holes**: Overlay of persistence barcodes (1D homology)
- **Orderbook Heatmap**: Combined view for comprehensive market understanding

## Integration Points

### Main Trading System
The TDA components should be integrated into the main trading system:

```python
# In main.py or TradingSystem class
from hean.core.intelligence.swarm import SwarmIntelligence
from hean.core.intelligence.topological_watchdog import TopologicalWatchdog

class TradingSystem:
    def __init__(self):
        # ... existing initialization ...
        
        # TDA Integration
        self._swarm = SwarmIntelligence(self._bus)
        self._topological_watchdog = TopologicalWatchdog(self._bus)
    
    async def start(self):
        # ... existing start logic ...
        
        # Start TDA components
        await self._swarm.start()
        await self._topological_watchdog.start()
```

### Orderbook Updates
Orderbook updates must feed into FastWarden:

```python
# When receiving orderbook updates
async def handle_orderbook_update(event):
    orderbook_data = event.data["orderbook"]
    symbol = orderbook_data["symbol"]
    
    # Update FastWarden for TDA computation
    if fast_warden:
        fast_warden.update_orderbook(
            symbol,
            bid_prices, bid_sizes,
            ask_prices, ask_sizes
        )
```

## Performance Considerations

- **Update Frequency**: Topology scores updated every 100ms
- **Background Thread**: Separate thread for TDA computation to avoid blocking
- **Lock-Free Access**: Shared memory optimized for zero-copy point cloud access
- **Caching**: Topology scores cached and updated asynchronously

## Future Enhancements

1. **Multi-Dimensional Homology**: Extend to 2D homology (voids)
2. **Machine Learning Integration**: Use topology features for ML models
3. **Distributed TDA**: Scale TDA computation across multiple symbols
4. **Real-time Barcode Updates**: Stream persistence barcodes to UI in real-time

## References

- **Persistent Homology**: Mathematical framework for TDA
- **Vietoris-Rips Complex**: Simplicial complex construction method
- **Riemannian Geometry**: Curvature-based slippage prediction
- **Topological Data Analysis**: General theory and applications
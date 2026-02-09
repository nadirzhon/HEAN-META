# HEAN Absolute+ System Implementation

## Overview

The Absolute+ system synthesizes the Metamorphic Core (C++) and Causal Brain (Python) into a self-aware economic engine designed for 100% autonomy and exponential profit growth from a $300 seed capital.

## Architecture

### 1. Deep Integration: Brain ↔ Body Connection

**Components:**
- **CausalBrain** (Python): The "Brain" that learns from market patterns
- **MetamorphicCore** (C++): The "Body" that executes trading logic
- **Communication**: ZeroMQ PUB/SUB pattern (100ms mutation signals)

**Location:**
- `src/hean/core/absolute_plus/integration.py`: Integration layer
- `src/hean/core/python/causal_brain.py`: Python Brain implementation
- `src/hean/core/cpp/MetamorphicCore.cpp`: C++ Body implementation

**How it works:**
1. CausalBrain analyzes market patterns and generates Logic Mutation signals every 100ms
2. Signals are sent via ZeroMQ to MetamorphicCore
3. MetamorphicCore adapts execution context (risk, spread thresholds, execution speed)
4. Performance feedback flows back to Brain for continuous learning

### 2. Infinite Scaling: Tensorized Orderflow

**Component:** `src/hean/core/absolute_plus/tensorized_orderflow.py`

**Features:**
- Processes ALL 200+ Bybit perpetual pairs as a single multi-dimensional matrix
- Tensor shape: `(num_symbols, num_features, time_window)`
- Vectorized operations across all pairs simultaneously
- Features tracked: price, volume, spread, volatility, returns, orderflow, momentum
- Automatic symbol discovery from Bybit API

**Usage:**
```python
from hean.core.absolute_plus.tensorized_orderflow import TensorizedOrderflow

tensorized = TensorizedOrderflow(bus=bus, bybit_client=bybit_client)
await tensorized.initialize()  # Fetches all perpetual symbols
await tensorized.start()

# Get tensor
tensor = tensorized.get_tensor()  # Shape: (200+, 10, 100)

# Get correlation matrix
correlation_matrix = tensorized.compute_correlation_matrix()

# Get top correlated pairs
top_pairs = tensorized.get_top_correlated_pairs(top_k=50)
```

### 3. Self-Healing Mechanism: Autonomous Proxy Routing

**Component:** `src/hean/core/absolute_plus/self_healing_proxy.py`

**Features:**
- Monitors Bybit API latency continuously
- Automatically switches to fastest proxy when latency exceeds threshold (default 200ms)
- Integrates with Phase 19 proxy sharding system
- Proxy switch cooldown (60s) to prevent rapid switching
- Real-time latency tracking and statistics

**Usage:**
```python
from hean.core.absolute_plus.self_healing_proxy import SelfHealingProxyRouter
from hean.core.network.proxy_sharding import ProxyConfig, ProxyType

proxy_configs = [
    ProxyConfig(
        id="proxy1",
        type=ProxyType.RESIDENTIAL,
        host="proxy1.example.com",
        port=8080,
    ),
]

router = SelfHealingProxyRouter(
    bus=bus,
    bybit_client=bybit_client,
    proxy_configs=proxy_configs,
    latency_threshold_ms=200.0,
)
await router.start()
```

### 4. Evolution Terminal UI

**Files:**
- `control-center/` (integrated UI; see components)

**Features:**
- **Causal Web**: 3D visualization of asset correlations using Three.js
  - Nodes represent assets (200+ perpetual pairs)
  - Edges represent correlations (green=positive, red=negative)
  - Interactive 3D navigation with OrbitControls
  - Real-time updates every 2 seconds

- **System Intelligence Quotient (SIQ)**: Real-time metric display
  - SIQ score (0-100)
  - Learning velocity
  - Pattern recognition rate
  - Adaptation rate
  - Win rate

- **Sidebar Statistics**:
  - Tensorized Orderflow stats (number of symbols, top correlations)
  - Self-Healing Proxy stats (current proxy, latency, switches)

**Access:**
Navigate to `http://localhost:3000`

## API Endpoints

All endpoints are under `/api/absolute-plus/`:

- `GET /api/absolute-plus/siq`: Get System Intelligence Quotient
- `GET /api/absolute-plus/causal-web`: Get Causal Web data for visualization
- `GET /api/absolute-plus/tensorized-orderflow/stats`: Get tensorized orderflow statistics
- `GET /api/absolute-plus/self-healing-proxy/stats`: Get proxy router statistics

## Integration with Main System

To integrate Absolute+ into the main trading system:

```python
from hean.core.absolute_plus import AbsolutePlusIntegration, TensorizedOrderflow, SelfHealingProxyRouter

# In TradingSystem.__init__()
self._absolute_plus = AbsolutePlusIntegration(bus=self._bus)
self._tensorized_orderflow = TensorizedOrderflow(bus=self._bus, bybit_client=self._bybit_http)
self._self_healing_proxy = SelfHealingProxyRouter(
    bus=self._bus,
    bybit_client=self._bybit_http,
    proxy_configs=proxy_configs,  # Optional
)

# In TradingSystem.start()
await self._absolute_plus.start()
await self._tensorized_orderflow.start()
await self._self_healing_proxy.start()
```

## Dependencies

**Python:**
- `pyzmq>=25.0.0`: ZeroMQ for Brain-Body communication
- `numpy>=1.24.0`: Tensor operations (already in dependencies)

**C++:**
- ZeroMQ library (libzmq3-dev on Linux, zeromq on macOS)
- pybind11 (for Python bindings - already in dependencies)

**Web:**
- Three.js (loaded via CDN in evolution-terminal.html)
- OrbitControls (for 3D navigation)

## Configuration

**ZeroMQ Endpoint:**
- Default: `ipc:///tmp/hean_metamorphic`
- Configurable in `AbsolutePlusIntegration.__init__()`

**Latency Threshold:**
- Default: 200ms
- Configurable in `SelfHealingProxyRouter.__init__()`

**Tensor Window:**
- Default: 100 ticks per symbol
- Configurable in `TensorizedOrderflow.__init__()`

## Status

✅ **Deep Integration**: CausalBrain ↔ MetamorphicCore via ZeroMQ (100ms signals)  
✅ **Tensorized Orderflow**: Processes 200+ pairs as single matrix  
✅ **Self-Healing Proxy**: Autonomous latency-based proxy switching  
✅ **Evolution Terminal UI**: Causal Web 3D visualization + SIQ display  
⚠️ **Note**: Python bindings for MetamorphicCore need to be added to `python_bindings.cpp`

## Next Steps

1. Add Python bindings for MetamorphicCore in `src/hean/core/cpp/python_bindings.cpp`
2. Integrate Absolute+ components into `TradingSystem` class in `src/hean/main.py`
3. Add proxy configurations to settings/config
4. Test ZeroMQ communication between Brain and Body
5. Verify tensorized orderflow with real Bybit data
6. Test proxy switching under latency conditions

## Notes

- The CausalBrain sends mutation signals every 100ms via ZeroMQ PUB socket
- MetamorphicCore receives signals via ZeroMQ SUB socket
- Tensorized Orderflow automatically discovers all perpetual symbols from Bybit API
- Self-Healing Proxy monitors latency every 5 seconds
- Evolution Terminal polls API every 2 seconds for real-time updates

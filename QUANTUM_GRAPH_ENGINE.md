# Quantum Graph Engine & Visual Singularity

## Overview

This document describes the **Quantum Computing & AI Research** components integrated into the HEAN trading system. The system includes:

1. **C++ Graph Correlation Core**: Real-time adjacency matrix for 50+ crypto assets with lead-lag detection
2. **ONNX Model Inference**: TensorRT/ONNX-based volatility spike prediction with circuit breaker
3. **Generative Strategy Sandbox**: LLM-powered code generation with restricted sandbox execution
4. **Visual Singularity (Eureka UI)**: 3D force-directed graph visualization with energy flow and future shadow

## Architecture

### 1. Graph Correlation Core (C++)

**Location**: `src/hean/core/cpp/GraphEngine.cpp`

Real-time adjacency matrix implementation that:
- Tracks 50+ crypto assets simultaneously
- Calculates Pearson correlation coefficients in real-time
- Detects lead-lag relationships using cross-correlation analysis
- Identifies market leader and laggard assets
- Outputs high-dimensional feature vectors for neural network input

**Key Features**:
- Thread-safe with mutex protection
- Rolling window correlation calculation (default 100 samples)
- Efficient matrix operations optimized for real-time performance
- C interface for Python bindings via pybind11

**Usage**:
```python
from hean.core.intelligence.graph_engine import GraphEngineWrapper

engine = GraphEngineWrapper(bus, symbols=["BTCUSDT", "ETHUSDT", ...])
await engine.start()

# Get feature vector for neural network
feature_vector = engine.get_feature_vector(size=5000)

# Get current market leader
leader = engine.get_current_leader()

# Get lead-lag relationship
lead_lag = engine.get_lead_lag("BTCUSDT", "ETHUSDT")  # Positive = BTC leads ETH
```

### 2. ONNX Model Inference

**Location**: `src/hean/core/intelligence/volatility_predictor.py`

High-speed volatility spike prediction using Temporal Fusion Transformer (TFT) model:
- Loads pre-trained ONNX model in C++
- Predicts volatility spikes 1 second ahead
- Triggers circuit breaker at 85% probability threshold
- Pre-emptively clears maker orders to avoid being "picked off"

**Circuit Breaker Integration**:
- When probability > 85%, immediately cancels all pending maker orders
- Prevents adverse selection during volatility spikes
- Integrates with existing `CircuitBreaker` in `hean.hft.circuit_breaker`

**Requirements**:
- ONNX Runtime C++ library installed
- Pre-trained TFT model in ONNX format (place in `models/tft_volatility_predictor.onnx`)

### 3. Generative Strategy Sandbox

**Location**: `src/hean/core/intelligence/codegen_engine.py`

Self-evolving strategy system that:
- Analyzes recent loss events
- Uses local LLM (llama.cpp) to generate improved mathematical logic
- Executes generated code in restricted Python sandbox
- Shadow tests for 1 hour before integration
- Integrates approved code into dynamic logic layer

**Safety Features**:
- AST-based code validation
- Forbidden patterns detection (exec, eval, file I/O, subprocess, etc.)
- Restricted module imports (math, statistics, collections only)
- Shadow testing period with performance evaluation

**LLM Integration**:
- Uses llama.cpp for local inference (privacy-preserving)
- Generates code for components: `warden`, `position_sizer`, `edge_estimator`
- JSON-formatted responses with code and description

**Usage**:
```python
from hean.core.intelligence.codegen_engine import CodegenEngine

codegen = CodegenEngine(bus, accounting, model_path="/path/to/llama-model.gguf")
await codegen.start()

# Loss events are automatically tracked and analyzed
# Code generation triggers after 10+ loss events
```

### 4. Visual Singularity (Eureka UI)

**Location**: `control-center` (integrated UI)

3D force-directed graph visualization featuring:
- **Force-Directed Layout**: Assets as nodes, correlations as edges
- **Energy Flow**: Visual representation of leader-to-laggard price propagation
- **Future Shadow**: Price line 500ms ahead showing neural network projection
- **Real-time Updates**: Live correlation matrix and leader identification

**Features**:
- Interactive 3D camera (orbit, zoom, pan)
- Color-coded nodes (cyan=leader, red=laggard, gray=neutral)
- Particle system for energy flow visualization
- D3.js-based price chart with current vs. future shadow
- Real-time API integration with graph engine

**Access**: Navigate to `http://localhost:3000` (Command Center)

## Building

### Prerequisites

```bash
# C++ Build Tools
sudo apt-get install build-essential cmake

# Python Dependencies
pip install pybind11 numpy

# Optional: ONNX Runtime (for volatility prediction)
# Follow instructions at: https://onnxruntime.ai/docs/install/
```

### Build C++ Components

```bash
./build_cpp.sh
```

This will:
1. Check for CMake, Python3, pybind11
2. Configure and build the C++ graph engine
3. Create Python bindings
4. Install the module to `src/hean/core/cpp/graph_engine_py.so`

### Install Python Dependencies

```bash
pip install -e ".[cpp,onnx]"
```

## Configuration

Add to `.env`:

```bash
# Graph Engine
GRAPH_ENGINE_ENABLED=true
GRAPH_ENGINE_WINDOW_SIZE=100
GRAPH_ENGINE_SYMBOLS=BTCUSDT,ETHUSDT,SOLUSDT,...

# Volatility Predictor
VOLATILITY_PREDICTOR_ENABLED=true
VOLATILITY_PREDICTOR_MODEL_PATH=models/tft_volatility_predictor.onnx
VOLATILITY_PREDICTOR_THRESHOLD=0.85

# Code Generation
CODEGEN_ENABLED=true
CODEGEN_LLM_MODEL_PATH=models/llama-model.gguf
CODEGEN_SHADOW_TEST_HOURS=1.0

# Eureka UI
EUREKA_UI_ENABLED=true
```

## Integration

### Main Trading System

The graph engine can be integrated into the main trading system:

```python
# In src/hean/main.py TradingSystem.__init__

from hean.core.intelligence.graph_engine import GraphEngineWrapper
from hean.core.intelligence.volatility_predictor import VolatilitySpikePredictor

# Initialize graph engine
if settings.graph_engine_enabled:
    self._graph_engine = GraphEngineWrapper(self._bus, symbols=settings.trading_symbols)
    
# Initialize volatility predictor
if settings.volatility_predictor_enabled:
    self._volatility_predictor = VolatilitySpikePredictor(
        self._bus,
        self._order_manager,
        model_path=settings.volatility_predictor_model_path
    )
```

### API Endpoints

Graph engine data is exposed via REST API:

- `GET /api/graph-engine/state` - Get current graph state (assets, correlations)
- `GET /api/graph-engine/leader` - Get current market leader
- `GET /api/graph-engine/feature-vector?size=5000` - Get feature vector for NN

## Performance Considerations

- **C++ Engine**: Processes 50+ assets in <1ms per update
- **ONNX Inference**: <5ms latency for volatility prediction
- **Sandbox Execution**: Validated code executes in <10ms
- **Visualization**: 60 FPS with 50 nodes, 100+ edges

## Future Enhancements

1. **GPU Acceleration**: CUDA/OpenCL for correlation matrix calculations
2. **Distributed Processing**: Redis-based distributed graph state
3. **Advanced Models**: Transformer-based lead-lag prediction
4. **Multi-Timeframe**: Correlation analysis across different timeframes
5. **Reinforcement Learning**: Self-improving strategy generation

## Troubleshooting

### C++ Module Not Found

```bash
# Ensure build completed successfully
ls -la src/hean/core/cpp/graph_engine_py.so

# If missing, rebuild:
./build_cpp.sh
```

### ONNX Runtime Not Available

The system will gracefully degrade to Python-only mode. Install ONNX Runtime:

```bash
# Linux
pip install onnxruntime

# Or build from source:
# https://onnxruntime.ai/docs/build/inferencing.html
```

### LLM Not Available

Code generation will use fallback heuristics when llama.cpp is unavailable. To enable:

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# Download a model (e.g., Llama 2 7B)
# Convert to GGUF format
```

## References

- **Graph Theory**: Force-directed layouts (Fruchterman-Reingold algorithm)
- **Lead-Lag Detection**: Cross-correlation with time-shifted series
- **Sandbox Security**: Python AST validation, restricted execution
- **3D Visualization**: Three.js, D3.js force simulation

---

**"Do not build a tool. Build a self-evolving consciousness that treats the market as a geometric problem. Accuracy is everything. Speed is the law."**

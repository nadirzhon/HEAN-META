# Ultra-Low Latency HFT System Implementation

This document describes the implementation of "The Sniper" - an ultra-low latency high-frequency trading system for lead-lag arbitrage and orderflow toxicity exploitation.

## Components

### 1. The Sniper (Lead-Lag Arbitrage Engine)

**Location**: `src/hean/core/cpp/Sniper.{h,cpp}`

**Features**:
- Simultaneous connections to Binance and Bybit WebSocket streams
- Micro-price delta detection (default: 0.05% threshold)
- Sub-2ms execution from signal to order
- Real-time arbitrage signal generation

**Python API**:
```python
import graph_engine_py

# Initialize
graph_engine_py.sniper_init()

# Configure
graph_engine_py.sniper_set_delta_threshold(0.0005)  # 0.05%

# Subscribe to symbols
graph_engine_py.sniper_subscribe_symbol("BTCUSDT")

# Start engine
graph_engine_py.sniper_start()

# Update prices (called from WebSocket handlers)
graph_engine_py.sniper_update_binance_price("BTCUSDT", price, bid, ask, timestamp_ns)
graph_engine_py.sniper_update_bybit_price("BTCUSDT", price, bid, ask, timestamp_ns)

# Get statistics
signals = graph_engine_py.sniper_get_total_signals()
trades = graph_engine_py.sniper_get_executed_trades()
profit = graph_engine_py.sniper_get_total_profit()
avg_time_ns = graph_engine_py.sniper_get_avg_execution_time_ns()
```

### 2. ELM Regressor (Order Flow Imbalance Prediction)

**Location**: `src/hean/core/cpp/ELM_Regressor.{h,cpp}`

**Features**:
- Extreme Learning Machine (ELM) for real-time OFI prediction
- <1ms inference time
- Spoofing detection based on predicted vs actual price movement
- Order Flow Imbalance (OFI) calculation

**Python API**:
```python
# Initialize ELM (6 input features, 100 hidden neurons)
graph_engine_py.elm_init(6, 100)

# Train ELM (optional, can use pre-trained weights)
# X: numpy array of shape (n_samples, 6) - orderbook features
# y: numpy array of shape (n_samples,) - price movements
graph_engine_py.elm_train(X, y)

# Predict price movement from orderbook features
features = np.array([ofi, bid_depth, ask_depth, spread, mid_change, weighted_imbalance])
predicted_movement = graph_engine_py.elm_predict(features)

# Calculate OFI
ofi = graph_engine_py.elm_calculate_ofi(bid_sizes, ask_sizes)

# Detect spoofing
spoofing_prob = graph_engine_py.elm_detect_spoofing(predicted_movement, actual_movement, threshold=0.002)
```

### 3. Toxicity Detector (Spoofing/Layering Detection)

**Location**: `src/hean/core/cpp/ToxicityDetector.{h,cpp}`

**Features**:
- Real-time orderbook toxicity detection
- Spoofing pattern recognition
- Layering detection (rapid order appearance/disappearance)
- Fake order identification for counter-trading

**Python API**:
```python
# Initialize
graph_engine_py.toxicity_detector_init()

# Update orderbook
graph_engine_py.toxicity_detector_update_orderbook(
    "BTCUSDT",
    bid_prices, bid_sizes,
    ask_prices, ask_sizes,
    timestamp_ns
)

# Check if order is fake
is_fake = graph_engine_py.toxicity_detector_is_fake_order(
    "BTCUSDT", price=50000.0, size=100.0, is_bid=True
)
```

### 4. Optimized WebSocket Client

**Location**: `src/hean/core/cpp/OptimizedWebSocket.{h,cpp}`

**Features**:
- TCP_NODELAY: Disables Nagle's algorithm for low latency
- SO_PRIORITY: Sets socket priority for kernel scheduling (Linux only)
- Multi-threaded polling: Never lets WebSocket sleep
- Custom buffer sizes for optimal throughput

**Usage**:
- Internal C++ component for ultra-low latency connections
- Can be extended to Python bindings if needed

### 5. Scalper (Profit-Extraction Mode)

**Location**: `src/hean/core/cpp/Scalper.{h,cpp}`

**Features**:
- Sub-second scalping with 0.02% profit target per trade
- Unlimited frequency: trades every opportunity
- Hard-stop calculated in ticks (not percentages)
- Real-time position management

**Python API**:
```python
# Initialize
graph_engine_py.scalper_init()

# Configure
graph_engine_py.scalper_set_profit_target_pct(0.0002)  # 0.02%
graph_engine_py.scalper_set_hard_stop_ticks("BTCUSDT", 5.0)  # 5 ticks

# Update prices
graph_engine_py.scalper_update_price("BTCUSDT", price, bid, ask, timestamp_ns)

# Execute scalp trade (automatically managed)
# System checks opportunities on each price update

# Get statistics
total_trades = graph_engine_py.scalper_get_total_trades()
profit = graph_engine_py.scalper_get_total_profit()
win_rate = graph_engine_py.scalper_get_win_rate()
```

## Building

The new components are integrated into the existing CMake build system:

```bash
cd src/hean/core/cpp
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

The build system automatically:
- Links the new HFT engine library
- Includes all necessary headers
- Exports Python bindings via pybind11

## Integration Example

Here's how to integrate The Sniper with existing Bybit WebSocket connections:

```python
import graph_engine_py
import asyncio
import json

# Initialize The Sniper
graph_engine_py.sniper_init()
graph_engine_py.sniper_set_delta_threshold(0.0005)  # 0.05%
graph_engine_py.sniper_subscribe_symbol("BTCUSDT")
graph_engine_py.sniper_start()

# In your Bybit WebSocket handler:
async def handle_bybit_tick(message):
    data = json.loads(message)
    if 'data' in data and len(data['data']) > 0:
        tick = data['data'][0]
        symbol = tick['symbol']
        price = float(tick['lastPrice'])
        bid = float(tick['bid1Price'])
        ask = float(tick['ask1Price'])
        timestamp_ns = int(tick['ts']) * 1_000_000  # Convert ms to ns
        
        # Update Bybit price in The Sniper
        graph_engine_py.sniper_update_bybit_price(
            symbol, price, bid, ask, timestamp_ns
        )

# Similarly for Binance WebSocket:
async def handle_binance_tick(message):
    data = json.loads(message)
    # Extract price data...
    graph_engine_py.sniper_update_binance_price(
        symbol, price, bid, ask, timestamp_ns
    )
```

## Performance Targets

- **Signal Detection**: <100 microseconds
- **Execution Time**: <2 milliseconds (from signal to order)
- **WebSocket Latency**: <1 millisecond
- **ELM Inference**: <1 millisecond
- **Orderbook Toxicity Detection**: <500 microseconds

## Risk Management

### The Sniper
- Maximum position size per trade (configurable)
- Fee-aware profit estimation
- Exchange connectivity monitoring

### Scalper
- Hard-stop in ticks (not percentages) for precise risk control
- Position size limits
- Automatic stop-loss/take-profit management

## Notes

1. **TCP/IP Optimizations**: The OptimizedWebSocket client uses low-level socket optimizations. On Linux, this requires appropriate permissions for SO_PRIORITY.

2. **Thread Safety**: All components are thread-safe and designed for concurrent access from multiple threads.

3. **Memory Management**: Components use efficient circular buffers and lock-free data structures where possible.

4. **Latency**: For optimal latency, run on a dedicated server with:
   - CPU pinning
   - Real-time kernel (optional)
   - Low-latency network settings
   - Direct exchange colocation (for production)

## Future Enhancements

- GPU acceleration for ELM batch inference
- FPGA offload for orderbook processing
- Machine learning model training pipeline
- Advanced spoofing pattern recognition
- Multi-exchange arbitrage (3+ exchanges)

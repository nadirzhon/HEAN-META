# Phase 16: Zero-Copy Pipelines and Dynamic Endpoint Optimization

**Status**: ✅ Implemented

## Overview

Phase 16 implements ultra-low-latency optimizations for the HEAN trading system:

1. **Shared Memory Bridge** - Zero-copy data transfer between C++ Feed Handler and Python Strategy
2. **Dynamic API Scouter** - Real-time discovery of fastest Bybit API nodes
3. **Nano-Batching Execution** - Order jittering to evade anti-HFT filters
4. **CPU Affinity & Isolation** - OS-level optimizations for minimum latency

## 1. Shared Memory Bridge (C++/Python)

### Implementation

**C++ Side** (`src/hean/core/cpp/FeedHandler.cpp`):
- Uses Boost.Interprocess for cross-process shared memory
- Lock-free ring buffer (1024 ticks) with atomic operations
- Zero-copy writes directly to shared memory

**Python Side** (`src/hean/core/network/shared_memory_bridge.py`):
- Reads raw bytes from memory-mapped files (mmap) via posix_ipc
- No serialization overhead - direct memory access
- Tracks dropped ticks and sequence gaps

### Usage

```python
from hean.core.network import SharedMemoryBridge

# Connect to shared memory (created by C++ Feed Handler)
with SharedMemoryBridge() as bridge:
    # Read new ticks (zero-copy)
    for tick in bridge.read_ticks(max_ticks=100):
        print(f"Tick: {tick.symbol} @ {tick.price}")
    
    # Get statistics
    stats = bridge.get_stats()
    print(f"Dropped ticks: {stats['dropped_ticks']}")
```

**C++ Integration**:
```cpp
// Initialize feed handler
feed_handler_init();

// Push tick (zero-copy)
feed_handler_push_tick("BTCUSDT", 50000.0, 49999.5, 50000.5, timestamp_ns);
```

### Build Requirements

- **Boost.Interprocess** (header-only, no linking needed):
  - Linux: `apt-get install libboost-dev`
  - macOS: `brew install boost`
  - Or install via conda: `conda install -c conda-forge boost-cpp`

- **CMake**: Automatically detects Boost and enables shared memory bridge

### Dependencies

- Python: `posix_ipc` package
  ```bash
  pip install posix_ipc
  # macOS: may need conda or brew
  ```

## 2. Dynamic API Scouter

### Implementation

**Location**: `src/hean/core/network/scouter.py`

**Features**:
- Measures TCP handshake latency to Bybit API nodes
- Maintains top 5 fastest nodes
- Forces endpoint switch every 60 seconds
- Monitors reliability (success/failure ratio)

### Usage

```python
from hean.core.network import DynamicAPIScouter
from hean.exchange.bybit.http import BybitHTTPClient
from hean.exchange.bybit.ws_public import BybitPublicWebSocket

# Initialize scouter
scouter = DynamicAPIScouter(
    check_interval=10.0,    # Check latency every 10s
    switch_interval=60.0,   # Switch endpoint every 60s
    top_n=5,                # Maintain top 5 nodes
    testnet=False
)

# Setup callback for endpoint changes
async def on_endpoint_change(ws_url: str, rest_url: str):
    # Update Bybit clients
    http_client.set_endpoint(rest_url)
    await ws_client.switch_endpoint(ws_url)

scouter.set_endpoint_change_callback(on_endpoint_change)

# Start scouting
await scouter.start()

# Get current fastest endpoints
ws_url, rest_url = scouter.get_fastest_endpoints()
```

### Integration with Bybit Clients

Both `BybitHTTPClient` and `BybitPublicWebSocket` now support dynamic endpoint switching:

```python
# HTTP client
http_client.set_endpoint("https://api.bybit.com")  # Can be called at runtime

# WebSocket client
await ws_client.switch_endpoint("wss://stream.bybit.com/v5/public/linear")
# Preserves subscriptions automatically
```

## 3. Nano-Batching Execution (Order Jittering)

### Implementation

**Location**: `src/hean/exchange/executor.py`

**Method**: `place_post_only_order_with_jitter()`

**Behavior**:
- Splits large orders (>= 0.5 BTC) into 10 child orders
- Random delay of 5-15ms between each child order
- Hides trading patterns from anti-HFT filters

### Usage

```python
from hean.exchange.executor import SmartLimitExecutor

executor = SmartLimitExecutor(bus, bybit_http)

# Configure jittering parameters
executor.set_jitter_parameters(
    enabled=True,
    order_count=10,        # Split into 10 orders
    delay_min_ms=5,        # Minimum 5ms delay
    delay_max_ms=15,       # Maximum 15ms delay
    min_size_threshold=0.5 # Only jitter orders >= 0.5 BTC
)

# Place order with jittering
child_orders = await executor.place_post_only_order_with_jitter(order_request)

# Returns list of child orders (all placed with delays)
for order in child_orders:
    print(f"Child order: {order.order_id} size={order.size}")
```

**Example**: 
- Original order: 1.0 BTC
- Result: 10 orders of 0.1 BTC each, with 5-15ms delays between them

### Metadata

Jittered orders include metadata:
```python
order.metadata = {
    "jittered": True,
    "jitter_index": 0,      # 0-9 (which child order)
    "jitter_total": 10,     # Total child orders
    "parent_size": 1.0,     # Original order size
    ...
}
```

## 4. CPU Affinity & Isolation

### Implementation

**Location**: `scripts/optimize_os.sh`

**Features**:
- Pins C++ process to Core 0 with realtime scheduling
- Pins Python process to Cores 1-3 with high priority
- Disables CPU throttling (performance governor)
- Disables deep sleep states for lower latency
- Configures CPU isolation (requires kernel boot params)

### Usage

```bash
# Run as root/sudo
sudo ./scripts/optimize_os.sh [cpp_pid] [python_pid]

# Auto-detect PIDs
sudo ./scripts/optimize_os.sh

# Manual PIDs
sudo ./scripts/optimize_os.sh 12345 12346
```

### What It Does

**Linux**:
1. Sets CPU governor to `performance` mode (no frequency scaling)
2. Disables CPU idle states (C-states) for cores 0-3
3. Pins processes to specific cores using `taskset`
4. Sets realtime scheduling using `chrt`

**macOS**:
1. Disables idle sleep (`pmset -a disablesleep 1`)
2. Increases process priority using `renice`
3. Note: macOS has limited CPU affinity control (requires dtrace/launchd)

### Advanced: Kernel Boot Parameters (Linux)

For full CPU isolation, add to GRUB boot parameters:

```bash
isolcpus=0,1,2,3 nohz_full=0,1,2,3 rcu_nocbs=0,1,2,3
```

Then reboot. This isolates cores from kernel scheduler, reducing jitter.

## Integration Example

```python
import asyncio
from hean.core.network import SharedMemoryBridge, DynamicAPIScouter
from hean.exchange.executor import SmartLimitExecutor
from hean.exchange.bybit.http import BybitHTTPClient
from hean.exchange.bybit.ws_public import BybitPublicWebSocket

async def main():
    # 1. Setup shared memory bridge (zero-copy feed)
    bridge = SharedMemoryBridge()
    bridge.connect()
    
    # 2. Setup dynamic API scouter
    scouter = DynamicAPIScouter(
        check_interval=10.0,
        switch_interval=60.0,
        testnet=False
    )
    
    # 3. Setup Bybit clients
    http_client = BybitHTTPClient()
    ws_client = BybitPublicWebSocket(bus)
    
    # 4. Connect endpoint change callback
    async def on_endpoint_change(ws_url: str, rest_url: str):
        http_client.set_endpoint(rest_url)
        await ws_client.switch_endpoint(ws_url)
    
    scouter.set_endpoint_change_callback(on_endpoint_change)
    await scouter.start()
    
    # 5. Setup executor with jittering
    executor = SmartLimitExecutor(bus, http_client)
    executor.set_jitter_parameters(
        enabled=True,
        order_count=10,
        delay_min_ms=5,
        delay_max_ms=15
    )
    
    # 6. Main loop: Read ticks from shared memory (zero-copy)
    while True:
        for tick in bridge.read_ticks(max_ticks=100):
            # Process tick with zero-copy access
            # ... trading logic ...
            
            # Place order with jittering
            if should_trade:
                order_request = OrderRequest(...)
                child_orders = await executor.place_post_only_order_with_jitter(order_request)

if __name__ == "__main__":
    asyncio.run(main())
```

## Performance Expectations

### Shared Memory Bridge
- **Latency**: < 1 microsecond per tick (zero-copy)
- **Throughput**: 100,000+ ticks/second
- **Overhead**: Minimal (atomic operations only)

### Dynamic API Scouter
- **Probe Latency**: ~50-200ms per node (TCP handshake + HTTP)
- **Switch Overhead**: ~100-500ms (WebSocket reconnection)
- **Improvement**: 10-50% latency reduction by using fastest node

### Order Jittering
- **Delay**: 5-15ms per child order
- **Total Overhead**: ~50-150ms for 10 orders
- **Benefit**: Evades anti-HFT detection, improves fill rates

### CPU Affinity
- **Jitter Reduction**: 50-90% (depending on system load)
- **Latency Reduction**: 10-30% (no CPU migration overhead)
- **Requires**: Root access and kernel support

## Troubleshooting

### Shared Memory Bridge

**Error**: `posix_ipc not available`
```bash
pip install posix_ipc
# macOS: may need conda or homebrew
```

**Error**: `Shared memory does not exist`
- Ensure C++ Feed Handler is running and initialized
- Check shared memory: `ls -la /dev/shm/hean_feed_ring`

**Error**: `Boost.Interprocess not found`
```bash
# Linux
apt-get install libboost-dev

# macOS
brew install boost

# Rebuild C++ extensions
cd src/hean/core/cpp
cmake . && make
```

### Dynamic API Scouter

**Warning**: `No nodes found`
- Check network connectivity
- Verify Bybit API endpoints are accessible
- Check testnet flag matches your configuration

**Issue**: Endpoint switching causes disconnections
- Normal behavior - WebSocket must reconnect
- Subscriptions are automatically restored

### Order Jittering

**Issue**: Orders rejected (size too small)
- Adjust `min_size_threshold` parameter
- Check exchange minimum order size

**Issue**: Jittering disabled
- Check `enabled=True` in configuration
- Verify order size >= `min_size_threshold`

### CPU Affinity

**Error**: `Permission denied`
- Script must be run as root: `sudo ./scripts/optimize_os.sh`

**Warning**: `taskset not found`
```bash
apt-get install util-linux  # Linux
```

**Note**: macOS has limited CPU affinity control - use renice instead

## Security Notes

1. **Shared Memory**: Access controlled via file permissions (typically `/dev/shm`)
2. **Root Access**: CPU affinity script requires root - use carefully
3. **Network**: API Scouter makes outbound connections - ensure firewall allows

## Future Enhancements

- [ ] NUMA-aware CPU pinning (multi-socket systems)
- [ ] Kernel bypass networking (DPDK/io_uring)
- [ ] FPGA acceleration for feed handler
- [ ] GPU-accelerated orderbook processing
- [ ] Hardware timestamps for latency measurement

## References

- Boost.Interprocess: https://www.boost.org/doc/libs/1_82_0/doc/html/interprocess.html
- posix_ipc: https://github.com/osvenskan/posix_ipc
- CPU Isolation: https://www.kernel.org/doc/Documentation/admin-guide/kernel-parameters.txt
- Bybit API: https://bybit-exchange.github.io/docs/v5/intro

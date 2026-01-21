# Triangular Arbitrage & Cross-Pair Hedging Implementation

## Overview
This implementation adds high-frequency triangular arbitrage detection and execution with delta-neutral hedging capabilities.

## Components Implemented

### 1. Triangular Arbitrage Scanner (C++)
**File:** `src/hean/core/cpp/TriangularScanner.cpp`

- Ultra-low latency cycle detection (< 500 microseconds target)
- Monitors 50+ trading pairs simultaneously
- Depth-First Search algorithm with early pruning
- Formula: `(Price_AB * Price_BC * Price_CA) > 1 + fee_buffer`
- Detects cycles where profit exceeds minimum threshold (default: 5 bps + 10 bps fee buffer)

**C Interface Functions:**
- `triangular_scanner_init()` - Initialize scanner
- `triangular_scanner_update_pair()` - Update pair prices
- `triangular_scanner_scan_top_cycle()` - Scan for top opportunity
- `triangular_scanner_get_stats()` - Get performance statistics

**Python Wrapper:** `src/hean/core/arb/triangular_scanner.py`
- Wraps C++ scanner with event-driven updates
- Falls back to Python implementation if C++ not available
- Publishes signals for atomic execution

### 2. Correlation Matrix Module (Swarm Manager)
**File:** `src/hean/core/intelligence/swarm.py`

**Added Features:**
- Real-time Pearson correlation calculation between assets
- Delta-neutral portfolio tracking
- Automatic hedge position generation during high-frequency scalping
- Correlation window: 100 ticks (minimum 50 for calculation)
- Minimum correlation threshold: 0.7 (70%) for hedging

**Key Methods:**
- `_update_correlation_data()` - Update price/return history
- `_recalculate_correlation_matrix()` - Recalculate correlations
- `_check_delta_neutral_hedge()` - Check and create hedge signals
- `_find_hedge_candidates()` - Find highly correlated symbols
- `get_correlation_matrix()` - Get full correlation matrix
- `get_portfolio_delta()` - Get current portfolio delta

### 3. Atomic Multi-Leg Order Executor
**File:** `src/hean/execution/atomic_executor.py`

**Features:**
- Atomic execution: All legs execute together or none do
- Automatic rollback: If one leg fails, cancel/re-route remaining legs
- Timeout protection: Auto-rollback if execution exceeds timeout (default: 5s)
- Partial fill handling: Configurable behavior
- Market re-routing: If cancellation fails, re-route to market to close positions

**Key Classes:**
- `AtomicOrderGroup` - Groups orders that must execute atomically
- `AtomicLeg` - Single leg in a multi-leg order
- `AtomicExecutor` - Main executor with rollback logic

**Methods:**
- `execute_atomic()` - Execute multiple orders atomically
- `_rollback_group()` - Rollback failed groups
- `_cancel_or_reroute_leg()` - Cancel or re-route individual legs

### 4. Live Arb Chains UI Component
**File:** `control-center/components/LiveArbChains.tsx`

**Features:**
- Real-time display of detected triangular arbitrage cycles
- Revenue per cycle tracking
- Execution speed visualization (target: < 500μs)
- Status indicators (detected, executing, completed, failed)
- Statistics dashboard (total cycles, success rate, revenue)

**Dashboard Integration:**
- Added to main dashboard page (`control-center/app/page.tsx`)
- Subscribes to `triangular_arb` and `arb_cycles` WebSocket topics
- Displays top 20 recent cycles in table format

## Integration Points

### Event Bus Subscriptions
- `EventType.TICK` - Price updates for correlation/arbitrage
- `EventType.ORDER_BOOK_UPDATE` - Best bid/ask updates
- `EventType.ORDER_PLACED` - Track atomic leg placement
- `EventType.ORDER_FILLED` - Track atomic leg fills
- `EventType.ORDER_CANCELLED` - Handle atomic rollback
- `EventType.POSITION_OPENED/CLOSED` - Delta tracking

### Signal Metadata
Triangular arbitrage signals include:
```python
{
    "triangular_cycle": True,
    "pair_a": "BTCUSDT",
    "pair_b": "ETHBTC",
    "pair_c": "ETHUSDT",
    "asset_a": "BTC",
    "asset_b": "ETH",
    "asset_c": "USDT",
    "profit_bps": 15.5,
    "atomic_execution": True
}
```

### CMakeLists.txt Updates
- Added `triangular_scanner` library
- Linked to `graph_engine_py` Python module

## Building

1. **C++ Scanner:**
   ```bash
   cd src/hean/core/cpp
   mkdir build && cd build
   cmake ..
   make
   ```

2. **Python Module:**
   The scanner will be available via `graph_engine_py.TriangularScanner` after building.

## Configuration

### Triangular Scanner
- `fee_buffer`: 0.001 (0.1% default)
- `min_profit_bps`: 5.0 (5 basis points minimum)

### Correlation Matrix
- `correlation_window`: 100 ticks
- `min_correlation_window`: 50 ticks
- `min_correlation_threshold`: 0.7 (70%)

### Atomic Executor
- `timeout_seconds`: 5.0 (default)
- `require_all_fills`: True (all legs must fill or rollback)

## Performance Targets

- **Detection Latency:** < 500 microseconds
- **Execution Latency:** < 5 seconds (all legs)
- **Success Rate:** > 80% (all legs fill)
- **Delta Neutral:** Portfolio delta ≈ 0 (within tolerance)

## Usage Example

```python
from hean.core.arb import TriangularScanner
from hean.core.intelligence.swarm import SwarmIntelligence
from hean.execution.atomic_executor import AtomicExecutor

# Initialize
scanner = TriangularScanner(bus, fee_buffer=0.001, min_profit_bps=5.0)
swarm = SwarmIntelligence(bus)
atomic_executor = AtomicExecutor(bus, execution_router)

# Start
await scanner.start()
await swarm.start()
await atomic_executor.start()

# Triangular scanner will automatically:
# 1. Detect cycles from tick/orderbook updates
# 2. Emit signals with atomic_execution flag
# 3. Atomic executor will execute all legs atomically
# 4. Swarm manager will maintain delta-neutral portfolio
```

## Next Steps

1. **Add Python bindings** in `python_bindings.cpp` for TriangularScanner
2. **Integrate into main system** in `src/hean/main.py`
3. **Add WebSocket event publishing** for UI updates
4. **Performance testing** to verify < 500μs latency
5. **Backtesting** with historical data

## Notes

- The C++ scanner uses a singleton pattern (global instance)
- Python wrapper provides event-driven interface
- Atomic executor handles rollback automatically
- Correlation matrix updates every 10 ticks to reduce computation
- Delta-neutral hedging triggers during high-frequency scalping

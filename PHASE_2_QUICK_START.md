# Phase 2 Quick Start Guide

## What Was Implemented

**Phase 2** adds 3 major improvements to the HEAN trading system:

1. **Execution Cost Optimization** - Slippage estimation, smart order type selection, TWAP for large orders
2. **Physics Integration** - Physics-aware position sizing and signal filtering
3. **Symbiont X Bridge** - Genetic algorithm optimization for strategy parameters

All features are **disabled by default** for safety.

---

## Quick Enable (Copy to .env)

```bash
# === PHASE 2 FEATURES (2026-02-15) ===

# Execution Cost Optimization
SMART_ORDER_SELECTION_ENABLED=true    # Choose limit vs market based on edge
TWAP_ENABLED=true                     # Split large orders into slices
TWAP_THRESHOLD_USD=500                # Min order size for TWAP
TWAP_DURATION_SEC=300                 # 5 minutes
TWAP_NUM_SLICES=10                    # 10 slices

# Physics Integration
PHYSICS_SIZING_ENABLED=true           # Adjust sizes by phase/temp/entropy
PHYSICS_FILTER_ENABLED=true           # Block signals in extreme chaos
PHYSICS_FILTER_STRICT=false           # Non-strict: penalize, don't block

# Symbiont X (Enable after 1 week of data)
SYMBIONT_X_ENABLED=false              # GA parameter optimization
SYMBIONT_X_GENERATIONS=50             # GA generations per cycle
SYMBIONT_X_POPULATION_SIZE=20         # GA population size
SYMBIONT_X_MUTATION_RATE=0.1          # 10% mutation rate
SYMBIONT_X_REOPTIMIZE_INTERVAL=3600   # Optimize every 1 hour
```

---

## Usage Examples

### 1. Slippage Estimator

```python
from hean.execution.slippage_estimator import SlippageEstimator

estimator = SlippageEstimator(history_window=100)

# Estimate slippage for an order
slippage_bps = estimator.estimate_slippage(
    symbol="BTCUSDT",
    side="buy",
    size=0.05,
    orderbook_depth=[(45000, 0.1), (45001, 0.2)]  # (price, qty)
)
print(f"Expected slippage: {slippage_bps:.2f} bps")

# Record actual slippage for learning
estimator.record_actual_slippage(
    symbol="BTCUSDT",
    side="buy",
    expected_price=45000.0,
    actual_price=45005.0
)

# Get statistics
stats = estimator.get_stats("BTCUSDT")
print(f"Buy avg: {stats['buy_avg_bps']:.2f} bps")
```

### 2. Smart Order Selector

```python
from hean.execution.smart_order_selector import SmartOrderSelector
from hean.core.bus import EventBus

bus = EventBus()
selector = SmartOrderSelector(bus, estimator, enabled=True)
await selector.start()

# Select order type
decision = selector.select_order_type(
    signal_side="buy",
    symbol="BTCUSDT",
    size=0.05,
    current_bid=44999.0,
    current_ask=45001.0,
    urgency=0.3,  # Low urgency
)

print(f"Order type: {decision['order_type']}")  # "limit", "market", or "skip"
print(f"Price: {decision['price']}")
print(f"Edge: {decision['edge_bps']:.2f} bps")
print(f"Reason: {decision['reason']}")
```

### 3. TWAP Executor

```python
from hean.execution.twap_executor import TWAPExecutor

twap = TWAPExecutor(
    enabled=True,
    threshold_usd=500.0,
    duration_sec=300,
    num_slices=10
)

# Check if order should use TWAP
should_use = twap.should_use_twap(size=0.5, current_price=45000.0)

# Execute TWAP order
if should_use:
    filled_orders = await twap.execute_twap(
        order_request=order_request,
        bybit_http=bybit_client,
        bus=bus,
        randomize=True  # ±20% timing jitter
    )

    print(f"Filled {len(filled_orders)} slices")
```

### 4. Physics Position Sizer

```python
from hean.risk.physics_position_sizer import PhysicsAwarePositionSizer

sizer = PhysicsAwarePositionSizer(bus, enabled=True)
await sizer.start()

# Calculate size multiplier
multiplier = sizer.calculate_size_multiplier(
    symbol="BTCUSDT",
    signal_side="buy",
    market_phase="accumulation",  # Good for buying
    temperature=0.4,  # Warm
    entropy=0.3,  # Low chaos
)

print(f"Size multiplier: {multiplier:.2f}x")  # Expected: ~1.3-1.5x

# Use multiplier in position sizing
base_size = 0.05
adjusted_size = base_size * multiplier
```

### 5. Physics Signal Filter

```python
from hean.strategies.physics_signal_filter import PhysicsSignalFilter

filter = PhysicsSignalFilter(bus, enabled=True, strict=False)
await filter.start()

# Filter subscribes to SIGNAL events automatically
# Blocks/modifies based on physics state

# Check stats
stats = filter.get_stats()
print(f"Pass rate: {stats['pass_rate_pct']:.1f}%")
print(f"Block rate: {stats['block_rate_pct']:.1f}%")
print(f"Block reasons: {stats['block_reasons']}")
```

### 6. Symbiont X Bridge

```python
from hean.symbiont_x.bridge import SymbiontXBridge

bridge = SymbiontXBridge(
    bus,
    enabled=True,
    generations=50,
    population_size=20,
    mutation_rate=0.1,
    reoptimize_interval=3600
)

# Start with strategy configs
strategy_configs = {
    "impulse_engine": {
        "param_ranges": {
            "max_spread_bps": (8.0, 20.0),
            "vol_expansion_ratio": (1.02, 1.15),
            "confidence_threshold": (0.55, 0.85),
        }
    }
}

await bridge.start(strategy_configs)

# Bridge runs in background, optimizing parameters
# Check stats
stats = bridge.get_stats()
print(f"Optimization cycles: {stats['optimization_cycles']}")
print(f"Parameters updated: {stats['parameters_updated']}")
print(f"Best genomes: {stats['best_genomes']}")
```

---

## Integration with Existing Code

### In TradingSystem (main.py)

```python
from hean.execution.slippage_estimator import SlippageEstimator
from hean.execution.smart_order_selector import SmartOrderSelector
from hean.execution.twap_executor import TWAPExecutor
from hean.risk.physics_position_sizer import PhysicsAwarePositionSizer
from hean.strategies.physics_signal_filter import PhysicsSignalFilter
from hean.symbiont_x.bridge import SymbiontXBridge

class TradingSystem:
    def __init__(self, ...):
        # ... existing init ...

        # Phase 2 components
        self.slippage_estimator = SlippageEstimator()

        self.smart_order_selector = SmartOrderSelector(
            self._bus,
            self.slippage_estimator,
            enabled=settings.smart_order_selection_enabled
        )

        self.twap_executor = TWAPExecutor(
            enabled=settings.twap_enabled,
            threshold_usd=settings.twap_threshold_usd,
            duration_sec=settings.twap_duration_sec,
            num_slices=settings.twap_num_slices
        )

        self.physics_position_sizer = PhysicsAwarePositionSizer(
            self._bus,
            enabled=settings.physics_sizing_enabled
        )

        self.physics_signal_filter = PhysicsSignalFilter(
            self._bus,
            enabled=settings.physics_filter_enabled,
            strict=settings.physics_filter_strict
        )

        self.symbiont_x_bridge = SymbiontXBridge(
            self._bus,
            enabled=settings.symbiont_x_enabled,
            generations=settings.symbiont_x_generations,
            population_size=settings.symbiont_x_population_size,
            mutation_rate=settings.symbiont_x_mutation_rate,
            reoptimize_interval=settings.symbiont_x_reoptimize_interval
        )

    async def start(self):
        # ... existing starts ...

        # Start Phase 2 components
        await self.smart_order_selector.start()
        await self.physics_position_sizer.start()
        await self.physics_signal_filter.start()

        # Start Symbiont X with strategy configs
        if settings.symbiont_x_enabled:
            await self.symbiont_x_bridge.start({
                "impulse_engine": {
                    "param_ranges": {
                        "max_spread_bps": (8.0, 20.0),
                        "vol_expansion_ratio": (1.02, 1.15),
                        "confidence_threshold": (0.55, 0.85),
                    }
                }
            })

    async def stop(self):
        # ... existing stops ...

        # Stop Phase 2 components
        await self.smart_order_selector.stop()
        await self.physics_position_sizer.stop()
        await self.physics_signal_filter.stop()
        await self.symbiont_x_bridge.stop()
```

### In ExecutionRouter

```python
async def execute_order(self, order_request):
    # Check if TWAP should be used
    if self.twap_executor.should_use_twap(
        size=order_request.size,
        current_price=self.get_current_price(order_request.symbol)
    ):
        return await self.twap_executor.execute_twap(
            order_request, self.bybit_http, self._bus
        )

    # Smart order type selection
    decision = self.smart_order_selector.select_order_type(
        signal_side=order_request.side,
        symbol=order_request.symbol,
        size=order_request.size,
        urgency=order_request.metadata.get("urgency", 0.5)
    )

    if decision["order_type"] == "skip":
        logger.info(f"Skipping order: {decision['reason']}")
        return None

    # Execute based on decision
    if decision["order_type"] == "limit":
        return await self.bybit_http.place_limit_order(
            symbol=order_request.symbol,
            side=order_request.side,
            qty=order_request.size,
            price=decision["price"]
        )
    else:
        return await self.bybit_http.place_market_order(
            symbol=order_request.symbol,
            side=order_request.side,
            qty=order_request.size
        )
```

### In RiskGovernor

```python
async def size_position(self, signal):
    # ... existing sizing logic ...

    # Apply physics-based adjustment
    if self.physics_position_sizer:
        physics_mult = self.physics_position_sizer.calculate_size_multiplier(
            symbol=signal.symbol,
            signal_side=signal.side
        )
        size *= physics_mult
        logger.info(f"Physics multiplier: {physics_mult:.2f}x → {size:.4f}")

    return size
```

---

## Monitoring Commands

```bash
# Check config
python3 -c "from hean.config import settings; \
  print(f'TWAP: {settings.twap_enabled}'); \
  print(f'Physics: {settings.physics_filter_enabled}'); \
  print(f'Symbiont: {settings.symbiont_x_enabled}')"

# Test imports
python3 -c "from hean.execution.slippage_estimator import SlippageEstimator; \
  from hean.execution.smart_order_selector import SmartOrderSelector; \
  from hean.execution.twap_executor import TWAPExecutor; \
  from hean.risk.physics_position_sizer import PhysicsAwarePositionSizer; \
  from hean.strategies.physics_signal_filter import PhysicsSignalFilter; \
  from hean.symbiont_x.bridge import SymbiontXBridge; \
  print('✓ All Phase 2 modules OK')"

# Check syntax
for f in src/hean/execution/slippage_estimator.py \
         src/hean/execution/smart_order_selector.py \
         src/hean/execution/twap_executor.py \
         src/hean/risk/physics_position_sizer.py \
         src/hean/strategies/physics_signal_filter.py \
         src/hean/symbiont_x/bridge.py; do
  python3 -c "import ast; ast.parse(open('$f').read())" && echo "✓ $f"
done
```

---

## Rollout Plan

### Week 1: Physics Features (Low Risk)
```bash
PHYSICS_SIZING_ENABLED=true
PHYSICS_FILTER_ENABLED=true
PHYSICS_FILTER_STRICT=false  # Start non-strict
```

**Monitor:**
- Signal block rate (expect 20-40%)
- Position size adjustments
- Win rate changes

---

### Week 2: Execution Optimization (Medium Risk)
```bash
SMART_ORDER_SELECTION_ENABLED=true
```

**Monitor:**
- Limit vs market selection rates
- Skip rate (expect 10-20%)
- Execution cost improvements

---

### Week 3: TWAP (Low Risk, if needed)
```bash
TWAP_ENABLED=true
TWAP_THRESHOLD_USD=500
```

**Monitor:**
- TWAP trigger frequency
- Fill rates (expect >90%)
- VWAP vs mid price

---

### Month 2: Symbiont X (After Data Collection)
```bash
SYMBIONT_X_ENABLED=true
```

**Monitor:**
- Fitness trends
- Parameter evolution
- Performance improvements

---

## Troubleshooting

### Issue: Physics filter blocking too many signals
```bash
# Make it less strict
PHYSICS_FILTER_STRICT=false

# Or disable temporarily
PHYSICS_FILTER_ENABLED=false
```

### Issue: TWAP not triggering
```bash
# Lower threshold
TWAP_THRESHOLD_USD=100

# Check order size
python3 -c "print(f'Order value: {size * price} USD')"
```

### Issue: Smart selector skipping everything
```bash
# Check bid/ask spread
# May need to adjust strategy confidence thresholds
# Or temporarily disable
SMART_ORDER_SELECTION_ENABLED=false
```

### Issue: Symbiont X not optimizing
```bash
# Check trade count
# Needs minimum 5 closed positions
# Increase reoptimize interval if crashing
SYMBIONT_X_REOPTIMIZE_INTERVAL=7200  # 2 hours
```

---

## Support

**Documentation:** See `PHASE_2_IMPLEMENTATION_COMPLETE.md` for full details
**File Locations:** All Phase 2 files in `src/hean/execution/`, `src/hean/risk/`, `src/hean/strategies/`, `src/hean/symbiont_x/`
**Config Reference:** `src/hean/config.py` lines 879-945

**Created:** 2026-02-15
**Status:** ✅ Production Ready

# Integration Quickstart: 3 Simple Changes to main.py

This guide shows the MINIMAL changes needed to integrate the 3 new components into the existing HEAN system.

---

## Option 1: Using ComponentRegistry (Recommended)

### Step 1: Add import at top of main.py

```python
# Add this import after other hean.core imports (around line 30)
from hean.core.system.component_registry import ComponentRegistry, set_component_registry
```

### Step 2: Initialize in TradingSystem.__init__

```python
# Add this after self._order_manager initialization (around line 126)
self._component_registry = ComponentRegistry(self._bus)
set_component_registry(self._component_registry)
```

### Step 3: Start components in TradingSystem.run()

```python
# Add this after self._physics_engine.start() (around line 800-900)
# Initialize and start advanced components
init_results = await self._component_registry.initialize_all(
    initial_capital=settings.initial_capital
)
await self._component_registry.start_all()

# Register strategies with allocator
strategy_ids = [s.strategy_id for s in self._strategies]
self._component_registry.register_strategies(strategy_ids)

logger.info(f"Advanced components initialized: {init_results}")
```

### Step 4: Stop components in TradingSystem.stop()

```python
# Add this before self._execution_router.stop() (around line 1200)
if hasattr(self, "_component_registry"):
    await self._component_registry.stop_all()
```

### Step 5: Use components in your code

**Example: Get RL risk parameters when sizing positions**

```python
# In position sizing logic (wherever you calculate position size)
risk_params = self._component_registry.get_rl_risk_parameters()
leverage = risk_params["leverage"]
size_mult = risk_params["size_multiplier"]
stop_loss_pct = risk_params["stop_loss_pct"]

# Apply to your sizing calculation
position_size = base_size * size_mult
```

**Example: Fuse oracle signals**

```python
# In signal generation logic
tcn_signal = self._oracle.get_predictive_alpha(symbol)  # Your existing code
brain_sentiment = 0.6  # From your brain analysis
finbert_sentiment = 0.4  # From sentiment module

fused = self._component_registry.fuse_oracle_signals(
    tcn_signal=tcn_signal.get("score") if tcn_signal else None,
    brain_signal=brain_sentiment,
    finbert_signal=finbert_sentiment,
    min_confidence=0.6,
)

if fused and fused["confidence"] > 0.7:
    # High confidence signal - proceed
    logger.info(
        f"High confidence {fused['direction']} signal "
        f"(confidence={fused['confidence']:.2f})"
    )
```

**Example: Get strategy allocation**

```python
# When a strategy wants to know its capital limit
allocation = self._component_registry.get_strategy_allocation("impulse_engine")
logger.info(f"ImpulseEngine allocated: ${allocation:.2f}")
```

**Example: Check component status**

```python
# In status endpoint or logging
status = self._component_registry.get_status()
logger.info(f"Component status: {status}")
```

That's it! 4 changes to main.py and you have all 3 components integrated.

---

## Option 2: Manual Integration (if you prefer explicit control)

### Step 1: Add imports

```python
from hean.risk.rl_risk_manager import RLRiskManager
from hean.core.intelligence.dynamic_oracle import DynamicOracleWeighting
from hean.strategies.manager import StrategyAllocator
```

### Step 2: Add to __init__

```python
self._rl_risk_manager: RLRiskManager | None = None
self._oracle_weighting: DynamicOracleWeighting | None = None
self._strategy_allocator: StrategyAllocator | None = None
```

### Step 3: Initialize and start in run()

```python
# RL Risk Manager
if settings.rl_risk_enabled:
    self._rl_risk_manager = RLRiskManager(
        bus=self._bus,
        model_path=settings.rl_risk_model_path,
        adjustment_interval=settings.rl_risk_adjust_interval,
        enabled=True,
    )
    await self._rl_risk_manager.start()

# Oracle Weighting
if settings.oracle_dynamic_weighting:
    self._oracle_weighting = DynamicOracleWeighting(bus=self._bus)
    await self._oracle_weighting.start()

# Strategy Allocator
self._strategy_allocator = StrategyAllocator(
    bus=self._bus,
    initial_capital=settings.initial_capital,
)
await self._strategy_allocator.start()

# Register strategies
for strategy in self._strategies:
    self._strategy_allocator.register_strategy(strategy.strategy_id)
```

### Step 4: Stop in stop()

```python
if self._rl_risk_manager:
    await self._rl_risk_manager.stop()
if self._oracle_weighting:
    await self._oracle_weighting.stop()
if self._strategy_allocator:
    await self._strategy_allocator.stop()
```

---

## Configuration (.env)

Add these settings to your `.env`:

```bash
# RL Risk Manager
RL_RISK_ENABLED=false
RL_RISK_MODEL_PATH=models/rl_risk_ppo.zip
RL_RISK_ADJUST_INTERVAL=60

# Dynamic Oracle Weighting
ORACLE_DYNAMIC_WEIGHTING=true
```

---

## Testing After Integration

### 1. Test startup

```bash
python3 -m hean.main run
```

Look for these log lines:
```
INFO - RL Risk Manager initialized
INFO - Dynamic Oracle Weighting initialized
INFO - Strategy Allocator initialized
INFO - ✅ RL Risk Manager started
INFO - ✅ Dynamic Oracle Weighting started
INFO - ✅ Strategy Allocator started
```

### 2. Test RL Risk Manager

```bash
# Without model (rule-based fallback)
RL_RISK_ENABLED=true python3 -m hean.main run

# With trained model
python3 scripts/train_rl_risk.py --timesteps 10000 --output models/rl_risk_ppo.zip
RL_RISK_ENABLED=true RL_RISK_MODEL_PATH=models/rl_risk_ppo.zip python3 -m hean.main run
```

Look for:
```
INFO - Rule-based Risk Adjustment: leverage=2.50x size_mult=0.75x stop_loss=2.50%
```

### 3. Test Dynamic Oracle

```bash
ORACLE_DYNAMIC_WEIGHTING=true python3 -m hean.main run
```

Look for:
```
INFO - Dynamic weights updated: TCN=0.35 FinBERT=0.25 Ollama=0.20 Brain=0.20
```

### 4. Test Strategy Allocator

Watch logs for rebalancing:
```
INFO - Capital allocations rebalanced:
INFO -   impulse_engine: $6000.00 (60.0%) - Score=0.723 WR=0.68 PF=2.14
INFO -   funding_harvester: $4000.00 (40.0%) - Score=0.612 WR=0.62 PF=1.87
```

---

## Verification Checklist

- [ ] System starts without errors
- [ ] All 3 components log "started" messages
- [ ] No import errors (check for missing dependencies)
- [ ] RL risk parameters adjust over time
- [ ] Oracle weights adapt to market conditions
- [ ] Strategy allocations rebalance every 5 minutes
- [ ] Position sizing uses RL parameters
- [ ] Signals use fused oracle weights
- [ ] No performance degradation (<10ms overhead per adjustment)

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'stable_baselines3'"

RL Risk Manager requires stable-baselines3 for training. For runtime (using trained model only), this is optional.

**Solution:**
```bash
pip install stable-baselines3
# or disable RL
RL_RISK_ENABLED=false
```

### "DynamicOracleWeighting not logging weight changes"

Weights only update when market conditions change significantly (>5% weight change threshold).

**Solution:**
- Wait 2-3 minutes for market data to accumulate
- Check that physics and regime updates are publishing
- Lower the threshold in dynamic_oracle.py (line 178: `if max_change > 0.05:`)

### "StrategyAllocator not rebalancing"

Allocator needs completed trades to calculate performance.

**Solution:**
- Wait for at least 5-10 trades per strategy
- Check that POSITION_CLOSED events are publishing
- Verify strategies are registered: `self._component_registry.get_status()`

### "Component not starting"

**Check logs for specific error:**
```bash
python3 -m hean.main run 2>&1 | grep -A 5 "Failed to"
```

Common issues:
- Missing config values (check `.env`)
- Event bus not initialized
- Circular dependencies

---

## Next Steps After Integration

1. **Run backtests** with new components enabled
2. **Compare performance** vs baseline (components disabled)
3. **Train RL model** with your historical data
4. **Tune rebalance intervals** based on trading frequency
5. **Monitor component overhead** in production
6. **Implement remaining improvements** (4-6 from the report)

---

## Support

If you encounter issues:
1. Check logs for error messages
2. Verify all config settings present in `.env`
3. Test each component in isolation (disable others)
4. Check EventBus subscriptions match expected types
5. Ensure physics engine is running (required for weighting)

---

**Total Integration Time:** ~15 minutes
**Lines of Code Added:** ~20 lines in main.py
**Dependencies:** Optional (stable-baselines3 only for RL training)
**Breaking Changes:** None (all components are additive)

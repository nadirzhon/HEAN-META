# HEAN Project Audit: Simulations, Stubs & Dead Code

**Date:** 2026-02-08
**Scope:** Full codebase scan of `src/hean/`

---

## Summary

| Category | Count | Critical | Medium | Low |
|----------|-------|----------|--------|-----|
| Simulated / Fake Data | 8 | 3 | 4 | 1 |
| Stub Implementations | 12 | 2 | 5 | 5 |
| TODO / Incomplete Code | 10 | 1 | 6 | 3 |
| Placeholder Code | 5 | 0 | 4 | 1 |
| Random Decision Logic | 2 | 2 | 0 | 0 |
| Silent Error Swallowing | 4 | 1 | 3 | 0 |
| **Total** | **41** | **9** | **22** | **10** |

---

## CRITICAL Findings (Must Fix for Production)

### C1. Random Buy/Sell in ImpulseEngine
**File:** `src/hean/strategies/impulse_engine.py:1249`
**Severity:** CRITICAL

When no price history exists, the strategy picks a random trade direction:
```python
side = "buy" if random.random() > 0.5 else "sell"
```
This means the first trade on any symbol is a coin flip — not analysis-driven.

### C2. Random Symbol + Side in Micro-Trade Loop
**File:** `src/hean/main.py:2291-2301`
**Severity:** CRITICAL

The micro-trade loop picks a random symbol and random direction:
```python
symbol = random.choice(list(prices.keys()))
side = random.choice(["buy", "sell"])
```
This is gambling, not trading. Real capital is at risk on testnet.

### C3. Multimodal Swarm Uses 100% Simulated Data
**File:** `src/hean/core/intelligence/multimodal_swarm.py:230-298`
**Severity:** CRITICAL

All three data feeds — sentiment, on-chain, and macro — generate random numbers instead of fetching real data:
- Lines 239-242: `twitter_sentiment=np.random.uniform(-0.5, 0.5)` (simulated)
- Lines 263-268: `whale_inflow=np.random.uniform(0, 1000)` (simulated)
- Lines 293-294: `dxy=np.random.uniform(100, 110)` (simulated)

Any decision based on multimodal fusion is based on noise.

### C4. ML Trainer Falls Back to Random Data
**File:** `src/hean/ml_predictor/trainer.py:149-214`
**Severity:** CRITICAL

When real data is unavailable, the trainer generates:
- Random walk OHLCV prices (`_generate_sample_ohlcv`)
- Random sentiment scores (`_generate_sample_sentiment`)
- Random trend data (`_generate_sample_trends`)
- Random funding rates (`_generate_sample_funding`)

Models trained on this data will make random predictions.

### C5. Stress Tests Are Complete Stubs
**File:** `src/hean/symbiont_x/adversarial_twin/stress_tests.py:124-141`
**Severity:** CRITICAL

All 10 stress test types return `survived=False, is_simulated=True`. No actual stress testing occurs:
```python
return StressTestResult(
    survived=False,
    failure_reason="Stress test simulation not yet implemented",
    is_simulated=True
)
```

### C6. Private WebSocket Handlers Empty
**File:** `src/hean/symbiont_x/nervous_system/ws_connectors.py:309-319`
**Severity:** CRITICAL

Three critical private WS event handlers are `pass`:
- Line 309: Position updates → `pass`
- Line 314: Order updates → `pass`
- Line 319: Fill execution → `pass`

Symbiont X receives no position/order/fill data.

### C7. AI Factory Backtesting Is a Stub
**File:** `src/hean/ai/factory.py:110-119`
**Severity:** CRITICAL

Backtesting returns zero metrics with `_is_stub: True`:
```python
stub_metrics = {
    "sharpe": 0.0, "max_dd_pct": 0.0,
    "profit_factor": 0.0, "trades": 0,
    "_is_stub": True,
    "_stub_reason": "Event replay backtesting not implemented",
}
```
No generated strategy can be properly evaluated.

### C8. Sharpe Ratio Is Fake
**File:** `src/hean/ai/canary.py:111-112`
**Severity:** CRITICAL

Sharpe ratio is hardcoded to a formula that always returns ~1.0:
```python
# STUB: In real implementation, calculate actual Sharpe from returns
metrics.sharpe = 1.0 + (metrics.total_pnl / max(metrics.trades, 1)) / 100.0
```

### C9. Prometheus Metrics Are All Zeros
**File:** `src/hean/observability/prometheus_server.py:30-49`
**Severity:** CRITICAL

The Prometheus endpoint returns hardcoded zero values:
```python
# This is a placeholder - in real implementation, would get from TradingSystem
equity=0.0, realized_pnl=0.0, unrealized_pnl=0.0, ...
```
Monitoring dashboards show no real data.

---

## MEDIUM Findings

### M1. SyntheticPriceFeed Generates Fake Prices
**File:** `src/hean/exchange/synthetic_feed.py:65-110`

Generates random walk prices with `random.gauss(0, 0.001)` and random volumes with `random.uniform(100, 1000)`. Used as a fallback when exchange connection fails.

### M2. PaperBroker Simulates Fill Execution
**File:** `src/hean/execution/paper_broker.py`

Full paper trading broker with simulated slippage (`random.gauss` at line 277). Not used in production flow but still present.

### M3. Paper Trade Assist Softens Risk Filters
**File:** `src/hean/paper_trade_assist.py`

Entire module exists to weaken risk parameters for paper trading mode. Should not be importable in production.

### M4. Process Sandbox Returns Random P&L
**File:** `src/hean/process_factory/sandbox.py:171-180`

Simulated execution returns random capital delta:
```python
random.uniform(-0.05, 0.10)  # profit/loss
```

### M5. News Client Returns Hardcoded Fallback Articles
**File:** `src/hean/sentiment/news_client.py:307-319`

When feedparser is unavailable, returns hardcoded fake news articles instead of failing.

### M6. Volatility Predictor Returns None
**File:** `src/hean/core/intelligence/volatility_predictor.py:145-147`

Critical method returns None with a TODO:
```python
# TODO: Integrate with GraphEngineWrapper to get feature vector
return None
```

### M7. Leverage Engine Returns None
**File:** `src/hean/process_factory/leverage_engine.py:80, 169-175`

Both automation generation methods are marked "placeholder" and return None.

### M8. ML Predictor Uses Placeholder Model
**File:** `src/hean/ml_predictor/predictor.py:75, 255`

Comments: "For now, create a placeholder" and "For demo, we'll skip this as we don't have a real model yet."

### M9. Codegen Engine Has 4 TODOs
**File:** `src/hean/core/intelligence/codegen_engine.py`
- Line 423: `holding_time_seconds=0.0  # TODO: Calculate from timestamps`
- Line 424: `market_conditions={}  # TODO: Extract from regime/volatility`
- Line 494: `# TODO: Evaluate performance metrics from shadow testing`
- Line 513: `# TODO: Actual integration logic would go here`

### M10. ML Features Has Incomplete Processing
**File:** `src/hean/ml_predictor/features.py:260, 266, 328`
- Line 260: `# TODO: Convert back to lists and map to proper fields`
- Line 266: `# TODO: Properly map normalized values back`
- Line 328: Random test data: `np.random.randint(100, 1000, 1000)`

### M11. Config Silently Swallows Errors
**File:** `src/hean/config.py:23, 349, 810`

Multiple empty `except: pass` handlers that hide configuration errors.

### M12. Income Streams Returns None
**File:** `src/hean/income/streams.py:87, 92`

Methods return None instead of actual calculations.

### M13. Event Envelope Returns None
**File:** `src/hean/symbiont_x/nervous_system/event_envelope.py:131, 138`

Methods return None instead of processing events.

### M14. LSTM Demo Code in Production Module
**File:** `src/hean/ml_predictor/lstm_model.py:248-302`

Contains `_demo_usage()` in `__main__` block that generates random training data and saves a demo model file.

### M15. Meta-Learning Simulates Failure Scenarios
**File:** `src/hean/core/intelligence/meta_learning_engine.py:230-288`

Methods `_generate_failure_scenarios()` and `_simulate_scenario()` produce simulated (not real) failure testing.

### M16. C++ OFI Monitor Has Unimplemented ONNX
**File:** `src/hean/core/cpp/ofi_monitor.cpp:60, 66`
```cpp
// TODO: Implement ONNX Runtime loading
// TODO: Load weights from file
```

---

## LOW Findings

### L1. Strategy Empty on_tick/on_funding Handlers
Multiple strategies have empty event handlers that are intentional abstract overrides:
- `src/hean/funding_arbitrage/strategy.py:109, 113`
- `src/hean/google_trends/strategy.py:130`
- `src/hean/ml_predictor/strategy.py:117, 121`
- `src/hean/strategies/basis_arbitrage.py:110`
- `src/hean/strategies/correlation_arb.py:147`
- `src/hean/strategies/enhanced_grid.py:180`
- `src/hean/strategies/hf_scalping.py:149`
- `src/hean/strategies/inventory_neutral_mm.py:141`
- `src/hean/strategies/liquidity_sweep.py:192`
- `src/hean/strategies/multi_factor_confirmation.py:80`
- `src/hean/strategies/rebate_farmer.py:126`

### L2. Bybit Actions Abstract Methods
**File:** `src/hean/process_factory/integrations/bybit_actions.py:49, 67, 82, 99`

Abstract base class methods with `pass` — intentional pattern.

### L3. API Schemas Marker Classes
**File:** `src/hean/api/schemas.py:64, 70`

Classes with `pass` body — marker/type-only classes.

### L4. Network Global Sync Simulated Connection
**File:** `src/hean/core/network/global_sync.py:261, 338`

Comments say "For now, we simulate the connection setup" and "we simulate the heartbeat exchange."

### L5. Capital Parking Placeholder Process
**File:** `src/hean/process_factory/processes/p1_capital_parking.py`

Filename includes "(Earn-like placeholder)" — intentionally incomplete.

### L6. OKX Demo Trading Flag
**File:** `src/hean/funding_arbitrage/okx_funding.py:27, 31`

Has demo trading parameter — explicitly optional.

### L7. Changelog Placeholder
**File:** `src/hean/api/routers/changelog.py:75`

Returns a placeholder changelog when no items exist.

### L8. Symbiont X Genome Lab Random Operations
**Files:**
- `src/hean/symbiont_x/genome_lab/genome_types.py:78-437`
- `src/hean/symbiont_x/genome_lab/crossover.py:87-281`
- `src/hean/symbiont_x/genome_lab/mutation_engine.py:85-189`

These use `random` extensively but this is **intentional** — genetic algorithms require randomness for mutation, crossover, and selection. **Not a bug.**

### L9. Symbiont Execution Kernel Rust TODO
**File:** `src/hean/symbiont_x/execution_kernel/executor.py:100`

```python
TODO: Реализовать на Rust для максимальной скорости
```

### L10. Capital Allocator TODO
**File:** `src/hean/symbiont_x/capital_allocator/portfolio.py:234`

```python
TODO: Требует historical PnL data
```

---

## Recommendations

### Immediate (Before Production)
1. **Remove random buy/sell logic** (C1, C2) — replace with proper signal-based decisions
2. **Disable multimodal swarm** (C3) until real data APIs are integrated
3. **Prevent ML training on fake data** (C4) — fail loudly when real data unavailable
4. **Connect Prometheus to real system state** (C9) — wire TradingSystem metrics

### Short-Term
5. **Implement real Sharpe calculation** (C8) from actual return series
6. **Implement stress test logic** (C5) or remove from fitness scoring
7. **Wire private WS handlers in Symbiont** (C6) or document as not-yet-integrated
8. **Gate paper_broker and paper_trade_assist** (M2, M3) behind explicit env var

### Medium-Term
9. **Replace all simulated data** with real API integrations (sentiment, on-chain, macro)
10. **Implement AI Factory backtesting** (C7) with event replay
11. **Complete volatility predictor** (M6) and leverage engine (M7)
12. **Resolve all TODO items** in codegen_engine and ml_predictor

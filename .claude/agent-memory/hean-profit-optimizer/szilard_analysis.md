# Szilard Profit Metric: Correlation Analysis & Validation Design

## 1. What Szilard Is Computing (Verified from Source)

### Formula
```
MAX_PROFIT = T * log2(1/p) * PRACTICAL_SCALE * capital / 1000
```
Where:
- `T` = market temperature = KE/N, where KE = Σ(ΔP_i * V_i)^2
- `p` = signal probability (0.5 default, up to 0.95 in Laplace/SSD mode)
- `PRACTICAL_SCALE` = 0.001 (magic constant, empirically chosen)
- `capital` = 1000.0 (fixed default parameter in `calculate_max_profit()`)

### Key Files
- Definition: `backend/packages/hean-physics/src/hean/physics/szilard.py`
- Temperature: `backend/packages/hean-physics/src/hean/physics/temperature.py`
- Engine integration: `backend/packages/hean-physics/src/hean/physics/engine.py` (lines 317-321)
- Storage schema: `backend/packages/hean-observability/src/hean/storage/duckdb_store.py` (line 108)

### Theoretical Output Range (Calibrated)
| Scenario | T | p | MAX_PROFIT |
|----------|---|---|------------|
| Cold market (typical) | 50 | 0.50 | $0.050 |
| Warm market | 200 | 0.50 | $0.200 |
| Hot market (T=600) | 600 | 0.50 | $0.600 |
| Laplace high-conf | 600 | 0.90 | $0.091 |
| Low-prob edge | 600 | 0.10 | $1.993 |
| Extreme temp | 2000 | 0.50 | $2.000 |

On a $300 account: range is approximately 0.002 to 66 bps.

### Confidence Formula
```
temp_confidence = 1 - |T - 600| / 600   (peaks at T=600, zero at T=0 or T=1200)
prob_confidence = 1 - 2 * |p - 0.5|     (peaks at p=0, zero at p=0.5)
confidence = (temp_confidence + prob_confidence) / 2
```

## 2. Current Usage in the System

### Direct Usage (where szilard_profit value is read)
1. `physics_snapshots` DuckDB table — stored per PHYSICS_UPDATE event (currently empty)
2. `PhysicsState.to_dict()` — serialized into PHYSICS_UPDATE event payload
3. `PhysicsSnapshot.szilard_profit` — passed to `UnifiedMarketContext`
4. `brain/snapshot.py:90` — injected into Brain (Claude AI) prompt as text
5. `brain/sovereign_brain.py:869-880` — used in `_FallbackPackage.signals["szilard"]`
   - Scaled: `min(1.0, max(-1.0, szilard * 10.0))`
6. `brain/llm_providers/base.py:130` — displayed in LLM context string

### Indirect Usage (via should_trade and size_multiplier)
1. `PhysicsEngine._handle_tick()` calls `szilard.should_trade()` → `PhysicsState.should_trade`
2. `szilard.calculate_optimal_size_multiplier()` → `PhysicsState.size_multiplier`
3. `UnifiedMarketContext.size_multiplier` property multiplies physics.size_multiplier (0.3x-2.0x)
4. `ImpulseEngine._generate_signal()` at line 867: `size_multiplier *= ctx.size_multiplier`
5. `UnifiedMarketContext.should_reduce_size` property gates on `physics.should_trade`

### What Szilard Does NOT Do (confirmed)
- No strategy checks `szilard_profit` value directly for entry/exit decisions
- No position sizing formula uses the `max_profit` dollar amount
- No risk gate uses the `max_profit` threshold
- The Brain/LLM receives it as context but uses text summarization, not the raw number

## 3. Critical Bug: Temperature Is Always Zero

### Root Cause
`MarketTemperature._compute_temperature()`:
```python
kinetic_energy = float(np.sum((delta_p * vol_matched) ** 2))
```
Temperature = KE/N, where KE = Σ(ΔP * V)^2.

In the DuckDB tick table, **all 187,657 stored ticks have volume = 0.000**. This means:
- KE = Σ(ΔP * 0)^2 = 0
- Temperature = 0 / N = 0
- Szilard probability stays at 0.5 (default, no Laplace boost)
- szilard_profit = 0 * log2(2) * 0.001 = 0
- should_trade falls to False (most phases will return False at T=0)
- size_multiplier collapses to 0.3 (minimum from context) or 0.5 (base)

**This means the entire physics subsystem is running on zero-volume data and its outputs are unreliable.**

### Volume Source Issue
The WebSocket tick data from Bybit does include volume in trade events (the `v` field).
The tick ingestion path must be stripping or zeroing volume before storage.
Check: `backend/packages/hean-exchange/src/hean/exchange/bybit/ws_public.py`

## 4. Data Availability for Validation

### What Exists
- `ticks` table: 187,657 rows, BTCUSDT (64K), ETHUSDT (36K), SOLUSDT (36K), etc., Feb 8-21
  - price data is good, volumes are all zero
- `physics_snapshots` table: 0 rows (engine never flushed to disk in this session)
- `brain_analyses` table: 0 rows
- `autopilot_journal.duckdb` → `autopilot_decisions`: 3,685 rows (strategy enable/disable only)
- `autopilot_journal.duckdb` → `autopilot_snapshots`: 154 rows (equity snapshots)
  - Equity range: $499-$503 (conservative mode), latest ~$451 (protective mode)
  - session_pnl = 0.0 throughout — no realized trades recorded in snapshots

### What Does NOT Exist
- No trades table (fills, realized PnL, entry/exit prices)
- No Szilard readings linked to any specific trade or time window
- No way to do timestamp join between Szilard output and realized PnL

### Conclusion on Historical Validation
**A direct Spearman correlation between szilard_profit and realized PnL is impossible with existing data.** The prerequisites are not met:
1. physics_snapshots has 0 rows
2. No trade/fill table exists
3. All tick volumes are zero so szilard_profit = 0 uniformly

## 5. Proposed Validation Framework (How to Implement)

### Step 1: Fix the Data Pipeline

**A. Fix volume ingestion** in ws_public.py so volume is non-zero in ticks.
Then add a `trades` table to DuckDB:

```sql
CREATE TABLE IF NOT EXISTS trades (
    trade_id VARCHAR PRIMARY KEY,
    strategy_id VARCHAR,
    symbol VARCHAR,
    side VARCHAR,
    entry_price DOUBLE,
    exit_price DOUBLE,
    size DOUBLE,
    entry_timestamp DOUBLE,
    exit_timestamp DOUBLE,
    realized_pnl DOUBLE,
    fees_paid DOUBLE,
    -- Physics context at entry time
    physics_temperature DOUBLE,
    physics_entropy DOUBLE,
    physics_phase VARCHAR,
    physics_szilard_profit DOUBLE,
    physics_should_trade BOOLEAN,
    physics_size_multiplier DOUBLE,
    physics_ssd_mode VARCHAR,
    -- Regime at entry
    regime VARCHAR
)
```

Add this table in `DuckDBStore._create_tables()` and write to it in `_handle_position_closed`.

**B. Capture physics context at signal time** in ImpulseEngine:
```python
# In _generate_signal(), when emitting SIGNAL, attach current physics context
ctx = self._unified_context.get(symbol)
if ctx:
    signal_metadata["physics_at_signal"] = {
        "szilard_profit": ctx.physics.szilard_profit,
        "temperature": ctx.physics.temperature,
        "entropy": ctx.physics.entropy,
        "phase": ctx.physics.phase,
        "size_multiplier": ctx.physics.size_multiplier,
    }
```

Then in `_handle_position_closed` in main.py, persist the physics context from position metadata.

### Step 2: Define the Validation Metrics

**Metric 1: Information Coefficient (IC)**
```
IC = Spearman rank correlation(szilard_profit_at_signal, forward_return_N_minutes)
```
Where forward_return_N = (exit_price - entry_price) / entry_price for longs.
- N = 5 min, 15 min (matching typical hold duration of max 15 min)
- Accept if |IC| > 0.05 and p-value < 0.05 with N >= 30 trades

**Metric 2: Quintile Analysis (Regime Indicator Test)**
```
Bucket trades into quintiles by szilard_profit at entry.
Compare average realized_pnl per quintile (net of fees).
```
Hypothesis: Q5 (highest Szilard) should outperform Q1 (lowest Szilard).
- If monotonically increasing Q1→Q5: Szilard is a valid regime indicator
- If no pattern: Szilard has no predictive value

**Metric 3: should_trade Gate Effectiveness**
```
Compare average realized_pnl when physics.should_trade=True vs False.
```
This directly tests whether the Szilard-derived trade gate adds value.

**Metric 4: size_multiplier Attribution**
```
Regression: realized_pnl ~ alpha + beta * size_multiplier + epsilon
```
Tests whether size_multiplier (driven by Szilard's calculate_optimal_size_multiplier)
correlates with better outcomes.

**Metric 5: Optimal PRACTICAL_SCALE Estimation**
Once enough trade data exists:
```
OLS: actual_profit = alpha + PRACTICAL_SCALE_estimate * (T * log2(1/p)) * capital / 1000 + epsilon
```
Compare estimated scale to 0.001 (current hardcoded value).

### Step 3: Minimum Sample Size
For IC test with power 0.80, alpha 0.05, and expected IC of 0.15:
- Required N ≈ 350 trades
At 2-8 trades/day (ImpulseEngine + other strategies), this requires 45-175 trading days.
At current observed equity (no realized PnL in autopilot snapshots), trading may be blocked.

## 6. Hypothesis: What Szilard Likely Measures vs What It Should Measure

### What it currently is
A monotonic function of temperature × information content. Since volume is zero:
- T = 0 always → MAX_PROFIT = 0 always
- Even if volume were non-zero: T = KE/N = Σ(ΔP*V)² / N — this has units of ($/BTC * BTC)²/tick
  = $² per tick. Not dimensionally consistent with "temperature" in physics.

### What it intends to be
A theoretical upper bound on extractable profit per unit capital per period, informed by:
- Market volatility (T high = more energy = more opportunity)
- Signal quality (p close to 0 or 1 = more information = more extractable profit)

### The thermodynamic analogy problem
In Szilard's engine, information about the particle position converts to work. But:
- The "temperature" T in that context is in Kelvin (thermal energy kT)
- The work extracted = kT * log(2) per bit of information
- Here, T is defined as kinetic energy per tick, not thermal energy per degree of freedom
- The analogy breaks when T can be arbitrarily large (BTC * volume squared ≫ kT)
- PRACTICAL_SCALE = 0.001 is an ad-hoc rescaling with no physical basis

### More Accurate Interpretation
Szilard_profit is best understood as a heuristic regime quality score, not a profit prediction:
- High Szilard: market has energy (volatility) AND a measurable information edge
- Low Szilard: either market is too cold (no volatility) or signal is too uncertain
- It should be used as a regime filter (binary: trade/no-trade) not a profit estimator

## 7. Concrete Recommendations

### Priority 1: Fix Volume Ingestion (Impact: Restores entire physics subsystem)
File: `backend/packages/hean-exchange/src/hean/exchange/bybit/ws_public.py`
Find where tick volume is parsed and ensure `v` field from Bybit trade events is captured.
Without this fix, szilard_profit = 0 always, physics is disabled in effect.

### Priority 2: Add Trades Persistence (Impact: Enables all future validation)
File: `backend/packages/hean-observability/src/hean/storage/duckdb_store.py`
Add `trades` table and wire to POSITION_CLOSED event in main.py.
Also capture physics context at signal time from position metadata.

### Priority 3: Normalize Szilard Output for Comparability
Current range: 0 to $2.00 (0 to 66 bps on $300).
Recommended: publish as a z-score normalized value (like temperature_normalized and entropy_normalized
which already exist in PhysicsState). Add `szilard_profit_normalized` to PhysicsState.

### Priority 4: Add Szilard to Phase Gate Logic More Explicitly
Currently should_trade uses temperature + entropy + phase. Szilard_profit value itself is not
used as a gate threshold. Propose adding:
```python
# In szilard.should_trade():
if max_profit < min_profit_threshold:
    return False, f"Szilard profit too low: ${max_profit:.4f} < ${min_profit_threshold:.4f}"
```

### Priority 5: Replace PRACTICAL_SCALE with Empirical Calibration
After 300+ trades, fit OLS to estimate the correct scaling factor.
The current 0.001 is arbitrary and was never calibrated to actual HEAN trade results.

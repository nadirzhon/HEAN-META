# HEAN Capital Allocation Analysis — 2026-02-21

## Context
- System: $300 USDT testnet, 11 strategies
- Deployable capital: $240 (80% after 20% cash reserve)
- Min/max per strategy: 5%/$15 — 40%/$120 (StrategyCapitalAllocator hard limits)
- Analysis: Pre-track-record (no real trade history available)
- Method used: qualitative risk-adjusted allocation based on:
  (a) signal frequency / slot contention risk
  (b) regime correlation
  (c) capital efficiency given $300 constraint
  (d) impact of three signal chain fixes

## Base Recommended Allocation (equal weight starting point)
11 strategies × ~9.1% = 100%
At $240 deployable: $21.8/strategy

## Optimal Starting Allocation (post-fix, risk-adjusted)
Strategy                | Weight | Amount  | Rationale
------------------------|--------|---------|-------------------------------------------
ImpulseEngine           | 18%    | $43.2   | Primary alpha source, 12-layer filter = high precision
FundingHarvester        | 14%    | $33.6   | Uncorrelated to momentum, near-zero directional risk
BasisArbitrage          | 10%    | $24.0   | Correlated with FundingHarvester; cap to avoid overlap
InventoryNeutralMM      | 10%    | $24.0   | Spread income, delta-neutral, benefits from InFlight fix
HFScalping              | 8%     | $19.2   | High frequency benefits from InFlight fairness fix
LiquiditySweep          | 10%    | $24.0   | Uncorrelated (institutional sweep detection)
MomentumTrader          | 8%     | $19.2   | Complements ImpulseEngine in trending regimes
EnhancedGrid            | 6%     | $14.4   | Range-only; low capital needed, diversifier
CorrelationArbitrage    | 6%     | $14.4   | Mean-reversion, low correlation to trend strategies
RebateFarmer            | 5%     | $12.0   | Passive, lowest risk, minimal capital needed
SentimentStrategy       | 5%     | $12.0   | New/trial status; sentiment orthogonal to all others

Total: 100% / $240.0

## Key Allocation Decisions Explained

### ImpulseEngine at 18% (highest)
- 12-layer filter cascade rejects 70-95% of raw signals
- Signals that pass have high statistical edge
- OFI + Oracle Engine integration = superior signal quality
- ImpulseEngine is the least likely to get POSITION_SIZER_ZERO (clear stop_loss mandatory in filter)
- Slot contention reduced by InFlightTracker — its 2min cooldown makes it less
  aggressive than HFScalping; it benefits from fair access more than it wins races

### FundingHarvester at 14%
- ~0% correlation to ImpulseEngine (orthogonal signal source: funding rate)
- Fills are rare but highly predictable
- DLQ fix matters most here: a missed ORDER_FILLED event for funding position
  previously caused accounting divergence. Now reliable.
- Benefits from larger allocation because it uses tiny position sizes per trade

### BasisArbitrage capped at 10%
- Correlated with FundingHarvester (both ice/accumulation phase)
- Holding both at high weight = redundant risk; cap enforced
- Provides marginal diversification vs FundingHarvester alone

### InventoryNeutralMM at 10%
- The strategy most impacted by InFlightTracker: 30s order intervals × 10 symbols
  = high contention previously. Now gets fair slot access.
- OFI-gated: only fires in truly balanced markets → high precision signals
- Delta-neutral by design = very low directional risk

### HFScalping at 8% (below neutral)
- The most aggressive racer in the old system (30s cooldown, tick-driven)
- Post InFlightTracker: fair access → fewer but cleaner fills
- Small allocation appropriate: high trade frequency but thin edge per trade
- Sizer=0 risk: at 8% = $19.2 budget. BTC at $100k → min_notional=$100/$100k=0.001 BTC
  At $19.2 equity, 1% risk = $0.19. Position = $0.19/0.02 = $9.5 notional < $100 min
  → POSITION_SIZER_ZERO will fire frequently for BTC but not ETH/SOL
  Recommendation: route HFScalping to lower-priced symbols (SOL, XRP, DOGE)

### LiquiditySweep at 10%
- Event-driven (institutional sweeps), low frequency, 15min cooldown
- Excellent diversifier: fires in markup/markdown/distribution — opposite of calm strategies
- Medium-risk but high reward when pattern fires correctly

### EnhancedGrid + CorrelationArbitrage at 6% each
- Range-bound and mean-reversion strategies
- Provide diversification against trending strategies at low capital cost
- Each needs only enough capital for a few grid levels ($14.4 is sufficient)

### RebateFarmer + SentimentStrategy at 5% minimum floor
- RebateFarmer: needs almost no capital (places deep limit orders at ±0.5%)
  DLQ fix matters here: fill events for deep limits were timing-sensitive
- SentimentStrategy: trial allocation until track record builds
  Sentiment is an orthogonal signal source — valuable diversifier if proven

## PositionSizer=0 Risk by Strategy (at $300 capital)
Strategy                | Budget   | Risk amt | Min notional issue?
------------------------|----------|----------|---------------------
ImpulseEngine           | $43.2    | $0.43    | BTC: $0.43/0.02=$21.5 < $100 min → ZERO
                        |          |          | ETH: $0.43/0.02=$21.5 < $100 → ZERO
                        |          |          | Note: ImpulseEngine sets stop explicitly
FundingHarvester        | $33.6    | $0.34    | Rare fills anyway, manageable
HFScalping              | $19.2    | $0.19    | HIGH RISK of SIZER_ZERO on all majors
InventoryNeutralMM      | $24.0    | $0.24    | Medium risk — uses 2% equity per side
RebateFarmer            | $12.0    | $0.12    | Low risk — 1% equity per side at far offset

## Critical Insight: $100 Minimum Notional Problem
The current PositionSizer enforces a $100 USD minimum notional (line 225 of position_sizer.py):
  min_notional_usd = 100.0
  absolute_min = min_notional_usd / current_price

At $300 total capital with 11 strategies, the 1% risk per trade produces:
  risk_amount = $27 × 0.01 = $0.27 USD per trade

For BTC at $100,000: required size = $0.27 / 0.02 = $13.5 → < $100 min → SIZER_ZERO
For ETH at $3,500: required size = $0.27 / 0.02 = $13.5 → < $100 min → SIZER_ZERO
For SOL at $200: required size = $0.27 / 0.02 = $13.5 → < $100 min → SIZER_ZERO

Conclusion: The $100 minimum notional is a MAINNET requirement that is inappropriate
for a $300 testnet account. The sizer returns absolute_min (the $100 minimum)
which costs 33% of total capital per trade — this is what actually allows trades
to execute, but it means effective risk per trade = 33%, not 1%.

This is NOT a problem introduced by the PositionSizer=0 fix — it was pre-existing.
The fix correctly blocks the case where final_size=0.0 (line 306: if final_size > 0 else 0.0)
which happens when equity * 0.001 < absolute_min AND stop_distance check fails.

## Regime-Phase Capital Adjustment (phase_matched component)
The 30% phase-matched component in hybrid mode will dynamically shift capital
based on dominant physics phase. Expected behavior:
- ICE/ACCUMULATION: FundingHarvester + BasisArbitrage + EnhancedGrid get boost
- MARKUP/WATER: ImpulseEngine + MomentumTrader + LiquiditySweep get boost
- MARKDOWN: LiquiditySweep + ImpulseEngine (short bias) + CorrelationArb get boost
- DISTRIBUTION: SentimentStrategy + LiquiditySweep get boost
This is already implemented in PHASE_AFFINITY dict in StrategyCapitalAllocator.

## Rebalancing Schedule
- Automatic: Every trade close triggers _maybe_reallocate() with 1-hour cooldown
- Manual recommendation: Bi-weekly review comparing actual vs target allocations
- Trigger threshold: >5% drift on any strategy (matches _allocations_changed_significantly)

## Scaling Plan (verified 2026-02-22)

### $300-$1000 (Small Account)
- Enable: ImpulseEngine, FundingHarvester, InventoryNeutralMM, LiquiditySweep, RebateFarmer
- Disable: HFScalping (POSITION_SIZER_ZERO too frequent), BasisArbitrage (redundant with FH),
  EnhancedGrid ($14 capital too small for 20 grid levels), CorrelationArb, Sentiment
- Rationale: min_notional_usd=$100 kills any strategy at <$100 budget;
  focus on strategies where 1% risk produces a viable notional size

### $1000-$10000 (Medium Account)
- Enable all 11 strategies; HFScalping and Grid become viable
- min_notional_usd should be reduced to $10 in config (Bybit testnet allows this)
- Target: 40% strategies with direct alpha + 60% structural/passive income

### $10000+ (Large Account)
- All 11 strategies fully active
- HFScalping can be bumped to 12%; InventoryNeutralMM to 12%; ImpulseEngine to 20%
- Strategies that scale poorly: RebateFarmer (flat rebate income regardless of size)
- Strategies that scale well: FundingHarvester, BasisArbitrage, ImpulseEngine

## min_notional_usd Issue (critical for $300 testnet)
File: backend/packages/hean-risk/src/hean/risk/position_sizer.py line 225
`min_notional_usd = 100.0  # $100 minimum order value (Bybit mainnet requirement)`
This is WRONG for testnet. Bybit testnet allows $1 minimum. Change to $5 to enable
meaningful position sizing at $300 total capital. Otherwise most strategies SIZER_ZERO.

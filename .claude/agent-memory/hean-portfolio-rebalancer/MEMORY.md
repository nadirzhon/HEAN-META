# HEAN Portfolio Rebalancer — Agent Memory

## Key Architecture Facts

### File Paths (verified 2026-02-21)
- Config: `backend/packages/hean-core/src/hean/config.py` (HEANSettings)
- Main/TradingSystem: `backend/packages/hean-app/src/hean/main.py`
- StrategyCapitalAllocator: `backend/packages/hean-portfolio/src/hean/portfolio/strategy_capital_allocator.py`
- RiskSentinel: `backend/packages/hean-risk/src/hean/risk/risk_sentinel.py`
- InFlightTracker: `backend/packages/hean-core/src/hean/core/inflight_tracker.py`
- PositionSizer: `backend/packages/hean-risk/src/hean/risk/position_sizer.py`
- DLQ: `backend/packages/hean-core/src/hean/core/bus_dlq.py`
- Strategy files: `backend/packages/hean-strategies/src/hean/strategies/`

### Capital Configuration
- Initial capital: 300 USDT (default, set in .env as INITIAL_CAPITAL)
- Cash reserve: 20% (cash_reserve_rate=0.2) → ~$240 deployable
- Max single strategy: 40% ($120) — hard limit in StrategyCapitalAllocator
- Min strategy allocation: 5% ($15) — hard limit in StrategyCapitalAllocator
- Capital allocation method: "hybrid" (70% performance-weighted + 30% phase-matched)
- Feature flag: `strategy_capital_allocation=True` (already enabled by default)
- Max position shift per reallocation: 10% per strategy (gradual)
- Reallocation cooldown: 1 hour (REALLOCATION_COOLDOWN_SECONDS=3600)

### Strategy Signal Frequency Profile (estimated from code analysis)
- HIGH frequency (tick-driven, short cooldowns):
  - HFScalping: 40-60 trades/day, 30s cooldown per symbol, RANGE/NORMAL only
  - InventoryNeutralMM: 30s order interval, RANGE/NORMAL only, OFI-gated
  - RebateFarmer: 5min refresh, passive/deep limit orders, very low directional
- MEDIUM frequency (event-driven, moderate cooldowns):
  - ImpulseEngine: up to 120 attempts/day, 2min cooldown, IMPULSE/NORMAL, 12-layer filter cascade
  - LiquiditySweep: event-driven (sweep detection), 15min cooldown per symbol
  - MomentumTrader: trend-following, moderate frequency
  - EnhancedGrid: range-bound, grid management style
- LOW frequency (rate-driven or sentiment-driven):
  - FundingHarvester: every 8 hours (funding intervals), all regimes
  - BasisArbitrage: funding + basis spread driven, low frequency
  - CorrelationArbitrage: pair divergence driven, low frequency
  - SentimentStrategy: sentiment poll-driven (300s+ intervals)

### The Three Signal Chain Fixes (analysed 2026-02-21)

**Fix 1: InFlightTracker** (in-flight order slot reservation)
- Prevents concurrent signal overflow between RiskSentinel recomputes
- Uses GIL-safe synchronous reserve() call before ORDER_REQUEST publish
- 30s TTL safety backstop; GC loop every 5s
- Effective capacity = open_positions + in_flight < max_positions
- Impact: HIGH FAIRNESS gain for low-frequency strategies. HFScalping and
  InventoryNeutralMM were the most likely racers (30s cooldown = many slots
  contested). With tracker, slot reservation is atomic — race eliminated.

**Fix 2: PositionSizer=0 now blocks trades** (line 1787 in main.py)
- Old behavior: size=0 was overridden with a micro-size minimum
- New behavior: size=0 → hard REJECT with reason_code="POSITION_SIZER_ZERO"
- Strategies affected: those generating signals without clear stop_loss or with
  very low equity budget (e.g., 5% floor = $15 → sizer may return 0 at $100k BTC)
- Impact: Cleans signal-to-noise ratio. Low-equity micro-cap strategies get fewer
  but better trades. Equal-allocation on 11 strategies = $27/strategy — borderline.

**Fix 3: DLQ Auto-Retry** (bus_dlq.py)
- Captures failed SIGNAL, ORDER_REQUEST, ORDER_FILLED, POSITION_CLOSED, etc.
- maxsize=1000, max_retries=3
- Strategies that had timing-sensitive handlers (POSITION_CLOSED → PnL update →
  reallocation trigger) may have been silently losing events. DLQ recovery means
  the StrategyCapitalAllocator now reliably sees all trade outcomes.
- FundingHarvester and BasisArbitrage benefit most (rare fills matter enormously
  for PnL tracking; missing one fill = wrong allocation signal)

### Correlation Structure (qualitative, from code regime/phase affinity)
- HIGH correlation pairs (same regime, similar logic):
  - FundingHarvester + BasisArbitrage: both ice/accumulation phase, funding-rate driven
  - HFScalping + InventoryNeutralMM: both RANGE/NORMAL, spread-capture, tick-driven
  - ImpulseEngine + MomentumTrader: both markup/trending, momentum logic
- LOW/NEGATIVE correlation pairs (diversification value):
  - FundingHarvester vs ImpulseEngine: opposite regime affinity (ice vs impulse)
  - BasisArbitrage vs LiquiditySweep: calm vs volatile transition phases
  - SentimentStrategy vs HFScalping: orthogonal signal sources (sentiment vs tick momentum)
  - RebateFarmer vs ImpulseEngine: counter-directional (rebate farmer avoids fills)
  - CorrelationArbitrage vs MomentumTrader: mean-reversion vs trend-following

### Recommended Allocation (post-fixes, $300 capital)
See: allocation-analysis.md

### AutoPilot Coordinator Note
AutoPilot (enabled by default) dynamically enables/disables strategies based on
Thompson Sampling. It respects min=2 / max=8 active strategies. Capital allocator
must be consistent with AutoPilot's active set — RiskSentinel handles this via
set_active_strategies() and equal-split fallback when allocator has no data.

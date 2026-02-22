# HEAN Profit Optimizer Memory

## Key Fee & Cost Facts (from source)
- Maker fee: -0.01% (rebate) — `backtest_maker_fee = 0.00005` (0.005% in config, but Bybit testnet = -0.01%)
- Taker fee: 0.03% in config (`backtest_taker_fee = 0.0003`), real Bybit = 0.06%
- Funding: 0.01%/8h typical, paid 3x/day = 0.03%/day on notional
- `maker_first = True` — system attempts maker orders by default
- `maker_ttl_ms = 150` — aggressive 150ms TTL, high expiry rate likely
- `allow_taker_fallback = True` — expired makers fall back to taker (costs 0.06%)
- `maker_price_offset_bps = 1` — 1 bps inside BBO, good fill probability
- http.py line 480: `timeInForce = "PostOnly"` for limit, `"IOC"` for market — CORRECT
- main.py line 3838: paper-mode close fee hardcoded as `0.00055` (5.5bps) — consistent with Bybit
- taker fallback edge check in router.py line 977: uses `settings.backtest_taker_fee` (3bps) NOT real 5.5bps
- VOLATILITY BUG: router.py line 1177 and router_bybit_only.py line 1190 compute `abs(...)` but discard
  the result — `_update_volatility_history` only appends PRICES, not RETURNS. Adaptive TTL/offset uses
  `_get_current_volatility` which correctly computes returns from prices, so no leakage here — but the
  intermediate calculation on line 1190 is dead code wasted on every tick.
- Position close in main.py uses `order_type="market"` always — forced taker on every exit (5.5bps)
- No limit-order TP exits — all closes are market orders (TP hit, SL hit, TTL all use market)

## Capital & Risk (from config defaults)
- `initial_capital = 300 USDT`
- `max_trade_risk_pct = 1.0%` — $3 max risk per trade
- `max_leverage = 3.0x` — capped at 3x (SmartLeverage can reach 4x with edge)
- `max_open_positions = 10`
- `max_hold_seconds = 900` — 15 min max hold, forced exit
- `killswitch_drawdown_pct = 30%` — $90 max total loss from capital
- `consecutive_losses_limit = 3` — 2h pause after 3 consecutive losses
- Position size floor: `min_notional_usd = 100.0` — $100 min order (Bybit mainnet req)
- Kelly: `fractional_kelly = 0.25` (quarter Kelly), adaptive 0.15-0.50

## Strategy Activity Rates
- ImpulseEngine: max 120 attempts/day, 2 min cooldown, 3-filter cascade (spread/vol/time)
  - Time windows: 07-11, 12-16, 17-21 UTC (9 hours active/day)
  - Filter rejection: 70-95% of signals blocked
  - Net trades: likely 2-8/day per symbol pair
- FundingHarvester: max 6 signals/day, 4h cooldown, 2 symbols (BTC+ETH)
  - Minimum funding threshold: 0.01%
  - Uses `prefer_maker=True`
- BasisArbitrage: max 4 signals/day, 2h cooldown
- HFScalpingStrategy: targets 40-60 trades/day, 30s cooldown, 4 symbols
  - TP=25bps, SL=15bps, min_move=10bps
- RebateFarmer: passive, refresh every 5 min, 1% equity per side
- Other 6 strategies (MomentumTrader, CorrelationArb, EnhancedGrid, InventoryNeutralMM, LiquiditySweep, SentimentStrategy): all enabled

## Execution Architecture (router_bybit_only.py)
- maker_first path → SmartLimitExecutor → post-only limit @ bid-1bps
- TTL 150ms → if expired → MakerRetryQueue → taker fallback (if edge > 2bps net)
- OFI imbalance detected → overrides to taker immediately with 1.5x size
- No paper trading; all orders to Bybit testnet

## Critical Sizing Constraint
- `min_notional_usd = 100.0` is hardcoded in PositionSizer for both `calculate_size` and `calculate_size_v2`
- With $300 capital, max 3 simultaneous $100 min positions = 100% capital deployed
- `max_trade_risk_pct = 1%` → $3 risk per trade → at 2% stop = $150 notional (0.5x leverage)
- Kelly not active until 10+ trades per strategy (returns 1.0 multiplier until then)

## Szilard Engine — Validated Findings (see szilard_analysis.md)
- Formula: MAX_PROFIT = T * log2(1/p) * 0.001 * capital / 1000
- With default capital=1000: effective scale = 1e-6 per unit temperature per bit
- Typical output range: $0.00005 - $2.00 (0.002 to 66 bps on $300 account)
- CRITICAL BUG: Temperature = KE/N where KE = Σ(ΔP * V)^2 — since ALL stored ticks have
  volume=0, temperature is always 0, making szilard_profit always 0 in practice
- physics_snapshots table is EMPTY (0 rows) — Szilard is never persisted in current session
- Szilard is PASSIVE: no strategy directly gates on szilard_profit value
- Szilard contributes indirectly via: should_trade (gate) + size_multiplier (0.0-2.0 multiplier)
- In _FallbackPackage (sovereign_brain.py): szilard signal = min(1, max(-1, szilard * 10.0))
  — this signal is a no-op when szilard_profit=0
- No trades table exists in DuckDB — realized PnL is pure in-memory only
- Autopilot journal has equity snapshots (max equity seen: $502.65, recent: $451.22)

## Data Availability for Validation
- DuckDB: ticks table has 187,657 rows (BTC/ETH/SOL/XRP/BNB), Feb 8-21
  - ALL volumes are 0.000 — data quality issue in tick ingestion
- physics_snapshots: 0 rows — never flushed from in-process buffer
- No trades/fills table — realized PnL not persisted anywhere
- autopilot_journal: 154 equity snapshots, 3685 strategy enable/disable decisions
- CONCLUSION: Cannot do historical Szilard vs PnL correlation with existing data

## Profit Decomposition Reference (see detailed analysis below)
- See: `profit_decomposition_300.md`
- See: `szilard_analysis.md`
- See: `profit_leakage_full_analysis.md` — full 10-point leakage audit (Feb 2026)

## Top 10 Profit Leakage Points (Feb 2026 Audit)
1. **min_notional hardcoded 100.0** in position_sizer.py lines 225/302 — ignores config's min_notional_usd=5.0
2. **maker_ttl_ms=150** too short for momentum signals — high expiry rate → taker fallback at 5.5bps cost
3. **HFScalping negative EV**: TP=25bps, SL=15bps, ~60% WR → gross EV=9bps < 11bps round-trip taker fee
4. **Filter cascade over-rejects**: 70-95% rejection + 2-step EdgeConfirmation + TimeWindow blocks 12h/day
5. **RiskGovernor SOFT_BRAKE at 10%** from peak — triggers on normal pullbacks during profitable periods
6. **FundingHarvester 4h cooldown** misses 1 of 3 daily payments (0:00, 8:00, 16:00 UTC)
7. **No trailing stop**: Fixed 1.5% TP exits too early in trending markets; break-even stop wastes 0.7% gains
8. **Kelly inactive <10 trades**: Returns 0.0 → 0.5x multiplier for all strategies until 10 trades each
9. **Correlation drag**: ImpulseEngine + HFScalping + MomentumTrader all momentum → 100% correlated losses
10. **Taker fee misconfigured**: router uses backtest_taker_fee=3bps for fallback check; actual Bybit=5.5bps

## Fee Reality (validated)
- backtest_taker_fee=0.0003 (3bps) in config — UNDERCOUNTS real Bybit taker of 5.5bps
- backtest_maker_fee=0.00005 (0.5bps) — but Bybit testnet maker = -1bps (rebate)
- Taker fallback edge check in router uses 3bps cost → approves trades that are actually negative edge

## HFScalping Fix Required
- Current: TP=25bps, SL=15bps — negative EV with taker fees
- Fix: TP=35bps, SL=12bps → R:R=2.9, EV=+8.85bps net after blended fees

## Signal Chain Reliability Fixes (Feb 2026)
- RISK_ENVELOPE was NORMAL priority in legacy code, now CRITICAL in bus.py line 107 — strategies
  received stale envelope under backpressure, trading without live risk limits
- InFlightTracker: `backend/packages/hean-core/src/hean/core/inflight_tracker.py` — atomic
  in-flight slot reservation, GC loop every 5s, TTL 30s default
- Legacy `_handle_signal_legacy` lines 2336-2340 forced min size when sizer returned 0;
  new `_handle_enriched_signal` lines 1787-1810 correctly vetos the trade
- DLQ (`bus_dlq.py`): auto-retry loop runs every 10s with 2^n * 10s backoff; covers all
  CRITICAL event types; max 3 retries then permanent failure list
- Close position retry: _handle_position_close_request (router line 448-459) — still
  single-attempt as of last read (no retry loop added yet)
- ORDER_FILLED dedup: `_republished` flag prevents double-processing in _handle_order_filled;
  dedup dict `_filled_order_ids` bounded to 5000 entries
- RiskGovernor state was NEVER checked in old _handle_signal (documented in sentinel line 58)
- risk_sentinel_update_interval_ms = 1000ms (1s debounce for TICK-triggered recomputes)
- Position changes (ORDER_FILLED, etc.) trigger IMMEDIATE recompute (no debounce)
- See: `signal_chain_reliability_analysis.md` for full profit impact breakdown

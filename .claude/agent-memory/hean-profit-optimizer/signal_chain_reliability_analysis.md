# Signal Chain Reliability Fixes — Profit Impact Analysis
# Created: 2026-02-21
# Capital base: $300 USDT initial, equity observed ~$450 at time of analysis

## Key Architectural Facts Used in Estimates
- 11 strategies active, ~8-25 net trades/day total across all strategies
- Bybit testnet taker fee: 0.03% (config), ~0.06% real; maker rebate: -0.01%
- min_notional_usd = 5.0 (config default), effective ~$100 due to Bybit minimums
- max_open_positions = 10 (config)
- InFlightTracker: timeout 30s, GC every 5s
- DLQ: 10s base interval, 3 max retries, exponential backoff

## Fix 1: RISK_ENVELOPE Priority NORMAL→CRITICAL
- Bug: strategies received stale envelope during TICK backpressure → RiskGovernor
  HARD_STOP / deposit_protection states were invisible to strategies for 100s of ms
- Source: bus.py line 107 (`EventType.RISK_ENVELOPE: EventPriority.CRITICAL`)
- RiskGovernor check was NEVER called in old _handle_signal (sentinel.py line 58)
- Estimated frequency: TICK rate ~10/s * 5 symbols = 50 TICK/s. At queue utilization
  >80%, NORMAL events delayed behind TICK storms. With 1s risk_sentinel interval,
  stale envelope window = up to 1-2s during active trading.

## Fix 2: InFlightTracker
- Bug: N strategies could each see 0 positions → all send ORDER_REQUEST simultaneously
  → exchange sees N orders → max_open_positions violated
- With max 10 positions and 11 strategies: worst case = 10 concurrent over-leverage orders
- At $100 notional min each: $1000 exposure on $300 capital = 3.3x unintended leverage
- InFlightTracker: O(1) dict lookup, GC every 5s, TTL 30s fallback

## Fix 3: PositionSizer=0 Respected
- Legacy code (main.py lines 2336-2340): if base_size <= 0, forced min 0.1% equity / price
- New code (main.py lines 1787-1810): if base_size <= 0, REJECT with POSITION_SIZER_ZERO
- PositionSizer returns 0 when: equity too low, drawdown too high, signal edge too weak,
  or min_notional_usd cannot be met
- Post-fix: enforces min_size = max(equity*0.001/price, 0.001) ONLY after size passes > 0 check

## Fix 4: DLQ Auto-Retry
- Bus DLQ covers all CRITICAL event types including ORDER_REQUEST
- Auto-retry: 10s base, backoff 2^n * 10s (10s, 20s, 40s then permanent failure)
- Recovery window: up to 70s total for 3 retries
- Relevant for transient Bybit API errors, coroutine exceptions in handlers

## Fix 5: Close Position Retry
- Current state: single-attempt only (router.py lines 448-459), no retry loop
- On failure: only logs error, position stays open, no follow-up
- Fix needed: add retry with exponential backoff (3x, 1s/2s/4s)

## Fix 6: ORDER_FILLED Dedup
- `_republished` flag: TradingSystem re-publishes enriched ORDER_FILLED for downstream
  listeners. The `if event.data.get("_republished"): return` guard prevents double
  accounting/position creation in _handle_order_filled
- Also: `_filled_order_ids` dict (bounded 5000) deduplicates partial fills from Bybit
- RiskSentinel also subscribes to ORDER_FILLED → triggers immediate envelope recompute
  (no debounce), so double-process would cause 2x envelope recompute per fill

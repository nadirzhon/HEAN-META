# HEAN Microstructure Analyst Memory

## Key File Locations
- Execution router: `backend/packages/hean-execution/src/hean/execution/router_bybit_only.py`
- OFI monitor: `backend/packages/hean-core/src/hean/core/ofi.py`
- Position reconciler: `backend/packages/hean-execution/src/hean/execution/position_reconciliation.py`
- Maker retry queue: `backend/packages/hean-execution/src/hean/execution/maker_retry_queue.py`
- Slippage estimator: `backend/packages/hean-execution/src/hean/execution/slippage_estimator.py`

## OFI Implementation (ofi.py) — Critical Gaps Found
- Python fallback uses STATIC orderbook snapshot (total bid vs ask volume), NOT delta-based OFI
- This measures book imbalance (depth ratio), not true Order Flow Imbalance
- True OFI = sum of (delta_bid_qty - delta_ask_qty) across levels between snapshots
- C++ path via `graph_engine_py.OFIMonitor` is the real implementation; Python is degraded
- `get_aggression_factor()` returns OFI bias in [0,1] — consumed by SmartLimitExecutor
- OFI publishes `OFI_UPDATE` events consumed by ContextAggregator

## ExecutionRouter — Existing Microstructure Hooks
- `detect_orderbook_imbalance()`: ratio threshold 3:1, size_multiplier 1.5x for taker
- `_calculate_optimal_ttl()`: spread/volatility/fill-rate adaptive TTL (50-500ms range)
- `_route_maker_first()`: uses `get_aggression_factor()` to adjust SmartLimitExecutor
- Market order timeout: soft=15s, hard=30s — tracked via `_pending_market_orders` dict
- Volatility gating: percentile thresholds 75/90/99 for medium/soft/hard block
- Taker fallback: requires net_edge >= 2 bps after spread + fee (backtest_taker_fee * 10000 + 5 bps slippage)

## Reconciler — Key Architecture Decisions
- Reconciliation interval: 60s (was 30s, relaxed to reduce exchange load)
- Orphan confirmation: 3 consecutive detections before closing (prevents race with ORDER_FILLED)
- Order grace period: 120s after ORDER_PLACED before treating as orphan
- Emergency halt threshold: 10 consecutive drifts (was 3, relaxed)
- Ghost positions (local only, not exchange): auto-closed immediately with POSITION_CLOSED event

## Retry Queue Logic
- Max retries: 2, min delay: 5s, max delay: 60s
- Retry condition: volatility improved 10%+ OR max delay exceeded
- Clears entirely on regime change; skips on capital preservation mode

## Bybit Testnet Fee Structure (for alpha calculations)
- Taker fee: `settings.backtest_taker_fee * 10000` bps (typically ~7 bps = 0.07%)
- Assumed slippage: 5 bps (hardcoded in taker fallback logic)
- Minimum edge for taker fallback: 2 bps net after all costs
- Round-trip minimum alpha needed: ~(2 * taker_fee) + spread + 2 bps = ~20 bps for BTCUSDT

## Market Order Timeout — Current State
- `_pending_market_orders` dict is populated but the `_market_order_check_task` is NEVER started
- This is a phantom order recovery gap — the timeout infrastructure exists but is not wired
- The `_market_order_check_task` field is set to None in __init__ and never assigned

## HFScalping Strategy — Current State (Confirmed)
- Momentum-only: compares prices[0] vs prices[-1] over a 5-tick window, no book data
- Cooldown: 30s between signals — severely limits frequency
- No OFI, no spread awareness, no side inference from trade data
- Trade side ("Buy"/"Sell") IS available in TICK events from publicTrade topic (event.data["trade_side"])
- Trade qty IS available (event.data["trade_qty"]) — can compute aggressor volume

## Bybit WebSocket — What Data Is Available
- TICK from ticker: price, bid1, ask1 (no volume/side)
- TICK from publicTrade: price, qty, side ("Buy"/"Sell"), timestamp_ms — key for VPIN/toxic flow
- ORDER_BOOK_UPDATE: bids/asks as [[price, qty], ...], update_id (u), timestamp_ns (ts)
  - Type "snapshot" vs "delta" — delta updates have qty=0 for removals
- OFI delta-based calculation requires tracking previous snapshot and diffing per price level

## ParticipantClassifier (physics module) — Available Context
- Publishes CONTEXT_UPDATE with context_type="participant_breakdown"
- Fields: mm_activity, institutional_flow, retail_sentiment, whale_activity, arb_pressure
- meta_signal: natural-language signal like "Retail FOMO, reversal likely"
- institutional_iceberg_detected: bool — useful for toxic flow detection
- All strategies can consume via on_context_ready() (BaseStrategy already subscribes CONTEXT_READY)

## Key Integration Points for New Microstructure Strategies
- Subscribe to ORDER_BOOK_UPDATE for true delta-OFI (track prev_bids/prev_asks per level)
- Subscribe to TICK; check event.data["trade_side"] and event.data["trade_qty"] for aggressor data
- Publish features as CONTEXT_UPDATE (context_type="microstructure_features") for downstream
- CONTEXT_READY is the aggregated event strategies receive — add microstructure data there
- SSD lens filter in BaseStrategy auto-suppresses signals in entropy divergence regimes
- Risk envelope blocks trading before on_tick() reaches strategy — no double-filtering needed

See: `execution-microstructure-patterns.md` for detailed analysis and formulas

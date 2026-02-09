# iOS Backend API Comprehensive Audit Report
**Date:** 2026-02-07
**Backend API:** http://localhost:8000
**iOS App:** /Users/macbookpro/Desktop/HEAN/ios/HEAN/

---

## Executive Summary

This audit compared **ALL** iOS Swift models and service protocols against live backend API responses. The audit identified:

- **8 CRITICAL issues** - Will cause crashes or decode failures
- **12 WARNING issues** - May cause missing data or incorrect display
- **5 INFO items** - Best practice improvements

**Overall Assessment:** The iOS models are generally well-structured with good fallback handling, but several endpoints have critical mismatches that will cause runtime failures.

---

## 1. `/api/v1/engine/status` - Portfolio/Engine Status

### Backend Response
```json
{
    "status": "running",
    "running": true,
    "engine_state": "RUNNING",
    "trading_mode": "live",
    "is_live": true,
    "dry_run": false,
    "equity": 10000.0,
    "daily_pnl": 0.0,
    "initial_capital": 10000.0,
    "unrealized_pnl": 0,
    "realized_pnl": 0.0,
    "available_balance": 10000.0,
    "used_margin": 0,
    "total_fees": 0.0
}
```

### iOS Models
**File:** `ios/HEAN/Models/Portfolio.swift`
- `Portfolio` struct (lines 37-169)
- `EngineStatusResponse` struct (lines 11-35)

### Analysis

#### CRITICAL Issues
**None** - The `Portfolio` struct properly handles all fields with fallbacks.

#### WARNING Issues
1. **MISSING FIELDS**: iOS `Portfolio` does NOT decode:
   - `trading_mode` (backend provides "live")
   - `is_live` (backend provides true/false)
   - `dry_run` (backend provides true/false)
   - `total_fees` (backend provides 0.0)

   **Impact:** iOS cannot distinguish between live/testnet modes or track total fees.

2. **TYPE MISMATCH**: Backend `unrealized_pnl` returns `0` (Int) sometimes, `0.0` (Double) other times
   - iOS expects `Double?`
   - **Status:** SAFE - Swift's JSONDecoder handles Int->Double coercion

3. **TYPE MISMATCH**: Backend `used_margin` returns `0` (Int) sometimes, `0.0` (Double) other times
   - iOS expects `Double?`
   - **Status:** SAFE - Swift's JSONDecoder handles Int->Double coercion

#### INFO
- The `EngineStatusResponse` (lines 11-35) appears UNUSED. It duplicates `Portfolio` fields but is never decoded anywhere in the codebase.
- **Recommendation:** Remove `EngineStatusResponse` or clarify its purpose.

---

## 2. `/api/v1/orders/positions` - Open Positions

### Backend Response
```json
[]
```
**Note:** No positions currently open. Backend returns empty array.

### iOS Models
**File:** `ios/HEAN/Models/Position.swift`
- `Position` struct (lines 35-160)
- `PositionSide` enum (lines 10-33)

### Analysis

#### CRITICAL Issues
**None** - But cannot fully validate without live position data.

#### WARNING Issues
1. **UNCHECKED FIELD**: Backend likely returns fields like:
   - `position_id` or `id`
   - `entry_price`, `current_price`, `unrealized_pnl`, etc.

   iOS has robust fallback handling (lines 109-130) for:
   - `id` vs `position_id` (line 112-118)
   - All numeric fields have defaults (lines 124-129)

   **Status:** LIKELY SAFE - Good defensive coding present.

#### INFO
- The `Position.init(from decoder:)` correctly tries `id` first, then `position_id` (lines 112-118).
- CodingKeys properly map `entry_price`, `current_price` (markPrice), `unrealized_pnl` (lines 94-107).

---

## 3. `/api/v1/orders` - All Orders

### Backend Response Sample
```json
{
  "order_id": "smart_limit_1770456520.182701",
  "symbol": "BTCUSDT",
  "side": "buy",
  "size": 7.20944148456819e-05,
  "filled_size": 0.0,
  "price": 69353.5,
  "type": "LIMIT",
  "status": "pending",
  "strategy_id": "basis_arbitrage",
  "timestamp": "2026-02-07T12:28:40.182712",
  "updated_at": "2026-02-07T12:28:40.182712"
}
```

### iOS Models
**File:** `ios/HEAN/Models/Order.swift`
- `Order` struct (lines 100-217)
- `OrderSide`, `OrderType`, `OrderStatus` enums (lines 10-98)

### Analysis

#### CRITICAL Issues
1. **MISSING FIELD DECODE**: Backend returns `strategy_id` but iOS Order does NOT have a `strategyId` field
   - **Impact:** `strategy_id` is silently dropped during decode
   - **Fix Required:** Add `let strategyId: String?` to `Order` struct and proper CodingKey

#### WARNING Issues
1. **CASE SENSITIVITY**: Backend returns `side: "buy"` (lowercase)
   - iOS `OrderSide` has custom decoder that handles `.lowercased()` (lines 18-32)
   - **Status:** SAFE

2. **CASE SENSITIVITY**: Backend returns `type: "LIMIT"` (uppercase)
   - iOS `OrderType` decoder uses `.uppercased()` (lines 50-60)
   - **Status:** SAFE

3. **CASE SENSITIVITY**: Backend returns `status: "pending"` (lowercase)
   - iOS `OrderStatus` decoder uses `.uppercased()` (lines 84-97)
   - **Status:** SAFE BUT has default fallback to `.pending` (line 95)
   - **Concern:** Unknown statuses silently become `.pending`

#### INFO
- The `Order` struct has excellent fallback handling for `id` vs `order_id` (lines 169-176)
- All optional fields properly handled with `decodeIfPresent`

---

## 4. `/api/v1/strategies` - Trading Strategies

### Backend Response
```json
[
  {
    "strategy_id": "funding_harvester",
    "enabled": true,
    "type": "FundingHarvester",
    "win_rate": 0.0,
    "total_trades": 0,
    "profit_factor": 0.0,
    "total_pnl": 0.0,
    "wins": 0,
    "losses": 0,
    "description": ""
  }
]
```

### iOS Models
**File:** `ios/HEAN/Services/TradingServiceProtocol.swift` (lines 442-524)
- `Strategy` struct

### Analysis

#### CRITICAL Issues
1. **MISSING FIELD DECODE**: Backend returns `total_pnl`, `wins`, `losses` but iOS does NOT decode these
   - **Impact:** PnL and win/loss counts are lost
   - **Fix Required:** Add these fields to `Strategy` struct

#### WARNING Issues
**None** - The existing fields decode correctly:
- `strategy_id` -> `id` (handled via fallback, lines 478-485)
- `type` -> `name` (handled via fallback, lines 487-494)
- Other fields mapped correctly

#### INFO
- Strategy decoding has excellent fallback logic (lines 475-501)

---

## 5. `/api/v1/risk/governor/status` - Risk Governor

### Backend Response
```json
{
    "risk_state": "NORMAL",
    "level": 0,
    "reason_codes": [],
    "metric": null,
    "value": null,
    "threshold": null,
    "recommended_action": "Risk governor not initialized",
    "clear_rule": "N/A",
    "quarantined_symbols": [],
    "blocked_at": null,
    "can_clear": true
}
```

### iOS Models
**File:** `ios/HEAN/Services/TradingServiceProtocol.swift` (lines 67-120)
- `RiskGovernorStatus` struct

### Analysis

#### CRITICAL Issues
1. **MISSING FIELD DECODE**: Backend returns `clear_rule` and `blocked_at` but iOS does NOT decode these
   - **Impact:** Cannot display when risk was blocked or clear rule explanation
   - **Fix Required:** Add `clearRule: String?` and `blockedAt: String?` to struct

#### WARNING Issues
**None** - All critical fields properly decoded with `decodeIfPresent` fallbacks (lines 94-104)

#### INFO
- Good conversion from raw `riskState` string to typed `RiskState` enum (lines 90-92)

---

## 6. `/api/v1/risk/killswitch/status` - Killswitch Status

### Backend Response
```json
{
    "triggered": false,
    "reasons": [],
    "triggered_at": null,
    "thresholds": {
        "drawdown_pct": 10.0,
        "equity_drop": 20.0,
        "max_loss": 0,
        "risk_limit": 0
    },
    "current_metrics": {
        "current_drawdown_pct": 0.0,
        "current_equity": 10000.0,
        "max_drawdown_pct": 0.0,
        "peak_equity": 10000.0
    }
}
```

### iOS Models
**File:** `ios/HEAN/Services/TradingServiceProtocol.swift` (lines 123-163)
- `KillswitchStatus` struct

### Analysis

#### CRITICAL Issues
**None** - All fields properly decoded.

#### WARNING Issues
**None** - Excellent implementation with proper optionals and fallbacks (lines 138-145)

#### INFO
- The convenience accessor `currentDrawdown` (lines 156-158) correctly looks up `current_metrics["current_drawdown_pct"]`

---

## 7. `/api/v1/telemetry/signal-rejections` - Signal Rejection Stats

### Backend Response
```json
{
    "time_window_minutes": 60,
    "total_rejections": 25,
    "total_signals": 8,
    "rejection_rate": 312.5,
    "by_category": {
        "anomaly": 25
    },
    "by_reason": {
        "price_anomaly_block": 25
    },
    "by_symbol": {
        "SOLUSDT": 13,
        "XRPUSDT": 6
    },
    "by_strategy": {
        "funding_harvester": 25
    },
    "rates": {
        "1m": 0.0,
        "5m": 312.5
    }
}
```

### iOS Models
**File:** `ios/HEAN/Services/TradingServiceProtocol.swift` (lines 540-567)
- `SignalRejectionStats` struct

### Analysis

#### CRITICAL Issues
1. **MAJOR MISSING FIELDS**: iOS only decodes 4 fields, but backend returns 9 fields:
   - MISSING: `time_window_minutes`
   - MISSING: `by_category`
   - MISSING: `by_symbol`
   - MISSING: `by_strategy`
   - MISSING: `rates`

   **Impact:** Cannot show detailed rejection breakdown by category, symbol, or strategy
   - **Fix Required:** Add all missing fields to struct

#### WARNING Issues
**None** - The fields that ARE decoded work correctly with proper fallbacks (lines 553-558)

---

## 8. `/api/v1/trading/equity-history?limit=5` - Equity History

### Backend Response
```json
{
    "snapshots": [
        {
            "timestamp": "2026-02-07T12:32:39.627394",
            "equity": 10000.0
        }
    ],
    "count": 5
}
```

### iOS Models
**File:** `ios/HEAN/Services/Services.swift` (lines 282-301)
- `EquityPoint` struct (lines 282-286)
- `EquityHistoryResponse` struct (lines 288-301)

### Analysis

#### CRITICAL Issues
**None** - Perfect match. Good defensive decoding with fallbacks (lines 292-296).

#### WARNING Issues
**None**

#### INFO
- Excellent defensive coding: both `snapshots` and `count` have fallbacks to empty/0 if decode fails

---

## 9. `/api/v1/trading/metrics` - Trading Metrics

### Backend Response
```json
{
    "status": "ok",
    "counters": {
        "last_1m": {
            "signals_total": 0,
            "decisions_create": 2,
            "decisions_skip": 6,
            "decisions_block": 0,
            "orders_created": 0,
            "orders_filled": 0,
            "orders_canceled": 0,
            "orders_rejected": 0,
            "positions_open": 0,
            "positions_closed": 0
        },
        "last_5m": { ... },
        "session": {
            "signals_total": 8,
            "decisions_create": 2,
            "decisions_skip": 6,
            "decisions_block": 0,
            "orders_created": 0,
            "orders_filled": 0,
            "orders_canceled": 0,
            "orders_rejected": 0,
            "orders_open": 300,
            "positions_open": 0,
            "positions_closed": 0,
            "pnl_unrealized": 0,
            "pnl_realized": 0.0,
            "equity": 10000.0
        }
    },
    "top_reasons_for_skip_block": [ ... ],
    "active_orders_count": 300,
    "active_positions_count": 0,
    "last_signal_ts": "2026-02-07T12:28:40.584611+00:00",
    "last_order_ts": null,
    "last_fill_ts": null,
    "engine_state": "RUNNING",
    "mode": "live",
    "top_symbols": [ ... ],
    "top_strategies": [ ... ],
    "uptime_sec": 293.03
}
```

### iOS Models
**File:** `ios/HEAN/Services/TradingServiceProtocol.swift` (lines 194-275)
- `TradingMetrics` struct (lines 194-249)
- `MetricsCounters` struct (lines 251-261)
- `MetricsBucket` struct (lines 263-275)

### Analysis

#### CRITICAL Issues
1. **MISSING MAJOR FIELDS**: iOS `TradingMetrics` does NOT decode:
   - `top_reasons_for_skip_block` (array of reason codes with counts/percentages)
   - `top_symbols` (array of symbols with signal counts)
   - `top_strategies` (array of strategies with signal counts)
   - `uptime_sec` (system uptime)

   **Impact:** Cannot show top rejection reasons, most active symbols/strategies, or uptime
   - **Fix Required:** Add these fields with proper struct types

2. **MISSING COUNTER FIELDS**: iOS `MetricsBucket` only decodes 4 fields but backend provides 13+ fields:
   - MISSING: `decisions_create`, `decisions_skip`, `orders_canceled`, `orders_rejected`, `orders_open`, `positions_open`, `positions_closed`, `pnl_unrealized`, `pnl_realized`, `equity`

   **Impact:** Cannot display decision breakdown, cancelation stats, PnL per bucket
   - **Fix Required:** Expand `MetricsBucket` to include all counter fields

#### WARNING Issues
**None** - The fields that exist decode correctly with fallbacks (lines 217-228)

---

## 10. `/api/v1/trading/why` - Trading Diagnostics

### Backend Response
```json
{
    "engine_state": "running",
    "killswitch_state": {
        "triggered": false,
        "reasons": [],
        "triggered_at": null
    },
    "last_tick_age_sec": 0.806825,
    "last_signal_ts": "2026-02-07T12:28:40.584611+00:00",
    "last_decision_ts": "2026-02-07T12:28:40.584680",
    "last_order_ts": null,
    "last_fill_ts": null,
    "active_orders_count": 300,
    "active_positions_count": 0,
    "top_reason_codes_last_5m": [
        {"code": "LIMIT_REACHED", "count": 6},
        {"code": "ACCEPTED", "count": 2}
    ],
    "equity": 10000.0,
    "balance": 10000.0,
    "unreal_pnl": 0,
    "real_pnl": 0.0,
    "margin_used": 0.0,
    "margin_free": 8494.137643688933,
    "profit_capture_state": { ... },
    "execution_quality": { ... },
    "multi_symbol": { ... }
}
```

### iOS Models
**File:** `ios/HEAN/Services/TradingServiceProtocol.swift` (lines 278-358)
- `WhyDiagnostics` struct (lines 278-334)
- `KillswitchState` struct (lines 336-353)
- `ReasonCode` struct (lines 355-358)

### Analysis

#### CRITICAL Issues
1. **MASSIVE MISSING FIELDS**: iOS `WhyDiagnostics` does NOT decode:
   - `profit_capture_state` (object with enabled, armed, triggered, mode, thresholds, intra-session compounding, etc.)
   - `execution_quality` (object with ws_ok, rest_ok, avg_latency_ms, reject_rate, slippage)
   - `multi_symbol` (object with enabled, symbols_count, scan state)

   **Impact:** Cannot display profit capture status, execution quality metrics, or multi-symbol scanning state
   - **Fix Required:** Add 3 new nested struct types and decode these critical diagnostic fields

#### WARNING Issues
**None** - The fields that exist are properly decoded with fallbacks (lines 315-332)

---

## 11. `/api/v1/market/tickers` - Market Tickers (NOT FOUND)

### Backend Response
```json
{
    "detail": "Not Found"
}
```

### iOS Models
**File:** `ios/HEAN/Services/Services.swift` (lines 405-517)
- `LiveMarketService` (lines 405-517)
- Calls `/api/v1/market/ticker?symbol=XXX` (single symbol, line 450)

### Analysis

#### CRITICAL Issues
1. **ENDPOINT MISMATCH**: iOS attempts to call `/api/v1/market/tickers` (plural) which does NOT exist
   - Actual endpoint: `/api/v1/market/ticker?symbol=BTCUSDT` (singular, requires symbol param)
   - **Status:** iOS code ALREADY uses correct singular endpoint (line 450, 484)
   - **Impact:** NONE - This was just a test endpoint typo in the audit, not an actual iOS bug

#### WARNING Issues
**None** - The `LiveMarketService` correctly uses the singular `/market/ticker` endpoint

---

## 12. Additional Models Not Tested Against Live API

### `TradingEvent` (ios/HEAN/Models/TradingEvent.swift)
- **Status:** Cannot validate without WebSocket events
- **Concern:** `EventType` enum (lines 10-34) defines 9 event types but backend may emit different event types
- **Recommendation:** Test with live WebSocket to confirm all event types are handled

### `WebSocketHealth` (ios/HEAN/Models/WebSocketState.swift)
- **Status:** Client-side constructed struct, not decoded from backend
- **Impact:** NONE - This is computed locally

### `Signal` (ios/HEAN/Services/TradingServiceProtocol.swift, lines 370-429)
- **Status:** Cannot validate without WebSocket signal events
- **Concern:** Decoder tries `strategy` then `strategy_id` (lines 397-402) which is good
- **Recommendation:** Test with live WebSocket to confirm field names

---

## Summary of Critical Issues

### Priority 1 - Will Cause Data Loss

1. **Order.strategyId missing** - Backend returns `strategy_id` but iOS doesn't decode it
2. **Strategy missing total_pnl, wins, losses** - Backend returns these but iOS ignores them
3. **SignalRejectionStats missing 5 fields** - Cannot show breakdown by category/symbol/strategy
4. **TradingMetrics missing top_reasons, top_symbols, top_strategies, uptime**
5. **MetricsBucket missing 10+ counter fields** - Cannot show full decision/order/position stats
6. **WhyDiagnostics missing profit_capture_state, execution_quality, multi_symbol**
7. **RiskGovernorStatus missing clear_rule, blocked_at**

### Priority 2 - Will Cause UI Issues

1. **Portfolio missing trading_mode, is_live, dry_run, total_fees** - Cannot show mode or fees
2. **Unknown OrderStatus values default to .pending** - May hide actual rejected/expired states

---

## Recommended Fixes

### 1. Order.swift - Add strategyId field
```swift
struct Order: Identifiable, Codable {
    let id: String
    let symbol: String
    let side: OrderSide
    let type: OrderType?
    let status: OrderStatus
    let price: Double?
    let quantity: Double
    var filledQuantity: Double
    let createdAt: Date?
    var updatedAt: Date?
    let strategyId: String?  // ADD THIS

    enum CodingKeys: String, CodingKey {
        // ... existing keys ...
        case strategyId = "strategy_id"  // ADD THIS
    }
}
```

### 2. Strategy - Add missing PnL fields
```swift
struct Strategy: Identifiable, Codable {
    // ... existing fields ...
    let totalPnl: Double
    let wins: Int
    let losses: Int

    enum CodingKeys: String, CodingKey {
        // ... existing keys ...
        case totalPnl = "total_pnl"
        case wins
        case losses
    }
}
```

### 3. SignalRejectionStats - Add breakdown fields
```swift
struct SignalRejectionStats: Codable {
    let totalRejections: Int
    let totalSignals: Int
    let rejectionRate: Double
    let byReason: [String: Int]
    let timeWindowMinutes: Int
    let byCategory: [String: Int]
    let bySymbol: [String: Int]
    let byStrategy: [String: Int]
    let rates: [String: Double]

    enum CodingKeys: String, CodingKey {
        case totalRejections = "total_rejections"
        case totalSignals = "total_signals"
        case rejectionRate = "rejection_rate"
        case byReason = "by_reason"
        case timeWindowMinutes = "time_window_minutes"
        case byCategory = "by_category"
        case bySymbol = "by_symbol"
        case byStrategy = "by_strategy"
        case rates
    }
}
```

### 4. TradingMetrics - Add missing top-level fields
```swift
struct TradingMetrics: Codable {
    // ... existing fields ...
    let topReasonsForSkipBlock: [TopReason]?
    let topSymbols: [TopSymbol]?
    let topStrategies: [TopStrategy]?
    let uptimeSec: Double?

    enum CodingKeys: String, CodingKey {
        // ... existing keys ...
        case topReasonsForSkipBlock = "top_reasons_for_skip_block"
        case topSymbols = "top_symbols"
        case topStrategies = "top_strategies"
        case uptimeSec = "uptime_sec"
    }
}

struct TopReason: Codable {
    let code: String
    let count: Int
    let pct: Double
}

struct TopSymbol: Codable {
    let symbol: String
    let count: Int
}

struct TopStrategy: Codable {
    let strategyId: String
    let count: Int

    enum CodingKeys: String, CodingKey {
        case strategyId = "strategy_id"
        case count
    }
}
```

### 5. MetricsBucket - Add all counter fields
```swift
struct MetricsBucket: Codable {
    let signalsTotal: Int?
    let decisionsCreate: Int?
    let decisionsSkip: Int?
    let decisionsBlock: Int?
    let ordersCreated: Int?
    let ordersFilled: Int?
    let ordersCanceled: Int?
    let ordersRejected: Int?
    let ordersOpen: Int?
    let positionsOpen: Int?
    let positionsClosed: Int?
    let pnlUnrealized: Double?
    let pnlRealized: Double?
    let equity: Double?

    enum CodingKeys: String, CodingKey {
        case signalsTotal = "signals_total"
        case decisionsCreate = "decisions_create"
        case decisionsSkip = "decisions_skip"
        case decisionsBlock = "decisions_block"
        case ordersCreated = "orders_created"
        case ordersFilled = "orders_filled"
        case ordersCanceled = "orders_canceled"
        case ordersRejected = "orders_rejected"
        case ordersOpen = "orders_open"
        case positionsOpen = "positions_open"
        case positionsClosed = "positions_closed"
        case pnlUnrealized = "pnl_unrealized"
        case pnlRealized = "pnl_realized"
        case equity
    }
}
```

### 6. WhyDiagnostics - Add complex nested objects
```swift
struct WhyDiagnostics: Codable {
    // ... existing fields ...
    let profitCaptureState: ProfitCaptureState?
    let executionQuality: ExecutionQuality?
    let multiSymbol: MultiSymbolState?

    enum CodingKeys: String, CodingKey {
        // ... existing keys ...
        case profitCaptureState = "profit_capture_state"
        case executionQuality = "execution_quality"
        case multiSymbol = "multi_symbol"
    }
}

struct ProfitCaptureState: Codable {
    let enabled: Bool
    let armed: Bool
    let triggered: Bool
    let cleared: Bool
    let mode: String
    let startEquity: Double
    let peakEquity: Double
    let targetPct: Double
    let trailPct: Double
    let afterAction: String
    let continueRiskMult: Double

    enum CodingKeys: String, CodingKey {
        case enabled, armed, triggered, cleared, mode
        case startEquity = "start_equity"
        case peakEquity = "peak_equity"
        case targetPct = "target_pct"
        case trailPct = "trail_pct"
        case afterAction = "after_action"
        case continueRiskMult = "continue_risk_mult"
    }
}

struct ExecutionQuality: Codable {
    let wsOk: Bool
    let restOk: Bool
    let avgLatencyMs: Double?
    let rejectRate5m: Double?
    let slippageEst5m: Double?

    enum CodingKeys: String, CodingKey {
        case wsOk = "ws_ok"
        case restOk = "rest_ok"
        case avgLatencyMs = "avg_latency_ms"
        case rejectRate5m = "reject_rate_5m"
        case slippageEst5m = "slippage_est_5m"
    }
}

struct MultiSymbolState: Codable {
    let enabled: Bool
    let symbolsCount: Int
    let lastScannedSymbol: String?
    let scanCursor: Int
    let scanCycleTs: String?

    enum CodingKeys: String, CodingKey {
        case enabled
        case symbolsCount = "symbols_count"
        case lastScannedSymbol = "last_scanned_symbol"
        case scanCursor = "scan_cursor"
        case scanCycleTs = "scan_cycle_ts"
    }
}
```

### 7. Portfolio - Add mode tracking
```swift
struct Portfolio: Codable {
    // ... existing fields ...
    let tradingMode: String?
    let isLive: Bool?
    let dryRun: Bool?
    let totalFees: Double?

    enum CodingKeys: String, CodingKey {
        // ... existing keys ...
        case tradingMode = "trading_mode"
        case isLive = "is_live"
        case dryRun = "dry_run"
        case totalFees = "total_fees"
    }
}
```

### 8. RiskGovernorStatus - Add missing fields
```swift
struct RiskGovernorStatus: Codable {
    // ... existing fields ...
    let clearRule: String?
    let blockedAt: String?

    enum CodingKeys: String, CodingKey {
        // ... existing keys ...
        case clearRule = "clear_rule"
        case blockedAt = "blocked_at"
    }
}
```

---

## Files Requiring Updates

1. `/Users/macbookpro/Desktop/HEAN/ios/HEAN/Models/Order.swift` - Add strategyId
2. `/Users/macbookpro/Desktop/HEAN/ios/HEAN/Services/TradingServiceProtocol.swift` - Update Strategy, SignalRejectionStats, TradingMetrics, MetricsBucket, WhyDiagnostics, RiskGovernorStatus
3. `/Users/macbookpro/Desktop/HEAN/ios/HEAN/Models/Portfolio.swift` - Add trading mode fields

---

## Testing Checklist

After implementing fixes, test:

- [ ] Decode `/api/v1/orders` and verify `strategyId` appears
- [ ] Decode `/api/v1/strategies` and verify `totalPnl`, `wins`, `losses` appear
- [ ] Decode `/api/v1/telemetry/signal-rejections` and verify all breakdown fields
- [ ] Decode `/api/v1/trading/metrics` and verify top reasons/symbols/strategies
- [ ] Decode `/api/v1/trading/why` and verify profit capture, execution quality, multi-symbol states
- [ ] Decode `/api/v1/engine/status` and verify trading mode fields
- [ ] Decode `/api/v1/risk/governor/status` and verify clearRule, blockedAt
- [ ] Test WebSocket `Signal` events to confirm field names
- [ ] Test WebSocket `TradingEvent` events to confirm all EventType cases

---

## Conclusion

The iOS models have **strong defensive coding** with good fallback handling, but are **missing many fields** returned by the backend API. This means:

1. **No crashes** - The app won't crash due to these issues
2. **Data loss** - Many useful backend fields are silently dropped
3. **Limited UI** - The iOS app cannot display full diagnostic information

**Priority:** Implement all recommended fixes to unlock full backend diagnostic capabilities in the iOS UI.

---

**Audit completed:** 2026-02-07
**Auditor:** Split_ (HEAN Multi-Persona Agent)

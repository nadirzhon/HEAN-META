# iOS API Integration Audit Report

**Date**: 2026-02-07
**Status**: CRITICAL ISSUES FOUND - Multiple integration mismatches blocking functionality

## Executive Summary

The iOS app has **significant API integration issues** causing:
1. Data refresh errors
2. Open orders not displaying
3. Buttons not working
4. Overall loss of integration

**Root Cause**: Field name mismatches, missing CodingKeys, and incorrect data structure expectations between iOS Swift models and Python FastAPI backend responses.

---

## Detailed Findings by Endpoint

### 1. `/api/v1/engine/status` - Portfolio/Engine Status

**iOS Service**: `LivePortfolioService.fetchPortfolio()` (line 232 in Services.swift)

**Backend Response**:
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
    "initial_capital": 10000.0
}
```

**iOS Model Expected**: `Portfolio` struct (Portfolio.swift)

**Issues**:
- ✅ GOOD: CodingKeys exist for snake_case conversion
- ❌ CRITICAL: Backend returns `daily_pnl` but iOS expects `realized_pnl` OR `daily_pnl`
- ❌ MISSING: Backend does NOT return:
  - `available_balance` (iOS has default fallback to equity)
  - `used_margin` (iOS defaults to 0)
  - `unrealized_pnl` (iOS defaults to 0)
  - `realized_pnl` (iOS tries `daily_pnl` as fallback - OK)

**Impact**: Portfolio displays incorrect margin usage, unrealized PnL, and available balance.

**iOS Decoder** (lines 115-134 in Portfolio.swift):
```swift
init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: DecodingKeys.self)
    self.equity = try container.decodeIfPresent(Double.self, forKey: .equity) ?? 0
    self.initialCapital = try container.decodeIfPresent(Double.self, forKey: .initialCapital) ?? 0
    self.availableBalance = try container.decodeIfPresent(Double.self, forKey: .availableBalance) ?? equity  // ✅ Fallback
    self.usedMargin = try container.decodeIfPresent(Double.self, forKey: .usedMargin) ?? 0  // ⚠️ Always 0
    self.unrealizedPnL = try container.decodeIfPresent(Double.self, forKey: .unrealizedPnL) ?? 0  // ⚠️ Always 0
    // ✅ Tries both realized_pnl and daily_pnl
    if let realized = try? container.decode(Double.self, forKey: .realizedPnL) {
        self.realizedPnL = realized
    } else if let daily = try? container.decode(Double.self, forKey: .dailyPnl) {
        self.realizedPnL = daily  // ✅ Works
    } else {
        self.realizedPnL = 0
    }
}
```

**Recommendation**: Backend should return all fields iOS expects.

---

### 2. `/api/v1/orders/positions` - Active Positions

**iOS Service**: `LiveTradingService.fetchPositions()` (line 157 in Services.swift)

**Backend Response**: `[]` (empty array, no positions currently)

**Backend Format** (from engine_facade.py lines 181-197):
```python
{
    "symbol": pos.symbol,
    "size": pos.size,
    "entry_price": pos.entry_price,
    "current_price": pos.current_price,
    "unrealized_pnl": pos.unrealized_pnl,
    "realized_pnl": pos.realized_pnl,
    "side": pos.side,
    "position_id": pos.position_id,
    "take_profit": pos.take_profit,
    "stop_loss": pos.stop_loss,
    "strategy_id": pos.strategy_id,
    "status": "open",
}
```

**iOS Model Expected**: `Position` struct (Position.swift)

**Issues**:
- ❌ CRITICAL: iOS expects `unrealized_pnl_percent` - Backend does NOT provide this
- ❌ CRITICAL: iOS expects `leverage` - Backend does NOT provide this
- ❌ CRITICAL: iOS expects `created_at` - Backend does NOT provide this
- ✅ GOOD: Custom `init(from:)` tries both `id` and `position_id` (lines 113-119)
- ✅ GOOD: CodingKey maps `current_price` to `markPrice` (line 100)
- ⚠️ PARTIAL: Side mapping handles "buy"/"sell" to LONG/SHORT (lines 22-25)

**iOS Decoder** (lines 109-130):
```swift
init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    // ✅ Tries id first, then position_id
    if let id = try? container.decode(String.self, forKey: .id) {
        self.id = id
    } else if let posId = try? container.decode(String.self, forKey: .positionId) {
        self.id = posId
    } else {
        self.id = UUID().uuidString  // ✅ Fallback
    }

    self.symbol = try container.decode(String.self, forKey: .symbol)
    self.side = try container.decode(PositionSide.self, forKey: .side)
    self.size = try container.decode(Double.self, forKey: .size)
    self.entryPrice = try container.decodeIfPresent(Double.self, forKey: .entryPrice) ?? 0
    self.markPrice = try container.decodeIfPresent(Double.self, forKey: .markPrice) ?? 0  // Maps to current_price
    self.unrealizedPnL = try container.decodeIfPresent(Double.self, forKey: .unrealizedPnL) ?? 0
    self.unrealizedPnLPercent = try container.decodeIfPresent(Double.self, forKey: .unrealizedPnLPercent) ?? 0  // ⚠️ Always 0
    self.leverage = try container.decodeIfPresent(Int.self, forKey: .leverage) ?? 1  // ⚠️ Always 1
    self.createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt)  // ⚠️ Always nil
}
```

**Impact**: Position list shows incorrect leverage (always 1x), no creation timestamp, and no PnL percentage.

**Recommendation**: Backend must add `leverage`, `unrealized_pnl_percent`, `created_at` fields.

---

### 3. `/api/v1/orders?status=open` - Open Orders

**iOS Service**: `LiveTradingService.fetchOrders(status:)` (lines 163-169 in Services.swift)

**Backend Response**:
```json
[
    {
        "order_id": "smart_limit_1770404421.787939",
        "symbol": "BNBUSDT",
        "side": "buy",
        "size": 0.007559721802237678,
        "filled_size": 0.0,
        "price": 661.4,
        "status": "pending",
        "strategy_id": "basis_arbitrage",
        "timestamp": "2026-02-06T22:00:21.787945"
    }
]
```

**iOS Model Expected**: `Order` struct (Order.swift)

**Issues**:
- ❌ CRITICAL: Backend returns `size` - iOS maps to `quantity` via CodingKey (line 158) ✅ OK
- ❌ CRITICAL: Backend returns `filled_size` - iOS expects `filled_size` via CodingKey (line 159) ✅ OK
- ❌ MISSING: Backend does NOT return `type` (MARKET/LIMIT) - iOS defaults to nil (line 180)
- ❌ CRITICAL: Backend returns `timestamp` - iOS expects `created_at` via CodingKey mapping (line 160)
- ❌ MISSING: Backend does NOT return `updated_at` - iOS defaults to nil (line 186)
- ⚠️ STATUS: Backend returns "pending" - iOS expects uppercase enum values but has case-insensitive decoder (lines 87-96) ✅ OK

**iOS Decoder** (lines 166-187):
```swift
init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)

    // ✅ Tries id first, then order_id
    if let id = try? container.decode(String.self, forKey: .id) {
        self.id = id
    } else if let ordId = try? container.decode(String.self, forKey: .orderId) {
        self.id = ordId
    } else {
        self.id = UUID().uuidString
    }

    self.symbol = try container.decode(String.self, forKey: .symbol)
    self.side = try container.decode(OrderSide.self, forKey: .side)
    self.type = try container.decodeIfPresent(OrderType.self, forKey: .type)  // ⚠️ Always nil
    self.status = try container.decodeIfPresent(OrderStatus.self, forKey: .status) ?? .pending
    self.price = try container.decodeIfPresent(Double.self, forKey: .price)
    self.quantity = try container.decodeIfPresent(Double.self, forKey: .quantity) ?? 0  // Maps to "size"
    self.filledQuantity = try container.decodeIfPresent(Double.self, forKey: .filledQuantity) ?? 0  // Maps to "filled_size"
    self.createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt)  // Maps to "timestamp"
    self.updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt)  // ⚠️ Always nil
}
```

**CodingKeys** (lines 151-164):
```swift
enum CodingKeys: String, CodingKey {
    case id
    case symbol
    case side
    case type
    case status
    case price
    case quantity = "size"  // ✅ Maps correctly
    case filledQuantity = "filled_size"  // ✅ Maps correctly
    case createdAt = "timestamp"  // ✅ Maps correctly
    case updatedAt = "updated_at"  // ⚠️ Backend doesn't provide
    case orderId = "order_id"  // ✅ Used as fallback
}
```

**Impact**: Orders display without type (MARKET/LIMIT indicator), no updated timestamp.

**Recommendation**: Backend should add `type` and `updated_at` fields.

---

### 4. `/api/v1/strategies` - Strategy List

**iOS Service**: `LiveStrategyService.fetchStrategies()` (line 276 in Services.swift)

**Backend Response**:
```json
[
    {
        "strategy_id": "funding_harvester",
        "enabled": true,
        "type": "FundingHarvester"
    }
]
```

**iOS Model Expected**: `Strategy` struct (TradingServiceProtocol.swift, lines 442-524)

**Issues**:
- ❌ CRITICAL: Backend returns minimal data - iOS expects rich performance metrics:
  - `win_rate` - Backend does NOT provide (iOS defaults to 0)
  - `total_trades` - Backend does NOT provide (iOS defaults to 0)
  - `profit_factor` - Backend does NOT provide (iOS defaults to 0)
  - `description` - Backend does NOT provide (iOS defaults to "")
- ✅ GOOD: Custom decoder tries `id` then `strategy_id` (lines 479-485)
- ✅ GOOD: Custom decoder tries `name` then `type` (lines 488-494)

**iOS Decoder** (lines 475-501):
```swift
init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: DecodingKeys.self)

    // ✅ id: try "id" first, then "strategy_id"
    if let id = try? container.decode(String.self, forKey: .id) {
        self.id = id
    } else if let stratId = try? container.decode(String.self, forKey: .strategyId) {
        self.id = stratId
    } else {
        self.id = UUID().uuidString
    }

    // ✅ name: try "name" first, then "type"
    if let name = try? container.decode(String.self, forKey: .name) {
        self.name = name
    } else if let type = try? container.decode(String.self, forKey: .type) {
        self.name = type
    } else {
        self.name = self.id
    }

    self.enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
    self.winRate = try container.decodeIfPresent(Double.self, forKey: .winRate) ?? 0  // ⚠️ Always 0
    self.totalTrades = try container.decodeIfPresent(Int.self, forKey: .totalTrades) ?? 0  // ⚠️ Always 0
    self.profitFactor = try container.decodeIfPresent(Double.self, forKey: .profitFactor) ?? 0  // ⚠️ Always 0
    self.description = try container.decodeIfPresent(String.self, forKey: .description) ?? ""  // ⚠️ Always ""
}
```

**Impact**: Strategy cards show no performance data (win rate, trade count, profit factor).

**Recommendation**: Backend must add strategy performance metrics or iOS should remove these UI elements.

---

### 5. `/api/v1/risk/governor/status` - Risk Status

**iOS Service**: `LiveRiskService.fetchRiskStatus()` (line 308 in Services.swift)

**Backend Response**:
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

**iOS Model Expected**: `RiskGovernorStatus` struct (TradingServiceProtocol.swift, lines 67-120)

**Issues**:
- ✅ GOOD: All expected fields present
- ✅ GOOD: CodingKeys handle snake_case conversion
- ⚠️ EXTRA: Backend returns extra fields iOS ignores: `clear_rule`, `blocked_at`

**Impact**: No issues - this endpoint works correctly.

---

### 6. `/api/v1/trading/metrics` - Trading Metrics

**iOS Service**: `LiveTradingService.fetchTradingMetrics()` (line 202 in Services.swift)

**Backend Response**:
```json
{
    "status": "ok",
    "counters": {
        "last_1m": {
            "signals_total": 0,
            "decisions_create": 0,
            "decisions_skip": 0,
            "decisions_block": 0,
            "orders_created": 0,
            "orders_filled": 0,
            "orders_canceled": 0,
            "orders_rejected": 0,
            "positions_open": 0,
            "positions_closed": 0
        },
        "last_5m": { ... },
        "session": { ... }
    },
    "top_reasons_for_skip_block": [...],
    "active_orders_count": 1127,
    "active_positions_count": 0,
    "last_signal_ts": "2026-02-06T22:00:21.753665+00:00",
    "last_order_ts": null,
    "last_fill_ts": null,
    "engine_state": "RUNNING",
    "mode": "live"
}
```

**iOS Model Expected**: `TradingMetrics` struct (TradingServiceProtocol.swift, lines 194-249)

**Issues**:
- ❌ CRITICAL: Backend returns `signals_total` - iOS expects `signals_detected`
- ❌ CRITICAL: Backend returns `decisions_create` - iOS expects `orders_created` (WRONG MAPPING)
- ❌ CRITICAL: Backend returns `orders_created` - iOS expects... `orders_created` ✅
- ❌ SEMANTIC MISMATCH: iOS maps session counters incorrectly:
  - `signalsDetected` should map to `signals_total` (backend) not `signals_detected`
  - `signalsBlocked` should map to `decisions_block` not `signals_blocked`

**iOS Accessor Properties** (lines 244-249):
```swift
// ⚠️ These accessors look for WRONG field names in backend response
var signalsDetected: Int { counters?.session?.signalsDetected ?? 0 }  // Backend: signals_total
var ordersCreated: Int { counters?.session?.ordersCreated ?? 0 }  // Backend: orders_created ✅
var ordersFilled: Int { counters?.session?.ordersFilled ?? 0 }  // Backend: orders_filled ✅
var signalsBlocked: Int { counters?.session?.signalsBlocked ?? 0 }  // Backend: decisions_block
```

**MetricsBucket Model** (lines 263-275):
```swift
struct MetricsBucket: Codable {
    let signalsDetected: Int?  // ❌ Backend: signals_total
    let ordersCreated: Int?    // ✅ Matches
    let ordersFilled: Int?     // ✅ Matches
    let signalsBlocked: Int?   // ❌ Backend: decisions_block

    enum CodingKeys: String, CodingKey {
        case signalsDetected = "signals_detected"  // ❌ Backend doesn't have this
        case ordersCreated = "orders_created"
        case ordersFilled = "orders_filled"
        case signalsBlocked = "signals_blocked"  // ❌ Backend doesn't have this
    }
}
```

**Impact**: Metrics display shows 0 for signals detected and signals blocked (always).

**Recommendation**: iOS CodingKeys must map to backend's actual field names:
- `signalsDetected` → `signals_total`
- `signalsBlocked` → `decisions_block`

---

### 7. `/api/v1/market/ticker?symbol=BTCUSDT` - Market Ticker

**iOS Service**: `LiveMarketService.refreshMarkets()` (lines 370-403 in Services.swift)

**Backend Response**:
```json
{
    "symbol": "BTCUSDT",
    "price": 69988.3,
    "bid": 69988.3,
    "ask": null,
    "volume": 0.0,
    "timestamp": "2026-02-06T22:05:03.323193"
}
```

**iOS Model Expected**: `TickerResponse` struct (Services.swift, lines 447-454)

**Issues**:
- ✅ GOOD: All fields present and optional
- ⚠️ WORKAROUND: iOS builds full `Market` object from minimal ticker data (lines 378-392)
- ❌ LIMITED: iOS sets `change24h`, `changePercent24h`, `high24h`, `low24h` to 0 (not provided)

**Impact**: Market list shows prices but no 24h change data.

**Recommendation**: Backend should provide 24h statistics or document that it's intentionally minimal.

---

## Summary of Critical Issues

### Backend Must Fix (Breaking Issues):

1. **`/api/v1/orders/positions`** - Add missing fields:
   - `leverage` (Int)
   - `unrealized_pnl_percent` (Double)
   - `created_at` (ISO8601 timestamp)

2. **`/api/v1/orders`** - Add missing fields:
   - `type` (String: "MARKET", "LIMIT", etc.)
   - `updated_at` (ISO8601 timestamp)

3. **`/api/v1/trading/metrics`** - Inconsistent field names in counters:
   - Backend uses: `signals_total`, `decisions_create`, `decisions_block`
   - iOS expects: `signals_detected`, `orders_created`, `signals_blocked`
   - **Decision**: Either backend changes field names OR iOS updates CodingKeys

4. **`/api/v1/strategies`** - Add performance metrics:
   - `win_rate` (Double)
   - `total_trades` (Int)
   - `profit_factor` (Double)
   - `description` (String)

5. **`/api/v1/engine/status`** - Add missing portfolio fields:
   - `available_balance` (Double)
   - `used_margin` (Double)
   - `unrealized_pnl` (Double)

### iOS Should Fix (Non-Breaking, Improve Robustness):

1. **`TradingMetrics` CodingKeys** - Update to match backend:
   ```swift
   case signalsDetected = "signals_total"  // Not "signals_detected"
   case signalsBlocked = "decisions_block"  // Not "signals_blocked"
   ```

2. **Error Handling** - Add better logging when decoding fails to help diagnose issues

3. **Fallback Values** - Current approach of defaulting to 0/nil is good but should log warnings

---

## Testing Checklist

After backend changes, test these flows:

- [ ] Portfolio screen shows correct equity, balance, margin usage
- [ ] Positions list displays leverage, PnL %, and creation date
- [ ] Orders list shows order type (MARKET/LIMIT) and timestamps
- [ ] Strategy cards display win rate, trade count, profit factor
- [ ] Dashboard metrics show correct signal and order counts
- [ ] Risk status updates correctly
- [ ] Market tickers display prices (24h change will remain 0)

---

## Files to Modify

### Backend (Python):
- `/Users/macbookpro/Desktop/HEAN/src/hean/api/engine_facade.py`
  - `get_positions()` - Add leverage, unrealized_pnl_percent, created_at
  - `get_orders()` - Add type, updated_at
  - `get_strategies()` - Add win_rate, total_trades, profit_factor, description

- `/Users/macbookpro/Desktop/HEAN/src/hean/api/routers/engine.py`
  - `/status` endpoint - Add available_balance, used_margin, unrealized_pnl

- `/Users/macbookpro/Desktop/HEAN/src/hean/api/services/trading_metrics.py`
  - Consider renaming fields to match iOS expectations OR document mismatch

### iOS (Swift):
- `/Users/macbookpro/Desktop/HEAN/ios/HEAN/Services/TradingServiceProtocol.swift`
  - `MetricsBucket` CodingKeys (lines 269-274) - Update to match backend field names

---

## Recommended Action Plan

1. **Immediate Fix** (Backend): Add missing fields to positions and orders endpoints
2. **Quick Win** (iOS): Update TradingMetrics CodingKeys to match backend
3. **Medium Priority** (Backend): Add strategy performance metrics
4. **Low Priority** (Backend): Add portfolio margin/balance fields (iOS has fallbacks)
5. **Optional** (Backend): Add 24h market statistics to ticker endpoint

This will restore full iOS app functionality.

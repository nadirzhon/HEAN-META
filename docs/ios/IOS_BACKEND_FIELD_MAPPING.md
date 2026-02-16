# iOS ↔️ Backend Field Mapping Reference

## Positions Endpoint: `/api/v1/orders/positions`

| iOS Field (Swift) | Backend Field (Python) | Status | Notes |
|------------------|------------------------|---------|-------|
| `id` | `position_id` | ✅ OK | Custom decoder tries both |
| `symbol` | `symbol` | ✅ OK | Direct match |
| `side` | `side` | ✅ OK | Maps buy/sell → LONG/SHORT |
| `size` | `size` | ✅ OK | Direct match |
| `entryPrice` | `entry_price` | ✅ OK | CodingKey mapping |
| `markPrice` | `current_price` | ✅ OK | CodingKey mapping |
| `unrealizedPnL` | `unrealized_pnl` | ✅ OK | CodingKey mapping |
| `unrealizedPnLPercent` | ❌ MISSING | ❌ BROKEN | iOS defaults to 0 |
| `leverage` | ❌ MISSING | ❌ BROKEN | iOS defaults to 1 |
| `createdAt` | ❌ MISSING | ❌ BROKEN | iOS defaults to nil |

**Backend Must Add**:
```python
"leverage": 10,  # Int
"unrealized_pnl_percent": 5.2,  # Double (PnL / Position Value * 100)
"created_at": "2026-02-06T22:00:21.787945"  # ISO8601 string
```

---

## Orders Endpoint: `/api/v1/orders`

| iOS Field (Swift) | Backend Field (Python) | Status | Notes |
|------------------|------------------------|---------|-------|
| `id` | `order_id` | ✅ OK | Custom decoder tries both |
| `symbol` | `symbol` | ✅ OK | Direct match |
| `side` | `side` | ✅ OK | Case-insensitive decoder |
| `type` | ❌ MISSING | ❌ BROKEN | iOS defaults to nil |
| `status` | `status` | ✅ OK | Case-insensitive decoder |
| `price` | `price` | ✅ OK | Direct match |
| `quantity` | `size` | ✅ OK | CodingKey mapping |
| `filledQuantity` | `filled_size` | ✅ OK | CodingKey mapping |
| `createdAt` | `timestamp` | ✅ OK | CodingKey mapping |
| `updatedAt` | ❌ MISSING | ⚠️ MINOR | iOS defaults to nil |

**Backend Must Add**:
```python
"type": "LIMIT",  # String: MARKET, LIMIT, STOP_MARKET, STOP_LIMIT
"updated_at": "2026-02-06T22:00:21.787945"  # ISO8601 string (optional)
```

---

## Strategies Endpoint: `/api/v1/strategies`

| iOS Field (Swift) | Backend Field (Python) | Status | Notes |
|------------------|------------------------|---------|-------|
| `id` | `strategy_id` | ✅ OK | Custom decoder tries both |
| `name` | `type` | ✅ OK | Custom decoder uses type as name |
| `enabled` | `enabled` | ✅ OK | Direct match |
| `winRate` | ❌ MISSING | ❌ BROKEN | iOS defaults to 0 |
| `totalTrades` | ❌ MISSING | ❌ BROKEN | iOS defaults to 0 |
| `profitFactor` | ❌ MISSING | ❌ BROKEN | iOS defaults to 0 |
| `description` | ❌ MISSING | ⚠️ MINOR | iOS defaults to "" |

**Backend Must Add**:
```python
"win_rate": 0.62,  # Double (0-1)
"total_trades": 89,  # Int
"profit_factor": 1.8,  # Double
"description": "Momentum-based strategy"  # String (optional)
```

---

## Portfolio Endpoint: `/api/v1/engine/status`

| iOS Field (Swift) | Backend Field (Python) | Status | Notes |
|------------------|------------------------|---------|-------|
| `equity` | `equity` | ✅ OK | Direct match |
| `availableBalance` | ❌ MISSING | ⚠️ FALLBACK | iOS uses equity as fallback |
| `usedMargin` | ❌ MISSING | ⚠️ FALLBACK | iOS defaults to 0 |
| `unrealizedPnL` | ❌ MISSING | ⚠️ FALLBACK | iOS defaults to 0 |
| `realizedPnL` | `daily_pnl` | ✅ OK | iOS tries both realized_pnl and daily_pnl |
| `initialCapital` | `initial_capital` | ✅ OK | CodingKey mapping |

**Backend Should Add** (Low Priority - iOS has fallbacks):
```python
"available_balance": 9500.0,  # Double (equity - used_margin)
"used_margin": 500.0,  # Double
"unrealized_pnl": 50.0  # Double
```

---

## Trading Metrics Endpoint: `/api/v1/trading/metrics`

### Backend Counter Fields vs iOS Expectations

| iOS Field | iOS CodingKey | Backend Actual Field | Match? |
|-----------|---------------|---------------------|---------|
| `signalsDetected` | `signals_detected` | `signals_total` | ❌ MISMATCH |
| `ordersCreated` | `orders_created` | `orders_created` | ✅ OK |
| `ordersFilled` | `orders_filled` | `orders_filled` | ✅ OK |
| `signalsBlocked` | `signals_blocked` | `decisions_block` | ❌ MISMATCH |

**FIX REQUIRED**: iOS must update CodingKeys:

```swift
// BEFORE (WRONG):
enum CodingKeys: String, CodingKey {
    case signalsDetected = "signals_detected"  // ❌ Backend: signals_total
    case signalsBlocked = "signals_blocked"    // ❌ Backend: decisions_block
}

// AFTER (CORRECT):
enum CodingKeys: String, CodingKey {
    case signalsDetected = "signals_total"     // ✅ Matches backend
    case signalsBlocked = "decisions_block"    // ✅ Matches backend
}
```

**Backend Counter Structure**:
```json
{
  "counters": {
    "session": {
      "signals_total": 8,        // iOS expects "signals_detected"
      "decisions_create": 8,
      "decisions_skip": 0,
      "decisions_block": 0,      // iOS expects "signals_blocked"
      "orders_created": 0,       // ✅ Match
      "orders_filled": 0,        // ✅ Match
      "orders_canceled": 0,
      "orders_rejected": 0,
      "orders_open": 1127,
      "positions_open": 0,
      "positions_closed": 0
    }
  }
}
```

---

## Risk Governor Endpoint: `/api/v1/risk/governor/status`

| iOS Field (Swift) | Backend Field (Python) | Status | Notes |
|------------------|------------------------|---------|-------|
| `riskState` | `risk_state` | ✅ OK | CodingKey mapping |
| `level` | `level` | ✅ OK | Direct match |
| `reasonCodes` | `reason_codes` | ✅ OK | CodingKey mapping |
| `quarantinedSymbols` | `quarantined_symbols` | ✅ OK | CodingKey mapping |
| `canClear` | `can_clear` | ✅ OK | CodingKey mapping |
| `metric` | `metric` | ✅ OK | Direct match |
| `value` | `value` | ✅ OK | Direct match |
| `threshold` | `threshold` | ✅ OK | Direct match |
| `recommendedAction` | `recommended_action` | ✅ OK | CodingKey mapping |

**Status**: ✅ **NO ISSUES** - This endpoint works perfectly!

---

## Market Ticker Endpoint: `/api/v1/market/ticker?symbol=X`

| iOS Field (Swift) | Backend Field (Python) | Status | Notes |
|------------------|------------------------|---------|-------|
| `symbol` | `symbol` | ✅ OK | Direct match |
| `price` | `price` | ✅ OK | Direct match |
| `bid` | `bid` | ✅ OK | Direct match |
| `ask` | `ask` | ✅ OK | Direct match |
| `volume` | `volume` | ✅ OK | Direct match |
| `timestamp` | `timestamp` | ✅ OK | Direct match |

**Status**: ✅ **NO ISSUES** - Works correctly (iOS accepts minimal data)

---

## Visual Summary

```
┌─────────────────────────────────────────────────────────────┐
│                   INTEGRATION HEALTH                        │
├─────────────────────────────────────────────────────────────┤
│ ✅ Risk Governor         100% Working                       │
│ ✅ Market Ticker         100% Working                       │
│ ⚠️  Portfolio             66% Working (missing margin data) │
│ ❌ Positions             50% Working (missing 3 fields)     │
│ ❌ Orders                80% Working (missing 2 fields)     │
│ ❌ Strategies            50% Working (missing 4 fields)     │
│ ❌ Trading Metrics       50% Working (field name mismatch)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Priority Matrix

| Endpoint | Missing Fields | Impact | Priority |
|----------|----------------|--------|----------|
| Trading Metrics | Field name mismatch | Dashboard broken | 🔥 CRITICAL |
| Positions | 3 fields | Position display incomplete | 🔥 CRITICAL |
| Orders | 2 fields | Order type missing | ⚠️ HIGH |
| Strategies | 4 fields | Performance data missing | ⚠️ MEDIUM |
| Portfolio | 3 fields | Margin data missing | ℹ️ LOW |

---

## Testing Commands

After backend changes, verify responses:

```bash
# Check positions response
curl -s "http://localhost:8000/api/v1/orders/positions" | jq '.[0]'

# Should see: leverage, unrealized_pnl_percent, created_at

# Check orders response
curl -s "http://localhost:8000/api/v1/orders?status=open" | jq '.[0]'

# Should see: type, updated_at

# Check strategies response
curl -s "http://localhost:8000/api/v1/strategies" | jq '.[0]'

# Should see: win_rate, total_trades, profit_factor, description

# Check metrics response
curl -s "http://localhost:8000/api/v1/trading/metrics" | jq '.counters.session'

# Should see: signals_total, decisions_block (iOS will map these)

# Check portfolio response
curl -s "http://localhost:8000/api/v1/engine/status" | jq '.'

# Should see: available_balance, used_margin, unrealized_pnl
```

---

## Field Type Reference

| Type | Example | Notes |
|------|---------|-------|
| ISO8601 timestamp | `"2026-02-06T22:00:21.787945"` | Swift decodes to Date |
| Double | `1234.56` | Precision decimals |
| Int | `42` | Whole numbers |
| String | `"MARKET"` | Enum values uppercase |
| Bool | `true` / `false` | Lowercase JSON |

---

## Common Patterns

### iOS Custom Decoders (Fallback Keys)

```swift
// Try primary key first, then fallback
if let id = try? container.decode(String.self, forKey: .id) {
    self.id = id
} else if let fallbackId = try? container.decode(String.self, forKey: .fallbackKey) {
    self.id = fallbackId
} else {
    self.id = UUID().uuidString  // Last resort
}
```

### Backend Response Pattern

```python
{
    "field_name": value,  # snake_case
    "nested_object": {
        "sub_field": value  # also snake_case
    },
    "timestamp": datetime.now().isoformat()  # ISO8601
}
```

### iOS CodingKeys Pattern

```swift
enum CodingKeys: String, CodingKey {
    case swiftName = "backend_field_name"  // Maps snake_case to camelCase
}
```

---

This mapping reference should be kept up-to-date as the API evolves.

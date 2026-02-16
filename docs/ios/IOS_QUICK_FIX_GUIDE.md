# iOS App Quick Fix Guide

## Problem
iOS app shows errors, no data refreshing, buttons not working.

## Root Cause
Backend API responses missing fields that iOS models expect.

---

## CRITICAL FIX #1: Trading Metrics (Dashboard showing 0 signals/orders)

**File**: `ios/HEAN/Services/TradingServiceProtocol.swift` lines 269-274

**Change CodingKeys** in `MetricsBucket` struct:

```swift
// BEFORE (WRONG):
enum CodingKeys: String, CodingKey {
    case signalsDetected = "signals_detected"  // ❌ Backend doesn't have this
    case ordersCreated = "orders_created"
    case ordersFilled = "orders_filled"
    case signalsBlocked = "signals_blocked"  // ❌ Backend doesn't have this
}

// AFTER (CORRECT):
enum CodingKeys: String, CodingKey {
    case signalsDetected = "signals_total"  // ✅ Matches backend
    case ordersCreated = "orders_created"
    case ordersFilled = "orders_filled"
    case signalsBlocked = "decisions_block"  // ✅ Matches backend
}
```

**Impact**: Fixes dashboard metrics showing 0 for signals/orders.

---

## CRITICAL FIX #2: Backend - Add Position Fields

**File**: `src/hean/api/engine_facade.py` lines 181-197

**Current**:
```python
return [
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
    for pos in positions_list
]
```

**Add these fields**:
```python
return [
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
        # NEW FIELDS FOR iOS:
        "leverage": getattr(pos, 'leverage', 1),  # Default to 1x
        "unrealized_pnl_percent": (pos.unrealized_pnl / (pos.entry_price * pos.size) * 100) if pos.entry_price and pos.size else 0,
        "created_at": getattr(pos, 'created_at', None),  # ISO8601 timestamp
    }
    for pos in positions_list
]
```

**Impact**: Positions show leverage, PnL %, creation date.

---

## CRITICAL FIX #3: Backend - Add Order Type and Updated Timestamp

**File**: `src/hean/api/engine_facade.py` lines 220-234

**Current**:
```python
return [
    {
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side,
        "size": order.size,
        "filled_size": order.filled_size,
        "price": order.price,
        "status": order.status.value,
        "strategy_id": order.strategy_id,
        "timestamp": order.timestamp.isoformat() if order.timestamp else None,
    }
    for order in orders
]
```

**Add these fields**:
```python
return [
    {
        "order_id": order.order_id,
        "symbol": order.symbol,
        "side": order.side,
        "size": order.size,
        "filled_size": order.filled_size,
        "price": order.price,
        "status": order.status.value,
        "strategy_id": order.strategy_id,
        "timestamp": order.timestamp.isoformat() if order.timestamp else None,
        # NEW FIELDS FOR iOS:
        "type": getattr(order, 'order_type', 'LIMIT').upper(),  # MARKET, LIMIT, etc.
        "updated_at": getattr(order, 'updated_at', order.timestamp).isoformat() if hasattr(order, 'updated_at') or order.timestamp else None,
    }
    for order in orders
]
```

**Impact**: Orders show type (MARKET/LIMIT) and last update time.

---

## MEDIUM PRIORITY: Backend - Add Strategy Performance Metrics

**File**: `src/hean/api/engine_facade.py` lines 435-443

**Current**:
```python
strategies = []
for strategy in self._trading_system._strategies:
    strategies.append({
        "strategy_id": strategy.strategy_id,
        "enabled": strategy._running,
        "type": type(strategy).__name__,
    })
return strategies
```

**Add performance metrics**:
```python
strategies = []
for strategy in self._trading_system._strategies:
    # Get strategy stats from accounting or strategy object
    stats = getattr(strategy, 'stats', {})

    strategies.append({
        "strategy_id": strategy.strategy_id,
        "enabled": strategy._running,
        "type": type(strategy).__name__,
        # NEW FIELDS FOR iOS:
        "win_rate": stats.get('win_rate', 0.0),
        "total_trades": stats.get('total_trades', 0),
        "profit_factor": stats.get('profit_factor', 0.0),
        "description": getattr(strategy, 'description', ''),
    })
return strategies
```

**Impact**: Strategy cards show win rate, trade count, profit factor.

---

## LOW PRIORITY: Backend - Add Portfolio Margin Fields

**File**: `src/hean/api/routers/engine.py` or wherever `/engine/status` returns data

**Current response**:
```json
{
    "status": "running",
    "running": true,
    "engine_state": "RUNNING",
    "equity": 10000.0,
    "daily_pnl": 0.0,
    "initial_capital": 10000.0
}
```

**Add these fields**:
```json
{
    "status": "running",
    "running": true,
    "engine_state": "RUNNING",
    "equity": 10000.0,
    "daily_pnl": 0.0,
    "initial_capital": 10000.0,
    "available_balance": 9500.0,  // equity - used_margin
    "used_margin": 500.0,
    "unrealized_pnl": 50.0
}
```

**Impact**: Portfolio shows accurate margin usage. (iOS has fallbacks so not critical)

---

## Testing After Fixes

1. **Restart backend**: `docker-compose restart api` or `make run`
2. **Restart iOS app**: Full app restart
3. **Test dashboard**: Should show signal/order counts
4. **Test positions**: Should show leverage, PnL %
5. **Test orders**: Should show order type (MARKET/LIMIT)
6. **Test strategies**: Should show performance metrics

---

## Quick Command to Apply iOS Fix

```bash
cd /Users/macbookpro/Desktop/HEAN
# Edit the file manually or apply patch
nano ios/HEAN/Services/TradingServiceProtocol.swift
# Change lines 269-274 as shown above
```

Then rebuild iOS app in Xcode.

---

## Priority Order

1. ✅ **Do First**: iOS TradingMetrics fix (5 minutes, no backend restart needed)
2. ✅ **Do Second**: Backend position fields (10 minutes, backend restart required)
3. ✅ **Do Third**: Backend order fields (5 minutes, same backend restart)
4. ⏭️ **Later**: Strategy metrics (30 minutes, may need accounting integration)
5. ⏭️ **Optional**: Portfolio margin fields (10 minutes, low impact)

---

## Files Referenced

**iOS**:
- `/Users/macbookpro/Desktop/HEAN/ios/HEAN/Services/Services.swift` (compiled service implementations)
- `/Users/macbookpro/Desktop/HEAN/ios/HEAN/Services/TradingServiceProtocol.swift` (models and protocols)
- `/Users/macbookpro/Desktop/HEAN/ios/HEAN/Models/Position.swift` (Position model)
- `/Users/macbookpro/Desktop/HEAN/ios/HEAN/Models/Order.swift` (Order model)
- `/Users/macbookpro/Desktop/HEAN/ios/HEAN/Models/Portfolio.swift` (Portfolio model)

**Backend**:
- `/Users/macbookpro/Desktop/HEAN/src/hean/api/engine_facade.py` (main API logic)
- `/Users/macbookpro/Desktop/HEAN/src/hean/api/routers/engine.py` (engine endpoints)
- `/Users/macbookpro/Desktop/HEAN/src/hean/api/routers/trading.py` (trading endpoints)
- `/Users/macbookpro/Desktop/HEAN/src/hean/api/routers/strategies.py` (strategy endpoints)

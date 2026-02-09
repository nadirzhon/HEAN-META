# Execution Router Fixes - 2026-02-07

## Issues Fixed

### 1. Iceberg Split Size=0 Error
**Problem**: Iceberg split produced orders with size=0 due to floating-point arithmetic, causing validation errors:
```
Error routing order request: 1 validation error for OrderRequest size Value error, Invalid size: 0.0, must be > 0
```

**Fix**:
- Added guard in `iceberg.py` to skip micro-orders with size <= 0 or < 0.000001
- Added fallback to return original order if all micro-orders are filtered out
- Added validation in router to filter out zero-size orders before scheduling

**Files Modified**:
- `/Users/macbookpro/Desktop/HEAN/src/hean/core/execution/iceberg.py`
- `/Users/macbookpro/Desktop/HEAN/src/hean/execution/router.py`

### 2. Maker Orders Stuck as "Pending" Forever
**Problem**: 1203 internal "pending" orders accumulated, never cleaned up after TTL expiry

**Fix**:
- Added `OrderStatus.PENDING` to TTL expiration check (was only checking PLACED/PARTIALLY_FILLED)
- Immediately remove expired orders from `_maker_orders` dict at start of expiration handler
- Clean up `_order_requests` dict in all code paths (retry, taker fallback, rejection)
- Prevent duplicate processing of same expired order

**Files Modified**:
- `/Users/macbookpro/Desktop/HEAN/src/hean/execution/router.py`

### 3. Orders Not Reaching Bybit
**Problem**: 0 orders sent to Bybit despite 1203 internal pending orders

**Root Cause**: Router created local `Order` objects but didn't properly route them to Bybit's HTTP API. The flow was:
1. Create local Order with local UUID
2. Try to route to Bybit
3. But routing logic created NEW OrderRequest which didn't match
4. Order stayed in pending state forever

**Fix**:
- In `_route_maker_first`: Create proper `OrderRequest` with maker price, place on Bybit, receive Bybit's order ID
- Update local tracking to use Bybit's order ID (not local UUID)
- Register Bybit's order with order_manager
- Store idempotency key for Bybit's order
- Publish ORDER_PLACED event with Bybit's order
- Added comprehensive error handling and cleanup on failure

**Files Modified**:
- `/Users/macbookpro/Desktop/HEAN/src/hean/execution/router.py`

### 4. Taker Fallback Cleanup
**Problem**: Taker fallback had proper logic but missing cleanup in some paths

**Fix**:
- Added cleanup of `_order_requests` dict after successful taker fallback
- Added cleanup when taker fallback is rejected
- Added cleanup when no market data available
- Added cleanup when taker fallback is disabled

**Files Modified**:
- `/Users/macbookpro/Desktop/HEAN/src/hean/execution/router.py`

## Code Changes Summary

### iceberg.py Changes
```python
# Added size validation to prevent zero-size orders
for idx in range(micro_count):
    size = micro_size if remaining > micro_size else remaining
    remaining -= size

    # Skip if size too small (prevents size=0 validation errors)
    if size <= 0 or size < 0.000001:
        logger.debug(f"Skipping micro-order {idx+1} with size={size:.8f} (too small)")
        continue
    # ... create order ...

# Fallback if all orders filtered out
if not micro_requests:
    logger.warning(f"Iceberg split resulted in no valid orders, using original")
    return [order_request]
```

### router.py Changes

#### Iceberg Processing
```python
# Validate all micro-requests have positive size
valid_micro_requests = [r for r in micro_requests if r.size > 0]

if len(valid_micro_requests) == 0:
    logger.error(f"Iceberg split produced no valid orders, using original request")
elif len(valid_micro_requests) > 1:
    # Schedule valid orders only
    await self._iceberg.schedule_micro_orders(valid_micro_requests)
    return
else:
    order_request = valid_micro_requests[0]
```

#### Maker Order Placement
```python
# Create OrderRequest with maker price for Bybit
bybit_order_request = OrderRequest(
    signal_id=order_request.signal_id,
    strategy_id=order_request.strategy_id,
    symbol=symbol,
    side=side,
    size=order.size,
    price=maker_price,
    order_type="limit",
    # ... other fields ...
)

# Place order directly on Bybit
placed_order = await self._bybit_http.place_order(bybit_order_request)

# Update tracking with Bybit's order ID
if placed_order and placed_order.order_id:
    # Remove old tracking
    self._maker_orders.pop(order.order_id, None)
    self._order_requests.pop(order.order_id, None)

    # Track with Bybit's order
    placed_order.metadata.update(order.metadata)
    placed_order.is_maker = True
    placed_order.placed_at = datetime.utcnow()

    self._maker_orders[placed_order.order_id] = placed_order
    self._order_requests[placed_order.order_id] = order_request
    self._order_manager.register_order(placed_order)

    # Store idempotency
    self._store_idempotency(order_request, placed_order.order_id, status="pending")

    # Publish event
    await self._bus.publish(Event(event_type=EventType.ORDER_PLACED, data={"order": placed_order}))
```

#### TTL Expiration Check
```python
# Include PENDING status in expiration check
if order.status in {OrderStatus.PENDING, OrderStatus.PLACED, OrderStatus.PARTIALLY_FILLED}:
    expired_orders.append(order)
```

#### Expiration Handler Cleanup
```python
async def _handle_expired_maker_order(self, order: Order) -> None:
    # Remove from tracking immediately
    self._maker_orders.pop(order.order_id, None)

    # Cancel order on Bybit
    # ...

    # Try retry queue
    if self._retry_queue.enqueue_for_retry(...):
        self._order_requests.pop(order.order_id, None)  # Cleanup
        return

    # Check taker fallback
    if not settings.allow_taker_fallback:
        self._order_requests.pop(order.order_id, None)  # Cleanup
        return

    # ... taker fallback logic with cleanup at each exit point ...
```

## Testing Recommendations

1. **Monitor Order Flow**:
   ```bash
   # Watch logs for successful Bybit placements
   docker-compose logs -f hean-api | grep "Maker order placed on Bybit"
   ```

2. **Check Pending Order Accumulation**:
   ```bash
   # Should not see 1203+ pending orders anymore
   curl http://localhost:8000/api/v1/orders | jq '.[] | select(.status=="pending") | length'
   ```

3. **Verify TTL Cleanup**:
   ```bash
   # Orders should expire and clean up properly
   docker-compose logs -f hean-api | grep "expired"
   ```

4. **Test Iceberg Split**:
   - Send large order (>$1000 notional)
   - Verify no size=0 errors
   - Verify valid micro-orders are created

## Expected Behavior After Fixes

1. **Orders reach Bybit**: Every signal that passes risk checks should result in actual Bybit API calls
2. **No size=0 errors**: Iceberg split handles edge cases gracefully
3. **No pending accumulation**: Expired orders get cleaned up within TTL window (8000ms default)
4. **Proper taker fallback**: When enabled, expired makers properly fallback to taker orders
5. **Clean tracking**: Internal order dicts stay synchronized with Bybit state

## Configuration

Current settings from .env:
- `MAKER_FIRST=true` - Use maker-first strategy
- `MAKER_TTL_MS=8000` - 8 second TTL for maker orders
- `ALLOW_TAKER_FALLBACK=true` - Enable taker fallback
- `TRADING_MODE=live` - Live trading mode
- `DRY_RUN=false` - Real orders (not simulation)
- `MAX_OPEN_ORDERS=12` - Maximum concurrent orders

## Risk Mitigation

All fixes maintain existing safety guarantees:
- Idempotency protection still active
- Risk governor still gates all orders
- Position limits still enforced
- Kill switch still operational
- Only valid, positive-size orders reach Bybit

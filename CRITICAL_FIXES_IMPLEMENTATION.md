# Critical Fixes - Implementation Guide

**Priority:** P0 - Implement Immediately
**Estimated Time:** 8-12 hours
**Impact:** Prevents capital loss, system freezes

---

## Fix #1: EventBus Queue Saturation Monitor

### File: `src/hean/core/bus.py`

**Add after line 47:**
```python
# Queue health thresholds
self._queue_alert_threshold = 0.80  # Alert at 80% full
self._queue_critical_threshold = 0.95  # Critical at 95% full
self._last_saturation_alert = 0.0  # Throttle alerts
```

**Replace `publish()` method (lines 70-133) with:**
```python
async def publish(self, event: Event) -> None:
    """Publish an event to the bus with queue health monitoring."""

    # Fast-path for time-critical events
    if event.event_type in FAST_PATH_EVENTS:
        logger.debug(f"Publishing {event.event_type} event via fast-path")
        await self._dispatch_fast(event)
        return

    # Check queue health BEFORE attempting to publish
    queue_size = self._queue.qsize()
    queue_capacity = self._queue.maxsize
    utilization = queue_size / queue_capacity if queue_capacity > 0 else 0

    # CRITICAL: Alert on high queue utilization
    now = time.time()
    if utilization >= self._queue_critical_threshold:
        logger.critical(
            f"EventBus CRITICAL: Queue {utilization*100:.1f}% full ({queue_size}/{queue_capacity}). "
            f"System at risk of event processing failure. "
            f"Metrics: published={self._metrics['events_published']}, "
            f"dropped={self._metrics['events_dropped']}, "
            f"delayed={self._metrics['events_delayed']}"
        )
    elif utilization >= self._queue_alert_threshold:
        # Throttle alerts to once per 30 seconds
        if (now - self._last_saturation_alert) > 30.0:
            logger.warning(
                f"EventBus WARNING: Queue {utilization*100:.1f}% full ({queue_size}/{queue_capacity}). "
                f"Event processing may be lagging. "
                f"Consider increasing max_queue_size or optimizing handlers."
            )
            self._last_saturation_alert = now

    # Existing publish logic...
    logger.debug(f"Publishing {event.event_type} event to queue")
    try:
        self._queue.put_nowait(event)
        self._metrics["events_published"] += 1
    except asyncio.QueueFull:
        # Drop low-value events instead of crashing
        if event.event_type == EventType.TICK:
            self._metrics["events_dropped"] += 1
            logger.warning(
                f"EventBus queue full ({queue_size}/{queue_capacity}). "
                f"Dropping TICK to relieve pressure. Total dropped: {self._metrics['events_dropped']}"
            )
            return

        # For critical events, wait briefly before giving up
        try:
            await asyncio.wait_for(self._queue.put(event), timeout=1.0)
            self._metrics["events_delayed"] += 1
            self._metrics["events_published"] += 1
            logger.warning(
                f"EventBus queue was full ({queue_size}/{queue_capacity}). "
                f"Backpressured and enqueued {event.event_type} after wait. "
                f"Total delayed: {self._metrics['events_delayed']}"
            )
        except Exception as exc:
            logger.error(
                f"EventBus queue FULL ({queue_size}/{queue_capacity}). "
                f"Cannot publish {event.event_type} event. Event processing severely behind.",
                exc_info=True,
            )
            raise RuntimeError(
                f"EventBus queue full ({queue_size}/{queue_capacity}). "
                "Event processing falling behind. Check handler performance."
            ) from exc
```

**Test:**
```bash
# Run existing tests
pytest tests/test_api.py::test_event_bus -v

# Add stress test
pytest tests/test_event_bus_saturation.py -v
```

---

## Fix #2: Order Idempotency Protection

### File: `src/hean/exchange/bybit/http.py`

**Add to `__init__` method (after line 36):**
```python
# Idempotency protection
self._recent_orders: dict[str, float] = {}  # order_fingerprint -> timestamp
self._order_dedup_window = 5.0  # 5 second deduplication window
self._order_cleanup_interval = 60.0  # Clean old entries every 60s
self._last_cleanup = time.time()
```

**Replace `place_order` method (lines 282-onwards) with:**
```python
async def place_order(self, order_request: OrderRequest) -> Order:
    """Place an order on Bybit with idempotency protection.

    CRITICAL: Prevents duplicate orders from network retries.
    """
    # DEFENSIVE CHECK: Prevent real API calls in simulation mode
    if settings.dry_run:
        raise RuntimeError(
            "CRITICAL: place_order() called in dry_run mode! "
            "This would place REAL orders. Route to PaperBroker instead."
        )

    if not settings.is_live:
        raise RuntimeError(
            "CRITICAL: place_order() called without is_live=true! "
            "This is a safety check to prevent accidental real trading."
        )

    # IDEMPOTENCY CHECK
    # Create fingerprint: symbol + side + size + price (rounded to avoid float issues)
    price_key = f"{order_request.price:.8f}" if order_request.price else "market"
    order_fingerprint = (
        f"{order_request.symbol}_{order_request.side}_"
        f"{order_request.size:.8f}_{price_key}"
    )

    now = time.time()

    # Check if identical order placed recently
    if order_fingerprint in self._recent_orders:
        last_placed = self._recent_orders[order_fingerprint]
        time_since_last = now - last_placed

        if time_since_last < self._order_dedup_window:
            logger.error(
                f"DUPLICATE ORDER BLOCKED: {order_fingerprint} "
                f"was placed {time_since_last:.2f}s ago (window: {self._order_dedup_window}s). "
                f"Idempotency violation prevented."
            )
            raise ValueError(
                f"Duplicate order detected: identical order placed {time_since_last:.2f}s ago. "
                "This may be a network retry or double submission. Order rejected for safety."
            )

    # Cleanup old entries periodically
    if (now - self._last_cleanup) > self._order_cleanup_interval:
        self._cleanup_order_history(now)
        self._last_cleanup = now

    # Rate limiting check
    time_since_last_order = now - self._last_order_time
    if time_since_last_order < self._min_order_interval:
        sleep_time = self._min_order_interval - time_since_last_order
        logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s before order")
        await asyncio.sleep(sleep_time)

    self._last_order_time = time.time()

    # Prepare order data
    order_data = {
        "category": "linear",
        "symbol": order_request.symbol,
        "side": "Buy" if order_request.side == "buy" else "Sell",
        "orderType": "Market" if order_request.order_type == "market" else "Limit",
        "qty": str(order_request.size),
    }

    if order_request.price and order_request.order_type == "limit":
        order_data["price"] = str(order_request.price)

    if order_request.stop_loss:
        order_data["stopLoss"] = str(order_request.stop_loss)

    if order_request.take_profit:
        order_data["takeProfit"] = str(order_request.take_profit)

    # Place order via API
    try:
        result = await self._request(
            "POST",
            "/v5/order/create",
            data=order_data,
        )

        # Record successful placement in idempotency cache
        self._recent_orders[order_fingerprint] = now

        # Create Order object
        order = Order(
            order_id=result["orderId"],
            strategy_id=order_request.strategy_id,
            symbol=order_request.symbol,
            side=order_request.side,
            size=order_request.size,
            price=order_request.price,
            order_type=order_request.order_type,
            status=OrderStatus.PLACED,
            stop_loss=order_request.stop_loss,
            take_profit=order_request.take_profit,
            timestamp=datetime.utcnow(),
            metadata=order_request.metadata,
        )

        logger.info(
            f"Order placed successfully: {order.order_id} "
            f"{order.symbol} {order.side} {order.size} @ {order.price or 'market'}"
        )

        return order

    except Exception as e:
        logger.error(f"Failed to place order: {e}", exc_info=True)
        raise

def _cleanup_order_history(self, current_time: float) -> None:
    """Remove old entries from order idempotency cache."""
    cutoff_time = current_time - self._order_dedup_window - 60  # Keep 1 extra minute

    old_count = len(self._recent_orders)
    self._recent_orders = {
        fp: ts for fp, ts in self._recent_orders.items()
        if ts > cutoff_time
    }

    cleaned_count = old_count - len(self._recent_orders)
    if cleaned_count > 0:
        logger.debug(f"Cleaned {cleaned_count} old order entries from idempotency cache")
```

**Test:**
```python
# tests/test_order_idempotency.py
async def test_duplicate_order_blocked():
    client = BybitHTTPClient()

    order_request = OrderRequest(
        signal_id="test",
        strategy_id="test_strategy",
        symbol="BTCUSDT",
        side="buy",
        size=0.001,
        price=50000.0,
        order_type="limit",
    )

    # First order should succeed
    await client.place_order(order_request)

    # Immediate duplicate should be blocked
    with pytest.raises(ValueError, match="Duplicate order detected"):
        await client.place_order(order_request)

    # After 5 seconds, should be allowed
    await asyncio.sleep(5.1)
    await client.place_order(order_request)  # Should succeed
```

---

## Fix #3: WebSocket Reconciliation

### File: `src/hean/exchange/bybit/ws_private.py`

**Add method to BybitPrivateWebSocket class:**
```python
async def reconcile_orders(self) -> None:
    """Reconcile orders after WebSocket reconnection.

    Fetches open orders and recent fills from REST API to catch any
    events missed during disconnection.
    """
    if not self._bybit_http:
        logger.warning("Cannot reconcile orders: no HTTP client available")
        return

    logger.info("🔄 Reconciling orders after WebSocket gap...")

    try:
        # Fetch all open orders
        open_orders_result = await self._bybit_http._request(
            "GET",
            "/v5/order/realtime",
            params={"category": "linear"},
        )

        open_orders = open_orders_result.get("list", [])
        logger.info(f"Found {len(open_orders)} open orders")

        # Fetch recent executions (last 5 minutes)
        five_min_ago = int((time.time() - 300) * 1000)  # 5 min in ms

        executions_result = await self._bybit_http._request(
            "GET",
            "/v5/execution/list",
            params={
                "category": "linear",
                "startTime": str(five_min_ago),
            },
        )

        executions = executions_result.get("list", [])
        logger.info(f"Found {len(executions)} recent executions")

        # Publish ORDER_FILLED events for any fills we missed
        reconciled_count = 0
        for execution in executions:
            order_id = execution["orderId"]

            # Only publish if this is a fill (not just an order placement)
            if execution["execType"] in ("Trade", "Funding"):
                # Create ORDER_FILLED event
                order = Order(
                    order_id=order_id,
                    strategy_id="reconciled",  # Will be updated by OrderManager
                    symbol=execution["symbol"],
                    side=execution["side"].lower(),
                    size=float(execution["execQty"]),
                    price=float(execution["execPrice"]),
                    order_type="limit",  # Unknown, default to limit
                    status=OrderStatus.FILLED,
                    filled_size=float(execution["execQty"]),
                    timestamp=datetime.utcfromtimestamp(int(execution["execTime"]) / 1000),
                )

                await self._bus.publish(Event(
                    event_type=EventType.ORDER_FILLED,
                    data={"order": order}
                ))

                reconciled_count += 1
                logger.info(f"✅ Reconciled fill: {order_id} {order.symbol} {order.side} {order.size}")

        logger.info(
            f"✅ Order reconciliation complete: "
            f"{len(open_orders)} open orders, {reconciled_count} fills reconciled"
        )

    except Exception as e:
        logger.error(f"❌ Order reconciliation failed: {e}", exc_info=True)
```

**Add to `_listen` method (after successful reconnect, around line 178):**
```python
try:
    await self.connect()

    # CRITICAL: Reconcile orders after reconnection
    await self.reconcile_orders()

    # Then re-subscribe to streams
    await self.subscribe_all()

    reconnect_delay = 5.0  # Reset delay on successful reconnect
except Exception as reconnect_err:
    logger.error(f"Reconnection failed: {reconnect_err}")
    reconnect_delay = min(reconnect_delay * 1.5, 60.0)
```

**Add periodic reconciliation (every 60s):**
```python
# In __init__
self._reconcile_task: asyncio.Task | None = None

# In connect()
self._reconcile_task = asyncio.create_task(self._periodic_reconciliation())

# In disconnect()
if self._reconcile_task:
    self._reconcile_task.cancel()
    try:
        await self._reconcile_task
    except asyncio.CancelledError:
        pass

# New method
async def _periodic_reconciliation(self) -> None:
    """Periodically reconcile orders (every 60s) as safety check."""
    while self._connected:
        try:
            await asyncio.sleep(60)  # Every 60 seconds
            await self.reconcile_orders()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Periodic reconciliation error: {e}")
```

---

## Fix #4: RiskGovernor Profit Mode

### File: `src/hean/risk/risk_governor.py`

**Replace `check_and_update` method (lines 76-133) with:**
```python
async def check_and_update(
    self,
    equity: float,
    initial_capital: float,
    positions_count: int,
    orders_count: int,
) -> RiskState:
    """Check risk conditions and update state if needed.

    Enhanced to handle profit mode: when equity > initial capital,
    only enforce drawdown from initial capital, not from daily high.
    """
    if not self._enabled:
        return RiskState.NORMAL

    # Calculate profit status
    profit_above_initial = equity > initial_capital * 1.05  # 5% buffer
    profit_pct = ((equity - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0

    if profit_above_initial:
        # IN PROFIT MODE: Only check drawdown from INITIAL capital
        logger.debug(
            f"Risk Governor in PROFIT MODE: equity ${equity:.2f} is "
            f"{profit_pct:.1f}% above initial ${initial_capital:.2f}"
        )

        drawdown_from_initial = ((initial_capital - equity) / initial_capital * 100) if initial_capital > 0 else 0

        # HARD_STOP only if equity drops BELOW initial by 20%
        # (i.e., net -20% loss from starting capital)
        if drawdown_from_initial >= 20.0:
            await self._escalate_to(
                RiskState.HARD_STOP,
                reason_codes=["PROFIT_ERODED_HARD"],
                metric="drawdown_from_initial",
                value=drawdown_from_initial,
                threshold=20.0,
            )
        elif drawdown_from_initial >= 15.0:
            # QUARANTINE if getting close
            await self._escalate_to(
                RiskState.QUARANTINE,
                reason_codes=["PROFIT_ERODED_QUARANTINE"],
                metric="drawdown_from_initial",
                value=drawdown_from_initial,
                threshold=15.0,
            )
        elif drawdown_from_initial >= 10.0:
            # SOFT_BRAKE as warning
            await self._escalate_to(
                RiskState.SOFT_BRAKE,
                reason_codes=["PROFIT_ERODED_SOFT"],
                metric="drawdown_from_initial",
                value=drawdown_from_initial,
                threshold=10.0,
            )
        else:
            # Still in profit zone, allow normal operation
            if self._state != RiskState.NORMAL:
                logger.info(
                    f"Risk Governor: drawdown from initial reduced to "
                    f"{drawdown_from_initial:.1f}%, de-escalating to NORMAL"
                )
                await self._deescalate_to(RiskState.NORMAL)

        return self._state

    # BELOW INITIAL CAPITAL: Strict enforcement
    drawdown_pct = ((initial_capital - equity) / initial_capital * 100) if initial_capital > 0 else 0

    logger.debug(
        f"Risk Governor in LOSS MODE: equity ${equity:.2f} is "
        f"{abs(profit_pct):.1f}% below initial ${initial_capital:.2f}"
    )

    # Check for HARD_STOP conditions
    if drawdown_pct >= 20.0:
        await self._escalate_to(
            RiskState.HARD_STOP,
            reason_codes=["MAX_DRAWDOWN_HARD"],
            metric="drawdown_pct",
            value=drawdown_pct,
            threshold=20.0,
        )
    # Check for QUARANTINE conditions
    elif drawdown_pct >= 15.0:
        await self._escalate_to(
            RiskState.QUARANTINE,
            reason_codes=["MAX_DRAWDOWN_QUARANTINE"],
            metric="drawdown_pct",
            value=drawdown_pct,
            threshold=15.0,
        )
    # Check for SOFT_BRAKE conditions
    elif drawdown_pct >= 10.0:
        await self._escalate_to(
            RiskState.SOFT_BRAKE,
            reason_codes=["MAX_DRAWDOWN_SOFT"],
            metric="drawdown_pct",
            value=drawdown_pct,
            threshold=10.0,
        )
    # Check for de-escalation (recovery)
    elif self._state != RiskState.NORMAL and drawdown_pct < 5.0:
        if self._can_deescalate():
            await self._deescalate_to(RiskState.NORMAL)

    return self._state
```

---

## Fix #5: Position Monitor

### File: `src/hean/execution/position_monitor.py` (NEW FILE)

```python
"""Position Monitor - Force-close stale positions."""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType, OrderRequest
from hean.exchange.bybit.http import BybitHTTPClient
from hean.logging import get_logger

logger = get_logger(__name__)


class PositionMonitor:
    """Monitor open positions and force-close stale ones.

    Prevents positions from being held indefinitely if:
    - Take profit / stop loss not hit
    - Strategy doesn't close position
    - Market moves sideways
    """

    def __init__(self, bus: EventBus, bybit_http: BybitHTTPClient):
        """Initialize position monitor.

        Args:
            bus: Event bus for publishing events
            bybit_http: Bybit HTTP client for fetching positions
        """
        self._bus = bus
        self._bybit_http = bybit_http
        self._position_timestamps: dict[str, datetime] = {}  # position_key -> opened_at
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start position monitoring."""
        if not settings.position_monitor_enabled:
            logger.info("Position monitor disabled in config")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"Position monitor started: checking every {settings.position_monitor_check_interval}s, "
            f"TTL={settings.max_hold_seconds}s"
        )

    async def stop(self) -> None:
        """Stop position monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Position monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self._check_stale_positions()
                await asyncio.sleep(settings.position_monitor_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitor loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Back off on error

    async def _check_stale_positions(self) -> None:
        """Check for stale positions and force-close them."""
        try:
            # Fetch all positions from exchange
            result = await self._bybit_http._request(
                "GET",
                "/v5/position/list",
                params={"category": "linear", "settleCoin": "USDT"},
            )

            positions = result.get("list", [])
            now = datetime.utcnow()

            for position in positions:
                symbol = position["symbol"]
                size = abs(float(position["size"]))

                if size == 0:
                    continue  # No position

                # Create unique position key
                side = "long" if float(position["size"]) > 0 else "short"
                position_key = f"{symbol}_{side}"

                # Track position opening time
                if position_key not in self._position_timestamps:
                    self._position_timestamps[position_key] = now
                    logger.debug(f"Tracking new position: {position_key} size={size}")
                    continue

                # Check if stale
                opened_at = self._position_timestamps[position_key]
                age_seconds = (now - opened_at).total_seconds()

                if age_seconds > settings.max_hold_seconds:
                    logger.warning(
                        f"⏰ STALE POSITION DETECTED: {symbol} {side} size={size} "
                        f"held for {age_seconds:.0f}s (max: {settings.max_hold_seconds}s). "
                        f"Force closing..."
                    )

                    await self._force_close_position(position)

                    # Remove from tracking
                    del self._position_timestamps[position_key]

                else:
                    # Log progress for long-held positions
                    remaining = settings.max_hold_seconds - age_seconds
                    if remaining < 180:  # Log when < 3 min remaining
                        logger.info(
                            f"Position {position_key} age={age_seconds:.0f}s, "
                            f"will force-close in {remaining:.0f}s"
                        )

        except Exception as e:
            logger.error(f"Error checking stale positions: {e}", exc_info=True)

    async def _force_close_position(self, position: dict[str, Any]) -> None:
        """Force close a position via market order.

        Args:
            position: Position dict from Bybit API
        """
        symbol = position["symbol"]
        size = abs(float(position["size"]))
        current_side = "long" if float(position["size"]) > 0 else "short"
        close_side = "sell" if current_side == "long" else "buy"

        logger.warning(
            f"🔴 FORCE CLOSING POSITION: {symbol} {current_side} size={size} "
            f"via {close_side} market order"
        )

        try:
            # Place market order to close
            order_data = {
                "category": "linear",
                "symbol": symbol,
                "side": "Sell" if close_side == "sell" else "Buy",
                "orderType": "Market",
                "qty": str(size),
                "reduceOnly": True,  # CRITICAL: Don't open new position
            }

            result = await self._bybit_http._request(
                "POST",
                "/v5/order/create",
                data=order_data,
            )

            order_id = result["orderId"]

            logger.info(
                f"✅ Force close order placed: {order_id} {symbol} {close_side} {size}"
            )

            # Publish event for tracking
            await self._bus.publish(Event(
                event_type=EventType.POSITION_FORCE_CLOSED,
                data={
                    "position": position,
                    "reason": "max_hold_seconds_exceeded",
                    "order_id": order_id,
                    "symbol": symbol,
                    "side": current_side,
                    "size": size,
                }
            ))

        except Exception as e:
            logger.error(
                f"❌ Failed to force close position {symbol} {current_side}: {e}",
                exc_info=True,
            )
            # Publish error event
            await self._bus.publish(Event(
                event_type=EventType.ERROR,
                data={
                    "error": f"Force close failed: {e}",
                    "position": position,
                }
            ))
```

**Add to `router_bybit_only.py` (in `__init__`):**
```python
# Add position monitor
from hean.execution.position_monitor import PositionMonitor

self._position_monitor = PositionMonitor(self._bus, self._bybit_http)
```

**Add to `router_bybit_only.py` (in `start()`):**
```python
# Start position monitor
await self._position_monitor.start()
```

**Add to `router_bybit_only.py` (in `stop()`):**
```python
# Stop position monitor
await self._position_monitor.stop()
```

**Add new event type to `src/hean/core/types.py`:**
```python
class EventType(str, Enum):
    # ... existing events ...
    POSITION_FORCE_CLOSED = "POSITION_FORCE_CLOSED"
```

---

## Testing All Fixes

### 1. Unit Tests
```bash
# Test each fix individually
pytest tests/test_eventbus_queue.py -v
pytest tests/test_order_idempotency.py -v
pytest tests/test_websocket_reconciliation.py -v
pytest tests/test_risk_governor_profit_mode.py -v
pytest tests/test_position_monitor.py -v
```

### 2. Integration Test
```bash
# Test full system with fixes
pytest tests/test_critical_fixes_integration.py -v
```

### 3. Manual Test (Testnet)
```bash
# Start system with fixes
docker-compose up --build

# Monitor logs for issues
docker-compose logs -f api

# Check metrics
curl http://localhost:8000/metrics
```

---

## Deployment Checklist

- [ ] Code review completed
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Testnet deployment successful
- [ ] No errors in logs for 1 hour
- [ ] Metrics showing expected behavior
- [ ] Alerts configured in Grafana
- [ ] Runbook updated with new components
- [ ] Team notified of changes

---

## Rollback Plan

If issues occur:

```bash
# Stop containers
docker-compose down

# Revert to previous version
git checkout <previous-commit-hash>

# Rebuild and restart
docker-compose up --build -d

# Monitor recovery
docker-compose logs -f
```

---

## Support

For issues:
1. Check logs: `docker-compose logs -f api`
2. Check metrics: `http://localhost:8000/metrics`
3. Check queue status: `curl http://localhost:8000/system/bus/metrics`
4. Review audit document: `PRODUCTION_AUDIT_2026.md`

---

**Last Updated:** 2026-01-31
**Version:** 1.0

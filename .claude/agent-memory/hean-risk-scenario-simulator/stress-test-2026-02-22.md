# Stress Test Analysis — 2026-02-22

## Files Examined
- `packages/hean-risk/src/hean/risk/risk_governor.py`
- `packages/hean-risk/src/hean/risk/killswitch.py`
- `packages/hean-execution/src/hean/execution/router_bybit_only.py`
- `packages/hean-exchange/src/hean/exchange/bybit/http.py`
- `packages/hean-exchange/src/hean/exchange/bybit/ws_public.py`
- `packages/hean-exchange/src/hean/exchange/bybit/ws_private.py`
- `packages/hean-core/src/hean/core/bus.py`
- `packages/hean-portfolio/src/hean/portfolio/accounting.py`
- `packages/hean-execution/src/hean/execution/position_reconciliation.py`
- `packages/hean-core/src/hean/core/microservices_bridge.py`
- `packages/hean-strategies/src/hean/strategies/impulse_filters.py`
- `packages/hean-app/src/hean/main.py` (selected sections)

## Critical Finding #1: Killswitch Poll Delay (10 SECONDS)

`check_drawdown()` is called ONLY inside `_print_status()` which is scheduled
at `timedelta(seconds=10)` (main.py:1190). During a BTC -30% flash crash,
the system can continue placing orders for up to 10 full seconds after the
threshold is crossed.

At $300 capital, 3x leverage, 5 open positions: potential 15+ USDT additional loss
before the killswitch fires.

The `RiskGovernor` subscribes to EQUITY_UPDATE events and fires in real-time
(via `_on_equity_update`), but EQUITY_UPDATE itself is only published inside
`_print_status()` too — same 10s cadence.

## Critical Finding #2: KillSwitch Does NOT Close Existing Positions

`_handle_killswitch()` (main.py:3154) sets `self._stop_trading = True` and
calls `risk_sentinel.set_stop_trading(True)`. It does NOT call `panic_close_all()`.
Open positions remain open at HARD_STOP. The only protection is TTL (900s = 15 min).

`panic_close_all()` exists at main.py:3232 but is only called on graceful shutdown.

## Critical Finding #3: IntelligenceGate TypeError Under Mock Context

`intelligence_gate.py:198` — `context.prediction.tcn_confidence > 0` raises
`TypeError` when `tcn_confidence` is a MagicMock. This is a real production risk
when context aggregator returns a partially-initialized context object (e.g., during
startup or after Oracle microservice restart). The exception is caught by bus.py
and the signal goes to DLQ instead of being enriched. Test confirms: 2 failures.

## Critical Finding #4: PositionSizer 3x Unexpected Leverage

test_risk.py::test_position_sizer fails: expected size 0.1, got 0.3 (3x leverage
applied unexpectedly). PositionSizer applies `max_leverage` multiplier when no
stop-loss distance limits it. At small $300 capital this triples effective risk.

## Redis Failure Handling

`MicroservicesBridge` consumer loops (microservices_bridge.py:485-490) catch all
exceptions and retry after 1s. This is GRACEFUL — Redis failure stops microservice
data (physics, brain, oracle) from reaching EventBus but does NOT crash the system.
Core in-process trading continues. However:
- Physics data becomes stale immediately
- Brain analysis stops updating
- Oracle signals stop

## WebSocket Reconnect — Public vs Private

Public WS (ws_public.py:182-195): reconnects and re-subscribes all symbols.
STALE DATA RISK: no gap-fill for missed ticks. Strategies will process
first tick after reconnect as if no time passed.

Private WS (ws_private.py:323-396): has `_reconcile_after_reconnect()` which
queries HTTP API for each pending order. REQUIRES http_client to be set via
`set_http_client()`. If not set (ws_private.py:328-333), logs WARNING and returns
without reconciling — missed fills possible.

Key: ExecutionRouter creates BybitPrivateWebSocket at line 63 WITHOUT passing
http_client. http_client is never set on ws_private in router_bybit_only.py.
So reconciliation on reconnect is ALWAYS skipped.

## HTTP Client Retry Logic

`_request_impl()` in http.py: 3 attempts with exponential backoff (1s, 2s).
Handles 429 and 5xx with retry. Circuit breaker opens after 5 failures in window.
When circuit open: raises RuntimeError immediately (fail fast). This is correct.
Total timeout for 3 attempts on 5xx: up to 10s client timeout + 1s + 2s = ~13s.
The httpx client timeout is 10.0s (line 158).

## Idempotency on Exchange Side

Bybit `orderLinkId` (http.py:421): format `{timestamp_ms}_{uuid8}`. This is
unique per `place_order()` call, NOT per signal. If `place_order()` fails after
the order lands on exchange but before the response returns, a retry will create
a SECOND order with a different orderLinkId. The in-process idempotency key
(router_bybit_only.py:106) prevents re-entry at the SIGNAL level but a timeout
retry within `_request_impl()` could duplicate the exchange order.

## EventBus Under Extreme Load

TICK events are LOW priority — dropped when circuit breaker opens (>95% util).
SIGNAL/ORDER_FILLED are FAST_PATH — bypass queue entirely, dispatched synchronously.
CRITICAL queue: 10,000 slots, waits up to 5s before RuntimeError.
If 5s timeout fires on CRITICAL queue, a RuntimeError propagates to the publisher.
Under flash crash with 10+ symbols all generating RISK_ALERT simultaneously,
the critical queue could fill.

## Accounting Race Condition

`record_fill()` (accounting.py:107) is synchronous (no lock). `record_fill_async()`
exists and uses asyncio.Lock. If TradingSystem calls the sync version from
multiple concurrent event handlers (possible with fast-path dispatch), race
condition on `self._cash` is possible. Need to verify which version main.py uses.

# execution -- Order Routing and Execution

Routes risk-approved `ORDER_REQUEST` events to the Bybit exchange, manages order lifecycle, handles position reconciliation, and provides execution quality diagnostics.

## Architecture

The `ExecutionRouter` (`router_bybit_only.py`) is the production execution path. It subscribes to `ORDER_REQUEST` events, validates them, generates correlation IDs for end-to-end tracing, enforces idempotency to prevent duplicate orders from the same signal, and sends orders to Bybit testnet via `BybitHTTPClient`. On fill confirmation (from private WebSocket), it publishes `ORDER_FILLED` and `POSITION_OPENED` events. The `OrderManager` tracks order state transitions (PENDING -> PLACED -> FILLED/CANCELLED/REJECTED). The `PositionReconciler` periodically compares local state with the exchange to detect ghost positions and missed fills. The `TWAPExecutor` can split large orders into time-weighted slices.

## Key Classes

- `ExecutionRouter` (`router_bybit_only.py`) -- Production router for Bybit testnet. Correlation ID tracking, idempotency via signal hash deduplication, smart order type selection (market vs limit based on spread and urgency). Integrates with `MakerRetryQueue` for maker order retries and `SmartLimitExecutor` for slippage prediction.
- `ExecutionRouter` (`router.py`) -- Generic router supporting both paper and live execution. Legacy; `router_bybit_only.py` is preferred for production.
- `OrderManager` (`order_manager.py`) -- Manages order lifecycle and state. Tracks orders by ID, strategy, and symbol. Provides `register_order()`, `update_order()`, `get_open_orders()`.
- `PositionReconciler` (`position_reconciliation.py`) -- Periodic reconciliation of local positions with exchange state. Detects missing positions, size mismatches, and ghost positions. Publishes events on discrepancy.
- `TWAPExecutor` (`twap_executor.py`) -- Time-Weighted Average Price executor. Splits orders above a USD threshold into randomized time slices to reduce market impact.
- `PaperBroker` (`paper_broker.py`) -- Deprecated paper trading broker with fee and slippage simulation. System now uses Bybit testnet instead.
- `ExecutionDiagnostics` (`execution_diagnostics.py`) -- Tracks execution quality metrics: fill rate, slippage, latency.
- `MakerRetryQueue` (`maker_retry_queue.py`) -- Queue for retrying failed maker (limit) orders.
- `SlippageEstimator` (`slippage_estimator.py`) -- Estimates expected slippage for an order.
- `EdgeEstimator` (`edge_estimator.py`) -- Estimates execution edge.
- `SignalDecay` (`signal_decay.py`) -- Detects stale signals that should not be executed.
- `AdaptiveTTL` (`adaptive_ttl.py`) -- Dynamically adjusts order time-to-live based on market conditions.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| ORDER_REQUEST | Subscribes | Risk-approved order to execute |
| TICK | Subscribes | Price updates for paper broker and reconciliation |
| ORDER_PLACED | Publishes | Order submitted to exchange |
| ORDER_FILLED | Publishes | Order filled (partial or complete) |
| ORDER_CANCELLED | Publishes | Order cancelled |
| ORDER_REJECTED | Publishes | Order rejected by exchange |
| POSITION_OPENED | Publishes | New position opened from fill |
| POSITION_CLOSED | Publishes | Position closed |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| DRY_RUN | true | Blocks real order placement with RuntimeError |
| LIVE_CONFIRM | "" | Must be "YES" to allow trading |

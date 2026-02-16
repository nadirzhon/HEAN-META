# exchange -- Bybit Exchange Connectors

Low-level HTTP REST and WebSocket clients for communicating with the Bybit exchange (testnet and mainnet).

## Architecture

All exchange interactions go through this package. The `BybitHTTPClient` handles REST API calls (order placement, position queries, balance checks) with HMAC authentication, rate limiting, circuit breaker protection, and an instrument info cache that saves approximately 150ms per order. The `BybitPublicWebSocket` streams real-time market data (ticks, orderbook, funding rates) and publishes them as EventBus events. The `BybitPrivateWebSocket` streams account-level updates (order fills, position changes) with reconnection reconciliation to prevent missed fills during disconnections. The `SmartLimitExecutor` uses geometric slippage prediction via Riemannian curvature of the orderbook to decide between market and limit order execution.

## Key Classes

- `BybitHTTPClient` (`bybit/http.py`) -- Async HTTP client for Bybit REST API v5. Features: HMAC-SHA256 authentication, instrument info caching, leverage caching, circuit breaker (opens after 5 failures, recovers after 60s), rate limiting (10 orders/sec, 20 total requests/sec), dynamic endpoint switching support. Methods include `place_order()`, `get_positions()`, `get_wallet_balance()`, `get_open_orders()`, `cancel_order()`, `set_leverage()`.
- `BybitPublicWebSocket` (`bybit/ws_public.py`) -- WebSocket client for public market data streams. Subscribes to ticker, orderbook (configurable depth), and funding rate channels. Publishes TICK, ORDER_BOOK_UPDATE, and FUNDING events to EventBus. Auto-reconnects on disconnect.
- `BybitPrivateWebSocket` (`bybit/ws_private.py`) -- WebSocket client for private account streams. Tracks order and position updates. Features reconnection reconciliation: records disconnect time, fetches missed events via HTTP API on reconnect, and recovers fills/cancellations. Publishes ORDER_FILLED, ORDER_CANCELLED, POSITION_UPDATE events.
- `SmartLimitExecutor` (`executor.py`) -- Uses TDA-based geometric slippage prediction (via optional C++ FastWarden module) to predict slippage before sending an order. If predicted slippage exceeds 1%, switches to smart-limit mode with tighter placement.
- `BybitTicker`, `BybitOrderResponse`, `BybitEarnProduct` (`bybit/models.py`) -- Pydantic models for Bybit API response data.

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| TICK | Publishes (ws_public) | Real-time price/volume data |
| ORDER_BOOK_UPDATE | Publishes (ws_public) | Orderbook depth snapshots |
| FUNDING | Publishes (ws_public) | Funding rate updates |
| FUNDING_UPDATE | Publishes (ws_public) | Funding rate changes |
| ORDER_FILLED | Publishes (ws_private) | Order fill confirmations |
| ORDER_CANCELLED | Publishes (ws_private) | Order cancellation confirmations |
| POSITION_UPDATE | Publishes (ws_private) | Position changes |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| BYBIT_API_KEY | "" | Bybit API key |
| BYBIT_API_SECRET | "" | Bybit API secret |
| BYBIT_TESTNET | true | Use testnet (always true for safety) |

# api -- FastAPI Server and WebSocket Gateway

Unified API gateway providing REST endpoints, WebSocket pub/sub with topic-based subscriptions, and real-time data streaming for the iOS app, web dashboard, and external integrations.

## Architecture

The FastAPI application (`main.py`) serves as the bridge between the trading engine and frontends. It exposes REST endpoints under `/api/v1/` via modular routers, and a WebSocket endpoint at `/ws` with topic-based subscriptions (system_status, order_decisions, ai_catalyst, etc.). The `EngineFacade` provides a unified interface to the `TradingSystem`, handling start/stop lifecycle and exposing component references. The `ConnectionManager` manages WebSocket connections with topic-based broadcasting. A cached trading state (orders, positions, account state) is maintained for fast dashboard priming. Authentication is handled via middleware (configurable), and rate limiting uses slowapi. Telemetry events are tracked and broadcast through a dedicated service.

## Key Classes

- `EngineFacade` (`engine_facade.py`) -- Unified facade for trading engine orchestration. Manages TradingSystem lifecycle (start/stop), exposes references to physics engine, brain client, DuckDB store, council, and advanced ML systems. Thread-safe via asyncio.Lock.
- `ConnectionManager` (`services/ws_manager.py`) -- WebSocket connection management with topic-based pub/sub. Clients subscribe to topics and receive broadcasts only for subscribed topics.
- `WebSocketService` (`services/websocket_service.py`) -- Bridges EventBus events to WebSocket clients, forwarding relevant events as real-time updates.
- `TradingMetrics` (`services/trading_metrics.py`) -- Aggregates trading metrics for the `/trading/metrics` endpoint with rolling window counters (1m, 5m, session).
- `MarketDataStore` (`services/market_data_store.py`) -- In-memory cache for latest market data (prices, orderbook, funding rates).
- `TelemetryService` (`telemetry/service.py`) -- Event envelope creation, sequence numbering, and broadcast coordination.

## Routers

| Router | Prefix | Key Endpoints |
|--------|--------|---------------|
| engine | /engine | `POST /start`, `POST /stop`, `GET /status` |
| trading | /trading | `GET /why` (diagnostic), `GET /metrics` |
| strategies | /strategies | `GET /` (list all strategies) |
| risk | /risk | `GET /killswitch/status` |
| risk_governor | /risk/governor | `GET /status` |
| physics | /physics | `GET /state?symbol=X` |
| brain | /brain | `GET /analysis` |
| council | /council | `GET /status` |
| archon | /archon | `GET /status` |
| analytics | /analytics | Analytical endpoints |
| market | /market | Market data endpoints |
| storage | /storage | DuckDB query endpoints |
| telemetry | /telemetry | Telemetry data access |
| meta_learning | /meta-learning | Meta-learning engine status |
| causal_inference | /causal | Causal inference status |
| graph_engine | /graph | Graph engine status |
| multimodal_swarm | /swarm | Multimodal swarm status |
| changelog | /changelog | System changelog |
| system | /system | System-level operations |

## WebSocket Topics

| Topic | Description |
|-------|-------------|
| system_status | Engine state, equity, PnL updates |
| order_decisions | Entry/exit decision notifications |
| ai_catalyst | Brain analysis and oracle predictions |
| risk_alerts | Risk state changes and killswitch events |

## Events

| Event | Direction | Description |
|-------|-----------|-------------|
| ORDER_DECISION | Subscribes | Cached for dashboard priming and WS broadcast |
| ORDER_EXIT_DECISION | Subscribes | Cached for dashboard priming and WS broadcast |
| ORDER_FILLED | Subscribes | Updates cached orders/positions |
| POSITION_OPENED | Subscribes | Updates cached positions |
| POSITION_CLOSED | Subscribes | Updates cached positions |
| PNL_UPDATE | Subscribes | Updates cached account state |
| RISK_ALERT | Subscribes | Forwarded to risk_alerts WS topic |
| BRAIN_ANALYSIS | Subscribes | Forwarded to ai_catalyst WS topic |

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| API_PORT | 8000 | FastAPI server port |
| API_AUTH_ENABLED | false | Enable authentication middleware |
| API_AUTH_TOKEN | "" | Bearer token for authentication |
| CORS_ORIGINS | ["*"] | Allowed CORS origins |

# HEAN API Documentation

## Base URL

- **Development**: `http://localhost:8000`
- **Production**: Configure via environment

## Authentication

Currently, API is unauthenticated. In production, add authentication middleware.

**Live Trading Protection**: All live trading actions require:
- `LIVE_CONFIRM=true` in environment
- `DRY_RUN=false` in environment
- `confirm_phrase: "I_UNDERSTAND_LIVE_TRADING"` in request body

If these conditions are not met, the API returns `403 Forbidden`.

## Endpoints

### Health Check

#### `GET /health`

Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "engine_running": false
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

---

### Settings

#### `GET /settings`

Get system settings (secrets are masked).

**Response:**
```json
{
  "trading_mode": "paper",
  "is_live": false,
  "dry_run": true,
  "initial_capital": 400.0,
  "trading_symbols": ["BTCUSDT", "ETHUSDT"],
  "max_daily_drawdown_pct": 15.0,
  "max_trade_risk_pct": 2.0,
  "bybit_testnet": false,
  "bybit_api_key": "***masked***",
  "bybit_api_secret": "***masked***"
}
```

**Example:**
```bash
curl http://localhost:8000/settings
```

---

### Engine Control

#### `POST /engine/start`

Start the trading engine.

**Request Body:**
```json
{
  "confirm_phrase": null
}
```

**For Live Trading:**
```json
{
  "confirm_phrase": "I_UNDERSTAND_LIVE_TRADING"
}
```

**Response:**
```json
{
  "status": "started",
  "message": "Engine started successfully",
  "trading_mode": "paper",
  "is_live": false,
  "dry_run": true
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/engine/start \
  -H "Content-Type: application/json" \
  -d '{"confirm_phrase": null}'
```

#### `POST /engine/stop`

Stop the trading engine.

**Request Body:**
```json
{}
```

**Response:**
```json
{
  "status": "stopped",
  "message": "Engine stopped successfully"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/engine/stop \
  -H "Content-Type: application/json" \
  -d '{}'
```

#### `POST /engine/pause`

Pause the trading engine (stop accepting new signals).

**Request Body:**
```json
{}
```

**Response:**
```json
{
  "status": "paused",
  "message": "Engine paused"
}
```

#### `POST /engine/resume`

Resume the trading engine.

**Response:**
```json
{
  "status": "resumed",
  "message": "Engine resumed"
}
```

#### `GET /engine/status`

Get current engine status.

**Response:**
```json
{
  "status": "running",
  "running": true,
  "trading_mode": "paper",
  "is_live": false,
  "dry_run": true,
  "equity": 400.50,
  "daily_pnl": 0.50,
  "initial_capital": 400.0
}
```

**Example:**
```bash
curl http://localhost:8000/engine/status
```

---

### Trading Operations

#### `GET /orders/positions`

Get current positions.

**Response:**
```json
[
  {
    "symbol": "BTCUSDT",
    "side": "long",
    "size": 0.001,
    "entry_price": 50000.0,
    "unrealized_pnl": 5.0,
    "realized_pnl": 0.0,
    "position_id": "pos_123"
  }
]
```

**Example:**
```bash
curl http://localhost:8000/orders/positions
```

#### `GET /orders`

Get orders.

**Query Parameters:**
- `status` (optional): Filter by status (`all`, `open`, `filled`)

**Response:**
```json
[
  {
    "order_id": "order_123",
    "symbol": "BTCUSDT",
    "side": "buy",
    "size": 0.001,
    "filled_size": 0.001,
    "price": 50000.0,
    "status": "filled",
    "strategy_id": "funding_harvester",
    "timestamp": "2024-01-01T12:00:00Z"
  }
]
```

**Example:**
```bash
# All orders
curl http://localhost:8000/orders

# Open orders only
curl http://localhost:8000/orders?status=open

# Filled orders only
curl http://localhost:8000/orders?status=filled
```

#### `POST /orders/test`

Place a test order (paper trading only).

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "side": "buy",
  "size": 0.001,
  "price": null
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Test order placed: buy 0.001 BTCUSDT",
  "order_id": "test_order_123"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/orders/test \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "BTCUSDT",
    "side": "buy",
    "size": 0.001
  }'
```

#### `POST /orders/close-position`

Close a position.

**Request Body:**
```json
{
  "position_id": "pos_123",
  "confirm_phrase": null
}
```

**For Live Trading:**
```json
{
  "position_id": "pos_123",
  "confirm_phrase": "I_UNDERSTAND_LIVE_TRADING"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Position pos_123 closed"
}
```

#### `POST /orders/cancel-all`

Cancel all open orders.

**Request Body:**
```json
{
  "confirm_phrase": null
}
```

**For Live Trading:**
```json
{
  "confirm_phrase": "I_UNDERSTAND_LIVE_TRADING"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "All orders cancelled"
}
```

---

### Strategies

#### `GET /strategies`

Get list of strategies.

**Response:**
```json
[
  {
    "strategy_id": "funding_harvester",
    "enabled": true,
    "type": "FundingHarvester"
  }
]
```

#### `POST /strategies/{strategy_id}/enable`

Enable or disable a strategy.

**Request Body:**
```json
{
  "enabled": true
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Strategy funding_harvester enabled"
}
```

#### `POST /strategies/{strategy_id}/params`

Update strategy parameters.

**Request Body:**
```json
{
  "params": {
    "max_position_size": 0.01
  }
}
```

---

### Risk Management

#### `GET /risk/status`

Get risk management status.

**Response:**
```json
{
  "killswitch_triggered": false,
  "stop_trading": false,
  "equity": 400.50,
  "daily_pnl": 0.50,
  "drawdown": 0.0,
  "drawdown_pct": 0.0,
  "max_open_positions": 2,
  "current_positions": 1
}
```

#### `GET /risk/limits`

Get current risk limits.

**Response:**
```json
{
  "max_open_positions": 2,
  "max_daily_attempts": 10,
  "max_exposure_usd": 100.0,
  "min_notional_usd": 5.0,
  "cooldown_seconds": 60
}
```

#### `POST /risk/limits`

Update risk limits (paper only).

**Request Body:**
```json
{
  "max_open_positions": 3,
  "max_daily_attempts": 15
}
```

---

### Analytics

#### `GET /analytics/summary`

Get analytics summary.

**Response:**
```json
{
  "total_trades": 10,
  "win_rate": 60.0,
  "profit_factor": 1.5,
  "max_drawdown": 5.0,
  "max_drawdown_pct": 1.25,
  "avg_trade_duration_sec": 3600.0,
  "trades_per_day": 2.5,
  "total_pnl": 15.0,
  "daily_pnl": 5.0
}
```

#### `GET /analytics/blocks`

Get blocked signals analytics.

**Response:**
```json
{
  "total_blocks": 50,
  "top_reasons": [
    {
      "code": "spread_too_wide",
      "count": 20,
      "message": "Spread exceeds threshold"
    }
  ],
  "blocks_by_hour": {
    "0": 5,
    "1": 3
  },
  "recent_blocks": []
}
```

#### `POST /analytics/backtest`

Run a backtest.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "initial_capital": 10000.0,
  "strategy_id": null
}
```

**Response:**
```json
{
  "job_id": "job_123",
  "status": "pending"
}
```

#### `POST /analytics/evaluate`

Run an evaluation.

**Request Body:**
```json
{
  "symbol": "BTCUSDT",
  "days": 7
}
```

**Response:**
```json
{
  "job_id": "job_456",
  "status": "pending"
}
```

---

### Jobs

#### `GET /jobs`

List recent jobs.

**Query Parameters:**
- `limit` (optional): Maximum number of jobs to return (default: 100)

**Response:**
```json
[
  {
    "job_id": "job_123",
    "status": "completed",
    "created_at": "2024-01-01T12:00:00Z",
    "started_at": "2024-01-01T12:00:01Z",
    "completed_at": "2024-01-01T12:05:00Z",
    "result": { ... },
    "error": null,
    "progress": 1.0
  }
]
```

#### `GET /jobs/{job_id}`

Get job by ID.

**Response:**
```json
{
  "job_id": "job_123",
  "status": "completed",
  "created_at": "2024-01-01T12:00:00Z",
  "started_at": "2024-01-01T12:00:01Z",
  "completed_at": "2024-01-01T12:05:00Z",
  "result": { ... },
  "error": null,
  "progress": 1.0
}
```

---

### System

#### `POST /reconcile/now`

Trigger manual reconcile.

**Response:**
```json
{
  "status": "success",
  "message": "Reconcile completed"
}
```

#### `POST /smoke-test/run`

Run execution smoke test.

**Response:**
```json
{
  "success": true,
  "order_id": "order_123",
  "symbol": "BTCUSDT",
  "side": "BUY",
  "message": "Smoke test completed successfully"
}
```

---

### Streaming Endpoints (SSE)

#### `GET /events/stream`

Stream events via Server-Sent Events (SSE).

**Response:** `text/event-stream`

**Event Format:**
```
data: {"event": "signal", "data": {...}, "timestamp": "2024-01-01T12:00:00Z"}

data: {"event": "order_placed", "data": {...}, "timestamp": "2024-01-01T12:00:01Z"}
```

**Example:**
```bash
curl -N http://localhost:8000/events/stream
```

#### `GET /logs/stream`

Stream logs via Server-Sent Events (SSE).

**Response:** `text/event-stream`

**Log Format:**
```
data: {"level": "info", "message": "Engine started", "timestamp": "2024-01-01T12:00:00Z", "module": "hean.main", "request_id": "uuid"}

data: {"level": "error", "message": "Order rejected", "timestamp": "2024-01-01T12:00:01Z", "module": "hean.execution", "request_id": "uuid"}
```

**Example:**
```bash
curl -N http://localhost:8000/logs/stream
```

---

### Metrics

#### `GET /metrics`

Get Prometheus metrics.

**Response:** `text/plain` (Prometheus format)

**Example:**
```bash
curl http://localhost:8000/metrics
```

**Sample Output:**
```
# HELP hean_engine_status Engine running status (1=running, 0=stopped)
# TYPE hean_engine_status gauge
hean_engine_status 1

# HELP hean_equity Current equity
# TYPE hean_equity gauge
hean_equity 400.50

# HELP hean_orders_total Total orders placed
# TYPE hean_orders_total counter
hean_orders_total 10
```

---

## Error Responses

All endpoints may return standard HTTP error codes:

- `400 Bad Request`: Invalid request parameters
- `403 Forbidden`: Live trading action without proper confirmation
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

**Error Response Format:**
```json
{
  "error": "Error message",
  "request_id": "uuid",
  "detail": "Detailed error information"
}
```

---

## Rate Limiting

Currently, no rate limiting is implemented. In production, add rate limiting middleware.

---

## CORS

CORS is enabled for all origins in development. In production, restrict to specific origins.

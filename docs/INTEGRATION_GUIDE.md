# HEAN Trading System - Integration Guide

## Quick Start

### Prerequisites
- Docker & Docker Compose v2.x
- Python 3.11+
- Redis 7+ (included in docker-compose)
- 8GB RAM minimum (16GB recommended)

### Basic Deployment (Redis + API + UI)

```bash
# 1. Clone and setup
git clone <repo-url>
cd HEAN-META

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys and settings

# 3. Start core services
docker-compose up -d

# 4. Verify services
docker-compose ps
curl http://localhost:8000/health
```

### Full Deployment (with Monitoring)

```bash
# Start all services including Prometheus + Grafana
docker-compose --profile full up -d

# Access services:
# - Trading API: http://localhost:8000
# - UI: http://localhost:3000
# - Grafana: http://localhost:3001 (admin/admin)
# - Prometheus: http://localhost:9090
```

### With Kafka (High Throughput)

```bash
# Start with Kafka for event streaming
docker-compose --profile kafka up -d
```

---

## Integration Points

### 1. EventBus Integration

All modules communicate via the EventBus. Here's how to integrate a new module:

```python
from hean.core.bus import EventBus
from hean.core.types import Event, EventType

class MyModule:
    def __init__(self, bus: EventBus):
        self.bus = bus

    async def start(self):
        # Subscribe to events
        self.bus.subscribe(EventType.TICK, self.on_tick)

    async def on_tick(self, event: Event):
        # Process event
        data = event.data

        # Publish new event
        await self.bus.publish(Event(
            event_type=EventType.SIGNAL,
            data={"symbol": "BTC/USDT", "side": "BUY"}
        ))
```

### 2. Orchestrator Integration

Register your module with the orchestrator for lifecycle management and health monitoring:

```python
from hean.core.orchestrator import TradingOrchestrator

# Create orchestrator
orchestrator = TradingOrchestrator(bus=event_bus)

# Register your module
orchestrator.register_component(
    name="my_module",
    component=my_module,
    circuit_breaker=True,  # Enable fault tolerance
    failure_threshold=5,   # Circuit opens after 5 failures
    timeout=60.0          # Retry after 60 seconds
)

# Start (initializes all components)
await orchestrator.start()
```

### 3. ML Model Integration

Integrate your ML models with the feature pipeline:

```python
from hean.core.types import EventType

class MyMLModel:
    def __init__(self, bus: EventBus):
        self.bus = bus
        self.model = self.load_model()

    async def start(self):
        # Subscribe to features
        self.bus.subscribe(EventType.FEATURES_READY, self.on_features)

    async def on_features(self, event: Event):
        features = event.data["features"]

        # Run inference (with timeout)
        prediction = await asyncio.wait_for(
            self.predict(features),
            timeout=0.1  # 100ms max
        )

        # Publish prediction
        await self.bus.publish(Event(
            event_type=EventType.ML_PREDICTION,
            data={"prediction": prediction}
        ))

    async def predict(self, features):
        # Your ML logic here
        return self.model.predict(features)
```

### 4. Risk Management Integration

Add custom risk checks:

```python
from hean.risk.risk_governor import RiskGovernor

class CustomRiskCheck:
    def __init__(self, bus: EventBus):
        self.bus = bus

    async def start(self):
        # Subscribe to order requests
        self.bus.subscribe(EventType.ORDER_REQUEST, self.validate_order)

    async def validate_order(self, event: Event):
        order = event.data

        # Your risk logic
        if self.is_too_risky(order):
            await self.bus.publish(Event(
                event_type=EventType.ORDER_REJECTED,
                data={
                    "order_id": order["id"],
                    "reason": "Risk limit exceeded"
                }
            ))
            return

        # Approve order
        await self.bus.publish(Event(
            event_type=EventType.ORDER_APPROVED,
            data=order
        ))
```

### 5. Execution Strategy Integration

Add custom execution strategies:

```python
from hean.execution.router import ExecutionRouter

class CustomExecutionStrategy:
    def __init__(self, bus: EventBus, exchange):
        self.bus = bus
        self.exchange = exchange

    async def start(self):
        self.bus.subscribe(EventType.ORDER_APPROVED, self.execute)

    async def execute(self, event: Event):
        order = event.data

        # Custom execution logic (e.g., TWAP, VWAP, Iceberg)
        filled_order = await self.execute_with_strategy(order)

        # Publish fill
        await self.bus.publish(Event(
            event_type=EventType.ORDER_FILLED,
            data=filled_order
        ))
```

---

## Event Flow

### Complete Trade Lifecycle

```
1. Market Data Arrives
   ↓ EventType.TICK

2. Feature Engineering
   ↓ Calculate indicators (RSI, MACD, etc.)
   ↓ EventType.FEATURES_READY

3. ML Inference
   ↓ LightGBM/XGBoost prediction
   ↓ EventType.ML_PREDICTION

4. Strategy Decision
   ↓ Combine ML + technical analysis
   ↓ EventType.SIGNAL

5. Risk Validation
   ↓ Kelly criterion sizing
   ↓ Check exposure limits
   ↓ EventType.ORDER_REQUEST
   ↓ EventType.ORDER_APPROVED (if passed)

6. Execution
   ↓ Smart routing (TWAP/VWAP)
   ↓ Exchange API call
   ↓ EventType.ORDER_FILLED

7. Portfolio Update
   ↓ Update positions & P&L
   ↓ EventType.PNL_UPDATE

8. Monitoring
   ↓ Record metrics
   ↓ Log to Prometheus
```

---

## Event Types Reference

| Event Type | Description | Data Schema |
|------------|-------------|-------------|
| `TICK` | Real-time price update | `{symbol, price, volume, timestamp}` |
| `CANDLE` | OHLCV candle | `{symbol, open, high, low, close, volume, timeframe}` |
| `FEATURES_READY` | Engineered features | `{symbol, features: {...}}` |
| `ML_PREDICTION` | ML model prediction | `{symbol, prediction, confidence}` |
| `SIGNAL` | Trading signal | `{symbol, side, size, signal_strength}` |
| `ORDER_REQUEST` | Order creation request | `{symbol, side, size, type, price?}` |
| `ORDER_APPROVED` | Risk-approved order | `{order_id, symbol, side, size, ...}` |
| `ORDER_PLACED` | Order sent to exchange | `{order_id, exchange_order_id, ...}` |
| `ORDER_FILLED` | Order execution | `{order_id, fill_price, filled_qty, ...}` |
| `ORDER_CANCELLED` | Order cancellation | `{order_id, reason}` |
| `ORDER_REJECTED` | Order rejected by risk | `{order_id, reason}` |
| `PNL_UPDATE` | Portfolio P&L update | `{total_pnl, realized_pnl, unrealized_pnl}` |
| `ERROR` | Error notification | `{component, error, timestamp}` |
| `KILLSWITCH_TRIGGERED` | Emergency stop | `{reason, timestamp}` |

---

## Circuit Breaker Pattern

Circuit breakers prevent cascading failures:

```python
# Circuit breaker states:
# - CLOSED: Normal operation
# - OPEN: Failing, reject all requests
# - HALF_OPEN: Testing recovery

# Execute with circuit breaker protection
result = await orchestrator.execute_with_circuit_breaker(
    component_name="ml_engine",
    func=ml_engine.predict,
    features=features
)
```

**Thresholds (configurable)**:
- Failure threshold: 5 consecutive failures → OPEN
- Timeout: 60 seconds before testing recovery
- Success threshold: 2 successful requests → CLOSED

---

## Health Monitoring

### Health Check API

```bash
# Overall system health
curl http://localhost:8000/health

# Component-specific health
curl http://localhost:8000/health/orchestrator
curl http://localhost:8000/health/ml_engine
curl http://localhost:8000/health/execution
```

### Health Status Response

```json
{
  "orchestrator": {
    "running": true,
    "uptime_seconds": 3600,
    "overall_state": "healthy"
  },
  "components": {
    "ml_engine": {
      "state": "healthy",
      "last_check": 1234567890,
      "failure_count": 0,
      "circuit_breaker": "closed"
    },
    "execution_router": {
      "state": "degraded",
      "last_check": 1234567890,
      "failure_count": 2,
      "circuit_breaker": "half_open",
      "error_message": "API timeout"
    }
  }
}
```

---

## Prometheus Metrics

### Available Metrics

**Trading Metrics:**
- `hean_orders_total` - Total orders placed (counter)
- `hean_pnl` - Current P&L (gauge)
- `hean_latency_seconds` - Processing latency (histogram)
- `hean_trades_total` - Total trades executed (counter)

**System Metrics:**
- `hean_orchestrator_uptime_seconds` - System uptime
- `hean_components_healthy` - Number of healthy components
- `hean_circuit_breakers_open` - Number of open circuit breakers
- `hean_eventbus_queue_size` - Event queue depth

### Query Examples

```promql
# P&L over time
hean_pnl

# Order rate (per second)
rate(hean_orders_total[5m])

# Average latency
histogram_quantile(0.95, hean_latency_seconds_bucket)

# Component health
hean_components_healthy / hean_components_total
```

---

## Grafana Dashboards

Access Grafana at http://localhost:3001 (default: admin/admin)

### Pre-configured Dashboards:

1. **Trading Performance**
   - Real-time P&L
   - Trade volume
   - Win rate & Sharpe ratio
   - Drawdown analysis

2. **System Health**
   - Component status
   - Circuit breaker states
   - Event bus queue depth
   - CPU & Memory usage

3. **Execution Quality**
   - Order fill rate
   - Slippage analysis
   - Latency distribution
   - Exchange API errors

---

## Redis Streams (Event Persistence)

### Why Redis Streams?

- **Persistence**: Events survive restarts
- **Replay**: Audit trail for debugging
- **Multi-consumer**: Multiple services can process same events
- **At-least-once delivery**: Guaranteed processing

### Event Persistence Strategy

```python
# High-frequency events: In-memory only (fast)
EventType.TICK → In-memory EventBus

# Important events: Both in-memory + Redis
EventType.SIGNAL → In-memory + Redis Streams
EventType.ORDER_* → In-memory + Redis Streams
EventType.PNL_UPDATE → In-memory + Redis Streams

# Audit events: Redis only
EventType.ERROR → Redis Streams
```

### Replay Events (Debugging)

```python
from hean.core.event_streams import RedisEventStream

redis_stream = RedisEventStream()
await redis_stream.connect()

# Replay last 100 ORDER_FILLED events
async for event in redis_stream.replay_events(
    event_type=EventType.ORDER_FILLED,
    start_id="-",  # Beginning
    end_id="+"     # End
):
    print(event)
```

---

## Environment Variables

### Core Settings

```bash
# Exchange
EXCHANGE=bybit
API_KEY=your_api_key
API_SECRET=your_api_secret

# Redis
REDIS_URL=redis://localhost:6379
EVENT_MAX_QUEUE_SIZE=50000

# Risk
MAX_POSITION_SIZE=10000  # USDT
MAX_LEVERAGE=3
DAILY_DRAWDOWN_LIMIT=0.05  # 5%

# ML
ML_MODEL_PATH=/models/lightgbm_v1.pkl
ML_INFERENCE_TIMEOUT=100  # milliseconds

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
LOG_LEVEL=INFO
```

---

## Troubleshooting

### Common Issues

**1. EventBus Queue Full**
```bash
# Symptom: "Queue is full" errors
# Solution: Increase queue size or add more consumers
EVENT_MAX_QUEUE_SIZE=100000
```

**2. Circuit Breaker Open**
```bash
# Symptom: "Circuit breaker is OPEN" errors
# Check component health
curl http://localhost:8000/health/ml_engine

# Review logs
docker-compose logs ml-service

# Manual recovery: Restart service
docker-compose restart ml-service
```

**3. Redis Connection Lost**
```bash
# Check Redis
docker-compose logs redis

# Verify connection
redis-cli -h localhost -p 6379 ping

# System has graceful degradation - will continue with in-memory only
```

**4. High Latency**
```bash
# Check Prometheus metrics
# Query: hean_latency_seconds{quantile="0.99"}

# Optimize:
# - Reduce ML inference timeout
# - Enable batching for features
# - Scale horizontally
```

---

## Performance Tuning

### Latency Optimization

```python
# 1. Enable batching for features
FEATURE_BATCH_SIZE=10
FEATURE_BATCH_TIMEOUT=50  # ms

# 2. ML model optimization
# - Use ONNX Runtime for inference
# - Quantize models (FP16)
# - Cache predictions

# 3. Event bus tuning
EVENT_BATCH_SIZE=10  # Process events in batches
EVENT_WORKER_COUNT=4  # Parallel event workers
```

### Throughput Optimization

```bash
# 1. Use Kafka instead of Redis Streams
docker-compose --profile kafka up -d

# 2. Horizontal scaling
docker-compose up -d --scale ml-service=3

# 3. Database connection pooling
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=10
```

---

## Security Best Practices

1. **API Keys**: Never commit to git
   ```bash
   # Use .env file
   echo "API_KEY=secret" >> .env
   echo ".env" >> .gitignore
   ```

2. **Redis**: Enable authentication
   ```bash
   # docker-compose.yml
   command: redis-server --requirepass your_password
   ```

3. **Grafana**: Change default password
   ```bash
   GF_SECURITY_ADMIN_PASSWORD=strong_password
   ```

4. **Network isolation**: Use Docker networks
   ```yaml
   # Only API exposed to host
   api:
     ports:
       - "8000:8000"
   # Internal services not exposed
   redis:
     expose:
       - "6379"
   ```

---

## Next Steps

1. **Customize Strategies**: Implement your own trading logic in `src/strategies/`
2. **Add ML Models**: Train and deploy models in `src/ml/`
3. **Configure Risk**: Adjust risk parameters in `src/risk/`
4. **Monitor Performance**: Create custom Grafana dashboards
5. **Scale Horizontally**: Deploy across multiple servers

---

## Support & Resources

- Architecture: [docs/ARCHITECTURE.md](./ARCHITECTURE.md)
- API Reference: http://localhost:8000/docs
- Grafana Dashboards: http://localhost:3001
- Prometheus Metrics: http://localhost:9090

---

**Happy Trading! 🚀**

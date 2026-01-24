# HEAN Trading System - Unified Architecture

## Overview
Event-driven, production-ready crypto trading system with full observability, fault tolerance, and modular design.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ORCHESTRATOR LAYER                          │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │            TradingOrchestrator (Core Coordinator)             │  │
│  │  - Lifecycle Management  - Health Monitoring  - Circuit Break  │  │
│  └─────────────────────┬───────────────────────────────────────────┘  │
└────────────────────────┼────────────────────────────────────────────┘
                         │
┌────────────────────────┼────────────────────────────────────────────┐
│                  EVENT BUS (Redis Streams + In-Memory)              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  AsyncEventBus: TICK, SIGNAL, ORDER_*, PNL_UPDATE, ERROR    │  │
│  │  Redis Streams: Persistent, Multi-Consumer, Replay Support   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────┬────────┬────────┬─────────┬──────────┬────────────┬──────────┘
      │        │        │         │          │            │
┌─────▼────┐ ┌▼────────▼─┐ ┌─────▼──────┐ ┌─▼─────────┐ ┌▼──────────┐
│   ML     │ │  FEATURE  │ │    RISK    │ │ EXECUTION │ │ MONITORING│
│  MODELS  │ │ ENGINEER  │ │ MANAGEMENT │ │   ENGINE  │ │  & HEALTH │
└──────────┘ └───────────┘ └────────────┘ └───────────┘ └───────────┘
```

## Layer Details

### 1. Orchestrator Layer
**Component**: `src/core/orchestrator.py`

**Responsibilities**:
- Initialize & coordinate all subsystems
- Manage lifecycle (startup, shutdown, graceful degradation)
- Monitor health of all components
- Implement circuit breakers
- Handle fault tolerance & retries

**Key Methods**:
- `async start()` - Initialize all modules in dependency order
- `async stop()` - Graceful shutdown with cleanup
- `async health_check()` - Real-time component status
- `async handle_failure(component)` - Circuit breaker logic

---

### 2. Event Bus Layer
**Components**:
- `src/core/bus.py` (In-memory, high-speed)
- `src/core/event_streams.py` (Redis Streams for persistence)

**Event Types**:
```python
EventType.TICK           # Market data updates (high-frequency)
EventType.CANDLE         # Aggregated OHLCV data
EventType.SIGNAL         # Trading signals from strategies
EventType.ORDER_REQUEST  # Order placement requests
EventType.ORDER_FILLED   # Order execution confirmations
EventType.PNL_UPDATE     # Portfolio performance updates
EventType.HEALTH_CHECK   # Component health status
EventType.ERROR          # Error notifications
```

**Features**:
- Async pub/sub pattern
- Batching for performance
- Backpressure handling
- Redis persistence for replay
- Multi-consumer groups

---

### 3. ML Models Layer
**Components**:
- `src/ml/model_manager.py` - Model lifecycle
- `src/ml/inference_engine.py` - Real-time predictions
- `src/ml/feature_store.py` - Feature caching

**Integrated Models**:
- **LightGBM**: Fast gradient boosting for price prediction
- **XGBoost**: Classification for trend direction
- **RL Agent**: Reinforcement learning for optimal execution

**Integration Points**:
```python
# Subscribe to features
bus.subscribe(EventType.FEATURES_READY, ml_manager.on_features)

# Publish predictions
await bus.publish(Event(
    event_type=EventType.ML_PREDICTION,
    data={"symbol": "BTC/USDT", "signal": 0.85, "confidence": 0.92}
))
```

**Fault Tolerance**:
- Model timeout (max 100ms per prediction)
- Fallback to previous model version
- Circuit breaker after 3 consecutive failures

---

### 4. Feature Engineering Layer
**Components**:
- `src/features/technical.py` - TA-Lib indicators
- `src/features/orderbook.py` - Depth, imbalance, liquidity
- `src/features/sentiment.py` - Market sentiment analysis
- `src/features/aggregator.py` - Feature pipeline

**Technical Indicators (TA-Lib)**:
- RSI, MACD, Bollinger Bands
- ATR (volatility), ADX (trend strength)
- Volume-weighted indicators

**Orderbook Features**:
- Bid-ask imbalance
- Depth pressure
- Liquidity score
- Order flow imbalance (OFI)

**Integration Points**:
```python
# Subscribe to tick data
bus.subscribe(EventType.TICK, feature_engine.on_tick)
bus.subscribe(EventType.CANDLE, feature_engine.on_candle)

# Publish engineered features
await bus.publish(Event(
    event_type=EventType.FEATURES_READY,
    data={"symbol": "BTC/USDT", "features": feature_vector}
))
```

---

### 5. Risk Management Layer
**Components**:
- `src/risk/kelly_criterion.py` - Position sizing
- `src/risk/risk_governor.py` - Portfolio limits
- `src/risk/killswitch.py` - Emergency stop
- `src/risk/dynamic_risk.py` - Adaptive risk based on volatility

**Risk Checks**:
1. **Pre-trade**: Position size, leverage, concentration
2. **Real-time**: Drawdown monitoring, exposure limits
3. **Post-trade**: P&L validation, slippage analysis

**Integration Points**:
```python
# Intercept order requests
bus.subscribe(EventType.ORDER_REQUEST, risk_governor.validate_order)

# Monitor portfolio health
bus.subscribe(EventType.PNL_UPDATE, risk_governor.check_drawdown)

# Emergency stop
if killswitch.should_stop():
    await bus.publish(Event(EventType.EMERGENCY_STOP))
```

**Kelly Criterion**:
- Optimal position sizing based on win rate & payoff ratio
- Dynamic adjustment based on recent performance

**Monte Carlo Simulation**:
- VAR (Value at Risk) estimation
- Stress testing under extreme scenarios

---

### 6. Execution Engine Layer
**Components**:
- `src/execution/router.py` - Smart order routing
- `src/execution/atomic_executor.py` - Atomic execution
- `src/execution/order_manager.py` - Order lifecycle
- `src/hft/twap.py`, `src/hft/vwap.py` - Advanced execution algos

**Execution Strategies**:
- **Market Orders**: Immediate execution
- **Limit Orders**: Maker rebate optimization
- **TWAP**: Time-weighted average price (reduce slippage)
- **VWAP**: Volume-weighted average price (large orders)
- **Iceberg Orders**: Hidden size for large positions

**Integration Points**:
```python
# Receive approved orders from risk layer
bus.subscribe(EventType.ORDER_APPROVED, execution_router.execute)

# Publish execution results
await bus.publish(Event(
    event_type=EventType.ORDER_FILLED,
    data={"order_id": "...", "fill_price": 50000, "filled_qty": 0.1}
))
```

**Fault Tolerance**:
- Retry logic with exponential backoff
- Order state recovery from Redis
- Exchange API circuit breaker

---

### 7. Monitoring & Observability Layer
**Components**:
- `src/observability/metrics.py` - Prometheus metrics
- `src/observability/health.py` - Health checks
- `src/observability/monitoring/self_healing.py` - Auto-recovery

**Prometheus Metrics**:
```python
# Trading metrics
orders_total = Counter('hean_orders_total', 'Total orders placed')
pnl_gauge = Gauge('hean_pnl', 'Current P&L')
latency_histogram = Histogram('hean_latency_seconds', 'Processing latency')

# System metrics
event_bus_queue_size = Gauge('hean_eventbus_queue_size', 'Event queue depth')
circuit_breaker_state = Enum('hean_circuit_breaker', 'Circuit breaker state')
```

**Grafana Dashboards**:
- Trading Performance: P&L, Sharpe, Drawdown
- System Health: CPU, Memory, Event Queue
- Execution Quality: Slippage, Fill Rate, Latency

**Logging (Loguru)**:
```python
logger.info("Order placed", order_id="...", symbol="BTC/USDT", side="BUY")
logger.warning("High queue depth", queue_size=45000, max_size=50000)
logger.error("Execution failed", error=str(e), exchange="bybit")
```

**Integration Points**:
```python
# Subscribe to all events for metrics
bus.subscribe(EventType.ORDER_FILLED, metrics.record_order)
bus.subscribe(EventType.PNL_UPDATE, metrics.update_pnl)
bus.subscribe(EventType.ERROR, metrics.record_error)
```

---

## Fault Tolerance Strategy

### Circuit Breakers
```python
class CircuitBreaker:
    """Prevents cascading failures."""

    CLOSED: Normal operation
    OPEN: Fail fast, reject requests
    HALF_OPEN: Test recovery with limited requests

    # Thresholds
    failure_threshold = 5   # Open after 5 failures
    timeout = 60            # Try recovery after 60s
    success_threshold = 2   # Close after 2 successes
```

### Retry Logic
```python
# Exponential backoff with jitter
async def retry_with_backoff(func, max_retries=3):
    for i in range(max_retries):
        try:
            return await func()
        except RetryableError as e:
            if i == max_retries - 1:
                raise
            delay = (2 ** i) + random.uniform(0, 1)
            await asyncio.sleep(delay)
```

### Health Checks
```python
# Component health endpoints
GET /health/orchestrator  → {"status": "healthy", "uptime": 3600}
GET /health/ml_models     → {"status": "degraded", "reason": "model_timeout"}
GET /health/execution     → {"status": "healthy", "queue_size": 5}
```

---

## Data Flow Example: Trade Lifecycle

```
1. TICK arrives from exchange
   ↓ EventBus (EventType.TICK)

2. Feature Engine processes tick
   ↓ Calculate RSI, MACD, OFI
   ↓ EventBus (EventType.FEATURES_READY)

3. ML Model generates prediction
   ↓ LightGBM inference (< 100ms)
   ↓ EventBus (EventType.ML_PREDICTION)

4. Strategy generates signal
   ↓ Combine ML + technical indicators
   ↓ EventBus (EventType.SIGNAL)

5. Risk Manager validates
   ↓ Kelly criterion sizing
   ↓ Check exposure limits
   ↓ EventBus (EventType.ORDER_REQUEST)

6. Execution Router places order
   ↓ Smart routing (TWAP/VWAP)
   ↓ Exchange API call
   ↓ EventBus (EventType.ORDER_FILLED)

7. Monitoring records metrics
   ↓ Update Prometheus
   ↓ Log to stdout (JSON)
```

---

## Configuration Management

### Environment Variables (.env)
```bash
# Exchange
EXCHANGE=bybit
API_KEY=***
API_SECRET=***

# Event Bus
REDIS_URL=redis://localhost:6379
EVENT_BATCH_SIZE=10
EVENT_MAX_QUEUE_SIZE=50000

# ML Models
ML_MODEL_PATH=/models/lightgbm_v1.pkl
ML_INFERENCE_TIMEOUT=100  # milliseconds

# Risk
MAX_POSITION_SIZE=10000  # USDT
MAX_LEVERAGE=3
DAILY_DRAWDOWN_LIMIT=0.05  # 5%

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3001
LOG_LEVEL=INFO
```

---

## Deployment Architecture

### Docker Compose Stack
```yaml
services:
  orchestrator:     # Main trading system
  redis:            # Event persistence + caching
  kafka:            # Optional: High-throughput event streaming
  prometheus:       # Metrics collection
  grafana:          # Visualization
  postgres:         # Historical data (trades, P&L)
  ml-service:       # Separate service for heavy ML inference
```

### Resource Allocation
```
orchestrator: 2 CPU, 4GB RAM
redis:        1 CPU, 2GB RAM
kafka:        2 CPU, 4GB RAM
prometheus:   1 CPU, 2GB RAM
grafana:      0.5 CPU, 1GB RAM
ml-service:   4 CPU, 8GB RAM (GPU optional)
```

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Tick-to-Signal Latency | < 50ms | TBD |
| Order Placement Latency | < 100ms | TBD |
| Event Bus Throughput | > 10k events/sec | TBD |
| ML Inference Time | < 100ms | TBD |
| System Uptime | > 99.9% | TBD |

---

## Security Considerations

1. **API Keys**: Stored in encrypted vault, never in code
2. **Redis**: Password-protected, network isolation
3. **Kafka**: SSL/TLS encryption, SASL authentication
4. **Grafana**: OAuth2 authentication
5. **Logs**: Sensitive data (keys, IPs) redacted

---

## Future Enhancements

1. **Multi-Exchange Support**: Binance, OKX, Kraken
2. **Advanced ML**: Transformer models, meta-learning
3. **Distributed Execution**: Multi-region deployment
4. **Hardware Acceleration**: FPGA for ultra-low latency
5. **Advanced Risk**: CVA (Credit Valuation Adjustment)

---

## References

- [Event-Driven Architecture Patterns](https://martinfowler.com/articles/201701-event-driven.html)
- [Circuit Breaker Pattern](https://martinfowler.com/bliki/CircuitBreaker.html)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Kelly Criterion](https://en.wikipedia.org/wiki/Kelly_criterion)

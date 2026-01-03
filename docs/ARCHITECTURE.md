# HEAN Architecture

## Overview

HEAN is a production-grade, event-driven crypto trading research system for Bybit. This document describes the system architecture, data flows, and key components.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Web UI)                        │
│  Dashboard | Strategies | Risk | Orders | Positions | Logs      │
└──────────────────────────────┬──────────────────────────────────┘
                                │ HTTP/REST
┌──────────────────────────────▼──────────────────────────────────┐
│                      FastAPI Backend (API)                        │
│  /health | /engine/* | /positions | /orders | /metrics           │
└──────────────────────────────┬──────────────────────────────────┘
                                │
┌──────────────────────────────▼──────────────────────────────────┐
│                    Engine Facade (Orchestration)                 │
│  Unified interface for TradingSystem + ProcessFactory            │
└──────────────────────────────┬──────────────────────────────────┘
                                │
        ┌───────────────────────┴───────────────────────┐
        │                                               │
┌───────▼────────┐                            ┌─────────▼────────┐
│ TradingSystem  │                            │ ProcessFactory   │
│ (Main Engine)  │                            │ (Extension)      │
└───────┬────────┘                            └─────────┬───────┘
        │                                               │
        └───────────────────────┬───────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │     Event Bus          │
                    │  (Async Event System)  │
                    └───────────┬───────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                         │
┌───────▼────────┐    ┌─────────▼────────┐    ┌─────────▼────────┐
│  Strategies    │    │   Risk Mgmt      │    │   Execution      │
│  - Funding     │    │   - Limits        │    │   - Router       │
│  - Basis       │    │   - Killswitch   │    │   - Paper/Live   │
│  - Impulse     │    │   - PositionSize │    │   - Reconcile    │
└───────┬────────┘    └─────────┬────────┘    └─────────┬───────┘
        │                       │                         │
        └───────────────────────┼─────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Portfolio           │
                    │   - Accounting        │
                    │   - Allocation        │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Exchange (Bybit)    │
                    │   - HTTP API          │
                    │   - WebSocket         │
                    └───────────────────────┘
```

## Data Flow: Signal → Decision → Execution → Reconcile

### 1. Signal Generation

**Strategies** generate trading signals based on market data:

```
Market Data (Ticks/Candles) → Strategy Analysis → Signal Event
```

- **Funding Harvester**: Monitors funding rates, generates signals for funding income
- **Basis Arbitrage**: Detects basis spreads, generates arbitrage signals
- **Impulse Engine**: Identifies momentum opportunities, generates impulse signals

### 2. Decision Layer

**Risk Management** and **Portfolio** components evaluate signals:

```
Signal Event → Risk Checks → Position Sizing → OrderRequest Event
```

- **Risk Limits**: Validates against drawdown, position limits, leverage caps
- **Position Sizer**: Calculates appropriate position size based on risk
- **Decision Memory**: Blocks bad contexts based on historical performance

### 3. Execution Layer

**Execution Router** routes orders to appropriate broker:

```
OrderRequest Event → Execution Router → Paper/Live Broker → Order Event
```

- **Paper Broker**: Simulates execution with fees and slippage
- **Live Broker**: Places real orders via Bybit HTTP API
- **Order Manager**: Tracks order lifecycle and state

### 4. Reconcile Layer

**Reconcile Service** periodically syncs internal state with exchange:

```
Internal State (Positions/Orders) ↔ Bybit API ↔ Exchange State
```

- **Position Reconcile**: Compares internal positions with exchange positions
- **Order Reconcile**: Verifies order status matches exchange state
- **Discrepancy Detection**: Identifies and logs any mismatches

## Key Components

### Engine Facade

The `EngineFacade` provides a unified interface for:
- Starting/stopping the trading engine
- Getting engine status and metrics
- Accessing positions and orders
- Orchestrating both TradingSystem and ProcessFactory

### TradingSystem

Main trading orchestrator that:
- Manages event bus and clock
- Initializes strategies and risk components
- Handles portfolio accounting
- Coordinates execution routing

### ProcessFactory

Extension layer for process-based workflows:
- Process definitions and execution
- Capital routing and allocation
- Process lifecycle management
- Environment scanning

### Reconcile Service

Ensures consistency between internal state and exchange:
- Periodic position reconciliation
- Order status verification
- Discrepancy reporting
- Lag tracking

## API Endpoints

### Engine Control
- `POST /engine/start` - Start trading engine
- `POST /engine/stop` - Stop trading engine
- `GET /engine/status` - Get engine status

### Data Access
- `GET /positions` - Get current positions (with reconcile)
- `GET /orders` - Get orders (with reconcile)
- `GET /settings` - Get system settings (secrets masked)

### Operations
- `POST /smoke-test/run` - Run execution smoke test
- `POST /backtest` - Start backtest
- `POST /evaluate` - Start evaluation

### Observability
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics

## Observability

### Logging

Structured logging with:
- **Request ID**: Unique ID per API request
- **Trace ID**: Correlation ID for distributed tracing
- **Context**: Additional context in log messages

### Metrics

Prometheus metrics include:
- `hean_engine_status` - Engine running status
- `hean_equity` - Current equity
- `hean_reconcile_lag_seconds` - Time since last reconcile
- `hean_orders_total` - Total orders
- `hean_errors_total` - Error count

### Health Checks

Health endpoints provide:
- System status (healthy/unhealthy)
- Trading mode (paper/live)
- Live trading flag
- Dry run status

## Safety Mechanisms

1. **Default Safety**: Paper trading by default, DRY_RUN=true
2. **Live Trading Gates**: Requires LIVE_CONFIRM=YES, confirm_phrase, DRY_RUN=false
3. **Killswitch**: Automatic stop on drawdown/error limits
4. **Risk Limits**: Per-trade, per-position, per-strategy limits
5. **Reconcile**: Periodic verification of state consistency

## Deployment

### Development

```bash
make dev  # Starts API + Frontend + Monitoring
```

### Production

```bash
docker-compose up -d  # Starts all services
```

Services:
- **API**: Port 8000 (FastAPI backend)
- **Frontend**: Port 3000 (Nginx + Dashboard)
- **Prometheus**: Port 9091 (Metrics)
- **Grafana**: Port 3000 (Dashboards)

## Future Enhancements

- WebSocket streaming for real-time updates
- Distributed event bus (Redis)
- Multi-exchange support
- Advanced reconciliation with auto-correction
- Machine learning integration for strategy optimization


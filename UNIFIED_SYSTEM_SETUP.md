# HEAN Unified System - Setup Guide

This guide covers the complete setup of the unified HEAN Trading System with FastAPI Gateway, Next.js Frontend, and C++ Core integration.

## Architecture Overview

```
┌─────────────────┐
│  Next.js        │  ← Singularity Dashboard (Cyber-Command Center)
│  Frontend       │
└────────┬────────┘
         │ WebSocket (Pub/Sub Topics)
         │ REST API
┌────────▼────────┐
│  FastAPI        │  ← Unified API Gateway (main.py)
│  Gateway        │     - WebSocket Pub/Sub
│                 │     - Emergency Kill-Switch
│                 │     - REST Endpoints
└────────┬────────┘
         │
    ┌────┴────┬──────────────┐
    │         │              │
┌───▼───┐ ┌──▼───┐    ┌──────▼──────┐
│ Redis │ │Event │    │ C++ Core    │
│State  │ │ Bus  │    │ GraphEngine │
└───────┘ └──────┘    └─────────────┘
```

## Prerequisites

- Python 3.11+
- Node.js 18+
- Redis server
- C++ compiler (for C++ core)

## Backend Setup

### 1. Install Python Dependencies

```bash
cd /Users/macbookpro/Desktop/HEAN
pip install -e ".[dev]"
```

### 2. Configure Environment

Ensure your `.env` file has:
```env
REDIS_URL=redis://localhost:6379/0
API_HOST=0.0.0.0
API_PORT=8000
```

### 3. Start Redis

```bash
# Using Docker
docker run -d -p 6379:6379 redis:7-alpine

# Or using local Redis
redis-server
```

### 4. Run the Unified API Gateway

The unified gateway (`src/hean/api/main.py`) replaces the old `app.py`:

```bash
# Option 1: Using uvicorn directly
uvicorn hean.api.main:app --host 0.0.0.0 --port 8000 --reload

# Option 2: Update docker-compose.yml to use main.py
# Change: command: uvicorn hean.api.main:app --host 0.0.0.0 --port 8000
```

## Frontend Setup

### 1. Install Dependencies

```bash
cd control-center
npm install
```

### 2. Configure Environment

Create `control-center/.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

### 3. Run Development Server

```bash
npm run dev
```

The dashboard will be available at `http://localhost:3000`

## Integration Testing

Run the integration test script to verify all components:

```bash
cd /Users/macbookpro/Desktop/HEAN
./scripts/integration_test.sh
```

The script tests:
1. ✅ Redis connectivity
2. ✅ C++ to Python shared memory
3. ✅ API response times (< 100ms)
4. ✅ WebSocket latency (< 20ms)
5. ✅ Emergency Kill-Switch endpoint

## WebSocket Pub/Sub Topics

The system uses topic-based subscriptions:

### Available Topics

- `ticker_{symbol}` - Real-time price updates (e.g., `ticker_btcusdt`)
- `signals` - Trading signals from strategies
- `orders` - Order fills and status updates
- `ai_reasoning` - AI decision reasoning
- `system_status` - System health and status updates

### Subscribing from Frontend

```typescript
const { subscribe, unsubscribe } = useWebSocket();

// Subscribe to BTC ticker
subscribe('ticker_btcusdt');

// Unsubscribe
unsubscribe('ticker_btcusdt');
```

## Emergency Kill-Switch

The Panic Button on the dashboard triggers the emergency kill-switch:

**Endpoint:** `POST /api/v1/emergency/killswitch`

This immediately:
1. Triggers the C++ Emergency Kill-Switch via EventBus
2. Broadcasts `KILLSWITCH_TRIGGERED` event
3. Stops all trading activity
4. Response time target: < 100ms

## Auto-Healing Reconnection

The frontend automatically:
- Detects WebSocket disconnections
- Reconnects with exponential backoff (max 10 attempts)
- Resubscribes to all topics on reconnect
- Shows "Reconnecting..." overlay during downtime
- Syncs state without page reload

## Design System

The HEAN Design System provides:

### Colors
- Primary: `#00ff88` (HEAN Green)
- Secondary: `#00d4ff` (Cyan)
- Danger: `#ff3366` (Red)
- Background: `#0a0e27` (Dark Blue)

### Components
- `.hean-button` - Standard button style
- `.hean-card` - Glass-morphism card
- `.hean-glow` - Glowing effects
- `.hean-text-glow` - Text with glow

All styles are in `control-center/app/globals.css`

## Production Deployment

### Docker Compose

Update `docker-compose.yml`:

```yaml
services:
  api:
    command: uvicorn hean.api.main:app --host 0.0.0.0 --port 8000
    
  frontend:
    build:
      context: ./control-center
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://api:8000
      - NEXT_PUBLIC_WS_URL=ws://api:8000
```

### Build Frontend

```bash
cd control-center
npm run build
npm start
```

## Monitoring

### Health Checks

- **API Health:** `GET /health`
- **System Status:** `GET /health/pulse`
- **Dashboard Data:** `GET /api/v1/dashboard`

### Metrics

- Prometheus metrics: `GET /metrics`
- WebSocket connection count: Monitored in ConnectionManager
- Latency tracking: Built into integration test

## Troubleshooting

### WebSocket Connection Issues

1. Check API is running: `curl http://localhost:8000/health`
2. Verify WebSocket endpoint: `wscat -c ws://localhost:8000/ws`
3. Check browser console for connection errors

### Redis Connection Issues

1. Verify Redis is running: `redis-cli ping`
2. Check `REDIS_URL` in `.env`
3. Review logs for connection errors

### C++ Core Issues

1. Verify Python bindings: `python -c "from hean.core.intelligence.graph_engine import GraphEngine"`
2. Check shared memory permissions
3. Review C++ compilation logs

## Performance Targets

- **API Response Time:** < 100ms (health checks)
- **WebSocket Latency:** < 20ms (ping-pong)
- **Kill-Switch Response:** < 100ms
- **Dashboard Refresh:** 1 second
- **Data Packet Latency:** < 20ms (C++ to UI)

## Next Steps

1. ✅ Unified API Gateway operational
2. ✅ Next.js Dashboard deployed
3. ✅ WebSocket Pub/Sub active
4. ✅ Emergency Kill-Switch connected
5. ✅ Auto-healing implemented
6. ✅ Integration tests passing

**The system is now production-ready!** 🚀
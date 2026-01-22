# HEAN Trading Command Center - UI Documentation

## Overview

The HEAN Trading Command Center is a production-grade web interface for monitoring and controlling the trading system. It provides real-time updates, comprehensive analytics, risk management, and full control over the trading engine.

## Access

- **URL**: `http://localhost:3000` (when running via `make dev`)
- **API Base**: `http://localhost:8000` (configurable via `NEXT_PUBLIC_API_URL`)
- **WebSocket**: `ws://localhost:8000/ws` (configurable via `NEXT_PUBLIC_WS_URL`)

## Pages

### Dashboard

The main control center showing:
- **Key Metrics**: Equity, Daily PnL, Open Positions, Orders, Drawdown, Win Rate
- **Event Feed**: Real-time events from the trading system (SSE stream)
- **Health Panel**: System health indicators

**Actions**:
- Start/Stop/Pause/Resume Engine
- Reconcile positions/orders with exchange
- Run smoke test

### Trading

Monitor and manage positions and orders:
- **Positions Table**: All open positions with PnL, entry price, current price
- **Orders Table**: All orders (open, filled, cancelled) with status
- **Actions**:
  - Place test order (paper only)
  - Close position
  - Cancel all orders

### Strategies

Manage trading strategies:
- View all available strategies
- Enable/disable strategies
- View strategy parameters
- Strategy performance metrics

### Analytics

Performance analysis and diagnostics:
- **Summary**: Total trades, win rate, profit factor, max drawdown
- **Blocked Signals**: Top reasons for blocked trades, frequency analysis
- **Jobs**: Backtest and evaluation job queue with progress

**Actions**:
- Run backtest
- Run evaluation
- View job history

### Risk

Risk management and monitoring:
- **Risk Status**: Killswitch status, stop trading flag, equity, drawdown
- **Risk Limits**: Max positions, daily attempts, exposure limits
- **Gate Inspector**: Detailed view of why signals are blocked

### Logs

Real-time log stream:
- Filter by log level (debug, info, warning, error)
- Search logs
- Auto-scroll option
- Ring buffer (last 2000 logs)

### Settings

System configuration:
- Trading mode (paper/live)
- DRY_RUN, LIVE_CONFIRM flags
- Live trading checklist
- System settings (masked secrets)

## Features

### Real-time Updates

- **WebSocket**: Real-time topics (system status, orders, signals)
- **Polling**: Dashboard metrics refresh every 2 seconds

### Command Palette

Press `Ctrl+K` (or `Cmd+K` on Mac) to open the command palette for quick actions:
- Start Engine
- Stop Engine
- Run Smoke Test
- Toggle Strategy
- Navigate to pages

### Hotkeys

- `Ctrl+K`: Open command palette
- More hotkeys to be added

### Theme Toggle

Click the theme toggle button (🌓) in the top bar to switch between light and dark themes.

### Confirm Danger Modal

For live trading actions, a confirmation modal appears requiring:
1. Checkbox: "I understand this action"
2. Confirmation phrase: `I_UNDERSTAND_LIVE_TRADING`
3. Display of current flags (DRY_RUN, LIVE_CONFIRM, Mode)

## API Client

The frontend uses a small hook-based client:

- `control-center/lib/hooks.ts`: polling data (`/v1/dashboard`, `/health`)
- `control-center/lib/websocket.ts`: WebSocket subscriptions and reconnects

## Event Schema

Events from `/api/events/stream` follow this structure:

```json
{
  "event": "signal|order_placed|position_opened|...",
  "data": { ... },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Log Schema

Logs from `/api/logs/stream` follow this structure:

```json
{
  "level": "debug|info|warning|error",
  "message": "Log message",
  "timestamp": "2024-01-01T00:00:00Z",
  "module": "module_name",
  "request_id": "uuid"
}
```

## Security

- All secrets are masked in the UI
- Live trading actions require explicit confirmation
- CORS is configured for development (restrict in production)
- Request IDs for traceability

## Development

### File Structure

```
control-center/
├── app/                    # Next.js app router
├── components/             # UI components
├── lib/                    # API + WebSocket hooks
├── public/                 # Static assets
├── next.config.js
├── tailwind.config.js
└── tsconfig.json
```

### Running Locally

1. Start backend: `uvicorn hean.api.main:app --reload --host 0.0.0.0 --port 8000`
2. Start frontend: `cd control-center && npm install && npm run dev`
3. Access: `http://localhost:3000`

### Building

The frontend is served via Next.js in Docker. To rebuild:

```bash
docker-compose up -d --build web
```

## Troubleshooting

### Events not showing

- Check browser console for WebSocket connection errors
- Verify backend is running and `/ws` is accessible

### API calls failing

- Verify backend is running on port 8000
- Check CORS settings
- Check browser console for errors

### Logs not streaming

- Check browser console for WebSocket connection errors
- Verify `/ws` endpoint is accessible
- Check backend logs for errors

## Future Enhancements

- [ ] Advanced filtering and search
- [ ] Export functionality (CSV, PDF)
- [ ] Customizable dashboards
- [ ] Alert system
- [ ] Multi-user support with RBAC
- [ ] Mobile responsive design improvements

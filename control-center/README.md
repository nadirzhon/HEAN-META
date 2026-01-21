# HEAN Singularity Dashboard

The production-ready Next.js dashboard for the HEAN Trading System - a cyber-command center interface for real-time trading operations.

## Features

- **Real-time Market Data** - Live price feeds with < 20ms latency
- **WebSocket Pub/Sub** - Topic-based subscriptions for efficient data streaming
- **Panic Button** - Emergency kill-switch with < 100ms response time
- **Auto-Healing** - Automatic reconnection and state synchronization
- **HEAN Design System** - Unified cyber-command center aesthetic
- **Live Activity Feed** - Orders, signals, and AI reasoning in real-time

## Quick Start

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Environment Variables

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
```

## Architecture

The dashboard connects to the FastAPI gateway via:
- **REST API** - For dashboard data and control endpoints
- **WebSocket** - For real-time pub/sub topics

### WebSocket Topics

Subscribe to real-time data:

```typescript
const { subscribe } = useWebSocket();

// Market data
subscribe('ticker_btcusdt');
subscribe('ticker_ethusdt');

// Trading events
subscribe('signals');
subscribe('orders');
subscribe('ai_reasoning');

// System status
subscribe('system_status');
```

## Components

- `PanicButton` - Emergency kill-switch trigger
- `MarketPulse` - Real-time price charts
- `TradingMetrics` - P&L, equity, win rate
- `OrderFeed` - Live order and signal feed
- `SystemStatus` - Health monitoring
- `ReconnectingOverlay` - Auto-healing UI

## Design System

The HEAN Design System provides a cohesive cyber-command center aesthetic:

- **Dark surfaces** with glass-morphism
- **Glowing data points** for live metrics
- **Real-time streaming charts** for market visualization
- **Unified color palette** (HEAN Green, Cyan, Red)

See `app/globals.css` for all design tokens.

## Development

The dashboard uses:
- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **SWR** - Data fetching with auto-refresh
- **Recharts** - Data visualization
- **Framer Motion** - Animations

## Production Build

```bash
npm run build
npm start
```

The production build is optimized for performance and includes:
- Code splitting
- Image optimization
- Static generation where possible
- WebSocket connection pooling
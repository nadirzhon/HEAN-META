# Phase 3 UI Integration - Architecture Diagram

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HEAN Trading System                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────┐
│   Physics Engine        │
│   (Python Module)       │
└──────────┬──────────────┘
           │ emit Event(PHYSICS_UPDATE)
           │ { temperature, entropy, phase, ... }
           ↓
┌─────────────────────────┐
│     Brain Module        │
│   (Claude AI Client)    │
└──────────┬──────────────┘
           │ emit Event(BRAIN_ANALYSIS)
           │ { stage, content, confidence, ... }
           ↓
┌─────────────────────────────────────────────────────────────────────┐
│                          EventBus                                    │
│  Central async event dispatcher with priority queues                │
└───────────┬──────────────────────────────────┬──────────────────────┘
            │                                   │
            │ subscribe                         │ subscribe
            ↓                                   ↓
┌──────────────────────────┐      ┌──────────────────────────┐
│ handle_physics_update    │      │ handle_brain_analysis    │
│ (src/hean/api/main.py)   │      │ (src/hean/api/main.py)   │
└───────────┬──────────────┘      └───────────┬──────────────┘
            │                                   │
            │ emit_topic_event                  │ emit_topic_event
            │ topic="physics_update"            │ topic="brain_update"
            ↓                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                   ConnectionManager (WebSocket)                      │
│                  broadcast_to_topic(topic, data)                     │
└───────────┬──────────────────────────────────┬──────────────────────┘
            │                                   │
            │ WebSocket /ws                     │ WebSocket /ws
            ↓                                   ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  React Frontend (apps/ui/)                           │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  RealtimeClient                                              │   │
│  │  - subscribes to: physics_update, brain_update               │   │
│  │  - normalizes EventEnvelope                                  │   │
│  │  - handles reconnection                                      │   │
│  └───────────┬──────────────────────────────────┬───────────────┘   │
│              │                                   │                   │
│              ↓                                   ↓                   │
│  ┌──────────────────────┐          ┌──────────────────────┐        │
│  │ usePhysicsWebSocket  │          │ useTradingData       │        │
│  │ - physics state      │          │ - trading metrics    │        │
│  │ - brain decisions    │          │ - orders/positions   │        │
│  │ - market price       │          │ - account state      │        │
│  └───────────┬──────────┘          └──────────┬───────────┘        │
│              │                                  │                    │
│              ↓                                  ↓                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  App.tsx - Tab Navigation                                    │   │
│  │                                                               │   │
│  │  ┌─────────────────┐         ┌──────────────────────────┐   │   │
│  │  │ Tab: Cockpit    │         │ Tab: Physics Dashboard   │   │   │
│  │  │                 │         │                          │   │   │
│  │  │ • Funnel        │         │ • DashboardV2            │   │   │
│  │  │ • Portfolio     │         │   - TemperatureGauge     │   │   │
│  │  │ • Controls      │         │   - EntropyGauge         │   │   │
│  │  │ • EventFeed     │         │   - PhaseIndicator       │   │   │
│  │  │ • Debug Panel   │         │   - GravityMap           │   │   │
│  │  │                 │         │   - PlayersRadar         │   │   │
│  │  │                 │         │   - BrainTimeline        │   │   │
│  │  └─────────────────┘         └──────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Physics Update Flow
```
PhysicsEngine
    ↓
Event(PHYSICS_UPDATE, data={temperature, entropy, ...})
    ↓
EventBus
    ↓
handle_physics_update()
    ↓
emit_topic_event(topic="physics_update", ...)
    ↓
ConnectionManager.broadcast_to_topic("physics_update", ...)
    ↓
WebSocket → All subscribed clients
    ↓
RealtimeClient (Frontend)
    ↓
usePhysicsWebSocket hook
    ↓
DashboardV2 component (renders gauges/maps)
```

### Brain Analysis Flow
```
BrainModule (Claude API)
    ↓
Event(BRAIN_ANALYSIS, data={stage, content, confidence, ...})
    ↓
EventBus
    ↓
handle_brain_analysis()
    ↓
emit_topic_event(topic="brain_update", ...)
    ↓
ConnectionManager.broadcast_to_topic("brain_update", ...)
    ↓
WebSocket → All subscribed clients
    ↓
RealtimeClient (Frontend)
    ↓
usePhysicsWebSocket hook
    ↓
BrainTimeline component (renders decisions)
```

## Event Payload Schemas

### PHYSICS_UPDATE
```typescript
{
  temperature: number;      // 0.0 - 1.0 (market heat)
  entropy: number;          // 0.0 - 1.0 (disorder/volatility)
  phase: 'ICE' | 'WATER' | 'VAPOR';
  regime: string;           // 'normal' | 'trending' | 'choppy'
  liquidations: Array<{
    price: number;
    volume: number;
    side: 'long' | 'short';
    isMagnet?: boolean;
  }>;
  players: {
    marketMaker: number;
    institutional: number;
    arbBot: number;
    retail: number;
    whale: number;
  };
}
```

### BRAIN_ANALYSIS
```typescript
{
  stage: 'detect' | 'analyze' | 'decide' | 'execute';
  content: string;          // Human-readable decision text
  confidence: number;       // 0.0 - 1.0
  analysis: object;         // Detailed analysis data
}
```

## WebSocket Topics

| Topic | Description | Event Source |
|-------|-------------|-------------|
| `physics_update` | Market physics state | PhysicsEngine → EventBus |
| `brain_update` | AI analysis decisions | BrainModule → EventBus |
| `market_data` | Price ticks | Exchange → EventBus |
| `market_ticks` | Tick events | Exchange → EventBus |
| `orders` | Order updates | ExecutionRouter → EventBus |
| `positions` | Position changes | Portfolio → EventBus |
| `risk_events` | Risk blocks/alerts | RiskGovernor → EventBus |
| `strategy_events` | Strategy signals | Strategies → EventBus |
| `trading_metrics` | Funnel metrics | TradingMetrics service |

## Component Tree

```
App.tsx
├── ErrorBoundary
│   └── Tabs
│       ├── TabsList
│       │   ├── TabsTrigger (Cockpit)
│       │   └── TabsTrigger (Physics Dashboard)
│       │
│       ├── TabsContent (cockpit)
│       │   ├── StatusBar
│       │   ├── TradingFunnelDashboard
│       │   ├── PortfolioCard
│       │   ├── WhyNotTradingPanel
│       │   ├── ControlPanel
│       │   ├── DebugPanel
│       │   └── EventFeed
│       │
│       └── TabsContent (physics)
│           └── DashboardV2
│               ├── TemperatureGauge
│               ├── EntropyGauge
│               ├── PhaseIndicator
│               ├── GravityMap
│               ├── PlayersRadar
│               ├── BrainTimeline
│               └── PositionsTable
```

## Key Integration Points

1. **EventType enum** (`src/hean/core/types.py`)
   - PHYSICS_UPDATE = "physics_update"
   - BRAIN_ANALYSIS = "brain_analysis"

2. **Backend handlers** (`src/hean/api/main.py`)
   - `handle_physics_update()` - line ~1299
   - `handle_brain_analysis()` - line ~1320
   - Event subscriptions - line ~1340

3. **Frontend types** (`apps/ui/src/app/api/client.ts`)
   - WsTopic: "physics_update" | "brain_update"

4. **Frontend hook** (`apps/ui/src/app/hooks/usePhysicsWebSocket.ts`)
   - Topics: ['market_data', 'market_ticks', 'physics_update', 'brain_update']

5. **UI navigation** (`apps/ui/src/app/App.tsx`)
   - Tabs component wrapping both views
   - DashboardV2 imported and rendered

## Defense Mechanisms

✅ **Defensive Rendering**: All components use optional chaining and defaults
✅ **Connection Health**: WebSocket status, staleness indicators, auto-reconnect
✅ **Error Boundaries**: Wrap all component trees to prevent crashes
✅ **No Silent Failures**: All errors visible to user with actionable messages
✅ **Component Preservation**: Original Cockpit untouched, additive changes only

---

**Architecture Status**: ✅ VERIFIED AND DOCUMENTED
**Integration Status**: ✅ COMPLETE
**Build Status**: ✅ PASSING (UI + Backend)

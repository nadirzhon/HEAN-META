# Mock Data System Architecture

Visual guide to the HEAN iOS mock data system architecture.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                         iOS App                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              SwiftUI Views                            │  │
│  │  (MarketListView, PositionsView, PortfolioView, etc) │  │
│  └────────────┬─────────────────────────────────────────┘  │
│               │                                              │
│               │ Fetch data / Subscribe to updates           │
│               ▼                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         MockServiceContainer (Singleton)              │  │
│  │                                                        │  │
│  │  ┌──────────────────────────────────────────────┐    │  │
│  │  │  MarketServiceProtocol                       │    │  │
│  │  │  TradingServiceProtocol                      │    │  │
│  │  │  PortfolioServiceProtocol                    │    │  │
│  │  │  EventServiceProtocol                        │    │  │
│  │  └──────────────────────────────────────────────┘    │  │
│  └────────────┬─────────────────────────────────────────┘  │
│               │                                              │
│               │ Implemented by                               │
│               ▼                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Mock Services                            │  │
│  │                                                        │  │
│  │  MockMarketService    ─┐                              │  │
│  │  MockTradingService   ─┼─ All depend on               │  │
│  │  MockPortfolioService ─┤  MockDataProvider            │  │
│  │  MockEventService     ─┘                              │  │
│  └────────────┬─────────────────────────────────────────┘  │
│               │                                              │
│               │ Generate data using                          │
│               ▼                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            MockDataProvider                           │  │
│  │                                                        │  │
│  │  Static generators for:                               │  │
│  │  - Markets (12 pairs)                                 │  │
│  │  - Candles (OHLCV)                                    │  │
│  │  - Positions (3 active)                               │  │
│  │  - Orders (6 total)                                   │  │
│  │  - Portfolio                                          │  │
│  │  - Events                                             │  │
│  │  - Sparklines                                         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Data Flow

### Initial Load

```
View.task
  │
  ├─> await service.fetchMarkets()
  │     │
  │     ├─> MockDataProvider.generateMarkets()
  │     │     │
  │     │     └─> [Market] with realistic data
  │     │
  │     └─> Simulate 200ms network delay
  │
  └─> Update @State var markets
```

### Real-Time Updates

```
Timer (2s)
  │
  ├─> MockMarketService.updatePrices()
  │     │
  │     ├─> For each market:
  │     │     │
  │     │     ├─> Apply random walk (±0.08-0.15%)
  │     │     ├─> Update 24h high/low
  │     │     └─> Recalculate 24h change%
  │     │
  │     └─> marketUpdatesSubject.send(updatedMarkets)
  │
  └─> View receives via Combine
        │
        └─> Updates UI
```

## Service Dependencies

```
MockServiceContainer
  │
  ├─> MockMarketService
  │     │
  │     ├─> Internal: [Market]
  │     ├─> Timer: updatePrices() every 2s
  │     └─> Publisher: marketUpdates
  │
  ├─> MockTradingService
  │     │
  │     ├─> Dependency: MockMarketService
  │     ├─> Internal: [Position], [Order]
  │     ├─> Timer: updatePositionPrices() every 2s
  │     ├─> Timer: fillRandomLimitOrder() every 10s
  │     ├─> Publisher: positionUpdates
  │     └─> Publisher: orderUpdates
  │
  ├─> MockPortfolioService
  │     │
  │     ├─> Dependency: MockTradingService
  │     ├─> Internal: Portfolio
  │     ├─> Subscribe: tradingService.positionUpdates
  │     │              └─> updatePortfolio()
  │     └─> Publisher: portfolioUpdates
  │
  └─> MockEventService
        │
        ├─> Internal: [TradingEvent], WebSocketHealth
        ├─> Timer: scheduleNextEvent() (2-5s random)
        ├─> Timer: publishHealthUpdate() every 1s
        ├─> Publisher: eventStream
        └─> Publisher: wsHealthUpdates
```

## Update Cycles

```
Timeline (seconds):
0         1         2         3         4         5         6
│─────────│─────────│─────────│─────────│─────────│─────────│
│
├─ Market prices update
│  └─> Positions recalculate PnL
│       └─> Portfolio updates equity
│
├─ WS health update
│
├─ Market prices update
│  └─> Positions recalculate PnL
│       └─> Portfolio updates equity
│
├─ WS health update
│  └─> New event generated
│
├─ Market prices update
│  └─> Positions recalculate PnL
│       └─> Portfolio updates equity
│
├─ WS health update
│
...continues every 1-2 seconds
```

## Protocol Implementations

```
┌──────────────────────────────────┐
│   MarketServiceProtocol          │
│                                  │
│   + marketUpdates                │
│   + fetchMarkets()               │
│   + fetchMarket(symbol:)         │
│   + fetchCandles(...)            │
└──────────────────────────────────┘
            △
            │ implements
            │
┌──────────────────────────────────┐
│     MockMarketService            │
│                                  │
│   - markets: [Market]            │
│   - updateTimer: Timer?          │
│   - marketUpdatesSubject         │
│                                  │
│   + init()                       │
│     └─> startPriceUpdates()      │
│   + updatePrices()               │
└──────────────────────────────────┘
```

## Data Models

```
Market
├─ id: String
├─ symbol: String
├─ baseCurrency: String
├─ quoteCurrency: String
├─ price: Double ← updates every 2s
├─ change24h: Double ← recalculated
├─ changePercent24h: Double ← recalculated
├─ volume24h: Double
├─ high24h: Double ← updated if new high
└─ low24h: Double ← updated if new low

Position
├─ id: String
├─ symbol: String
├─ side: PositionSide (.long/.short)
├─ size: Double
├─ entryPrice: Double ← never changes
├─ markPrice: Double ← syncs with market
├─ unrealizedPnL: Double ← calculated
├─ unrealizedPnLPercent: Double ← calculated
├─ leverage: Int
└─ createdAt: Date

Portfolio
├─ equity: Double ← calculated
├─ availableBalance: Double ← calculated
├─ usedMargin: Double ← calculated
├─ unrealizedPnL: Double ← sum of positions
├─ realizedPnL: Double
├─ initialCapital: Double
└─ lastUpdated: Date

Order
├─ id: String
├─ symbol: String
├─ side: OrderSide (.buy/.sell)
├─ type: OrderType (.market/.limit)
├─ status: OrderStatus (.new/.filled/etc)
├─ price: Double?
├─ quantity: Double
├─ filledQuantity: Double ← updates on fill
├─ createdAt: Date
└─ updatedAt: Date

TradingEvent
├─ id: String
├─ type: EventType
├─ symbol: String?
├─ message: String
├─ timestamp: Date
└─ metadata: [String: String]?
```

## Price Update Algorithm

```swift
// Every 2 seconds for each market:

currentPrice = 67234.50 (BTC example)
volatility = 0.0008 (0.08% for BTC)

priceChange = currentPrice * random(-volatility...volatility)
            = 67234.50 * random(-0.0008...0.0008)
            = random(-53.79...53.79)

newPrice = currentPrice + priceChange
         = 67234.50 + random(-53.79...53.79)
         = range: 67180.71 to 67288.29

// Result: smooth, realistic price movement
// Over 30 updates (1 minute): natural drift
// No jumps, no spikes, just realistic flow
```

## Position PnL Calculation

```swift
// For LONG positions:
priceDiff = markPrice - entryPrice
          = 67234.50 - 65000.00
          = 2234.50

unrealizedPnL = priceDiff * size
              = 2234.50 * 0.05
              = 111.73

unrealizedPnL% = (priceDiff / entryPrice) * 100 * leverage
               = (2234.50 / 65000.00) * 100 * 5
               = 3.438% * 5
               = 17.19%
               // But displayed as 3.44% (without leverage multiplier in UI)

// For SHORT positions:
priceDiff = entryPrice - markPrice
          = 3500.00 - 3456.78
          = 43.22

unrealizedPnL = priceDiff * size
              = 43.22 * 0.5
              = 21.61
```

## Order Lifecycle

```
Market Order:
User places → Instant fill → Position created
  ↓             ↓              ↓
NEW         → FILLED      → positionUpdates.send()
  ↓
orderUpdates.send()

Limit Order:
User places → Wait for price → Fill randomly → Position created
  ↓             ↓                ↓               ↓
NEW         → NEW            → FILLED        → positionUpdates.send()
  ↓             (30% chance       ↓
orderUpdates.send()  every 10s)  orderUpdates.send()

Cancel Order:
User cancels → Remove from active
  ↓              ↓
NEW         → CANCELLED
  ↓
orderUpdates.send()
```

## Event Generation

```
Timer with random delay (2-5 seconds)
  ↓
Select random event type:
  - Signal (Long/Short detected)
  - Order Placed
  - Order Filled
  - Position Opened/Closed
  - Risk Alert
  - System Info
  ↓
Select random symbol (if applicable)
  ↓
Format message with template
  ↓
Create TradingEvent with timestamp
  ↓
eventStreamSubject.send(event)
  ↓
View receives and displays
```

## Memory Management

```
MockServiceContainer (Singleton)
  │
  ├─> Holds strong references to all services
  │
  └─> Services hold:
        ├─> Timers (invalidated in deinit)
        ├─> Publishers (PassthroughSubject)
        └─> [weak self] in closures
              └─> Prevents retain cycles

Views
  │
  ├─> Hold service references via container
  ├─> Store Combine subscriptions in Set<AnyCancellable>
  └─> Use [weak self] in sink closures
        └─> Prevents memory leaks
```

## Thread Safety

```
@MainActor
  │
  ├─> All UI updates happen on main thread
  │
  └─> MockTradingService methods marked @MainActor:
        - updatePositionPrices()
        - fillRandomLimitOrder()
        - createPositionFromOrder()

Timer callbacks
  │
  └─> Run on main run loop
        └─> Safe to update @Published properties

Combine .receive(on: DispatchQueue.main)
  │
  └─> Ensures UI updates on main thread
```

## Extension Points

```
Want to add new data?
  │
  └─> Add to MockDataProvider
        └─> static func generateNewData() -> [NewType]

Want to add new service?
  │
  ├─> Define protocol in Services/
  ├─> Implement mock in Mock/
  └─> Add to MockServiceContainer

Want to change behavior?
  │
  ├─> Update intervals: change Timer.scheduledTimer()
  ├─> Change volatility: modify random walk algorithm
  └─> Add custom events: extend generateEvents()
```

## Testing Strategy

```
Unit Tests
  │
  ├─> Test each service in isolation
  ├─> Verify data generation
  └─> Check Combine publishers

Integration Tests
  │
  ├─> Test service dependencies
  ├─> Verify update propagation
  └─> Check portfolio calculations

UI Tests
  │
  ├─> Use mock services (not real network)
  ├─> Verify display logic
  └─> Test user interactions
```

## Production Migration

```
Development                Production
─────────────────         ────────────────────

MockServiceContainer      ProductionServiceContainer
  │                         │
  ├─ MockMarketService      ├─ APIMarketService
  │    ↓                    │    ↓
  │  Timers/Random          │  HTTP + WebSocket
  │                         │
  ├─ MockTradingService     ├─ APITradingService
  │    ↓                    │    ↓
  │  Simulated fills        │  Real order execution
  │                         │
  └─ Same protocols!        └─ Same protocols!
       │                         │
       └─────────────────────────┘
                 │
         Views don't change!
```

## Summary

- Clean architecture with protocol-based design
- Dependency injection via singleton container
- Real-time updates via Combine publishers
- Realistic data with smooth random walk
- Proper memory management with weak references
- Thread-safe with @MainActor
- Easy to extend and customize
- Seamless migration to production

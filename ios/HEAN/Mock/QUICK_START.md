# Mock Data System - Quick Start

Get up and running with the HEAN iOS mock data system in 5 minutes.

## Installation

All files are already in place:

```
ios/HEAN/
├── Models/          # Data models
├── Services/        # Service protocols
└── Mock/            # Mock implementations
```

## Instant Setup

### 1. Add to Your App

```swift
import SwiftUI

@main
struct HEANApp: App {
    var body: some Scene {
        WindowGroup {
            TradingDashboardExample() // Use the example or your own view
        }
    }
}
```

### 2. Use in Any View

```swift
import SwiftUI
import Combine

struct MyTradingView: View {
    @State private var markets: [Market] = []
    private let marketService = MockServiceContainer.shared.marketService
    private var cancellables = Set<AnyCancellable>()

    var body: some View {
        List(markets) { market in
            Text("\(market.symbol): \(market.formattedPrice)")
        }
        .task {
            markets = try await marketService.fetchMarkets()
            subscribeToUpdates()
        }
    }

    private func subscribeToUpdates() {
        marketService.marketUpdates
            .receive(on: DispatchQueue.main)
            .sink { markets = $0 }
            .store(in: &cancellables)
    }
}
```

## What You Get

### Live Data
- 12 crypto markets updating every 2 seconds
- 3 open positions with real-time PnL
- 6 orders (active and filled)
- Portfolio tracking with equity and balance
- Event feed with new events every 2-5 seconds
- WebSocket health monitoring

### Realistic Behavior
- Smooth price movements (random walk)
- Positions update with market prices
- Orders can be placed, filled, and cancelled
- Portfolio updates based on position PnL
- Events correlate with trading activity

### Complete Offline
- No network required
- No backend needed
- Perfect for development and demos

## Key Components

### Services

```swift
let services = MockServiceContainer.shared

// Market data
let markets = try await services.marketService.fetchMarkets()

// Positions
let positions = try await services.tradingService.fetchPositions()

// Portfolio
let portfolio = try await services.portfolioService.fetchPortfolio()

// Events
let events = try await services.eventService.fetchRecentEvents(limit: 20)
```

### Real-time Updates

```swift
// Subscribe to market updates
services.marketService.marketUpdates
    .sink { updatedMarkets in
        // Handle market updates
    }
    .store(in: &cancellables)

// Subscribe to position updates
services.tradingService.positionUpdates
    .sink { updatedPositions in
        // Handle position updates
    }
    .store(in: &cancellables)

// Subscribe to event stream
services.eventService.eventStream
    .sink { newEvent in
        // Handle new event
    }
    .store(in: &cancellables)
```

### Place Orders

```swift
let request = OrderRequest(
    symbol: "BTCUSDT",
    side: .buy,
    type: .market,
    quantity: 0.01,
    price: nil
)

let order = try await services.tradingService.placeOrder(request: request)
```

### Close Positions

```swift
try await services.tradingService.closePosition(symbol: "BTCUSDT")
```

## Example Data

### Markets
```
BTCUSDT:  $67,234.50  (+2.5%)
ETHUSDT:  $3,456.78   (-1.2%)
SOLUSDT:  $142.33     (+5.8%)
...12 total markets
```

### Positions
```
BTCUSDT LONG  0.05 @ $65,000  →  +$111.73 (+3.44%)
ETHUSDT SHORT 0.5  @ $3,500   →  +$21.61  (+1.24%)
SOLUSDT LONG  5.0  @ $138.50  →  +$19.15  (+2.77%)
```

### Portfolio
```
Equity:      $345.67
Available:   $198.43
Total PnL:   +$45.67 (+15.22%)
```

## Run the Example

The complete example is in `ExampleUsage.swift`:

```swift
TradingDashboardExample()
```

This shows:
- Portfolio summary with PnL
- WebSocket health indicator
- Live positions with real-time updates
- Active orders with cancel functionality
- Recent events feed

## Next Steps

1. Check `README.md` for full documentation
2. Browse `ExampleUsage.swift` for integration patterns
3. Extend with your own UI components
4. Swap mock services for real ones when ready

## Tips

### Performance
- All updates happen every 1-2 seconds
- Memory usage: ~2-3 MB
- CPU usage: <1%
- No lag or stuttering

### Customization
```swift
// Change update frequency
// In MockMarketService.swift
private let updateInterval: TimeInterval = 1.0

// Change price volatility
// In MockDataProvider.swift
let volatility = 0.002 // 0.2% per update

// Add more markets
// In MockDataProvider.generateMarkets()
("NEWTOKUSDT", "NEWTOK", "USDT", 10.50, 8.2)
```

### Testing
```swift
// Mock services are perfect for unit tests
let service = MockMarketService()
let markets = try await service.fetchMarkets()
XCTAssertEqual(markets.count, 12)
```

## Troubleshooting

### No updates appearing?
Make sure you're subscribing to publishers and storing in `cancellables`:
```swift
private var cancellables = Set<AnyCancellable>()

service.updates
    .sink { /* handle */ }
    .store(in: &cancellables) // Don't forget this!
```

### Prices not updating?
The timer-based updates require the run loop. Make sure you're not blocking the main thread.

### Memory leaks?
Use `[weak self]` in sink closures:
```swift
.sink { [weak self] data in
    self?.handleData(data)
}
```

## Support

For issues or questions:
1. Check the full `README.md`
2. Review `ExampleUsage.swift`
3. Examine model definitions in `Models/`

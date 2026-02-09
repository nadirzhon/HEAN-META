# HEAN iOS Mock Data System

Comprehensive mock data system for the HEAN trading app that simulates realistic market behavior, order execution, and portfolio management.

## Overview

The mock system provides:

- **Realistic market data** with smooth price updates
- **Live position tracking** with PnL calculations
- **Order simulation** including fills, cancellations, and partial fills
- **Portfolio updates** based on position changes
- **Event stream** with trading activity
- **WebSocket health monitoring**
- **Complete offline functionality**

## Architecture

```
MockServiceContainer (DI Container)
├── MockMarketService (implements MarketServiceProtocol)
├── MockTradingService (implements TradingServiceProtocol)
├── MockPortfolioService (implements PortfolioServiceProtocol)
└── MockEventService (implements EventServiceProtocol)
```

All services use **Combine publishers** for real-time updates.

## Data Models

### Market
- 12 crypto pairs (BTC, ETH, SOL, BNB, XRP, ADA, DOGE, MATIC, LINK, AVAX, ATOM, DOT)
- Realistic pricing and 24h changes
- Smooth price updates every 2 seconds
- Volume and high/low tracking

### Position
- LONG/SHORT positions with leverage
- Real-time mark price updates
- Automatic PnL calculation
- Entry price and unrealized PnL tracking

### Order
- Market and limit orders
- Order states: NEW, PARTIALLY_FILLED, FILLED, CANCELLED
- Simulated fills over time
- Fill percentage tracking

### Portfolio
- Equity, available balance, used margin
- Realized and unrealized PnL
- Total PnL vs initial capital
- Margin usage percentage

### TradingEvent
- Signal, order, position, and system events
- Timestamped event feed
- Event rate tracking
- Correlated with actual trading activity

## Usage

### Basic Setup

```swift
import SwiftUI

@main
struct HEANApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(MockServiceContainer.shared)
        }
    }
}
```

### Market Data

```swift
import SwiftUI
import Combine

struct MarketListView: View {
    @State private var markets: [Market] = []
    private let marketService = MockServiceContainer.shared.marketService
    private var cancellables = Set<AnyCancellable>()

    var body: some View {
        List(markets) { market in
            HStack {
                VStack(alignment: .leading) {
                    Text(market.symbol)
                        .font(.headline)
                    Text(market.formattedPrice)
                        .font(.subheadline)
                }
                Spacer()
                Text(market.formattedChangePercent)
                    .foregroundColor(market.isPositiveChange ? .green : .red)
            }
        }
        .task {
            await loadMarkets()
            subscribeToUpdates()
        }
    }

    private func loadMarkets() async {
        do {
            markets = try await marketService.fetchMarkets()
        } catch {
            print("Error loading markets: \(error)")
        }
    }

    private func subscribeToUpdates() {
        marketService.marketUpdates
            .receive(on: DispatchQueue.main)
            .sink { updatedMarkets in
                markets = updatedMarkets
            }
            .store(in: &cancellables)
    }
}
```

### Positions

```swift
struct PositionsView: View {
    @State private var positions: [Position] = []
    private let tradingService = MockServiceContainer.shared.tradingService
    private var cancellables = Set<AnyCancellable>()

    var body: some View {
        List(positions) { position in
            VStack(alignment: .leading) {
                HStack {
                    Text(position.symbol)
                        .font(.headline)
                    Spacer()
                    Text(position.side.displayName)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(position.side == .long ? Color.green : Color.red)
                        .cornerRadius(4)
                }

                HStack {
                    Text("Entry: \(position.formattedEntryPrice)")
                    Spacer()
                    Text("Mark: \(position.formattedMarkPrice)")
                }
                .font(.caption)

                HStack {
                    Text("PnL: \(position.formattedPnL)")
                        .foregroundColor(position.isProfit ? .green : .red)
                    Spacer()
                    Text(position.formattedPnLPercent)
                        .foregroundColor(position.isProfit ? .green : .red)
                }
            }
        }
        .task {
            await loadPositions()
            subscribeToUpdates()
        }
    }

    private func loadPositions() async {
        do {
            positions = try await tradingService.fetchPositions()
        } catch {
            print("Error loading positions: \(error)")
        }
    }

    private func subscribeToUpdates() {
        tradingService.positionUpdates
            .receive(on: DispatchQueue.main)
            .sink { updatedPositions in
                positions = updatedPositions
            }
            .store(in: &cancellables)
    }
}
```

### Portfolio

```swift
struct PortfolioView: View {
    @State private var portfolio: Portfolio?
    private let portfolioService = MockServiceContainer.shared.portfolioService
    private var cancellables = Set<AnyCancellable>()

    var body: some View {
        if let portfolio = portfolio {
            VStack(spacing: 16) {
                HStack {
                    Text("Equity")
                    Spacer()
                    Text(portfolio.formattedEquity)
                        .font(.title2)
                }

                HStack {
                    Text("Total PnL")
                    Spacer()
                    Text(portfolio.formattedTotalPnL)
                        .foregroundColor(portfolio.isProfit ? .green : .red)
                }

                HStack {
                    Text("Return")
                    Spacer()
                    Text(portfolio.formattedTotalPnLPercent)
                        .foregroundColor(portfolio.isProfit ? .green : .red)
                }

                Divider()

                HStack {
                    Text("Available")
                    Spacer()
                    Text(portfolio.formattedAvailableBalance)
                }

                HStack {
                    Text("Margin Used")
                    Spacer()
                    Text(String(format: "%.1f%%", portfolio.marginUsagePercent))
                }
            }
            .padding()
        }
        .task {
            await loadPortfolio()
            subscribeToUpdates()
        }
    }

    private func loadPortfolio() async {
        do {
            portfolio = try await portfolioService.fetchPortfolio()
        } catch {
            print("Error loading portfolio: \(error)")
        }
    }

    private func subscribeToUpdates() {
        portfolioService.portfolioUpdates
            .receive(on: DispatchQueue.main)
            .sink { updatedPortfolio in
                portfolio = updatedPortfolio
            }
            .store(in: &cancellables)
    }
}
```

### Event Feed

```swift
struct EventFeedView: View {
    @State private var events: [TradingEvent] = []
    private let eventService = MockServiceContainer.shared.eventService
    private var cancellables = Set<AnyCancellable>()

    var body: some View {
        List(events) { event in
            HStack {
                VStack(alignment: .leading) {
                    Text(event.type.displayName)
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Text(event.message)
                        .font(.body)
                }
                Spacer()
                Text(event.formattedTime)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
        }
        .task {
            await loadEvents()
            subscribeToEvents()
        }
    }

    private func loadEvents() async {
        do {
            events = try await eventService.fetchRecentEvents(limit: 50)
        } catch {
            print("Error loading events: \(error)")
        }
    }

    private func subscribeToEvents() {
        eventService.eventStream
            .receive(on: DispatchQueue.main)
            .sink { newEvent in
                events.insert(newEvent, at: 0)
                if events.count > 50 {
                    events.removeLast()
                }
            }
            .store(in: &cancellables)
    }
}
```

### Place Order

```swift
struct PlaceOrderView: View {
    @State private var quantity: String = ""
    @State private var isProcessing = false
    @State private var error: String?

    let symbol: String
    private let tradingService = MockServiceContainer.shared.tradingService

    var body: some View {
        VStack {
            TextField("Quantity", text: $quantity)
                .keyboardType(.decimalPad)
                .textFieldStyle(.roundedBorder)

            if let error = error {
                Text(error)
                    .foregroundColor(.red)
                    .font(.caption)
            }

            Button("Place Market Order") {
                Task {
                    await placeOrder()
                }
            }
            .disabled(isProcessing || quantity.isEmpty)
        }
        .padding()
    }

    private func placeOrder() async {
        guard let qty = Double(quantity) else {
            error = "Invalid quantity"
            return
        }

        isProcessing = true
        error = nil

        do {
            let request = OrderRequest(
                symbol: symbol,
                side: .buy,
                type: .market,
                quantity: qty,
                price: nil
            )

            let order = try await tradingService.placeOrder(request: request)
            print("Order placed: \(order.id)")

            quantity = ""
        } catch {
            self.error = error.localizedDescription
        }

        isProcessing = false
    }
}
```

## Features

### Realistic Price Simulation

Prices update every 2 seconds with smooth random walk:
- BTC/ETH: 0.08% volatility per update
- BNB/SOL: 0.12% volatility per update
- Other pairs: 0.15% volatility per update

### Automatic Position Updates

Positions automatically update their mark prices and PnL based on market movements.

### Order Simulation

- Market orders fill immediately
- Limit orders have 30% chance to fill every 10 seconds
- Partial fills supported
- Orders create positions when filled

### Event Generation

New events generated every 2-5 seconds, including:
- Trading signals
- Order placements and fills
- Position open/close
- Risk alerts
- System information

### WebSocket Health

Simulated connection health with:
- Connection state tracking
- Heartbeat monitoring
- Event rate calculation
- Staleness detection

## Default Mock Data

### Markets (12 pairs)
- BTCUSDT: $67,234.50 (+2.5%)
- ETHUSDT: $3,456.78 (-1.2%)
- SOLUSDT: $142.33 (+5.8%)
- And 9 more...

### Positions (3 active)
- BTCUSDT LONG 0.05 @ $65,000 → PnL: +$111.73 (+3.44%)
- ETHUSDT SHORT 0.5 @ $3,500 → PnL: +$21.61 (+1.24%)
- SOLUSDT LONG 5.0 @ $138.50 → PnL: +$19.15 (+2.77%)

### Portfolio
- Equity: $345.67
- Initial Capital: $300.00
- Total PnL: +$45.67 (+15.22%)
- Available Balance: $198.43

### Orders (6 total)
- 2 active limit orders
- 1 partially filled
- 2 filled (market)
- 1 cancelled

## Extending the Mock System

### Adding New Markets

```swift
// In MockDataProvider.generateMarkets()
("NEWTOKUSDT", "NEWTOK", "USDT", 10.50, 8.2)
```

### Customizing Update Intervals

```swift
// In MockMarketService
private let updateInterval: TimeInterval = 1.0 // Faster updates
```

### Adding Custom Events

```swift
// In MockDataProvider.generateEvents()
(.customEvent, ["Custom message here"])
```

## Performance

- Memory efficient: ~2-3 MB for all mock data
- CPU usage: <1% on average
- No network calls
- Fully deterministic (except random walks)

## Testing

The mock system is perfect for:
- UI development without backend
- Offline demos
- Screenshot generation
- Unit testing view logic
- Integration testing

## Notes

- All prices use smooth random walk (not jumpy)
- PnL calculations match real trading formulas
- Events correlate with actual order/position changes
- No external dependencies required
- Thread-safe using MainActor where needed

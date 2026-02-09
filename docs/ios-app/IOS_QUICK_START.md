# HEAN iOS - Quick Start Guide

Quick reference for building the HEAN iOS application.

---

## Setup (5 minutes)

### 1. Create Xcode Project

```bash
# Open Xcode → Create new project → iOS App
# Name: HEAN-iOS
# Interface: SwiftUI
# Language: Swift
# Minimum: iOS 17.0
```

### 2. Add Dependencies

**Package.swift** (if using SPM for modular architecture):

```swift
// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "HEAN-iOS",
    platforms: [.iOS(.v17)],
    products: [
        .library(name: "HEANCore", targets: ["HEANCore"]),
        .library(name: "HEANDesignSystem", targets: ["HEANDesignSystem"]),
    ],
    dependencies: [
        // Add third-party dependencies here if needed
    ],
    targets: [
        .target(name: "HEANCore", dependencies: []),
        .target(name: "HEANDesignSystem", dependencies: []),
        .testTarget(name: "HEANCoreTests", dependencies: ["HEANCore"]),
    ]
)
```

### 3. Configure Build Settings

**Info.plist**:
```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <false/>
    <key>NSExceptionDomains</key>
    <dict>
        <key>localhost</key>
        <dict>
            <key>NSExceptionAllowsInsecureHTTPLoads</key>
            <true/>
        </dict>
    </dict>
</dict>
```

**.swiftlint.yml**:
```yaml
disabled_rules:
  - trailing_whitespace
opt_in_rules:
  - explicit_self
  - force_unwrapping
included:
  - HEAN-iOS
excluded:
  - Pods
  - DerivedData
line_length: 120
```

---

## Architecture Overview (2-minute read)

```
App Launch
    ↓
HEANApp.swift → Configure DIContainer
    ↓
    ├─ APIClient (HTTP)
    ├─ WebSocketClient (WS)
    └─ Services (Market, Order, Position, etc.)
    ↓
Features (Dashboard, Markets, Trade, etc.)
    ↓
    ├─ View (SwiftUI)
    ├─ ViewModel (@Observable)
    └─ Coordinator (Navigation)
```

**Key Concepts**:
- **MVVM**: Views observe ViewModels
- **Dependency Injection**: Protocol-based, testable
- **Actor Model**: Thread-safe shared state
- **Event-Driven**: WebSocket → Service → ViewModel → View

---

## Step-by-Step Implementation

### Step 1: Create App Entry Point (2 min)

**HEANApp.swift**:
```swift
import SwiftUI

@main
struct HEANApp: App {
    init() {
        setupDependencies()
    }

    var body: some Scene {
        WindowGroup {
            AppCoordinator()
        }
    }

    private func setupDependencies() {
        #if DEBUG
        if FeatureFlags.shared.useMockData {
            DIContainer.shared.configureMock(scenario: .happyPath)
        } else {
            configureProduction()
        }
        #else
        configureProduction()
        #endif
    }

    private func configureProduction() {
        let baseURL = URL(string: "http://localhost:8000")!
        let wsURL = URL(string: "ws://localhost:8000/ws")!

        DIContainer.shared.configureProduction(
            baseURL: baseURL,
            wsURL: wsURL
        )
    }
}
```

### Step 2: Implement Core Networking (10 min)

Copy from `IOS_CODE_EXAMPLES.md`:
1. `APIClient.swift` - HTTP client with retry
2. `APIError.swift` - Error types
3. `APIEndpoint.swift` - Endpoint definitions

**Quick test**:
```swift
let client = APIClient(baseURL: URL(string: "http://localhost:8000")!)
let health: HealthResponse = try await client.request(.health, method: .get)
print("API Status: \(health.status)")
```

### Step 3: Implement WebSocket (10 min)

Copy from `IOS_CODE_EXAMPLES.md`:
1. `WebSocketClient.swift` - WS client
2. `WebSocketState.swift` - State enum
3. `MessageRouter.swift` - Event routing

**Quick test**:
```swift
let wsClient = WebSocketClient()
try await wsClient.connect(to: URL(string: "ws://localhost:8000/ws")!)

for await message in wsClient.messageStream {
    print("Received: \(message)")
}
```

### Step 4: Create Data Models (5 min)

**Market.swift**:
```swift
struct Market: Identifiable, Codable, Equatable, Sendable {
    let id: String
    let symbol: String
    let baseAsset: String
    let quoteAsset: String
    let minOrderSize: Decimal
    let priceScale: Int

    enum CodingKeys: String, CodingKey {
        case id, symbol
        case baseAsset = "base_asset"
        case quoteAsset = "quote_asset"
        case minOrderSize = "min_order_size"
        case priceScale = "price_scale"
    }
}
```

**Position.swift**:
```swift
struct Position: Identifiable, Codable, Equatable, Sendable {
    let id: String
    let symbol: String
    let side: OrderSide
    let size: Decimal
    let entryPrice: Decimal
    let currentPrice: Decimal
    let unrealizedPnL: Decimal
    let leverage: Int
    let timestamp: Date

    var pnlPercentage: Decimal {
        guard entryPrice > 0 else { return 0 }
        return (unrealizedPnL / (entryPrice * size)) * 100
    }

    enum CodingKeys: String, CodingKey {
        case id, symbol, side, size, leverage, timestamp
        case entryPrice = "entry_price"
        case currentPrice = "current_price"
        case unrealizedPnL = "unrealized_pnl"
    }
}

enum OrderSide: String, Codable, Sendable {
    case buy = "Buy"
    case sell = "Sell"
}
```

### Step 5: Build First Feature - Dashboard (15 min)

1. **ViewModel** (see `DashboardViewModel.swift` in examples)
2. **View** (see `DashboardView.swift` in examples)
3. **Coordinator**:

```swift
@MainActor
struct AppCoordinator: View {
    @State private var selectedTab = 0

    var body: some View {
        TabView(selection: $selectedTab) {
            DashboardView()
                .tabItem {
                    Label("Dashboard", systemImage: "chart.bar.fill")
                }
                .tag(0)

            MarketsView()
                .tabItem {
                    Label("Markets", systemImage: "chart.line.uptrend.xyaxis")
                }
                .tag(1)

            TradeView()
                .tabItem {
                    Label("Trade", systemImage: "arrow.left.arrow.right")
                }
                .tag(2)

            ActivityView()
                .tabItem {
                    Label("Activity", systemImage: "bell.fill")
                }
                .tag(3)
        }
    }
}
```

### Step 6: Add Design System Components (10 min)

**GlassCard.swift**:
```swift
struct GlassCard<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .background(
                RoundedRectangle(cornerRadius: 16)
                    .fill(.ultraThinMaterial)
                    .shadow(color: .black.opacity(0.1), radius: 10, y: 5)
            )
    }
}
```

**ColorPalette.swift**:
```swift
extension Color {
    static let brandPrimary = Color(hex: "3B82F6")
    static let brandSecondary = Color(hex: "8B5CF6")

    static let successGreen = Color(hex: "10B981")
    static let errorRed = Color(hex: "EF4444")
    static let warningYellow = Color(hex: "F59E0B")

    init(hex: String) {
        let scanner = Scanner(string: hex)
        var rgbValue: UInt64 = 0
        scanner.scanHexInt64(&rgbValue)

        let r = Double((rgbValue & 0xFF0000) >> 16) / 255.0
        let g = Double((rgbValue & 0x00FF00) >> 8) / 255.0
        let b = Double(rgbValue & 0x0000FF) / 255.0

        self.init(red: r, green: g, blue: b)
    }
}
```

### Step 7: Implement Services (5 min each)

**MarketService.swift** - See `IOS_CODE_EXAMPLES.md`

**OrderService.swift**:
```swift
@MainActor
final class OrderService {
    private let dataProvider: DataProviderProtocol
    private let logger: Logger

    init(dataProvider: DataProviderProtocol) {
        self.dataProvider = dataProvider
        self.logger = Logger.shared
    }

    func placeOrder(
        symbol: String,
        side: OrderSide,
        orderType: OrderType,
        quantity: Decimal,
        price: Decimal? = nil,
        leverage: Int = 1
    ) async throws -> Order {
        let request = PlaceOrderRequest(
            symbol: symbol,
            side: side,
            orderType: orderType,
            quantity: quantity,
            price: price,
            leverage: leverage
        )

        logger.info("Placing order", context: [
            "symbol": symbol,
            "side": side.rawValue,
            "type": orderType.rawValue,
            "quantity": quantity.description
        ])

        let order = try await dataProvider.placeOrder(request: request)

        logger.info("Order placed", context: [
            "order_id": order.id,
            "status": order.status.rawValue
        ])

        return order
    }

    func cancelOrder(orderId: String) async throws {
        logger.info("Cancelling order", context: ["order_id": orderId])
        try await dataProvider.cancelOrder(orderId: orderId)
        logger.info("Order cancelled", context: ["order_id": orderId])
    }

    func fetchOrders(status: OrderStatus? = nil) async throws -> [Order] {
        let orders = try await dataProvider.fetchOrders(status: status)

        logger.info("Orders fetched", context: [
            "count": orders.count,
            "status": status?.rawValue ?? "all"
        ])

        return orders
    }

    func observeOrderUpdates() -> AsyncStream<OrderUpdateEvent> {
        AsyncStream { continuation in
            Task {
                for await event in dataProvider.eventStream {
                    if let orderEvent = event as? OrderUpdateEvent {
                        continuation.yield(orderEvent)
                    }
                }
                continuation.finish()
            }
        }
    }
}
```

---

## Mock Data for Development

**MockDataProvider.swift**:
```swift
@MainActor
final class MockDataProvider: DataProviderProtocol {
    let scenario: MockScenario
    private let eventContinuation: AsyncStream<WSEvent>.Continuation

    var eventStream: AsyncStream<WSEvent>

    init(scenario: MockScenario = .happyPath) {
        self.scenario = scenario

        var continuation: AsyncStream<WSEvent>.Continuation!
        self.eventStream = AsyncStream { continuation = $0 }
        self.eventContinuation = continuation

        startMockEventStream()
    }

    func fetchMarkets() async throws -> [Market] {
        try await Task.sleep(for: .milliseconds(500)) // Simulate network delay

        return [
            Market(
                id: "1",
                symbol: "BTCUSDT",
                baseAsset: "BTC",
                quoteAsset: "USDT",
                minOrderSize: 0.001,
                priceScale: 2
            ),
            Market(
                id: "2",
                symbol: "ETHUSDT",
                baseAsset: "ETH",
                quoteAsset: "USDT",
                minOrderSize: 0.01,
                priceScale: 2
            )
        ]
    }

    func fetchPositions() async throws -> [Position] {
        try await Task.sleep(for: .milliseconds(300))

        return [
            Position(
                id: UUID().uuidString,
                symbol: "BTCUSDT",
                side: .buy,
                size: 0.1,
                entryPrice: 50000,
                currentPrice: 51000,
                unrealizedPnL: 100,
                leverage: 5,
                timestamp: Date()
            )
        ]
    }

    func fetchPortfolioSummary() async throws -> Portfolio {
        try await Task.sleep(for: .milliseconds(400))

        return Portfolio(
            totalBalance: 10000,
            availableBalance: 8000,
            marginUsed: 2000,
            totalPnL: 500,
            dailyPnL: 50
        )
    }

    func placeOrder(request: PlaceOrderRequest) async throws -> Order {
        try await Task.sleep(for: .milliseconds(600))

        return Order(
            id: UUID().uuidString,
            symbol: request.symbol,
            side: request.side,
            orderType: request.orderType,
            quantity: request.quantity,
            price: request.price,
            status: .filled,
            filledQuantity: request.quantity,
            averagePrice: request.price ?? 0,
            timestamp: Date()
        )
    }

    // ... implement other methods

    private func startMockEventStream() {
        Task {
            while true {
                try? await Task.sleep(for: .seconds(2))

                // Send mock ticker event
                let tickerEvent = TickerEvent(
                    symbol: "BTCUSDT",
                    price: Decimal(Int.random(in: 50000...52000)),
                    volume24h: 1000000,
                    change24h: 2.5,
                    timestamp: Date()
                )

                eventContinuation.yield(tickerEvent)
            }
        }
    }
}

enum MockScenario {
    case happyPath
    case networkError
    case serverError
    case slowNetwork
}
```

---

## Testing

### Unit Test Example

```swift
@MainActor
final class MarketServiceTests: XCTestCase {
    private var sut: MarketService!
    private var mockProvider: MockDataProvider!
    private var mockCache: MockCacheManager!

    override func setUp() {
        super.setUp()
        mockProvider = MockDataProvider(scenario: .happyPath)
        mockCache = MockCacheManager()
        sut = MarketService(dataProvider: mockProvider, cache: mockCache)
    }

    func testFetchMarkets_Success() async throws {
        // When
        let markets = try await sut.fetchMarkets()

        // Then
        XCTAssertEqual(markets.count, 2)
        XCTAssertEqual(markets.first?.symbol, "BTCUSDT")
    }

    func testFetchMarkets_CachesResult() async throws {
        // Given
        _ = try await sut.fetchMarkets()

        // When - second call
        let markets = try await sut.fetchMarkets()

        // Then - should use cache
        XCTAssertNotNil(mockCache.getValue(forKey: "markets"))
    }
}
```

### SwiftUI Preview

```swift
#Preview("Dashboard - Loaded") {
    DashboardView()
        .environment(\.container, {
            let container = DIContainer()
            container.configureMock(scenario: .happyPath)
            return container
        }())
}

#Preview("Dashboard - Loading") {
    DashboardView()
        .environment(\.container, {
            let container = DIContainer()
            container.configureMock(scenario: .slowNetwork)
            return container
        }())
}

#Preview("Dashboard - Error") {
    DashboardView()
        .environment(\.container, {
            let container = DIContainer()
            container.configureMock(scenario: .networkError)
            return container
        }())
}
```

---

## Common Tasks

### Add New Feature

1. Create folder: `Features/NewFeature/`
2. Add files:
   - `NewFeatureView.swift`
   - `NewFeatureViewModel.swift`
   - `NewFeatureCoordinator.swift`
3. Register in AppCoordinator
4. Add tests

### Add New API Endpoint

1. Add case to `APIEndpoint.swift`:
```swift
case newEndpoint(param: String)

var path: String {
    switch self {
    case .newEndpoint(let param):
        return "/api/new/\(param)"
    }
}
```

2. Add request/response models
3. Add method to service
4. Update mock provider

### Add New WebSocket Event

1. Create event model:
```swift
struct NewEvent: WSEvent {
    let type = "new_event"
    let data: String
    let timestamp: Date

    func toDomainEvent() -> Event {
        // Convert to domain model
    }
}
```

2. Update MessageRouter
3. Handle in ViewModel

---

## Performance Tips

1. **Use AsyncImage for remote images**:
```swift
AsyncImage(url: URL(string: imageURL)) { image in
    image.resizable()
} placeholder: {
    ProgressView()
}
```

2. **Lazy load lists**:
```swift
LazyVStack {
    ForEach(items) { item in
        ItemRow(item: item)
    }
}
```

3. **Cache network responses**:
```swift
cache.set(key: cacheKey, value: data, ttl: 300)
```

4. **Debounce search**:
```swift
@Published var searchText = ""
    .debounce(for: .milliseconds(300), scheduler: DispatchQueue.main)
```

---

## Debugging

### Enable Network Logging

```swift
// In APIClient
#if DEBUG
URLProtocol.registerClass(LoggingURLProtocol.self)
#endif
```

### View Memory Graph

Xcode → Debug → Memory Graph → Find cycles

### Monitor Network Activity

Instruments → Network → Profile app

### Check WebSocket Connection

```swift
// In DebugMenuView
Button("Test WebSocket") {
    Task {
        let client = DIContainer.shared.webSocketClient
        print("WS State: \(client.state)")
    }
}
```

---

## Deployment Checklist

- [ ] Update version and build number
- [ ] Set production API URL
- [ ] Disable debug logs
- [ ] Test on real device
- [ ] Run all tests
- [ ] Check for memory leaks
- [ ] Validate accessibility
- [ ] Submit to TestFlight

---

## Resources

- **Architecture**: `/docs/IOS_ARCHITECTURE.md`
- **Code Examples**: `/docs/IOS_CODE_EXAMPLES.md`
- **API Docs**: `/docs/API_REFERENCE.md`
- **Design System**: `/docs/DESIGN_SYSTEM.md`

## Support

- GitHub Issues: `https://github.com/yourusername/HEAN/issues`
- Documentation: `https://docs.hean.trading`
- Email: `support@hean.trading`

# HEAN iOS Application Architecture

## Overview
Production-grade iOS trading application for HEAN system built with SwiftUI, following MVVM+Coordinator pattern with protocol-based dependency injection.

**Target:** iOS 17+, Swift 5.9+

---

## 1. Project Structure

```
HEAN-iOS/
├── HEAN-iOS/
│   ├── App/
│   │   ├── HEANApp.swift                          # App entry point
│   │   ├── AppCoordinator.swift                   # Root coordinator
│   │   └── AppEnvironment.swift                   # Environment configuration
│   │
│   ├── Core/
│   │   ├── Networking/
│   │   │   ├── APIClient.swift                    # Main HTTP client
│   │   │   ├── APIClientProtocol.swift            # Protocol definition
│   │   │   ├── APIEndpoint.swift                  # Endpoint definitions
│   │   │   ├── APIError.swift                     # Error types
│   │   │   ├── HTTPMethod.swift                   # HTTP method enum
│   │   │   ├── RequestBuilder.swift               # URLRequest builder
│   │   │   ├── ResponseHandler.swift              # Response parsing
│   │   │   ├── RetryPolicy.swift                  # Exponential backoff
│   │   │   ├── RequestInterceptor.swift           # Auth/logging interceptor
│   │   │   └── NetworkMonitor.swift               # Reachability monitoring
│   │   │
│   │   ├── WebSocket/
│   │   │   ├── WebSocketClient.swift              # Main WS client
│   │   │   ├── WebSocketClientProtocol.swift      # Protocol definition
│   │   │   ├── WebSocketMessage.swift             # Message types
│   │   │   ├── WebSocketState.swift               # Connection state enum
│   │   │   ├── WebSocketReconnector.swift         # Auto-reconnect logic
│   │   │   ├── WebSocketHeartbeat.swift           # Ping/pong handler
│   │   │   ├── MessageRouter.swift                # Route messages to handlers
│   │   │   └── WebSocketError.swift               # WS-specific errors
│   │   │
│   │   ├── DI/
│   │   │   ├── DIContainer.swift                  # Dependency container
│   │   │   ├── Injectable.swift                   # Injection protocol
│   │   │   ├── Resolver.swift                     # Dependency resolver
│   │   │   └── EnvironmentKeys.swift              # SwiftUI environment keys
│   │   │
│   │   ├── Storage/
│   │   │   ├── CacheManager.swift                 # Memory/disk cache
│   │   │   ├── KeychainManager.swift              # Secure storage
│   │   │   └── UserDefaultsManager.swift          # Preferences
│   │   │
│   │   ├── Logging/
│   │   │   ├── Logger.swift                       # Unified logger
│   │   │   ├── LogLevel.swift                     # Log level enum
│   │   │   ├── LogDestination.swift               # Console/file/remote
│   │   │   ├── CorrelationIDManager.swift         # Request correlation
│   │   │   └── PerformanceTracker.swift           # Performance metrics
│   │   │
│   │   └── Utils/
│   │       ├── DateFormatter+Extensions.swift     # ISO8601 formatters
│   │       ├── Decimal+Extensions.swift           # Price formatting
│   │       ├── Publisher+Extensions.swift         # Combine helpers
│   │       ├── Task+Extensions.swift              # Async helpers
│   │       └── Throttle.swift                     # Rate limiting
│   │
│   ├── DesignSystem/
│   │   ├── Colors/
│   │   │   ├── ColorPalette.swift                 # Brand colors
│   │   │   └── SemanticColors.swift               # Semantic (success/error)
│   │   │
│   │   ├── Typography/
│   │   │   ├── Typography.swift                   # Font styles
│   │   │   └── TextStyles.swift                   # Predefined styles
│   │   │
│   │   ├── Components/
│   │   │   ├── Buttons/
│   │   │   │   ├── PrimaryButton.swift
│   │   │   │   ├── SecondaryButton.swift
│   │   │   │   └── IconButton.swift
│   │   │   │
│   │   │   ├── Cards/
│   │   │   │   ├── GlassCard.swift                # Glassmorphism card
│   │   │   │   ├── StatCard.swift                 # Metric display
│   │   │   │   └── PositionCard.swift             # Position display
│   │   │   │
│   │   │   ├── Charts/
│   │   │   │   ├── LineChart.swift                # Price chart
│   │   │   │   ├── CandlestickChart.swift         # OHLCV chart
│   │   │   │   └── SparklineChart.swift           # Mini chart
│   │   │   │
│   │   │   ├── Indicators/
│   │   │   │   ├── LoadingSpinner.swift
│   │   │   │   ├── ProgressBar.swift
│   │   │   │   └── HealthIndicator.swift          # System health badge
│   │   │   │
│   │   │   └── Lists/
│   │   │       ├── OrderRow.swift
│   │   │       ├── PositionRow.swift
│   │   │       └── EventRow.swift
│   │   │
│   │   └── Motion/
│   │       ├── AnimationTokens.swift              # Duration/easing constants
│   │       ├── TransitionStyles.swift             # Reusable transitions
│   │       └── HapticFeedback.swift               # Haptic patterns
│   │
│   ├── Features/
│   │   ├── Dashboard/
│   │   │   ├── DashboardView.swift                # Main dashboard UI
│   │   │   ├── DashboardViewModel.swift           # Business logic
│   │   │   ├── DashboardCoordinator.swift         # Navigation
│   │   │   ├── Components/
│   │   │   │   ├── PortfolioSummaryView.swift
│   │   │   │   ├── QuickStatsView.swift
│   │   │   │   └── RecentActivityView.swift
│   │   │   └── Models/
│   │   │       └── DashboardState.swift
│   │   │
│   │   ├── Markets/
│   │   │   ├── MarketsView.swift                  # Market list
│   │   │   ├── MarketsViewModel.swift
│   │   │   ├── MarketsCoordinator.swift
│   │   │   ├── MarketDetailView.swift             # Single market detail
│   │   │   ├── MarketDetailViewModel.swift
│   │   │   ├── Components/
│   │   │   │   ├── MarketTickerView.swift
│   │   │   │   ├── OrderBookView.swift
│   │   │   │   └── RecentTradesView.swift
│   │   │   └── Models/
│   │   │       ├── MarketListState.swift
│   │   │       └── MarketDetailState.swift
│   │   │
│   │   ├── Trade/
│   │   │   ├── TradeView.swift                    # Trade execution UI
│   │   │   ├── TradeViewModel.swift
│   │   │   ├── TradeCoordinator.swift
│   │   │   ├── Components/
│   │   │   │   ├── OrderFormView.swift
│   │   │   │   ├── LeverageSliderView.swift
│   │   │   │   └── RiskCalculatorView.swift
│   │   │   └── Models/
│   │   │       └── TradeFormState.swift
│   │   │
│   │   ├── Orders/
│   │   │   ├── OrdersView.swift                   # Active/history orders
│   │   │   ├── OrdersViewModel.swift
│   │   │   ├── OrdersCoordinator.swift
│   │   │   ├── Components/
│   │   │   │   ├── OrderFilterView.swift
│   │   │   │   └── OrderDetailSheet.swift
│   │   │   └── Models/
│   │   │       └── OrdersListState.swift
│   │   │
│   │   ├── Positions/
│   │   │   ├── PositionsView.swift                # Open positions
│   │   │   ├── PositionsViewModel.swift
│   │   │   ├── PositionsCoordinator.swift
│   │   │   ├── Components/
│   │   │   │   ├── PositionDetailSheet.swift
│   │   │   │   └── ClosePositionView.swift
│   │   │   └── Models/
│   │   │       └── PositionsState.swift
│   │   │
│   │   ├── Activity/
│   │   │   ├── ActivityView.swift                 # Event feed
│   │   │   ├── ActivityViewModel.swift
│   │   │   ├── ActivityCoordinator.swift
│   │   │   ├── Components/
│   │   │   │   ├── EventFilterView.swift
│   │   │   │   └── EventTimelineView.swift
│   │   │   └── Models/
│   │   │       └── ActivityState.swift
│   │   │
│   │   └── Settings/
│   │       ├── SettingsView.swift                 # App settings
│   │       ├── SettingsViewModel.swift
│   │       ├── SettingsCoordinator.swift
│   │       ├── Components/
│   │       │   ├── APIConfigView.swift
│   │       │   ├── NotificationSettingsView.swift
│   │       │   └── DebugMenuView.swift
│   │       └── Models/
│   │           └── SettingsState.swift
│   │
│   ├── Models/
│   │   ├── Domain/
│   │   │   ├── Market.swift                       # Market model
│   │   │   ├── Ticker.swift                       # Price ticker
│   │   │   ├── Candle.swift                       # OHLCV candle
│   │   │   ├── Order.swift                        # Order model
│   │   │   ├── Position.swift                     # Position model
│   │   │   ├── Trade.swift                        # Trade execution
│   │   │   ├── Portfolio.swift                    # Portfolio summary
│   │   │   ├── Event.swift                        # System event
│   │   │   ├── Strategy.swift                     # Strategy metadata
│   │   │   └── HealthStatus.swift                 # System health
│   │   │
│   │   ├── API/
│   │   │   ├── Requests/
│   │   │   │   ├── PlaceOrderRequest.swift
│   │   │   │   ├── CancelOrderRequest.swift
│   │   │   │   └── UpdateStrategyRequest.swift
│   │   │   │
│   │   │   └── Responses/
│   │   │       ├── HealthResponse.swift
│   │   │       ├── MarketsResponse.swift
│   │   │       ├── PositionsResponse.swift
│   │   │       ├── OrdersResponse.swift
│   │   │       └── PlaceOrderResponse.swift
│   │   │
│   │   └── WebSocket/
│   │       ├── WSEvent.swift                      # Base WS event
│   │       ├── TickerEvent.swift                  # Ticker update
│   │       ├── CandleEvent.swift                  # Candle update
│   │       ├── OrderUpdateEvent.swift             # Order status change
│   │       ├── PositionUpdateEvent.swift          # Position change
│   │       ├── TelemetryEvent.swift               # System telemetry
│   │       └── EventEnvelope.swift                # Event wrapper
│   │
│   ├── Services/
│   │   ├── DataProvider/
│   │   │   ├── DataProviderProtocol.swift         # Main provider protocol
│   │   │   ├── RealDataProvider.swift             # Production backend
│   │   │   └── MockDataProvider.swift             # Mock for development
│   │   │
│   │   ├── MarketService.swift                    # Market data service
│   │   ├── OrderService.swift                     # Order management
│   │   ├── PositionService.swift                  # Position tracking
│   │   ├── PortfolioService.swift                 # Portfolio aggregation
│   │   ├── EventService.swift                     # Event stream service
│   │   ├── StrategyService.swift                  # Strategy control
│   │   ├── HealthService.swift                    # System health monitoring
│   │   └── NotificationService.swift              # Push notifications
│   │
│   └── Mock/
│       ├── MockAPIClient.swift                    # Mock HTTP client
│       ├── MockWebSocketClient.swift              # Mock WS client
│       ├── MockDataGenerator.swift                # Random test data
│       ├── Fixtures/
│       │   ├── MarketFixtures.swift
│       │   ├── OrderFixtures.swift
│       │   ├── PositionFixtures.swift
│       │   └── EventFixtures.swift
│       └── Scenarios/
│           ├── HappyPathScenario.swift            # Normal operation
│           ├── ErrorScenario.swift                # Error handling
│           └── EdgeCaseScenario.swift             # Edge cases
│
├── HEAN-iOSTests/
│   ├── Core/
│   │   ├── Networking/
│   │   │   ├── APIClientTests.swift
│   │   │   ├── RetryPolicyTests.swift
│   │   │   └── ResponseHandlerTests.swift
│   │   │
│   │   ├── WebSocket/
│   │   │   ├── WebSocketClientTests.swift
│   │   │   ├── ReconnectorTests.swift
│   │   │   └── MessageRouterTests.swift
│   │   │
│   │   └── DI/
│   │       └── DIContainerTests.swift
│   │
│   ├── Features/
│   │   ├── DashboardViewModelTests.swift
│   │   ├── MarketsViewModelTests.swift
│   │   ├── TradeViewModelTests.swift
│   │   └── OrdersViewModelTests.swift
│   │
│   ├── Services/
│   │   ├── MarketServiceTests.swift
│   │   ├── OrderServiceTests.swift
│   │   └── PortfolioServiceTests.swift
│   │
│   └── Mocks/
│       ├── MockDependencies.swift
│       └── TestHelpers.swift
│
├── HEAN-iOSUITests/
│   ├── DashboardFlowTests.swift
│   ├── OrderPlacementFlowTests.swift
│   └── SettingsFlowTests.swift
│
├── HEAN-iOS.xcodeproj
├── Package.swift                                   # Swift Package dependencies
├── .swiftlint.yml                                  # Linting rules
├── .gitignore
└── README.md
```

---

## 2. Key Protocol Definitions

### 2.1 DataProviderProtocol

```swift
/// Core protocol for data access - enables mock/real switching
@MainActor
protocol DataProviderProtocol: Sendable {
    // Market Data
    func fetchMarkets() async throws -> [Market]
    func fetchTicker(symbol: String) async throws -> Ticker
    func fetchCandles(symbol: String, interval: String, limit: Int) async throws -> [Candle]

    // Portfolio
    func fetchPositions() async throws -> [Position]
    func fetchOrders(status: OrderStatus?) async throws -> [Order]
    func fetchPortfolioSummary() async throws -> Portfolio

    // Trading
    func placeOrder(request: PlaceOrderRequest) async throws -> Order
    func cancelOrder(orderId: String) async throws -> Void

    // Strategies
    func fetchStrategies() async throws -> [Strategy]
    func updateStrategy(strategyId: String, enabled: Bool) async throws -> Strategy

    // Health
    func fetchHealth() async throws -> HealthStatus

    // WebSocket
    var eventStream: AsyncStream<WSEvent> { get }
    func connect() async throws
    func disconnect() async
}
```

### 2.2 APIClientProtocol

```swift
/// HTTP client protocol for REST API communication
protocol APIClientProtocol: Sendable {
    /// Execute a request and decode response
    func request<T: Decodable>(
        _ endpoint: APIEndpoint,
        method: HTTPMethod,
        parameters: [String: Any]?,
        headers: [String: String]?
    ) async throws -> T

    /// Execute request without response body
    func request(
        _ endpoint: APIEndpoint,
        method: HTTPMethod,
        parameters: [String: Any]?,
        headers: [String: String]?
    ) async throws

    /// Cancel all ongoing requests
    func cancelAll()
}
```

### 2.3 WebSocketClientProtocol

```swift
/// WebSocket client protocol for real-time event streaming
protocol WebSocketClientProtocol: Sendable {
    /// Current connection state
    var state: WebSocketState { get }

    /// Stream of incoming messages
    var messageStream: AsyncStream<WebSocketMessage> { get }

    /// Connect to WebSocket endpoint
    func connect(to url: URL) async throws

    /// Disconnect gracefully
    func disconnect() async

    /// Send message to server
    func send(_ message: WebSocketMessage) async throws

    /// Observable state changes
    var statePublisher: AnyPublisher<WebSocketState, Never> { get }
}
```

### 2.4 Injectable Protocol

```swift
/// Marks types that can be dependency-injected
protocol Injectable {
    associatedtype DependencyType
    static var dependency: DependencyType { get }
}
```

---

## 3. Dependency Injection Container

### 3.1 DIContainer Design

```swift
/// Centralized dependency container using property wrappers
@MainActor
final class DIContainer {
    static let shared = DIContainer()

    // MARK: - Core Dependencies
    private(set) var apiClient: APIClientProtocol!
    private(set) var webSocketClient: WebSocketClientProtocol!
    private(set) var dataProvider: DataProviderProtocol!
    private(set) var logger: Logger!

    // MARK: - Services
    private(set) var marketService: MarketService!
    private(set) var orderService: OrderService!
    private(set) var positionService: PositionService!
    private(set) var portfolioService: PortfolioService!
    private(set) var eventService: EventService!
    private(set) var strategyService: StrategyService!
    private(set) var healthService: HealthService!

    // MARK: - Storage
    private(set) var cacheManager: CacheManager!
    private(set) var keychainManager: KeychainManager!

    private init() {}

    /// Configure production dependencies
    func configureProduction(baseURL: URL, wsURL: URL) {
        logger = Logger(destinations: [.console, .file])

        apiClient = APIClient(
            baseURL: baseURL,
            retryPolicy: RetryPolicy.exponential(),
            interceptors: [
                AuthInterceptor(),
                LoggingInterceptor(logger: logger)
            ]
        )

        webSocketClient = WebSocketClient(
            heartbeatInterval: 30,
            reconnector: WebSocketReconnector.exponential()
        )

        dataProvider = RealDataProvider(
            apiClient: apiClient,
            webSocketClient: webSocketClient,
            wsURL: wsURL
        )

        configureServices()
    }

    /// Configure mock dependencies for testing/development
    func configureMock(scenario: MockScenario = .happyPath) {
        logger = Logger(destinations: [.console])

        apiClient = MockAPIClient(scenario: scenario)
        webSocketClient = MockWebSocketClient(scenario: scenario)
        dataProvider = MockDataProvider(scenario: scenario)

        configureServices()
    }

    private func configureServices() {
        cacheManager = CacheManager()
        keychainManager = KeychainManager()

        marketService = MarketService(
            dataProvider: dataProvider,
            cache: cacheManager
        )

        orderService = OrderService(
            dataProvider: dataProvider,
            logger: logger
        )

        positionService = PositionService(
            dataProvider: dataProvider,
            cache: cacheManager
        )

        portfolioService = PortfolioService(
            dataProvider: dataProvider,
            positionService: positionService
        )

        eventService = EventService(
            dataProvider: dataProvider,
            logger: logger
        )

        strategyService = StrategyService(
            dataProvider: dataProvider
        )

        healthService = HealthService(
            dataProvider: dataProvider
        )
    }

    /// Reset container (useful for testing)
    func reset() {
        Task { @MainActor in
            await dataProvider?.disconnect()
        }
        // Clear all references
    }
}
```

### 3.2 SwiftUI Environment Integration

```swift
/// Environment key for dependency injection
struct DIContainerKey: EnvironmentKey {
    static let defaultValue: DIContainer = .shared
}

extension EnvironmentValues {
    var container: DIContainer {
        get { self[DIContainerKey.self] }
        set { self[DIContainerKey.self] = newValue }
    }
}

/// Property wrapper for injecting dependencies
@propertyWrapper
struct Injected<T> {
    private let keyPath: KeyPath<DIContainer, T>

    init(_ keyPath: KeyPath<DIContainer, T>) {
        self.keyPath = keyPath
    }

    @MainActor
    var wrappedValue: T {
        DIContainer.shared[keyPath: keyPath]
    }
}

// Usage in ViewModels:
// @Injected(\.marketService) private var marketService
```

---

## 4. State Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        App Launch                                │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
                    ┌────────────────────────┐
                    │  DIContainer.shared    │
                    │  .configureProduction()│
                    └────────────┬───────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
        ┌────────────┐  ┌────────────┐  ┌────────────┐
        │ APIClient  │  │   WS Client│  │  Services  │
        └────────────┘  └────────────┘  └────────────┘
                 │               │               │
                 └───────────────┼───────────────┘
                                 ▼
                        ┌─────────────────┐
                        │  DataProvider   │
                        │  (Real or Mock) │
                        └────────┬────────┘
                                 │
                 ┌───────────────┼───────────────┐
                 ▼               ▼               ▼
        ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
        │   REST API   │ │  WebSocket   │ │   Cache      │
        │   Requests   │ │  Connection  │ │   Layer      │
        └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
               │                │                │
               ▼                ▼                ▼
        ┌─────────────────────────────────────────────────┐
        │               Service Layer                      │
        │  (MarketService, OrderService, etc.)            │
        └────────────────────┬────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  ViewModel   │    │  ViewModel   │    │  ViewModel   │
│  @Observable │    │  @Observable │    │  @Observable │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  SwiftUI     │    │  SwiftUI     │    │  SwiftUI     │
│  View        │◄───┤  Coordinator │───►│  View        │
└──────────────┘    └──────────────┘    └──────────────┘

Event Flow (WebSocket):
┌─────────────┐
│  WS Server  │
└──────┬──────┘
       │ Push event
       ▼
┌─────────────────┐
│ WebSocketClient │
│  .messageStream │
└────────┬────────┘
         │
         ▼
┌──────────────────┐
│  MessageRouter   │
│  Route by type   │
└────┬─────────────┘
     │
     ├──► TickerEvent ──────► MarketService ──► ViewModel ──► View Update
     │
     ├──► OrderUpdateEvent ─► OrderService ──► ViewModel ──► View Update
     │
     └──► PositionUpdateEvent ► PositionService ─► ViewModel ─► View Update

Error Flow:
┌─────────────┐
│   Error     │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ Error Mapper │  (APIError → UserFacingError)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Logger      │  (Correlation ID + Context)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  ViewModel   │  (Present to user via @Published)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Error Alert │
└──────────────┘
```

---

## 5. Error Handling Strategy

### 5.1 Error Type Hierarchy

```swift
/// Root error type for the application
protocol HEANError: LocalizedError, Sendable {
    var correlationID: UUID { get }
    var underlyingError: Error? { get }
    var severity: ErrorSeverity { get }
}

enum ErrorSeverity: String, Sendable {
    case low        // Informational, no action needed
    case medium     // Retry possible
    case high       // User action required
    case critical   // App unstable, restart needed
}

/// Network-specific errors
enum APIError: HEANError {
    case networkUnavailable
    case timeout
    case unauthorized
    case forbidden
    case notFound
    case serverError(statusCode: Int)
    case decodingFailed(underlyingError: Error)
    case invalidRequest
    case rateLimited(retryAfter: TimeInterval)

    var correlationID: UUID {
        // Generated at error creation
    }

    var severity: ErrorSeverity {
        switch self {
        case .networkUnavailable: return .medium
        case .timeout: return .medium
        case .unauthorized: return .high
        case .forbidden: return .high
        case .serverError: return .critical
        case .decodingFailed: return .high
        case .rateLimited: return .medium
        default: return .medium
        }
    }

    var errorDescription: String? {
        switch self {
        case .networkUnavailable:
            return "No internet connection. Please check your network."
        case .timeout:
            return "Request timed out. Please try again."
        case .unauthorized:
            return "Authentication failed. Please log in again."
        case .serverError(let code):
            return "Server error (\(code)). Please try again later."
        // ... other cases
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .networkUnavailable:
            return "Check your WiFi or cellular connection."
        case .timeout:
            return "Tap to retry."
        case .unauthorized:
            return "Go to Settings to update credentials."
        // ... other cases
        }
    }
}

/// WebSocket-specific errors
enum WebSocketError: HEANError {
    case connectionFailed(reason: String)
    case disconnectedUnexpectedly
    case messageDecodingFailed(underlyingError: Error)
    case sendFailed
    case reconnectLimitExceeded
}

/// Business logic errors
enum TradingError: HEANError {
    case insufficientBalance
    case invalidOrderSize
    case marketClosed
    case positionNotFound
    case riskLimitExceeded
}
```

### 5.2 Error Handling Flow

```swift
// In APIClient
func request<T: Decodable>(_ endpoint: APIEndpoint) async throws -> T {
    let correlationID = UUID()

    logger.debug("Request started", context: [
        "correlation_id": correlationID.uuidString,
        "endpoint": endpoint.path
    ])

    do {
        let urlRequest = try buildRequest(endpoint)
        let (data, response) = try await session.data(for: urlRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw APIError.invalidResponse
        }

        // Log response
        logger.debug("Response received", context: [
            "correlation_id": correlationID.uuidString,
            "status_code": httpResponse.statusCode
        ])

        // Handle status codes
        switch httpResponse.statusCode {
        case 200...299:
            do {
                let decoded = try JSONDecoder.iso8601.decode(T.self, from: data)
                return decoded
            } catch {
                logger.error("Decoding failed", context: [
                    "correlation_id": correlationID.uuidString,
                    "error": error.localizedDescription
                ])
                throw APIError.decodingFailed(underlyingError: error)
            }

        case 401:
            throw APIError.unauthorized

        case 403:
            throw APIError.forbidden

        case 429:
            let retryAfter = parseRetryAfter(from: httpResponse)
            throw APIError.rateLimited(retryAfter: retryAfter)

        case 500...599:
            throw APIError.serverError(statusCode: httpResponse.statusCode)

        default:
            throw APIError.invalidResponse
        }

    } catch let error as APIError {
        // Already an APIError, just rethrow
        throw error

    } catch {
        // Wrap system errors
        logger.error("Request failed", context: [
            "correlation_id": correlationID.uuidString,
            "error": error.localizedDescription
        ])

        if (error as NSError).domain == NSURLErrorDomain {
            switch (error as NSError).code {
            case NSURLErrorNotConnectedToInternet:
                throw APIError.networkUnavailable
            case NSURLErrorTimedOut:
                throw APIError.timeout
            default:
                throw APIError.networkError(underlyingError: error)
            }
        }

        throw error
    }
}
```

### 5.3 ViewModel Error Handling Pattern

```swift
@Observable
final class DashboardViewModel {
    @Injected(\.portfolioService) private var portfolioService
    @Injected(\.logger) private var logger

    var state: LoadingState<Portfolio> = .idle
    var presentedError: UserFacingError?

    enum LoadingState<T> {
        case idle
        case loading
        case loaded(T)
        case failed(HEANError)
    }

    @MainActor
    func loadPortfolio() async {
        state = .loading

        do {
            let portfolio = try await portfolioService.fetchPortfolioSummary()
            state = .loaded(portfolio)

        } catch let error as APIError {
            state = .failed(error)

            // Map to user-facing error
            presentedError = UserFacingError(from: error)

            // Log with context
            logger.error("Portfolio load failed", context: [
                "correlation_id": error.correlationID.uuidString,
                "severity": error.severity.rawValue
            ])

            // Attempt recovery if appropriate
            if case .medium = error.severity {
                scheduleRetry()
            }

        } catch {
            // Unexpected error
            let wrappedError = APIError.unknown(underlyingError: error)
            state = .failed(wrappedError)
            presentedError = UserFacingError(from: wrappedError)

            logger.error("Unexpected error", context: [
                "error": error.localizedDescription
            ])
        }
    }

    private func scheduleRetry() {
        Task {
            try? await Task.sleep(for: .seconds(3))
            await loadPortfolio()
        }
    }
}

/// User-facing error with actionable information
struct UserFacingError: Identifiable {
    let id = UUID()
    let title: String
    let message: String
    let severity: ErrorSeverity
    let actions: [ErrorAction]

    init(from error: HEANError) {
        self.title = error.errorDescription ?? "Error"
        self.message = error.recoverySuggestion ?? "Please try again."
        self.severity = error.severity

        // Generate contextual actions
        self.actions = Self.actions(for: error)
    }

    private static func actions(for error: HEANError) -> [ErrorAction] {
        switch error {
        case let apiError as APIError:
            switch apiError {
            case .networkUnavailable:
                return [.retry, .openSettings]
            case .unauthorized:
                return [.reauth]
            case .serverError:
                return [.retry, .contactSupport]
            default:
                return [.retry]
            }
        default:
            return [.dismiss]
        }
    }
}

enum ErrorAction {
    case retry
    case dismiss
    case openSettings
    case reauth
    case contactSupport

    var title: String {
        switch self {
        case .retry: return "Retry"
        case .dismiss: return "OK"
        case .openSettings: return "Settings"
        case .reauth: return "Log In"
        case .contactSupport: return "Support"
        }
    }
}
```

### 5.4 View Error Presentation

```swift
struct DashboardView: View {
    @State private var viewModel = DashboardViewModel()

    var body: some View {
        content
            .alert(item: $viewModel.presentedError) { error in
                Alert(
                    title: Text(error.title),
                    message: Text(error.message),
                    primaryButton: .default(Text(error.actions.first?.title ?? "OK")) {
                        handleErrorAction(error.actions.first)
                    },
                    secondaryButton: .cancel()
                )
            }
    }

    private func handleErrorAction(_ action: ErrorAction?) {
        guard let action = action else { return }

        switch action {
        case .retry:
            Task { await viewModel.loadPortfolio() }
        case .openSettings:
            openAppSettings()
        case .reauth:
            // Navigate to login
            break
        case .contactSupport:
            openSupportURL()
        case .dismiss:
            break
        }
    }
}
```

---

## 6. Thread Safety & Actor Model

### 6.1 Actor-Based Services

```swift
/// Thread-safe market data cache using Actor
actor MarketDataCache {
    private var tickers: [String: Ticker] = [:]
    private var candles: [String: [Candle]] = [:]
    private let expirationInterval: TimeInterval = 60

    func cacheTicker(_ ticker: Ticker) {
        tickers[ticker.symbol] = ticker
    }

    func getTicker(symbol: String) -> Ticker? {
        return tickers[symbol]
    }

    func cacheCandles(_ candles: [Candle], symbol: String) {
        self.candles[symbol] = candles
    }

    func getCandles(symbol: String) -> [Candle]? {
        return candles[symbol]
    }

    func clearExpired() {
        let now = Date()
        tickers = tickers.filter { _, ticker in
            now.timeIntervalSince(ticker.timestamp) < expirationInterval
        }
    }
}
```

### 6.2 MainActor for ViewModels

```swift
/// All ViewModels run on MainActor for UI safety
@MainActor
@Observable
final class MarketsViewModel {
    private let marketService: MarketService
    private let logger: Logger

    var markets: [Market] = []
    var selectedMarket: Market?
    var isLoading = false
    var error: UserFacingError?

    // Async operations automatically isolated to MainActor
    func loadMarkets() async {
        isLoading = true
        defer { isLoading = false }

        do {
            markets = try await marketService.fetchMarkets()
        } catch {
            self.error = UserFacingError(from: error as! HEANError)
        }
    }
}
```

---

## 7. Performance & Caching Strategy

### 7.1 Multi-Level Caching

```
┌──────────────────────────────────────────┐
│          Memory Cache (NSCache)          │  ← Fast, volatile
│  - Tickers (60s TTL)                     │
│  - Recent candles (5min TTL)             │
│  - Position snapshots (10s TTL)          │
└────────────────┬─────────────────────────┘
                 │ Cache miss
                 ▼
┌──────────────────────────────────────────┐
│        Disk Cache (FileManager)          │  ← Persistent
│  - Historical candles (1 day)            │
│  - Market metadata (1 hour)              │
└────────────────┬─────────────────────────┘
                 │ Cache miss
                 ▼
┌──────────────────────────────────────────┐
│             Network Request              │  ← Slowest
└──────────────────────────────────────────┘
```

### 7.2 Request Deduplication

```swift
actor RequestDeduplicator {
    private var inflightRequests: [String: Task<Any, Error>] = [:]

    func deduplicate<T>(
        key: String,
        operation: @Sendable () async throws -> T
    ) async throws -> T {
        // Check if already in flight
        if let existingTask = inflightRequests[key] {
            return try await existingTask.value as! T
        }

        // Start new task
        let task = Task<Any, Error> {
            defer {
                Task { await removeTask(key: key) }
            }
            return try await operation()
        }

        inflightRequests[key] = task
        return try await task.value as! T
    }

    private func removeTask(key: String) {
        inflightRequests.removeValue(forKey: key)
    }
}
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

```swift
@MainActor
final class MarketServiceTests: XCTestCase {
    private var sut: MarketService!
    private var mockDataProvider: MockDataProvider!
    private var mockCache: MockCacheManager!

    override func setUp() {
        super.setUp()
        mockDataProvider = MockDataProvider(scenario: .happyPath)
        mockCache = MockCacheManager()
        sut = MarketService(
            dataProvider: mockDataProvider,
            cache: mockCache
        )
    }

    func testFetchMarkets_Success() async throws {
        // When
        let markets = try await sut.fetchMarkets()

        // Then
        XCTAssertFalse(markets.isEmpty)
        XCTAssertEqual(mockDataProvider.fetchMarketsCallCount, 1)
    }

    func testFetchMarkets_NetworkError_ReturnsError() async {
        // Given
        mockDataProvider.scenario = .networkError

        // When/Then
        do {
            _ = try await sut.fetchMarkets()
            XCTFail("Expected error")
        } catch let error as APIError {
            XCTAssertEqual(error, .networkUnavailable)
        }
    }

    func testFetchMarkets_UsesCache_OnSubsequentCall() async throws {
        // Given - first call populates cache
        _ = try await sut.fetchMarkets()

        // When - second call
        _ = try await sut.fetchMarkets()

        // Then - only one network call made
        XCTAssertEqual(mockDataProvider.fetchMarketsCallCount, 1)
        XCTAssertEqual(mockCache.getCallCount, 1)
    }
}
```

### 8.2 Integration Tests

```swift
@MainActor
final class OrderPlacementIntegrationTests: XCTestCase {
    private var container: DIContainer!
    private var viewModel: TradeViewModel!

    override func setUp() async throws {
        container = DIContainer()
        container.configureMock(scenario: .happyPath)
        viewModel = TradeViewModel()
    }

    func testPlaceOrder_EndToEnd() async throws {
        // Given
        viewModel.symbol = "BTCUSDT"
        viewModel.side = .buy
        viewModel.quantity = 0.001
        viewModel.orderType = .market

        // When
        await viewModel.placeOrder()

        // Then
        XCTAssertEqual(viewModel.orderState, .success)
        XCTAssertNotNil(viewModel.placedOrder)
        XCTAssertEqual(viewModel.placedOrder?.symbol, "BTCUSDT")
    }
}
```

---

## 9. API Endpoint Mapping

```swift
enum APIEndpoint {
    case health
    case markets
    case ticker(symbol: String)
    case candles(symbol: String, interval: String, limit: Int)
    case positions
    case orders(status: OrderStatus?)
    case placeOrder
    case cancelOrder(orderId: String)
    case portfolio
    case strategies
    case updateStrategy(strategyId: String)

    var path: String {
        switch self {
        case .health: return "/health"
        case .markets: return "/markets"
        case .ticker(let symbol): return "/markets/\(symbol)/ticker"
        case .candles(let symbol, _, _): return "/markets/\(symbol)/candles"
        case .positions: return "/positions"
        case .orders: return "/orders"
        case .placeOrder: return "/order/place"
        case .cancelOrder(let id): return "/order/cancel/\(id)"
        case .portfolio: return "/portfolio"
        case .strategies: return "/strategies"
        case .updateStrategy(let id): return "/strategies/\(id)"
        }
    }

    var queryItems: [URLQueryItem]? {
        switch self {
        case .candles(_, let interval, let limit):
            return [
                URLQueryItem(name: "interval", value: interval),
                URLQueryItem(name: "limit", value: "\(limit)")
            ]
        case .orders(let status):
            if let status = status {
                return [URLQueryItem(name: "status", value: status.rawValue)]
            }
            return nil
        default:
            return nil
        }
    }
}
```

---

## 10. WebSocket Event Routing

```swift
actor MessageRouter {
    typealias Handler = @Sendable (WSEvent) async -> Void

    private var handlers: [String: [Handler]] = [:]

    func register(eventType: String, handler: @escaping Handler) {
        handlers[eventType, default: []].append(handler)
    }

    func route(_ message: WebSocketMessage) async {
        do {
            let envelope = try JSONDecoder().decode(EventEnvelope.self, from: message.data)

            // Parse specific event type
            let event: WSEvent
            switch envelope.type {
            case "ticker":
                event = try JSONDecoder().decode(TickerEvent.self, from: message.data)
            case "candle":
                event = try JSONDecoder().decode(CandleEvent.self, from: message.data)
            case "order_update":
                event = try JSONDecoder().decode(OrderUpdateEvent.self, from: message.data)
            case "position_update":
                event = try JSONDecoder().decode(PositionUpdateEvent.self, from: message.data)
            case "telemetry":
                event = try JSONDecoder().decode(TelemetryEvent.self, from: message.data)
            default:
                return
            }

            // Dispatch to registered handlers
            if let eventHandlers = handlers[envelope.type] {
                await withTaskGroup(of: Void.self) { group in
                    for handler in eventHandlers {
                        group.addTask {
                            await handler(event)
                        }
                    }
                }
            }

        } catch {
            // Log decoding error
        }
    }
}
```

---

## 11. Feature Flags & Environment Switching

```swift
enum AppEnvironment {
    case production
    case staging
    case mock

    var baseURL: URL {
        switch self {
        case .production:
            return URL(string: "https://api.hean.trading")!
        case .staging:
            return URL(string: "https://staging-api.hean.trading")!
        case .mock:
            return URL(string: "http://localhost:8000")!
        }
    }

    var wsURL: URL {
        switch self {
        case .production:
            return URL(string: "wss://api.hean.trading/ws")!
        case .staging:
            return URL(string: "wss://staging-api.hean.trading/ws")!
        case .mock:
            return URL(string: "ws://localhost:8000/ws")!
        }
    }
}

final class FeatureFlags {
    static let shared = FeatureFlags()

    @AppStorage("useMockData") var useMockData: Bool = false
    @AppStorage("enableDebugMenu") var enableDebugMenu: Bool = false
    @AppStorage("logLevel") var logLevel: String = "info"

    var environment: AppEnvironment {
        #if DEBUG
        return useMockData ? .mock : .staging
        #else
        return .production
        #endif
    }
}
```

---

## 12. Observability & Metrics

```swift
/// Performance tracking for critical paths
final class PerformanceTracker {
    static let shared = PerformanceTracker()

    func track<T>(_ operation: String, _ block: () async throws -> T) async rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        defer {
            let duration = CFAbsoluteTimeGetCurrent() - start
            Logger.shared.info("Performance", context: [
                "operation": operation,
                "duration_ms": Int(duration * 1000)
            ])
        }
        return try await block()
    }
}

// Usage:
let markets = await PerformanceTracker.shared.track("fetch_markets") {
    try await marketService.fetchMarkets()
}
```

---

## Summary

This architecture provides:

1. **Production-Ready Foundation**
   - Protocol-based dependency injection
   - Thread-safe actor model
   - Comprehensive error handling
   - Multi-level caching

2. **Testability**
   - Mock/real data provider switching
   - Protocol-based abstractions
   - Dependency injection for easy testing

3. **Maintainability**
   - Clear separation of concerns (MVVM+Coordinator)
   - Feature-based module organization
   - Consistent patterns across features

4. **Performance**
   - Request deduplication
   - Intelligent caching
   - Async/await throughout
   - Actor isolation for shared state

5. **Observability**
   - Correlation IDs for request tracing
   - Structured logging
   - Performance metrics
   - Error severity classification

The architecture is designed for scale, supporting multiple teams working on different features without conflicts.

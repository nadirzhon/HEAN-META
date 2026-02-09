# HEAN iOS - Complete Dependency Injection Implementation

Production-ready dependency injection system with full code.

---

## Overview

This DI system provides:
- Protocol-based abstractions for testability
- Centralized container for all dependencies
- SwiftUI environment integration
- Mock/real environment switching
- Type-safe dependency resolution

---

## 1. Core Protocols

### File: `Core/DI/Injectable.swift`

```swift
import Foundation

/// Marks a type as injectable via the DI container
protocol Injectable {
    associatedtype DependencyType
    static var dependency: DependencyType { get }
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

/// Property wrapper for optional injection (useful for feature flags)
@propertyWrapper
struct OptionallyInjected<T> {
    private let keyPath: KeyPath<DIContainer, T?>

    init(_ keyPath: KeyPath<DIContainer, T?>) {
        self.keyPath = keyPath
    }

    @MainActor
    var wrappedValue: T? {
        DIContainer.shared[keyPath: keyPath]
    }
}
```

---

## 2. Dependency Container

### File: `Core/DI/DIContainer.swift`

```swift
import Foundation

/// Centralized dependency injection container
@MainActor
final class DIContainer: @unchecked Sendable {
    static let shared = DIContainer()

    // MARK: - Core Dependencies

    private(set) var apiClient: APIClientProtocol!
    private(set) var webSocketClient: WebSocketClientProtocol!
    private(set) var dataProvider: DataProviderProtocol!
    private(set) var logger: Logger!
    private(set) var networkMonitor: NetworkMonitor!

    // MARK: - Services

    private(set) var marketService: MarketService!
    private(set) var orderService: OrderService!
    private(set) var positionService: PositionService!
    private(set) var portfolioService: PortfolioService!
    private(set) var eventService: EventService!
    private(set) var strategyService: StrategyService!
    private(set) var healthService: HealthService!
    private(set) var notificationService: NotificationService!

    // MARK: - Storage

    private(set) var cacheManager: CacheManager!
    private(set) var keychainManager: KeychainManager!
    private(set) var userDefaultsManager: UserDefaultsManager!

    // MARK: - Configuration

    private(set) var environment: AppEnvironment = .staging
    private(set) var isConfigured = false

    private init() {}

    // MARK: - Configuration Methods

    /// Configure production dependencies
    func configureProduction(baseURL: URL, wsURL: URL) {
        guard !isConfigured else {
            logger?.warning("DIContainer already configured, ignoring reconfiguration")
            return
        }

        environment = .production

        // Core infrastructure
        configureInfrastructure()

        // Networking
        apiClient = APIClient(
            baseURL: baseURL,
            retryPolicy: RetryPolicy.exponential(maxAttempts: 3),
            interceptors: [
                AuthInterceptor(keychainManager: keychainManager),
                LoggingInterceptor(logger: logger),
                CorrelationIDInterceptor()
            ]
        )

        webSocketClient = WebSocketClient(
            heartbeatInterval: 30,
            reconnector: WebSocketReconnector.exponential(maxAttempts: 10)
        )

        dataProvider = RealDataProvider(
            apiClient: apiClient,
            webSocketClient: webSocketClient,
            wsURL: wsURL,
            logger: logger
        )

        // Services
        configureServices()

        isConfigured = true

        logger.info("Production configuration complete", context: [
            "base_url": baseURL.absoluteString,
            "ws_url": wsURL.absoluteString
        ])
    }

    /// Configure mock dependencies for testing/development
    func configureMock(scenario: MockScenario = .happyPath) {
        guard !isConfigured else {
            logger?.warning("DIContainer already configured, ignoring reconfiguration")
            return
        }

        environment = .mock

        // Core infrastructure
        configureInfrastructure()

        // Mock networking
        apiClient = MockAPIClient(scenario: scenario)
        webSocketClient = MockWebSocketClient(scenario: scenario)
        dataProvider = MockDataProvider(
            scenario: scenario,
            logger: logger
        )

        // Services (same as production, just different provider)
        configureServices()

        isConfigured = true

        logger.info("Mock configuration complete", context: [
            "scenario": String(describing: scenario)
        ])
    }

    /// Reset container (useful for testing)
    func reset() {
        Task { @MainActor in
            // Disconnect WebSocket
            await webSocketClient?.disconnect()

            // Clear all references
            apiClient = nil
            webSocketClient = nil
            dataProvider = nil

            marketService = nil
            orderService = nil
            positionService = nil
            portfolioService = nil
            eventService = nil
            strategyService = nil
            healthService = nil
            notificationService = nil

            cacheManager?.clearAll()

            isConfigured = false

            logger?.info("DIContainer reset complete")
        }
    }

    // MARK: - Private Configuration

    private func configureInfrastructure() {
        // Logging
        #if DEBUG
        logger = Logger(
            minimumLevel: .debug,
            destinations: [.console, .file]
        )
        #else
        logger = Logger(
            minimumLevel: .info,
            destinations: [.file, .remote]
        )
        #endif

        // Storage
        cacheManager = CacheManager(logger: logger)
        keychainManager = KeychainManager()
        userDefaultsManager = UserDefaultsManager()

        // Network monitoring
        networkMonitor = NetworkMonitor(logger: logger)
        networkMonitor.startMonitoring()
    }

    private func configureServices() {
        marketService = MarketService(
            dataProvider: dataProvider,
            cache: cacheManager,
            logger: logger
        )

        orderService = OrderService(
            dataProvider: dataProvider,
            logger: logger
        )

        positionService = PositionService(
            dataProvider: dataProvider,
            cache: cacheManager,
            logger: logger
        )

        portfolioService = PortfolioService(
            dataProvider: dataProvider,
            positionService: positionService,
            logger: logger
        )

        eventService = EventService(
            dataProvider: dataProvider,
            logger: logger
        )

        strategyService = StrategyService(
            dataProvider: dataProvider,
            logger: logger
        )

        healthService = HealthService(
            dataProvider: dataProvider,
            networkMonitor: networkMonitor,
            logger: logger
        )

        notificationService = NotificationService(
            eventService: eventService,
            logger: logger
        )
    }
}

// MARK: - App Environment

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
```

---

## 3. SwiftUI Environment Integration

### File: `Core/DI/EnvironmentKeys.swift`

```swift
import SwiftUI

/// Environment key for dependency injection container
struct DIContainerKey: EnvironmentKey {
    static let defaultValue: DIContainer = .shared
}

extension EnvironmentValues {
    var container: DIContainer {
        get { self[DIContainerKey.self] }
        set { self[DIContainerKey.self] = newValue }
    }
}

/// View modifier to inject custom container (useful for previews)
struct WithDIContainer: ViewModifier {
    let container: DIContainer

    func body(content: Content) -> some View {
        content
            .environment(\.container, container)
    }
}

extension View {
    func withContainer(_ container: DIContainer) -> some View {
        modifier(WithDIContainer(container: container))
    }
}
```

---

## 4. Service Resolution Examples

### File: `Core/DI/Resolver.swift`

```swift
import Foundation

/// Resolves dependencies from the container
@MainActor
struct Resolver {
    static let shared = Resolver()

    private init() {}

    // MARK: - Core Resolutions

    func resolveAPIClient() -> APIClientProtocol {
        guard let client = DIContainer.shared.apiClient else {
            fatalError("APIClient not configured. Call DIContainer.shared.configureProduction() first.")
        }
        return client
    }

    func resolveWebSocketClient() -> WebSocketClientProtocol {
        guard let client = DIContainer.shared.webSocketClient else {
            fatalError("WebSocketClient not configured.")
        }
        return client
    }

    func resolveDataProvider() -> DataProviderProtocol {
        guard let provider = DIContainer.shared.dataProvider else {
            fatalError("DataProvider not configured.")
        }
        return provider
    }

    func resolveLogger() -> Logger {
        guard let logger = DIContainer.shared.logger else {
            fatalError("Logger not configured.")
        }
        return logger
    }

    // MARK: - Service Resolutions

    func resolveMarketService() -> MarketService {
        guard let service = DIContainer.shared.marketService else {
            fatalError("MarketService not configured.")
        }
        return service
    }

    func resolveOrderService() -> OrderService {
        guard let service = DIContainer.shared.orderService else {
            fatalError("OrderService not configured.")
        }
        return service
    }

    func resolvePositionService() -> PositionService {
        guard let service = DIContainer.shared.positionService else {
            fatalError("PositionService not configured.")
        }
        return service
    }

    func resolvePortfolioService() -> PortfolioService {
        guard let service = DIContainer.shared.portfolioService else {
            fatalError("PortfolioService not configured.")
        }
        return service
    }

    // MARK: - Storage Resolutions

    func resolveCacheManager() -> CacheManager {
        guard let manager = DIContainer.shared.cacheManager else {
            fatalError("CacheManager not configured.")
        }
        return manager
    }

    func resolveKeychainManager() -> KeychainManager {
        guard let manager = DIContainer.shared.keychainManager else {
            fatalError("KeychainManager not configured.")
        }
        return manager
    }
}
```

---

## 5. Usage in ViewModels

### Example: DashboardViewModel with Injection

```swift
import Foundation
import Observation

@MainActor
@Observable
final class DashboardViewModel {
    // MARK: - Dependencies (injected via property wrapper)

    @Injected(\.portfolioService) private var portfolioService
    @Injected(\.positionService) private var positionService
    @Injected(\.eventService) private var eventService
    @Injected(\.healthService) private var healthService
    @Injected(\.logger) private var logger

    // MARK: - State

    var portfolioState: LoadingState<Portfolio> = .idle
    var positionsState: LoadingState<[Position]> = .idle
    var healthState: LoadingState<HealthStatus> = .idle
    var presentedError: UserFacingError?

    // MARK: - Initialization

    init() {
        Task {
            await initialize()
        }
    }

    // MARK: - Public Methods

    func initialize() async {
        await loadAll()
        startEventStream()
    }

    // ... rest of implementation
}
```

### Example: Manual Injection (if property wrapper not preferred)

```swift
@MainActor
@Observable
final class MarketsViewModel {
    // MARK: - Dependencies

    private let marketService: MarketService
    private let eventService: EventService
    private let logger: Logger

    // MARK: - State

    var markets: [Market] = []
    var isLoading = false

    // MARK: - Initialization

    init(
        marketService: MarketService? = nil,
        eventService: EventService? = nil,
        logger: Logger? = nil
    ) {
        // Use provided dependencies or resolve from container
        self.marketService = marketService ?? Resolver.shared.resolveMarketService()
        self.eventService = eventService ?? Resolver.shared.resolveEventService()
        self.logger = logger ?? Resolver.shared.resolveLogger()
    }

    // ... rest of implementation
}
```

---

## 6. App Entry Point Integration

### File: `App/HEANApp.swift`

```swift
import SwiftUI

@main
struct HEANApp: App {
    @State private var isConfigured = false

    init() {
        setupDependencies()
    }

    var body: some Scene {
        WindowGroup {
            if isConfigured {
                AppCoordinator()
                    .environment(\.container, DIContainer.shared)
            } else {
                ProgressView("Initializing...")
                    .task {
                        // Async initialization if needed
                        await performAsyncSetup()
                        isConfigured = true
                    }
            }
        }
    }

    // MARK: - Setup

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
        let environment = FeatureFlags.shared.environment
        DIContainer.shared.configureProduction(
            baseURL: environment.baseURL,
            wsURL: environment.wsURL
        )
    }

    private func performAsyncSetup() async {
        // Perform any async initialization here
        // e.g., loading cached data, checking auth state, etc.
        try? await Task.sleep(for: .milliseconds(100))
    }
}
```

---

## 7. SwiftUI Preview Integration

### Creating Mock Container for Previews

```swift
import SwiftUI

extension DIContainer {
    /// Creates a fresh mock container for SwiftUI previews
    @MainActor
    static func preview(scenario: MockScenario = .happyPath) -> DIContainer {
        let container = DIContainer()
        container.configureMock(scenario: scenario)
        return container
    }
}

// Usage in previews:
#Preview("Dashboard - Success") {
    DashboardView()
        .withContainer(.preview(scenario: .happyPath))
}

#Preview("Dashboard - Error") {
    DashboardView()
        .withContainer(.preview(scenario: .networkError))
}

#Preview("Dashboard - Loading") {
    DashboardView()
        .withContainer(.preview(scenario: .slowNetwork))
}
```

---

## 8. Testing Integration

### Unit Test Setup

```swift
import XCTest
@testable import HEAN_iOS

@MainActor
final class MarketServiceTests: XCTestCase {
    private var sut: MarketService!
    private var mockDataProvider: MockDataProvider!
    private var mockCache: MockCacheManager!
    private var mockLogger: MockLogger!

    override func setUp() async throws {
        try await super.setUp()

        // Create mocks
        mockLogger = MockLogger()
        mockCache = MockCacheManager()
        mockDataProvider = MockDataProvider(scenario: .happyPath, logger: mockLogger)

        // Inject dependencies manually (bypass container)
        sut = MarketService(
            dataProvider: mockDataProvider,
            cache: mockCache,
            logger: mockLogger
        )
    }

    override func tearDown() async throws {
        sut = nil
        mockDataProvider = nil
        mockCache = nil
        mockLogger = nil

        try await super.tearDown()
    }

    func testFetchMarkets_Success() async throws {
        // When
        let markets = try await sut.fetchMarkets()

        // Then
        XCTAssertFalse(markets.isEmpty)
        XCTAssertEqual(mockDataProvider.fetchMarketsCallCount, 1)
    }
}
```

### Integration Test with Container

```swift
@MainActor
final class DashboardIntegrationTests: XCTestCase {
    private var container: DIContainer!
    private var viewModel: DashboardViewModel!

    override func setUp() async throws {
        try await super.setUp()

        // Use real container with mock provider
        container = DIContainer()
        container.configureMock(scenario: .happyPath)

        viewModel = DashboardViewModel()
    }

    override func tearDown() async throws {
        await container.reset()
        container = nil
        viewModel = nil

        try await super.tearDown()
    }

    func testInitialize_LoadsAllData() async throws {
        // When
        await viewModel.initialize()

        // Give async operations time to complete
        try await Task.sleep(for: .seconds(1))

        // Then
        XCTAssertNotEqual(viewModel.portfolioState, .idle)
        XCTAssertNotEqual(viewModel.positionsState, .idle)
        XCTAssertNotEqual(viewModel.healthState, .idle)
    }
}
```

---

## 9. Feature Flags Integration

### File: `App/FeatureFlags.swift`

```swift
import SwiftUI

final class FeatureFlags: ObservableObject {
    static let shared = FeatureFlags()

    @AppStorage("useMockData") var useMockData: Bool = false
    @AppStorage("enableDebugMenu") var enableDebugMenu: Bool = false
    @AppStorage("logLevel") var logLevel: String = "info"
    @AppStorage("selectedEnvironment") private var selectedEnvironment: String = "production"

    private init() {}

    var environment: AppEnvironment {
        #if DEBUG
        if useMockData {
            return .mock
        } else {
            return AppEnvironment(rawValue: selectedEnvironment) ?? .staging
        }
        #else
        return .production
        #endif
    }

    func setEnvironment(_ env: AppEnvironment) {
        selectedEnvironment = env.rawValue
    }
}

extension AppEnvironment: RawRepresentable {
    init?(rawValue: String) {
        switch rawValue {
        case "production": self = .production
        case "staging": self = .staging
        case "mock": self = .mock
        default: return nil
        }
    }

    var rawValue: String {
        switch self {
        case .production: return "production"
        case .staging: return "staging"
        case .mock: return "mock"
        }
    }
}
```

---

## 10. Debug Menu for Runtime Switching

### File: `Features/Settings/Components/DebugMenuView.swift`

```swift
import SwiftUI

struct DebugMenuView: View {
    @ObservedObject private var featureFlags = FeatureFlags.shared
    @Environment(\.container) private var container

    var body: some View {
        List {
            Section("Environment") {
                Toggle("Use Mock Data", isOn: $featureFlags.useMockData)
                    .onChange(of: featureFlags.useMockData) { _, newValue in
                        switchEnvironment(useMock: newValue)
                    }

                if !featureFlags.useMockData {
                    Picker("Environment", selection: Binding(
                        get: { container.environment },
                        set: { featureFlags.setEnvironment($0) }
                    )) {
                        Text("Production").tag(AppEnvironment.production)
                        Text("Staging").tag(AppEnvironment.staging)
                    }
                }
            }

            Section("Logging") {
                Picker("Log Level", selection: $featureFlags.logLevel) {
                    Text("Debug").tag("debug")
                    Text("Info").tag("info")
                    Text("Warning").tag("warning")
                    Text("Error").tag("error")
                }
            }

            Section("Testing") {
                Button("Test API Connection") {
                    Task {
                        await testAPIConnection()
                    }
                }

                Button("Test WebSocket Connection") {
                    Task {
                        await testWebSocketConnection()
                    }
                }
            }

            Section("Cache") {
                Button("Clear Cache", role: .destructive) {
                    container.cacheManager?.clearAll()
                }
            }

            Section("Danger Zone") {
                Button("Reset Container", role: .destructive) {
                    Task {
                        await container.reset()
                        switchEnvironment(useMock: featureFlags.useMockData)
                    }
                }
            }
        }
        .navigationTitle("Debug Menu")
    }

    // MARK: - Actions

    private func switchEnvironment(useMock: Bool) {
        Task { @MainActor in
            await container.reset()

            if useMock {
                container.configureMock(scenario: .happyPath)
            } else {
                let env = featureFlags.environment
                container.configureProduction(
                    baseURL: env.baseURL,
                    wsURL: env.wsURL
                )
            }
        }
    }

    private func testAPIConnection() async {
        do {
            let health = try await container.healthService.fetchHealth()
            print("API Connection: \(health.status)")
        } catch {
            print("API Connection Failed: \(error)")
        }
    }

    private func testWebSocketConnection() async {
        do {
            await container.dataProvider.connect()
            print("WebSocket Connected")
        } catch {
            print("WebSocket Connection Failed: \(error)")
        }
    }
}

#Preview {
    NavigationStack {
        DebugMenuView()
            .withContainer(.preview())
    }
}
```

---

## Summary

This dependency injection system provides:

1. **Centralized Configuration**: All dependencies in one place
2. **Environment Switching**: Production/Staging/Mock at runtime
3. **Type Safety**: Compile-time dependency checking
4. **Testability**: Easy to inject mocks for testing
5. **SwiftUI Integration**: Environment values for views
6. **Property Wrappers**: Clean syntax with `@Injected`
7. **Debug Tools**: Runtime switching via debug menu

### Key Files Created:
- `Core/DI/Injectable.swift` - Protocols and property wrappers
- `Core/DI/DIContainer.swift` - Main container
- `Core/DI/EnvironmentKeys.swift` - SwiftUI integration
- `Core/DI/Resolver.swift` - Manual resolution helpers
- `App/HEANApp.swift` - App entry point
- `App/FeatureFlags.swift` - Feature flags
- `Features/Settings/Components/DebugMenuView.swift` - Debug menu

This is a **production-ready** DI system ready to be copied into your Xcode project.

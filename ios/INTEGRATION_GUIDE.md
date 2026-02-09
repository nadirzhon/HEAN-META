# HEAN iOS - Backend Integration Guide

How to connect SwiftUI components to the HEAN backend API and WebSocket.

---

## Architecture Overview

```
┌─────────────────┐
│   SwiftUI UI    │  ← Premium components (PriceTicker, etc.)
└────────┬────────┘
         │
┌────────▼────────┐
│  ViewModels     │  ← @StateObject, @Published properties
└────────┬────────┘
         │
┌────────▼────────┐
│   Services      │  ← WebSocket, API clients
└────────┬────────┘
         │
┌────────▼────────┐
│   Backend       │  ← FastAPI + WebSocket (apps/ui/src/app/api/)
└─────────────────┘
```

---

## Step 1: WebSocket Service

Create a WebSocket client to receive real-time updates from the backend.

### File: `Core/WebSocket/WebSocketService.swift`

```swift
import Foundation
import Combine

/// Real-time WebSocket connection to HEAN backend
class WebSocketService: NSObject, ObservableObject {
    @Published var isConnected = false
    @Published var latency: Int = 0
    @Published var lastHeartbeat: Date?

    private var webSocket: URLSessionWebSocketTask?
    private var cancellables = Set<AnyCancellable>()

    private let baseURL: String

    init(baseURL: String = "ws://localhost:8000") {
        self.baseURL = baseURL
        super.init()
    }

    func connect() {
        guard let url = URL(string: "\(baseURL)/ws") else {
            print("Invalid WebSocket URL")
            return
        }

        let session = URLSession(configuration: .default)
        webSocket = session.webSocketTask(with: url)
        webSocket?.resume()

        isConnected = true
        receiveMessage()

        // Start heartbeat
        startHeartbeat()
    }

    func disconnect() {
        webSocket?.cancel(with: .goingAway, reason: nil)
        isConnected = false
    }

    private func receiveMessage() {
        webSocket?.receive { [weak self] result in
            switch result {
            case .success(let message):
                self?.handleMessage(message)
                self?.receiveMessage() // Keep receiving

            case .failure(let error):
                print("WebSocket error: \(error)")
                self?.isConnected = false
            }
        }
    }

    private func handleMessage(_ message: URLSessionWebSocketTask.Message) {
        switch message {
        case .string(let text):
            handleJSONMessage(text)
        case .data(let data):
            if let text = String(data: data, encoding: .utf8) {
                handleJSONMessage(text)
            }
        @unknown default:
            break
        }
    }

    private func handleJSONMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        // Dispatch to appropriate publisher based on message type
        if let type = json["type"] as? String {
            switch type {
            case "tick":
                handleTickMessage(json)
            case "heartbeat":
                handleHeartbeat(json)
            default:
                print("Unknown message type: \(type)")
            }
        }
    }

    private func handleTickMessage(_ json: [String: Any]) {
        // Will be implemented in TradingDataService
        NotificationCenter.default.post(
            name: .webSocketTickReceived,
            object: json
        )
    }

    private func handleHeartbeat(_ json: [String: Any]) {
        lastHeartbeat = Date()

        if let timestamp = json["timestamp"] as? Double {
            let serverTime = Date(timeIntervalSince1970: timestamp)
            latency = Int(Date().timeIntervalSince(serverTime) * 1000)
        }
    }

    private func startHeartbeat() {
        Timer.publish(every: 5, on: .main, in: .common)
            .autoconnect()
            .sink { [weak self] _ in
                self?.sendHeartbeat()
            }
            .store(in: &cancellables)
    }

    private func sendHeartbeat() {
        let message = """
        {"type": "heartbeat", "timestamp": \(Date().timeIntervalSince1970)}
        """
        webSocket?.send(.string(message)) { error in
            if let error = error {
                print("Heartbeat error: \(error)")
            }
        }
    }
}

// MARK: - Notification Names
extension Notification.Name {
    static let webSocketTickReceived = Notification.Name("webSocketTickReceived")
}
```

---

## Step 2: Trading Data Service

Service layer to manage trading data and expose it to SwiftUI views.

### File: `Core/Networking/TradingDataService.swift`

```swift
import Foundation
import Combine

/// Service for managing trading data from backend
class TradingDataService: ObservableObject {
    @Published var prices: [String: PriceData] = [:]
    @Published var portfolio: PortfolioData?
    @Published var riskState: RiskState = .normal
    @Published var positions: [Position] = []
    @Published var orders: [Order] = []

    private let webSocket: WebSocketService
    private let apiClient: APIClient
    private var cancellables = Set<AnyCancellable>()

    init(webSocket: WebSocketService, apiClient: APIClient) {
        self.webSocket = webSocket
        self.apiClient = apiClient

        setupWebSocketListeners()
    }

    private func setupWebSocketListeners() {
        // Listen for tick messages
        NotificationCenter.default.publisher(for: .webSocketTickReceived)
            .sink { [weak self] notification in
                guard let json = notification.object as? [String: Any] else { return }
                self?.handleTickMessage(json)
            }
            .store(in: &cancellables)
    }

    private func handleTickMessage(_ json: [String: Any]) {
        guard let symbol = json["symbol"] as? String,
              let price = json["price"] as? Double else {
            return
        }

        let changePercent = json["change_percent"] as? Double ?? 0.0

        DispatchQueue.main.async {
            self.prices[symbol] = PriceData(
                symbol: symbol,
                price: price,
                changePercent: changePercent,
                timestamp: Date()
            )
        }
    }

    func fetchPortfolio() async {
        do {
            portfolio = try await apiClient.getPortfolio()
        } catch {
            print("Failed to fetch portfolio: \(error)")
        }
    }

    func fetchPositions() async {
        do {
            positions = try await apiClient.getPositions()
        } catch {
            print("Failed to fetch positions: \(error)")
        }
    }

    func fetchRiskState() async {
        do {
            riskState = try await apiClient.getRiskState()
        } catch {
            print("Failed to fetch risk state: \(error)")
        }
    }
}

// MARK: - Data Models
struct PriceData {
    let symbol: String
    let price: Double
    let changePercent: Double
    let timestamp: Date
}

struct PortfolioData {
    let totalPnL: Double
    let realizedPnL: Double
    let unrealizedPnL: Double
    let equity: Double
}

struct Position {
    let symbol: String
    let size: Double
    let entryPrice: Double
    let currentPrice: Double
    let unrealizedPnL: Double
}

struct Order {
    let id: String
    let symbol: String
    let side: String
    let size: Double
    let price: Double?
    let status: String
}

enum RiskState: String {
    case normal = "NORMAL"
    case softBrake = "SOFT_BRAKE"
    case quarantine = "QUARANTINE"
    case hardStop = "HARD_STOP"
}
```

---

## Step 3: API Client

HTTP client for REST API calls.

### File: `Core/Networking/APIClient.swift`

```swift
import Foundation

/// HTTP client for HEAN backend API
class APIClient {
    private let baseURL: String

    init(baseURL: String = "http://localhost:8000") {
        self.baseURL = baseURL
    }

    func getPortfolio() async throws -> PortfolioData {
        let url = URL(string: "\(baseURL)/api/portfolio/summary")!
        let (data, _) = try await URLSession.shared.data(from: url)

        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]

        return PortfolioData(
            totalPnL: json["total_pnl"] as? Double ?? 0,
            realizedPnL: json["realized_pnl"] as? Double ?? 0,
            unrealizedPnL: json["unrealized_pnl"] as? Double ?? 0,
            equity: json["equity"] as? Double ?? 0
        )
    }

    func getPositions() async throws -> [Position] {
        let url = URL(string: "\(baseURL)/api/positions")!
        let (data, _) = try await URLSession.shared.data(from: url)

        let json = try JSONSerialization.jsonObject(with: data) as! [[String: Any]]

        return json.map { dict in
            Position(
                symbol: dict["symbol"] as? String ?? "",
                size: dict["size"] as? Double ?? 0,
                entryPrice: dict["entry_price"] as? Double ?? 0,
                currentPrice: dict["current_price"] as? Double ?? 0,
                unrealizedPnL: dict["unrealized_pnl"] as? Double ?? 0
            )
        }
    }

    func getRiskState() async throws -> RiskState {
        let url = URL(string: "\(baseURL)/api/risk/state")!
        let (data, _) = try await URLSession.shared.data(from: url)

        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let stateStr = json["state"] as? String ?? "NORMAL"

        return RiskState(rawValue: stateStr) ?? .normal
    }
}
```

---

## Step 4: ViewModel

ViewModel to bridge services and UI.

### File: `Features/Dashboard/DashboardViewModel.swift`

```swift
import Foundation
import Combine

/// ViewModel for main trading dashboard
class DashboardViewModel: ObservableObject {
    @Published var btcPrice: PriceData?
    @Published var ethPrice: PriceData?
    @Published var portfolio: PortfolioData?
    @Published var riskState: RiskState = .normal
    @Published var connectionStatus: ConnectionStatus = .disconnected
    @Published var latency: Int = 0

    private let tradingData: TradingDataService
    private let webSocket: WebSocketService
    private var cancellables = Set<AnyCancellable>()

    enum ConnectionStatus {
        case connected, disconnected, reconnecting
    }

    init(tradingData: TradingDataService, webSocket: WebSocketService) {
        self.tradingData = tradingData
        self.webSocket = webSocket

        setupSubscriptions()
        connect()
    }

    private func setupSubscriptions() {
        // Subscribe to price updates
        tradingData.$prices
            .sink { [weak self] prices in
                self?.btcPrice = prices["BTCUSDT"]
                self?.ethPrice = prices["ETHUSDT"]
            }
            .store(in: &cancellables)

        // Subscribe to portfolio updates
        tradingData.$portfolio
            .sink { [weak self] portfolio in
                self?.portfolio = portfolio
            }
            .store(in: &cancellables)

        // Subscribe to risk state
        tradingData.$riskState
            .sink { [weak self] state in
                self?.riskState = state
            }
            .store(in: &cancellables)

        // Subscribe to connection status
        webSocket.$isConnected
            .map { $0 ? .connected : .disconnected }
            .assign(to: &$connectionStatus)

        // Subscribe to latency
        webSocket.$latency
            .assign(to: &$latency)
    }

    func connect() {
        webSocket.connect()

        Task {
            await tradingData.fetchPortfolio()
            await tradingData.fetchRiskState()
        }
    }

    func disconnect() {
        webSocket.disconnect()
    }

    func refresh() {
        Task {
            await tradingData.fetchPortfolio()
            await tradingData.fetchPositions()
            await tradingData.fetchRiskState()
        }
    }
}
```

---

## Step 5: SwiftUI View Integration

Connect components to ViewModel.

### File: `Features/Dashboard/DashboardView.swift`

```swift
import SwiftUI

struct DashboardView: View {
    @StateObject private var viewModel: DashboardViewModel

    init(viewModel: DashboardViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    var body: some View {
        ScrollView {
            VStack(spacing: AppTypography.lg) {
                // Header
                headerSection

                // Live Prices
                pricesSection

                // Portfolio
                portfolioSection

                // Risk Management
                riskSection
            }
            .padding(AppTypography.md)
        }
        .background(AppColors.backgroundPrimary.ignoresSafeArea())
        .onAppear {
            viewModel.connect()
        }
        .onDisappear {
            viewModel.disconnect()
        }
    }

    private var headerSection: some View {
        GlassCard {
            HStack {
                VStack(alignment: .leading) {
                    Text("HEAN")
                        .font(AppTypography.title(28, weight: .bold))
                        .foregroundColor(AppColors.textPrimary)

                    Text("Trading Dashboard")
                        .font(AppTypography.caption())
                        .foregroundColor(AppColors.textSecondary)
                }

                Spacer()

                StatusIndicator(
                    status: mapConnectionStatus(viewModel.connectionStatus),
                    latency: viewModel.latency,
                    showLabel: true
                )
            }
            .padding(AppTypography.lg)
        }
    }

    private var pricesSection: some View {
        VStack(spacing: AppTypography.md) {
            HStack(spacing: AppTypography.md) {
                if let btc = viewModel.btcPrice {
                    PriceTicker(
                        symbol: btc.symbol,
                        price: btc.price,
                        changePercent: btc.changePercent,
                        size: .large
                    )
                    .frame(maxWidth: .infinity)
                } else {
                    skeletonPriceTicker
                }

                if let eth = viewModel.ethPrice {
                    PriceTicker(
                        symbol: eth.symbol,
                        price: eth.price,
                        changePercent: eth.changePercent,
                        size: .large
                    )
                    .frame(maxWidth: .infinity)
                } else {
                    skeletonPriceTicker
                }
            }
        }
    }

    private var portfolioSection: some View {
        GlassCard {
            VStack(spacing: AppTypography.md) {
                HStack {
                    Text("Total P&L")
                        .font(AppTypography.body())
                        .foregroundColor(AppColors.textSecondary)

                    Spacer()

                    if let portfolio = viewModel.portfolio {
                        PnLBadge(
                            value: portfolio.totalPnL,
                            format: .dollar,
                            size: .expanded
                        )
                    } else {
                        SkeletonView(isLoading: true) {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(AppColors.backgroundTertiary)
                                .frame(width: 100, height: 28)
                        }
                    }
                }

                Divider()
                    .background(AppColors.textTertiary.opacity(0.3))

                HStack {
                    Text("Unrealized P&L")
                        .font(AppTypography.body())
                        .foregroundColor(AppColors.textSecondary)

                    Spacer()

                    if let portfolio = viewModel.portfolio {
                        PnLBadge(
                            value: portfolio.unrealizedPnL,
                            format: .dollar,
                            size: .compact
                        )
                    } else {
                        SkeletonView(isLoading: true) {
                            RoundedRectangle(cornerRadius: 8)
                                .fill(AppColors.backgroundTertiary)
                                .frame(width: 80, height: 24)
                        }
                    }
                }
            }
            .padding(AppTypography.md)
        }
    }

    private var riskSection: some View {
        VStack(spacing: AppTypography.md) {
            Text("Risk Management")
                .font(AppTypography.headline(18, weight: .bold))
                .foregroundColor(AppColors.textPrimary)
                .frame(maxWidth: .infinity, alignment: .leading)

            RiskBadge(
                state: mapRiskState(viewModel.riskState),
                variant: .expanded
            )
        }
    }

    private var skeletonPriceTicker: some View {
        VStack(alignment: .leading, spacing: AppTypography.xs) {
            RoundedRectangle(cornerRadius: 8)
                .fill(AppColors.backgroundSecondary)
                .frame(width: 80, height: 16)

            RoundedRectangle(cornerRadius: 8)
                .fill(AppColors.backgroundSecondary)
                .frame(width: 120, height: 32)

            RoundedRectangle(cornerRadius: 8)
                .fill(AppColors.backgroundSecondary)
                .frame(width: 60, height: 14)
        }
        .padding(AppTypography.md)
        .skeleton(isLoading: true)
        .frame(maxWidth: .infinity)
    }

    private func mapConnectionStatus(_ status: DashboardViewModel.ConnectionStatus) -> StatusIndicator.ConnectionStatus {
        switch status {
        case .connected: return .connected
        case .disconnected: return .disconnected
        case .reconnecting: return .reconnecting
        }
    }

    private func mapRiskState(_ state: RiskState) -> RiskBadge.RiskState {
        switch state {
        case .normal: return .normal
        case .softBrake: return .softBrake
        case .quarantine: return .quarantine
        case .hardStop: return .hardStop
        }
    }
}
```

---

## Step 6: Dependency Injection

Wire up all services in the app container.

### File: `Core/DI/DIContainer.swift`

```swift
import Foundation

/// Dependency injection container
class DIContainer: ObservableObject {
    static let shared = DIContainer()

    let webSocket: WebSocketService
    let apiClient: APIClient
    let tradingData: TradingDataService

    private init() {
        // Initialize services
        self.webSocket = WebSocketService(baseURL: "ws://localhost:8000")
        self.apiClient = APIClient(baseURL: "http://localhost:8000")
        self.tradingData = TradingDataService(webSocket: webSocket, apiClient: apiClient)
    }

    func start() {
        webSocket.connect()
    }

    func stop() {
        webSocket.disconnect()
    }
}
```

### Update `HEANApp.swift`:

```swift
import SwiftUI

@main
struct HEANApp: App {
    @StateObject private var container = DIContainer.shared

    var body: some Scene {
        WindowGroup {
            DashboardView(
                viewModel: DashboardViewModel(
                    tradingData: container.tradingData,
                    webSocket: container.webSocket
                )
            )
            .preferredColorScheme(.dark)
            .onAppear {
                container.start()
            }
        }
    }
}
```

---

## Backend WebSocket Message Format

The backend sends messages in this format:

### Tick Message (Price Update)
```json
{
    "type": "tick",
    "symbol": "BTCUSDT",
    "price": 42350.75,
    "change_percent": 3.45,
    "timestamp": 1706745600.0
}
```

### Heartbeat Message
```json
{
    "type": "heartbeat",
    "timestamp": 1706745600.0
}
```

### Portfolio Update
```json
{
    "type": "portfolio",
    "total_pnl": 2456.78,
    "realized_pnl": 1200.00,
    "unrealized_pnl": 1256.78,
    "equity": 12456.78
}
```

### Risk State Update
```json
{
    "type": "risk_state",
    "state": "SOFT_BRAKE",
    "reason": "Drawdown approaching threshold"
}
```

---

## Backend REST API Endpoints

### GET `/api/portfolio/summary`
Returns portfolio summary.

**Response:**
```json
{
    "total_pnl": 2456.78,
    "realized_pnl": 1200.00,
    "unrealized_pnl": 1256.78,
    "equity": 12456.78,
    "initial_capital": 10000.00
}
```

### GET `/api/positions`
Returns open positions.

**Response:**
```json
[
    {
        "symbol": "BTCUSDT",
        "size": 0.5,
        "entry_price": 42000.00,
        "current_price": 42350.75,
        "unrealized_pnl": 175.38
    }
]
```

### GET `/api/risk/state`
Returns current risk state.

**Response:**
```json
{
    "state": "NORMAL",
    "drawdown": 0.05,
    "max_position_size": 1000.00
}
```

---

## Testing

### 1. Mock Services for Previews

```swift
class MockTradingDataService: TradingDataService {
    override init(webSocket: WebSocketService, apiClient: APIClient) {
        super.init(webSocket: webSocket, apiClient: apiClient)

        // Inject mock data
        self.prices = [
            "BTCUSDT": PriceData(symbol: "BTCUSDT", price: 42350.75, changePercent: 3.45, timestamp: Date()),
            "ETHUSDT": PriceData(symbol: "ETHUSDT", price: 2245.30, changePercent: -1.25, timestamp: Date())
        ]

        self.portfolio = PortfolioData(
            totalPnL: 2456.78,
            realizedPnL: 1200.00,
            unrealizedPnL: 1256.78,
            equity: 12456.78
        )

        self.riskState = .normal
    }
}

#Preview {
    let mockWebSocket = WebSocketService()
    let mockAPI = APIClient()
    let mockTradingData = MockTradingDataService(webSocket: mockWebSocket, apiClient: mockAPI)

    let viewModel = DashboardViewModel(tradingData: mockTradingData, webSocket: mockWebSocket)

    return DashboardView(viewModel: viewModel)
}
```

### 2. Unit Tests

```swift
import XCTest
@testable import HEAN

class TradingDataServiceTests: XCTestCase {
    var service: TradingDataService!

    override func setUp() {
        let mockWebSocket = WebSocketService()
        let mockAPI = APIClient()
        service = TradingDataService(webSocket: mockWebSocket, apiClient: mockAPI)
    }

    func testHandleTickMessage() {
        let json: [String: Any] = [
            "symbol": "BTCUSDT",
            "price": 42350.75,
            "change_percent": 3.45
        ]

        service.handleTickMessage(json)

        XCTAssertEqual(service.prices["BTCUSDT"]?.price, 42350.75)
        XCTAssertEqual(service.prices["BTCUSDT"]?.changePercent, 3.45)
    }
}
```

---

## Error Handling

### WebSocket Reconnection

```swift
extension WebSocketService {
    func connectWithRetry(maxRetries: Int = 3) {
        var retries = 0

        func attemptConnection() {
            connect()

            // Check connection after 2 seconds
            DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
                if !self.isConnected && retries < maxRetries {
                    retries += 1
                    print("Retry \(retries)/\(maxRetries)")
                    attemptConnection()
                }
            }
        }

        attemptConnection()
    }
}
```

### API Error Handling

```swift
extension APIClient {
    func getPortfolioSafe() async -> PortfolioData? {
        do {
            return try await getPortfolio()
        } catch {
            print("Failed to fetch portfolio: \(error)")
            return nil
        }
    }
}
```

### UI Error States

```swift
struct DashboardView: View {
    @State private var errorMessage: String?
    @State private var showError = false

    var body: some View {
        // ... existing code ...

        .alert("Error", isPresented: $showError) {
            Button("OK") { showError = false }
        } message: {
            Text(errorMessage ?? "Unknown error")
        }
        .onReceive(viewModel.$connectionStatus) { status in
            if status == .disconnected {
                errorMessage = "Lost connection to server"
                showError = true
            }
        }
    }
}
```

---

## Next Steps

1. **Implement WebSocketService**: Start with basic connection
2. **Test WebSocket locally**: Run backend with `make dev`
3. **Add API client**: Implement REST endpoints
4. **Create ViewModels**: Bridge services to UI
5. **Build Dashboard**: Assemble components
6. **Add Error Handling**: Handle edge cases
7. **Test End-to-End**: Full integration test

---

## Complete File Structure

```
ios/HEAN/
├── Core/
│   ├── Networking/
│   │   ├── APIClient.swift              ✓ REST API client
│   │   └── TradingDataService.swift     ✓ Trading data manager
│   │
│   ├── WebSocket/
│   │   └── WebSocketService.swift       ✓ WebSocket connection
│   │
│   └── DI/
│       └── DIContainer.swift            ✓ Dependency injection
│
├── Features/
│   └── Dashboard/
│       ├── DashboardView.swift          ✓ Main dashboard UI
│       └── DashboardViewModel.swift     ✓ Dashboard logic
│
└── App/
    └── HEANApp.swift                    ✓ App entry point
```

---

**Ready to integrate!** Start with WebSocketService and test connection to backend.

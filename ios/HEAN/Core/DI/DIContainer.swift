//
//  DIContainer.swift
//  HEAN
//
//  Created by HEAN Team on 2026-01-31.
//

import Foundation
import Combine
import OSLog

enum Environment: String, CaseIterable {
    case mock = "Mock"
    case dev = "Development"
    case prod = "Production"

    var baseURL: String {
        switch self {
        case .mock:
            return "http://localhost:8000"
        case .dev:
            return "http://localhost:8000"
        case .prod:
            return "https://api.hean.trade"
        }
    }

    var wsURL: String {
        switch self {
        case .mock:
            return "ws://localhost:8000/ws"
        case .dev:
            return "ws://localhost:8000/ws"
        case .prod:
            return "wss://api.hean.trade/ws"
        }
    }
}

@MainActor
class DIContainer: ObservableObject {
    static let shared = DIContainer()

    @Published var currentEnvironment: Environment = .dev

    // Core Services
    private(set) var apiClient: APIClient!
    private(set) var webSocketManager: WebSocketManager!

    // Feature Services
    private(set) var marketService: MarketServiceProtocol!
    private(set) var tradingService: TradingServiceProtocol!
    private(set) var portfolioService: PortfolioServiceProtocol!
    private(set) var eventService: EventServiceProtocol!
    private(set) var signalService: SignalServiceProtocol!
    private(set) var strategyService: StrategyServiceProtocol!
    private(set) var riskService: RiskServiceProtocol!

    private var cancellables = Set<AnyCancellable>()

    private init() {
        setupServices()
    }

    func switchEnvironment(_ environment: Environment) {
        guard environment != currentEnvironment else { return }

        Logger.app.info("Switching environment: \(environment.rawValue)")
        currentEnvironment = environment

        // Tear down existing services
        webSocketManager?.disconnect()

        // Rebuild services
        setupServices()

        // Restart connections
        start()
    }

    /// Reconfigure with custom URLs (called from Settings)
    func reconfigure(apiBaseURL: String, wsBaseURL: String) {
        Logger.app.info("Reconfiguring with custom URLs: api=\(apiBaseURL) ws=\(wsBaseURL)")
        webSocketManager?.disconnect()
        let authToken = KeychainStore.shared.get("api_auth_key")
        apiClient = APIClient(baseURL: apiBaseURL, authToken: authToken)
        webSocketManager = WebSocketManager(url: wsBaseURL)
        rebuildFeatureServices()
        start()
    }

    private func setupServices() {
        // Use user-configured URLs if available, otherwise use environment defaults
        let userAPI = UserDefaults.standard.string(forKey: "apiBaseURL")
        let userWS = UserDefaults.standard.string(forKey: "wsBaseURL")
        let effectiveAPI = (userAPI?.isEmpty == false) ? userAPI! : currentEnvironment.baseURL
        let effectiveWS = (userWS?.isEmpty == false) ? userWS! : currentEnvironment.wsURL

        // Core
        let authToken = KeychainStore.shared.get("api_auth_key")
        apiClient = APIClient(baseURL: effectiveAPI, authToken: authToken)
        webSocketManager = WebSocketManager(url: effectiveWS)

        rebuildFeatureServices()
    }

    private func rebuildFeatureServices() {
        if currentEnvironment == .mock {
            let mockMarketService = MockMarketService()
            let mockTradingService = MockTradingService(marketService: mockMarketService)
            let mockPortfolioService = MockPortfolioService(tradingService: mockTradingService)
            let mockEventService = MockEventService()
            let mockSignalService = MockSignalService()
            let mockStrategyService = MockStrategyService()
            let mockRiskService = MockRiskService()

            marketService = mockMarketService
            tradingService = mockTradingService
            portfolioService = mockPortfolioService
            eventService = mockEventService
            signalService = mockSignalService
            strategyService = mockStrategyService
            riskService = mockRiskService
        } else {
            let liveTradingService = LiveTradingService(apiClient: apiClient, websocket: webSocketManager)
            let livePortfolioService = LivePortfolioService(apiClient: apiClient, websocket: webSocketManager)
            let liveSignalService = LiveSignalService(websocket: webSocketManager)
            let liveStrategyService = LiveStrategyService(apiClient: apiClient)
            let liveRiskService = LiveRiskService(apiClient: apiClient, websocket: webSocketManager)
            let liveMarketService = LiveMarketService(apiClient: apiClient, websocket: webSocketManager)
            let liveEventService = LiveEventService(apiClient: apiClient, websocket: webSocketManager)

            tradingService = liveTradingService
            portfolioService = livePortfolioService
            signalService = liveSignalService
            strategyService = liveStrategyService
            riskService = liveRiskService
            marketService = liveMarketService
            eventService = liveEventService
        }
    }

    func start() {
        Logger.app.info("Starting services...")

        if currentEnvironment != .mock {
            webSocketManager.connect()
        }

        // Mock services automatically start their internal timers upon initialization
        // No need to call startUpdates() methods
    }
}

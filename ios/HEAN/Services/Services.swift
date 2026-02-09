//
//  Services.swift
//  HEAN
//
//  All service implementations consolidated for Xcode compilation
//  Created on 2026-01-31.
//

import Foundation
import Combine

// MARK: - Mock Signal Service

@MainActor
final class MockSignalService: SignalServiceProtocol {
    private nonisolated(unsafe) let signalSubject = PassthroughSubject<Signal, Never>()
    private var timer: Timer?

    var signalPublisher: AnyPublisher<Signal, Never> {
        signalSubject.eraseToAnyPublisher()
    }

    init() {}

    func subscribeToSignals() {
        timer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.emitMockSignal()
            }
        }
    }

    func unsubscribeFromSignals() {
        timer?.invalidate()
        timer = nil
    }

    private func emitMockSignal() {
        let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        let strategies = ["ImpulseEngine", "FundingHarvester"]
        let sides = ["buy", "sell"]

        let signal = Signal(
            symbol: symbols.randomElement()!,
            side: sides.randomElement()!,
            confidence: Double.random(in: 0.4...0.95),
            strategy: strategies.randomElement()!,
            timestamp: Date(),
            reason: "Mock signal"
        )
        signalSubject.send(signal)
    }
}

// MARK: - Mock Strategy Service

@MainActor
final class MockStrategyService: StrategyServiceProtocol {
    private var strategies: [Strategy] = []

    init() {
        strategies = [
            Strategy(id: "impulse-engine", name: "ImpulseEngine", enabled: true, winRate: 0.62, totalTrades: 89, profitFactor: 1.8, description: "Momentum-based strategy", totalPnl: 245.50, wins: 55, losses: 34),
            Strategy(id: "funding-harvester", name: "FundingHarvester", enabled: true, winRate: 0.78, totalTrades: 52, profitFactor: 2.1, description: "Funding rate arbitrage", totalPnl: 180.25, wins: 41, losses: 11),
            Strategy(id: "basis-arbitrage", name: "BasisArbitrage", enabled: false, winRate: 0.55, totalTrades: 18, profitFactor: 1.2, description: "Spot-futures arbitrage", totalPnl: -12.30, wins: 10, losses: 8)
        ]
    }

    func fetchStrategies() async throws -> [Strategy] {
        try await Task.sleep(for: .milliseconds(200))
        return strategies
    }

    func toggleStrategy(id: String, enabled: Bool) async throws {
        try await Task.sleep(for: .milliseconds(150))
        if let index = strategies.firstIndex(where: { $0.id == id }) {
            let old = strategies[index]
            strategies[index] = Strategy(
                id: old.id, name: old.name, enabled: enabled,
                winRate: old.winRate, totalTrades: old.totalTrades,
                profitFactor: old.profitFactor, description: old.description
            )
        }
    }

    func updateParameters(id: String, parameters: sending [String: Any]) async throws {
        try await Task.sleep(for: .milliseconds(150))
        // Mock implementation - no-op
    }
}

// MARK: - Mock Risk Service

@MainActor
final class MockRiskService: RiskServiceProtocol {
    private nonisolated(unsafe) let riskStateSubject = CurrentValueSubject<RiskState, Never>(.normal)
    private var killswitchTriggered = false

    var riskStatePublisher: AnyPublisher<RiskState, Never> {
        riskStateSubject.eraseToAnyPublisher()
    }

    init() {}

    func fetchRiskStatus() async throws -> RiskGovernorStatus {
        try await Task.sleep(for: .milliseconds(200))
        return RiskGovernorStatus(
            riskState: riskStateSubject.value.rawValue,
            level: riskStateSubject.value.severity,
            reasonCodes: killswitchTriggered ? ["KILLSWITCH_TRIGGERED"] : [],
            quarantinedSymbols: [],
            canClear: true
        )
    }

    func resetKillswitch() async throws {
        try await Task.sleep(for: .milliseconds(200))
        killswitchTriggered = false
        riskStateSubject.send(.normal)
    }

    func quarantineSymbol(_ symbol: String) async throws {
        try await Task.sleep(for: .milliseconds(150))
        riskStateSubject.send(.quarantine)
    }

    func clearRiskBlocks() async throws {
        try await Task.sleep(for: .milliseconds(150))
        riskStateSubject.send(.normal)
    }

    func fetchKillswitchStatus() async throws -> KillswitchStatus {
        try await Task.sleep(for: .milliseconds(100))
        return KillswitchStatus(triggered: killswitchTriggered, reasons: killswitchTriggered ? ["Mock killswitch"] : [])
    }

    func fetchSignalRejectionStats() async throws -> SignalRejectionStats {
        try await Task.sleep(for: .milliseconds(100))
        return SignalRejectionStats(totalRejections: 3, totalSignals: 45, rejectionRate: 6.67, byReason: ["RISK_LIMIT": 2, "COOLDOWN": 1])
    }
}

// MARK: - Live Trading Service

@MainActor
final class LiveTradingService: TradingServiceProtocol {
    private let apiClient: APIClient
    private let websocket: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    private nonisolated(unsafe) let positionsSubject = CurrentValueSubject<[Position], Never>([])
    private nonisolated(unsafe) let ordersSubject = CurrentValueSubject<[Order], Never>([])

    var positionsPublisher: AnyPublisher<[Position], Never> {
        positionsSubject.eraseToAnyPublisher()
    }

    var ordersPublisher: AnyPublisher<[Order], Never> {
        ordersSubject.eraseToAnyPublisher()
    }

    init(apiClient: APIClient, websocket: WebSocketManager) {
        self.apiClient = apiClient
        self.websocket = websocket
        subscribeToWebSocketTopics()
    }

    private func subscribeToWebSocketTopics() {
        // Subscribe to real-time order and position updates
        websocket.subscribe(topic: "orders")
        websocket.subscribe(topic: "positions")
        websocket.subscribe(topic: "system_heartbeat")
        websocket.subscribe(topic: "account_state")
        websocket.subscribe(topic: "physics_events")
        websocket.subscribe(topic: "brain_events")

        websocket.messagePublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] message in
                Task { @MainActor in
                    self?.handleWSMessage(message)
                }
            }
            .store(in: &cancellables)
    }

    private func handleWSMessage(_ message: WebSocketMessage) {
        switch message.topic {
        case "orders":
            // Refresh orders on any order event
            Task { try? await fetchOrders(status: "open") }
        case "positions":
            // Refresh positions on any position event
            Task { try? await fetchPositions() }
        case "system_heartbeat":
            // System heartbeat - could update connection status
            break
        case "account_state":
            // Account state updates - equity, balance changes
            break
        case "physics_events":
            // Physics engine events - temperature, entropy updates
            break
        case "brain_events":
            // AI brain events - new analysis, insights
            break
        default:
            break
        }
    }

    func fetchPositions() async throws -> [Position] {
        // Backend wraps response: {"positions": [...]}
        struct PositionsResponse: Codable {
            let positions: [Position]
        }
        let response: PositionsResponse = try await apiClient.get("/api/v1/orders/positions")
        positionsSubject.send(response.positions)
        return response.positions
    }

    func fetchOrders(status: String? = nil) async throws -> [Order] {
        // Backend wraps response: {"orders": [...]}
        struct OrdersResponse: Codable {
            let orders: [Order]
        }
        var path = "/api/v1/orders"
        if let status = status { path += "?status=\(status)" }
        let response: OrdersResponse = try await apiClient.get(path)
        ordersSubject.send(response.orders)
        return response.orders
    }

    func placeOrder(symbol: String, side: String, type: String, qty: Double, price: Double?) async throws -> Order {
        var body: [String: Any] = ["symbol": symbol, "side": side, "type": type, "qty": qty]
        if let price = price { body["price"] = price }
        return try await apiClient.post("/api/v1/orders/test", body: body)
    }

    func cancelOrder(id: String) async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/orders/cancel", body: ["order_id": id])
    }

    func cancelAllOrders() async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/orders/cancel-all", body: [:])
    }

    func closePosition(symbol: String) async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/orders/close-position-by-symbol", body: ["symbol": symbol])
    }

    func closeAllPositions() async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/orders/close-all-positions", body: [:])
    }

    func fetchWhyDiagnostics() async throws -> WhyDiagnostics {
        try await apiClient.get("/api/v1/trading/why")
    }

    func fetchTradingMetrics() async throws -> TradingMetrics {
        try await apiClient.get("/api/v1/trading/metrics")
    }
}

// MARK: - Live Portfolio Service

@MainActor
final class LivePortfolioService: PortfolioServiceProtocol {
    private let apiClient: APIClient
    private let websocket: WebSocketManager

    private nonisolated(unsafe) let equitySubject = CurrentValueSubject<Double, Never>(0)
    private nonisolated(unsafe) let pnlSubject = CurrentValueSubject<PnLSnapshot, Never>(
        PnLSnapshot(realized: 0, unrealized: 0, total: 0, percent: 0, fees: 0)
    )

    var equityPublisher: AnyPublisher<Double, Never> {
        equitySubject.eraseToAnyPublisher()
    }

    var pnlPublisher: AnyPublisher<PnLSnapshot, Never> {
        pnlSubject.eraseToAnyPublisher()
    }

    init(apiClient: APIClient, websocket: WebSocketManager) {
        self.apiClient = apiClient
        self.websocket = websocket
    }

    func fetchPortfolio() async throws -> Portfolio {
        let portfolio: Portfolio = try await apiClient.get("/api/v1/engine/status")
        equitySubject.send(portfolio.equity)
        return portfolio
    }

    func fetchEquityHistory(limit: Int = 50) async throws -> [EquityPoint] {
        let response: EquityHistoryResponse = try await apiClient.get("/api/v1/trading/equity-history?limit=\(limit)")
        return response.snapshots
    }
}

struct EquityPoint: Codable, Identifiable {
    let timestamp: String
    let equity: Double
    var id: String { timestamp }
}

struct EquityHistoryResponse: Codable {
    let snapshots: [EquityPoint]
    let count: Int

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.snapshots = (try? container.decode([EquityPoint].self, forKey: .snapshots)) ?? []
        self.count = (try? container.decode(Int.self, forKey: .count)) ?? 0
    }

    enum CodingKeys: String, CodingKey {
        case snapshots, count
    }
}

// MARK: - Live Signal Service

@MainActor
final class LiveSignalService: SignalServiceProtocol {
    private let websocket: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    private nonisolated(unsafe) let signalSubject = PassthroughSubject<Signal, Never>()

    var signalPublisher: AnyPublisher<Signal, Never> {
        signalSubject.eraseToAnyPublisher()
    }

    init(websocket: WebSocketManager) {
        self.websocket = websocket
    }

    func subscribeToSignals() {
        websocket.subscribe(topic: "signals")
    }

    func unsubscribeFromSignals() {
        websocket.unsubscribe(topic: "signals")
        cancellables.removeAll()
    }
}

// MARK: - Live Strategy Service

@MainActor
final class LiveStrategyService: StrategyServiceProtocol {
    private let apiClient: APIClient

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func fetchStrategies() async throws -> [Strategy] {
        try await apiClient.get("/api/v1/strategies")
    }

    func toggleStrategy(id: String, enabled: Bool) async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/strategies/\(id)/enable", body: ["enabled": enabled])
    }

    func updateParameters(id: String, parameters: sending [String: Any]) async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/strategies/\(id)/params", body: parameters)
    }
}

// MARK: - Live Risk Service

@MainActor
final class LiveRiskService: RiskServiceProtocol {
    private let apiClient: APIClient
    private let websocket: WebSocketManager

    private nonisolated(unsafe) let riskStateSubject = CurrentValueSubject<RiskState, Never>(.normal)

    var riskStatePublisher: AnyPublisher<RiskState, Never> {
        riskStateSubject.eraseToAnyPublisher()
    }

    init(apiClient: APIClient, websocket: WebSocketManager) {
        self.apiClient = apiClient
        self.websocket = websocket
    }

    func fetchRiskStatus() async throws -> RiskGovernorStatus {
        let status: RiskGovernorStatus = try await apiClient.get("/api/v1/risk/governor/status")
        riskStateSubject.send(status.state)
        return status
    }

    func resetKillswitch() async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/risk/killswitch/reset", body: [:])
    }

    func quarantineSymbol(_ symbol: String) async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/risk/governor/quarantine/\(symbol)", body: [:])
    }

    func clearRiskBlocks() async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/risk/governor/clear", body: [:])
    }

    func fetchKillswitchStatus() async throws -> KillswitchStatus {
        try await apiClient.get("/api/v1/risk/killswitch/status")
    }

    func fetchSignalRejectionStats() async throws -> SignalRejectionStats {
        try await apiClient.get("/api/v1/telemetry/signal-rejections")
    }
}

// MARK: - Live Market Service

@MainActor
final class LiveMarketService: MarketServiceProtocol {
    private let apiClient: APIClient
    private let websocket: WebSocketManager
    private var cancellables = Set<AnyCancellable>()
    private nonisolated(unsafe) var pollTimer: Timer?

    private nonisolated(unsafe) let marketsSubject = CurrentValueSubject<[Market], Never>([])

    private(set) var markets: [Market] = []

    var marketUpdates: AnyPublisher<[Market], Never> {
        marketsSubject.eraseToAnyPublisher()
    }

    init(apiClient: APIClient, websocket: WebSocketManager) {
        self.apiClient = apiClient
        self.websocket = websocket
        startPolling()
    }

    deinit {
        pollTimer?.invalidate()
    }

    private func startPolling() {
        // Poll market data every 5 seconds
        pollTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                try? await self?.refreshMarkets()
            }
        }
        // Initial fetch
        Task { @MainActor in
            try? await refreshMarkets()
        }
    }

    private func refreshMarkets() async throws {
        // Fetch tickers for main trading symbols
        let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        var updatedMarkets: [Market] = []

        for symbol in symbols {
            do {
                let ticker: TickerResponse = try await apiClient.get("/api/v1/market/ticker?symbol=\(symbol)")
                let market = Market(
                    id: symbol,
                    symbol: symbol,
                    baseCurrency: String(symbol.dropLast(4)),
                    quoteCurrency: "USDT",
                    name: symbol,
                    price: ticker.price ?? 0,
                    change24h: 0, // Not provided by this endpoint
                    changePercent24h: 0,
                    volume24h: ticker.volume ?? 0,
                    high24h: 0,
                    low24h: 0,
                    sparkline: nil
                )
                updatedMarkets.append(market)
            } catch {
                // Skip failed symbols
                continue
            }
        }

        if !updatedMarkets.isEmpty {
            markets = updatedMarkets
            marketsSubject.send(updatedMarkets)
        }
    }

    func fetchMarkets() async throws -> [Market] {
        try await refreshMarkets()
        return markets
    }

    func fetchMarket(symbol: String) async throws -> Market {
        let ticker: TickerResponse = try await apiClient.get("/api/v1/market/ticker?symbol=\(symbol)")
        return Market(
            id: symbol,
            symbol: symbol,
            baseCurrency: String(symbol.dropLast(4)),
            quoteCurrency: "USDT",
            name: symbol,
            price: ticker.price ?? 0,
            change24h: 0,
            changePercent24h: 0,
            volume24h: ticker.volume ?? 0,
            high24h: 0,
            low24h: 0,
            sparkline: nil
        )
    }

    func fetchCandles(symbol: String, interval: String, limit: Int) async throws -> [Candle] {
        let response: CandlesResponse = try await apiClient.get(
            "/api/v1/market/candles?symbol=\(symbol)&timeframe=\(interval)&limit=\(limit)"
        )
        return response.klines.enumerated().map { index, kline in
            Candle(
                id: "\(symbol)-\(index)",
                timestamp: Date(timeIntervalSince1970: kline.timestamp / 1000),
                open: kline.open,
                high: kline.high,
                low: kline.low,
                close: kline.close,
                volume: kline.volume
            )
        }
    }
}

// Response models for Live services
struct TickerResponse: Codable {
    let symbol: String
    let price: Double?
    let bid: Double?
    let ask: Double?
    let volume: Double?
    let timestamp: String?
}

struct CandlesResponse: Codable {
    let symbol: String
    let timeframe: String
    let klines: [KlineData]
    let count: Int
}

struct KlineData: Codable {
    let timestamp: Double
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Double

    init(from decoder: Decoder) throws {
        // Backend may return klines as arrays [timestamp, open, high, low, close, volume]
        if let container = try? decoder.singleValueContainer(),
           let array = try? container.decode([Double].self), array.count >= 6 {
            timestamp = array[0]
            open = array[1]
            high = array[2]
            low = array[3]
            close = array[4]
            volume = array[5]
        } else {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            timestamp = try container.decode(Double.self, forKey: .timestamp)
            open = try container.decode(Double.self, forKey: .open)
            high = try container.decode(Double.self, forKey: .high)
            low = try container.decode(Double.self, forKey: .low)
            close = try container.decode(Double.self, forKey: .close)
            volume = try container.decode(Double.self, forKey: .volume)
        }
    }

    enum CodingKeys: String, CodingKey {
        case timestamp, open, high, low, close, volume
    }
}

// MARK: - Live Event Service

@MainActor
final class LiveEventService: EventServiceProtocol {
    private let apiClient: APIClient
    private let websocket: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    private nonisolated(unsafe) let eventStreamSubject = PassthroughSubject<TradingEvent, Never>()
    private nonisolated(unsafe) let wsHealthSubject = PassthroughSubject<WebSocketHealth, Never>()

    private var connectionStartTime: Date?
    private var lastHeartbeat: Date?
    private var lastEventTimestamp: Date?
    private var eventCount = 0
    private var eventRateWindow: [Date] = []
    private nonisolated(unsafe) var healthTimer: Timer?

    var eventStream: AnyPublisher<TradingEvent, Never> {
        eventStreamSubject.eraseToAnyPublisher()
    }

    var wsHealthUpdates: AnyPublisher<WebSocketHealth, Never> {
        wsHealthSubject.eraseToAnyPublisher()
    }

    init(apiClient: APIClient, websocket: WebSocketManager) {
        self.apiClient = apiClient
        self.websocket = websocket
        connectionStartTime = Date()
        subscribeToWebSocket()
        startHealthUpdates()
    }

    deinit {
        healthTimer?.invalidate()
    }

    private func subscribeToWebSocket() {
        websocket.eventPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] event in
                Task { @MainActor in
                    self?.handleWebSocketEvent(event)
                }
            }
            .store(in: &cancellables)
    }

    private func handleWebSocketEvent(_ event: WebSocketEvent) {
        switch event {
        case .connected:
            connectionStartTime = Date()
            lastHeartbeat = Date()

        case .disconnected:
            break

        case .message(let data):
            parseAndEmitEvent(data)

        case .error:
            break
        }
    }

    private func parseAndEmitEvent(_ data: Data) {
        lastHeartbeat = Date()

        // Try to decode as TradingEvent
        if let event = try? JSONDecoder.hean.decode(TradingEvent.self, from: data) {
            lastEventTimestamp = event.timestamp
            eventCount += 1

            // Track event rate
            let now = Date()
            eventRateWindow.append(now)
            let cutoff = now.addingTimeInterval(-60)
            eventRateWindow.removeAll { $0 < cutoff }

            eventStreamSubject.send(event)
        } else if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let typeStr = json["type"] as? String,
                  let eventType = EventType(rawValue: typeStr) {
            // Parse from generic JSON
            let event = TradingEvent(
                id: json["id"] as? String ?? UUID().uuidString,
                type: eventType,
                symbol: json["symbol"] as? String,
                message: json["message"] as? String ?? typeStr,
                timestamp: Date(),
                metadata: nil
            )

            lastEventTimestamp = event.timestamp
            eventCount += 1
            eventStreamSubject.send(event)
        }
    }

    private func startHealthUpdates() {
        healthTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.publishHealthUpdate()
            }
        }
    }

    private func publishHealthUpdate() {
        let eventsPerSecond = Double(eventRateWindow.count) / 60.0
        let connectionDuration = connectionStartTime.map { Date().timeIntervalSince($0) } ?? 0

        let health = WebSocketHealth(
            state: websocket.isConnected ? .connected : .disconnected,
            lastHeartbeat: lastHeartbeat,
            lastEventTimestamp: lastEventTimestamp,
            eventsPerSecond: eventsPerSecond,
            connectionDuration: connectionDuration
        )

        wsHealthSubject.send(health)
    }

    func fetchRecentEvents(limit: Int) async throws -> [TradingEvent] {
        // Fetch from money log
        struct EventsResponse: Codable {
            let entries: [MoneyLogEntry]
            let count: Int
        }

        struct MoneyLogEntry: Codable {
            let id: String
            let event_type: String
            let symbol: String?
            let timestamp: String
            let details: [String: String]?
        }

        let response: EventsResponse = try await apiClient.get("/api/v1/telemetry/money-log/entries?limit=\(limit)")

        return response.entries.compactMap { entry -> TradingEvent? in
            guard let eventType = EventType(rawValue: entry.event_type) else { return nil }

            let dateFormatter = ISO8601DateFormatter()
            dateFormatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
            let timestamp = dateFormatter.date(from: entry.timestamp) ?? Date()

            return TradingEvent(
                id: entry.id,
                type: eventType,
                symbol: entry.symbol,
                message: "\(entry.event_type): \(entry.symbol ?? "N/A")",
                timestamp: timestamp,
                metadata: entry.details
            )
        }
    }
}

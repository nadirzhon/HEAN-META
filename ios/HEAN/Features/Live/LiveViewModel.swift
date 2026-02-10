//
//  LiveViewModel.swift
//  HEAN
//
//  ViewModel for Live tab — aggregates market, physics, AI, portfolio data
//

import SwiftUI
import Combine

@MainActor
final class LiveViewModel: ObservableObject {
    // MARK: - Published State

    @Published var physics: PhysicsState?
    @Published var participants: ParticipantBreakdown?
    @Published var brainAnalysis: BrainAnalysis?
    @Published var anomalies: [Anomaly] = []
    @Published var equity: Double = 0
    @Published var initialCapital: Double = 0
    @Published var pnl: PnLSnapshot = PnLSnapshot(realized: 0, unrealized: 0, total: 0, percent: 0, fees: 0)
    @Published var riskState: RiskState = .normal
    @Published var positionCount: Int = 0
    @Published var symbol: String = "BTCUSDT"
    @Published var marketPrice: Double = 0
    @Published var priceChange24h: Double = 0
    @Published var isLoading = true
    @Published var error: String?
    @Published var backendReachable = true
    @Published var failedEndpoints: Int = 0
    @Published var engineState: String = "UNKNOWN"

    var connectionStatus: StatusIndicator.ConnectionStatus {
        if !backendReachable { return .disconnected }
        switch riskState {
        case .normal: return .connected
        case .softBrake, .quarantine: return .reconnecting
        case .hardStop: return .disconnected
        }
    }

    // MARK: - Dependencies

    private var apiClient: APIClient?
    private var portfolioService: PortfolioServiceProtocol?
    private var tradingService: TradingServiceProtocol?
    private var riskService: RiskServiceProtocol?
    private var cancellables = Set<AnyCancellable>()
    private var isConfigured = false

    // MARK: - Configuration

    func configure(container: DIContainer) {
        guard !isConfigured else { return }
        isConfigured = true

        self.apiClient = container.apiClient
        self.portfolioService = container.portfolioService
        self.tradingService = container.tradingService
        self.riskService = container.riskService
        subscribeToUpdates()
    }

    // MARK: - Real-time Subscriptions

    private func subscribeToUpdates() {
        guard let portfolioService = portfolioService,
              let tradingService = tradingService,
              let riskService = riskService else { return }

        portfolioService.equityPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] equity in
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    self?.equity = equity
                }
            }
            .store(in: &cancellables)

        portfolioService.pnlPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] pnl in
                withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                    self?.pnl = pnl
                }
            }
            .store(in: &cancellables)

        tradingService.positionsPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] positions in
                self?.positionCount = positions.count
            }
            .store(in: &cancellables)

        riskService.riskStatePublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in
                withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
                    self?.riskState = state
                }
            }
            .store(in: &cancellables)
    }

    private var isRefreshing = false

    // MARK: - Refresh

    func refresh() async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        guard let apiClient = apiClient else {
            isLoading = false
            error = "Not configured"
            backendReachable = false
            return
        }

        var failures = 0
        let totalCalls = 7

        // Market price
        do {
            let market: Market = try await apiClient.get("/api/v1/market/ticker?symbol=BTCUSDT")
            self.marketPrice = market.price
            self.priceChange24h = market.changePercent24h
        } catch {
            failures += 1
            print("[LiveVM] Market error: \(error)")
        }

        // Physics state
        do {
            let state: PhysicsState = try await apiClient.get("/api/v1/physics/state?symbol=BTCUSDT")
            self.physics = state
        } catch { failures += 1 }

        // Participants
        do {
            let bd: ParticipantBreakdown = try await apiClient.get("/api/v1/physics/participants?symbol=BTCUSDT")
            self.participants = bd
        } catch { failures += 1 }

        // Brain analysis
        do {
            let analysis: BrainAnalysis = try await apiClient.get("/api/v1/brain/analysis")
            self.brainAnalysis = analysis
        } catch { failures += 1 }

        // Anomalies
        do {
            let response: AnomaliesWrapper = try await apiClient.get("/api/v1/physics/anomalies?limit=10")
            self.anomalies = response.anomalies
        } catch {
            failures += 1
        }

        // Engine state + Portfolio
        do {
            // Use inline struct — no CodingKeys needed since JSONDecoder.hean
            // uses .convertFromSnakeCase automatically
            struct EngineStatus: Codable {
                var engineState: String?
                var running: Bool?
                var equity: Double?
                var initialCapital: Double?
                var unrealizedPnl: Double?
                var realizedPnl: Double?
                var dailyPnl: Double?
                var availableBalance: Double?
                var usedMargin: Double?
                var totalFees: Double?
            }
            let status: EngineStatus = try await apiClient.get("/api/v1/engine/status")
            self.engineState = status.engineState ?? "UNKNOWN"
            let eq = status.equity ?? 0
            let cap = status.initialCapital ?? 0
            self.equity = eq
            self.initialCapital = cap
            let realized = status.realizedPnl ?? status.dailyPnl ?? 0
            let unrealized = status.unrealizedPnl ?? 0
            let total = eq - cap
            let pct = cap > 0 ? (total / cap) * 100 : 0
            self.pnl = PnLSnapshot(
                realized: realized,
                unrealized: unrealized,
                total: total,
                percent: pct,
                fees: status.totalFees ?? 0
            )
        } catch {
            failures += 1
        }

        // Risk
        if let riskService = riskService {
            do {
                let status = try await riskService.fetchRiskStatus()
                self.riskState = status.state
            } catch {
                failures += 1
            }
        }

        // Positions count (uses tradingService, already counted above if portfolio failed)
        if let tradingService = tradingService {
            do {
                let positions = try await tradingService.fetchPositions()
                self.positionCount = positions.count
            } catch { }
        }

        // Update connectivity state
        failedEndpoints = failures
        let engineDown = engineState == "ERROR" || engineState == "STOPPED"
        if failures >= totalCalls {
            backendReachable = false
            error = "Backend unreachable"
        } else if engineDown {
            backendReachable = true
            error = "Engine \(engineState.lowercased()) — data may be stale"
        } else if failures > 0 {
            backendReachable = true
            error = "\(failures) endpoint\(failures == 1 ? "" : "s") failed"
        } else {
            backendReachable = true
            error = nil
        }

        isLoading = false
    }
}

// Private wrapper for anomalies endpoint
private struct AnomaliesWrapper: Codable {
    let anomalies: [Anomaly]
    let activeCount: Int

    enum CodingKeys: String, CodingKey {
        case anomalies
        case activeCount = "active_count"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.anomalies = try container.decodeIfPresent([Anomaly].self, forKey: .anomalies) ?? []
        self.activeCount = try container.decodeIfPresent(Int.self, forKey: .activeCount) ?? 0
    }
}

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
    @Published var pnl: PnLSnapshot = PnLSnapshot(realized: 0, unrealized: 0, total: 0, percent: 0, fees: 0)
    @Published var riskState: RiskState = .normal
    @Published var positionCount: Int = 0
    @Published var marketPrice: Double = 0
    @Published var priceChange24h: Double = 0
    @Published var isLoading = true
    @Published var error: String?

    var connectionStatus: StatusIndicator.ConnectionStatus {
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

    // MARK: - Refresh

    func refresh() async {
        guard let apiClient = apiClient else {
            isLoading = false
            return
        }

        // Show UI immediately with defaults if first load
        if isLoading && physics == nil {
            physics = PhysicsState(
                temperature: 0, entropy: 0, phase: "ice",
                szilardProfit: 0, timestamp: ""
            )
            isLoading = false
        }

        // Market price
        do {
            let market: Market = try await apiClient.get("/api/v1/market/ticker?symbol=BTCUSDT")
            self.marketPrice = market.price
            self.priceChange24h = market.changePercent24h
            print("[LiveVM] Market loaded: \(market.price)")
        } catch {
            print("[LiveVM] Market error: \(error)")
        }

        // Physics state
        do {
            let state: PhysicsState = try await apiClient.get("/api/v1/physics/state?symbol=BTCUSDT")
            self.physics = state
            print("[LiveVM] Physics loaded: \(state.phase)")
        } catch {
            print("[LiveVM] Physics error: \(error)")
            if self.physics == nil {
                self.physics = PhysicsState(
                    temperature: 50, entropy: 0.5, phase: "water",
                    szilardProfit: 0, timestamp: ""
                )
            }
        }

        // Participants
        do {
            let bd: ParticipantBreakdown = try await apiClient.get("/api/v1/physics/participants?symbol=BTCUSDT")
            self.participants = bd
        } catch {
            if self.participants == nil {
                self.participants = ParticipantBreakdown(
                    mmActivity: 0.35, institutionalFlow: 250000,
                    retailSentiment: 0.65, whaleActivity: 0.12,
                    arbPressure: 0.08, dominantPlayer: "market_maker",
                    metaSignal: "Analyzing..."
                )
            }
        }

        // Brain analysis
        do {
            let analysis: BrainAnalysis = try await apiClient.get("/api/v1/brain/analysis")
            self.brainAnalysis = analysis
        } catch {
            if self.brainAnalysis == nil {
                self.brainAnalysis = BrainAnalysis(
                    timestamp: "", thoughts: [], forces: [],
                    signal: nil, summary: "Analyzing market conditions...",
                    marketRegime: "unknown"
                )
            }
        }

        // Anomalies
        do {
            let response: AnomaliesWrapper = try await apiClient.get("/api/v1/physics/anomalies?limit=10")
            self.anomalies = response.anomalies
        } catch {
            // Keep existing
        }

        // Portfolio
        if let portfolioService = portfolioService {
            do {
                let portfolio = try await portfolioService.fetchPortfolio()
                self.equity = portfolio.equity
                let total = portfolio.realizedPnL + portfolio.unrealizedPnL
                let pct = portfolio.initialCapital > 0 ? (total / portfolio.initialCapital) * 100 : 0
                self.pnl = PnLSnapshot(
                    realized: portfolio.realizedPnL,
                    unrealized: portfolio.unrealizedPnL,
                    total: total,
                    percent: pct,
                    fees: portfolio.totalFees ?? 0
                )
            } catch { }
        }

        // Risk
        if let riskService = riskService {
            do {
                let status = try await riskService.fetchRiskStatus()
                self.riskState = status.state
            } catch { }
        }

        // Positions count
        if let tradingService = tradingService {
            do {
                let positions = try await tradingService.fetchPositions()
                self.positionCount = positions.count
            } catch { }
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

//
//  ActionViewModel.swift
//  HEAN
//
//  ViewModel for Action tab — trading controls, positions, signals
//

import SwiftUI
import Combine

@MainActor
final class ActionViewModel: ObservableObject {
    // MARK: - Published State

    @Published var positions: [Position] = []
    @Published var orders: [Order] = []
    @Published var latestSignal: BrainSignal?
    @Published var weeklyStats: WeeklyStats = .empty
    @Published var tradingMetrics: TradingMetrics?
    @Published var isLoading = true
    @Published var error: String?
    @Published var signalSkipped = false

    // MARK: - Dependencies

    private var tradingService: TradingServiceProtocol?
    private var apiClient: APIClient?
    private var cancellables = Set<AnyCancellable>()
    private var isConfigured = false
    private var isRefreshing = false

    // MARK: - Configuration

    func configure(tradingService: TradingServiceProtocol, apiClient: APIClient) {
        guard !isConfigured else { return }
        isConfigured = true

        self.tradingService = tradingService
        self.apiClient = apiClient
        subscribeToUpdates()
    }

    // MARK: - Real-time Subscriptions

    private func subscribeToUpdates() {
        guard let tradingService = tradingService else { return }

        tradingService.positionsPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] positions in
                withAnimation(.spring(response: 0.4, dampingFraction: 0.8)) {
                    self?.positions = positions
                }
            }
            .store(in: &cancellables)

        tradingService.ordersPublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] orders in
                self?.orders = orders
            }
            .store(in: &cancellables)
    }

    // MARK: - Refresh

    func refresh() async {
        guard !isRefreshing else { return }
        isRefreshing = true
        defer { isRefreshing = false }

        guard let tradingService = tradingService,
              let apiClient = apiClient else { return }

        // Positions
        do {
            self.positions = try await tradingService.fetchPositions()
        } catch { }

        // Orders
        do {
            self.orders = try await tradingService.fetchOrders(status: nil)
        } catch { }

        // Brain signal
        do {
            let analysis: BrainAnalysis = try await apiClient.get("/api/v1/brain/analysis")
            self.latestSignal = analysis.signal
        } catch { }

        // Trading metrics
        do {
            self.tradingMetrics = try await tradingService.fetchTradingMetrics()
        } catch { }

        // Weekly stats (computed from metrics)
        if let metrics = tradingMetrics {
            let total = metrics.ordersFilled
            let winRate = total > 0 ? Double(metrics.decisionsCreate) / Double(total) : 0
            self.weeklyStats = WeeklyStats(
                totalTrades: total,
                winRate: min(winRate, 1.0),
                totalPnL: 0,
                bestTrade: 0,
                worstTrade: 0,
                avgRiskReward: 0
            )
        }

        signalSkipped = false
        isLoading = false
    }

    // MARK: - Actions

    func confirmSignal() async {
        guard let signal = latestSignal,
              let tradingService = tradingService else { return }

        let side = signal.isLong ? "Buy" : "Sell"
        do {
            _ = try await tradingService.placeOrder(
                symbol: "BTCUSDT",
                side: side,
                type: "Market",
                qty: 0.001,
                price: nil
            )
            Haptics.success()
            self.latestSignal = nil
        } catch {
            self.error = "Order failed: \(error.localizedDescription)"
            Haptics.error()
        }
    }

    func skipSignal() {
        signalSkipped = true
        Haptics.light()
    }

    func closePosition(symbol: String) async {
        guard let tradingService = tradingService else { return }
        do {
            try await tradingService.closePosition(symbol: symbol)
            Haptics.success()
        } catch {
            self.error = "Close failed: \(error.localizedDescription)"
            Haptics.error()
        }
    }

    func closeAllPositions() async {
        guard let tradingService = tradingService else { return }
        do {
            try await tradingService.closeAllPositions()
            Haptics.success()
        } catch {
            self.error = "Close all failed: \(error.localizedDescription)"
            Haptics.error()
        }
    }
}

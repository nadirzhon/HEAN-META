//
//  DashboardView.swift
//  HEAN
//
//  World-class trading dashboard with real-time updates
//

import SwiftUI
import Combine
import OSLog

private let refreshLog = Logger(subsystem: "com.hean.trading", category: "refresh")

// MARK: - System Status Enum

enum SystemStatus: String, Codable {
    case unknown = "UNKNOWN"
    case running = "RUNNING"
    case stopped = "STOPPED"
    case error = "ERROR"
}

// MARK: - Shimmer Modifier

struct ShimmerModifier: ViewModifier {
    @State private var phase: CGFloat = 0

    func body(content: Content) -> some View {
        content
            .overlay(
                LinearGradient(
                    colors: [
                        .clear,
                        .white.opacity(0.1),
                        .clear
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .offset(x: phase)
                .mask(content)
            )
            .onAppear {
                withAnimation(.linear(duration: 1.5).repeatForever(autoreverses: false)) {
                    phase = 300
                }
            }
    }
}

extension View {
    func shimmer() -> some View {
        modifier(ShimmerModifier())
    }
}

// MARK: - Dashboard ViewModel

@MainActor
final class DashboardViewModel: ObservableObject {
    // MARK: - Published State

    @Published var equity: Double = 0
    @Published var pnl: PnLSnapshot = PnLSnapshot(realized: 0, unrealized: 0, total: 0, percent: 0, fees: 0)
    @Published var positions: [Position] = []
    @Published var orders: [Order] = []
    @Published var riskState: RiskState = .normal
    @Published var systemStatus: SystemStatus = .unknown
    @Published var tradingMetrics: TradingMetrics?
    @Published var equityHistory: [Double] = []
    @Published var killswitchStatus: KillswitchStatus?
    @Published var rejectionStats: SignalRejectionStats?
    @Published var portfolio: Portfolio?
    @Published var isLoading = false
    @Published var error: String?
    @Published var connectionState: WebSocketState = .disconnected
    @Published var refreshTick: Int = 0

    // MARK: - Dependencies

    private var tradingService: TradingServiceProtocol?
    private var portfolioService: PortfolioServiceProtocol?
    private var riskService: RiskServiceProtocol?
    private var cancellables = Set<AnyCancellable>()
    private var isConfigured = false

    // MARK: - Configuration (called after container is available)

    func configure(
        tradingService: TradingServiceProtocol,
        portfolioService: PortfolioServiceProtocol,
        riskService: RiskServiceProtocol
    ) {
        refreshLog.info("configure() called, isConfigured=\(self.isConfigured)")
        guard !isConfigured else { return }
        isConfigured = true

        self.tradingService = tradingService
        self.portfolioService = portfolioService
        self.riskService = riskService
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

        riskService.riskStatePublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] state in
                withAnimation(.spring(response: 0.5, dampingFraction: 0.8)) {
                    self?.riskState = state
                }
            }
            .store(in: &cancellables)
    }

    // MARK: - Actions

    func refresh() async {
        guard let portfolioService = portfolioService,
              let tradingService = tradingService,
              let riskService = riskService else {
            return
        }

        error = nil

        // Each fetch is independent — one failure doesn't block others
        do {
            let portfolio = try await portfolioService.fetchPortfolio()
            self.portfolio = portfolio
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
        } catch {
            refreshLog.error("Portfolio FAILED: \(error)")
        }

        do {
            self.positions = try await tradingService.fetchPositions()
        } catch {
            refreshLog.error("Positions FAILED: \(error)")
        }

        do {
            self.orders = try await tradingService.fetchOrders(status: nil)
        } catch {
            refreshLog.error("Orders FAILED: \(error)")
        }

        do {
            let riskStatus = try await riskService.fetchRiskStatus()
            self.riskState = riskStatus.state
        } catch {
            refreshLog.error("Risk FAILED: \(error)")
        }

        do {
            self.tradingMetrics = try await tradingService.fetchTradingMetrics()
        } catch {
            refreshLog.error("Metrics FAILED: \(error)")
        }

        do {
            self.equityHistory = try await portfolioService.fetchEquityHistory(limit: 50).map(\.equity)
        } catch {
            refreshLog.error("EquityHistory FAILED: \(error)")
        }

        self.killswitchStatus = try? await riskService.fetchKillswitchStatus()
        self.rejectionStats = try? await riskService.fetchSignalRejectionStats()

        self.refreshTick += 1
        refreshLog.info("Refresh #\(self.refreshTick): equity=\(self.equity) orders=\(self.orders.count) pos=\(self.positions.count)")
        isLoading = false
    }

    func closePosition(symbol: String) async {
        guard let tradingService = tradingService else { return }
        do {
            try await tradingService.closePosition(symbol: symbol)
        } catch {
            self.error = "Failed to close position: \(error.localizedDescription)"
        }
    }

    func closeAllPositions() async {
        guard let tradingService = tradingService else { return }
        do {
            try await tradingService.closeAllPositions()
        } catch {
            self.error = "Failed to close all positions: \(error.localizedDescription)"
        }
    }
}


// MARK: - Dashboard View

struct DashboardView: View {
    @EnvironmentObject var container: DIContainer
    @StateObject private var viewModel = DashboardViewModel()
    @State private var showQuickOrder = false
    @State private var quickOrderSide: TradeView.OrderSide = .buy

    // Inject services when container becomes available
    private func injectServices() {
        viewModel.configure(
            tradingService: container.tradingService,
            portfolioService: container.portfolioService,
            riskService: container.riskService
        )
    }

    var body: some View {
        NavigationStack {
            ScrollView(.vertical, showsIndicators: true) {
                VStack(spacing: 16) {
                    // Connection Status Banner
                    connectionBanner

                    // Equity Hero Card
                    equityCard

                    // Quick Metrics Row
                    quickMetrics

                    // Risk State
                    riskStateCard

                    // Killswitch Warning
                    if let ks = viewModel.killswitchStatus, ks.triggered {
                        killswitchBanner(ks)
                    }

                    // Quick Actions
                    QuickActionsRow(
                        onBuy: { quickOrderSide = .buy; showQuickOrder = true },
                        onSell: { quickOrderSide = .sell; showQuickOrder = true },
                        onCloseAll: { Task { await viewModel.closeAllPositions() } }
                    )

                    // Trading Funnel
                    if let metrics = viewModel.tradingMetrics {
                        tradingFunnelCard(metrics)
                    }

                    // Active Positions
                    positionsSection

                    // Active Orders
                    ordersSection

                    // Performance Sparkline
                    performanceCard

                    // Signal Rejection Stats
                    if let stats = viewModel.rejectionStats, stats.totalSignals > 0 {
                        signalRejectionCard(stats)
                    }
                }
                .padding()
            }
            .id(viewModel.refreshTick)
            .background(Color(hex: "0A0A0F").ignoresSafeArea())
            .navigationTitle("HEAN")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 12) {
                        connectionDot
                        Button {
                            Task { await viewModel.refresh() }
                        } label: {
                            Image(systemName: "arrow.clockwise")
                                .foregroundColor(Color(hex: "00D4FF"))
                        }
                    }
                }
            }
            .refreshable {
                await viewModel.refresh()
            }
            .task {
                injectServices()
                await viewModel.refresh()
            }
            .onReceive(Timer.publish(every: 3, on: .main, in: .common).autoconnect()) { _ in
                Task { await viewModel.refresh() }
            }
            .sheet(isPresented: $showQuickOrder) {
                OrderSheet(
                    symbol: "BTCUSDT",
                    side: quickOrderSide,
                    isPresented: $showQuickOrder,
                    tradingService: container.tradingService
                )
            }
            .alert("Error", isPresented: .constant(viewModel.error != nil)) {
                Button("OK") { viewModel.error = nil }
            } message: {
                Text(viewModel.error ?? "")
            }
        }
    }

    // MARK: - Connection

    private var connectionDot: some View {
        Circle()
            .fill(viewModel.connectionState.isConnected ? Color(hex: "22C55E") : Color(hex: "EF4444"))
            .frame(width: 8, height: 8)
    }

    @ViewBuilder
    private var connectionBanner: some View {
        if !viewModel.connectionState.isConnected && viewModel.connectionState != .disconnected {
            HStack(spacing: 8) {
                ProgressView().tint(.white).scaleEffect(0.8)
                Text(viewModel.connectionState.displayText)
                    .font(.caption)
                    .foregroundColor(.white)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(Color(hex: "F59E0B").opacity(0.9))
            .cornerRadius(8)
            .transition(.move(edge: .top).combined(with: .opacity))
        }
    }

    // MARK: - Equity Card

    private var equityCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                Text("Portfolio Equity")
                    .font(.subheadline)
                    .foregroundColor(.gray)

                if let mode = viewModel.portfolio?.tradingMode {
                    Text(mode.uppercased())
                        .font(.caption2.bold())
                        .foregroundColor(viewModel.portfolio?.isLive == true ? Color(hex: "22C55E") : Color(hex: "3B82F6"))
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background((viewModel.portfolio?.isLive == true ? Color(hex: "22C55E") : Color(hex: "3B82F6")).opacity(0.15))
                        .cornerRadius(6)
                }

                if viewModel.isLoading && viewModel.equity == 0 {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.gray.opacity(0.2))
                        .frame(height: 40)
                        .shimmer()
                } else {
                    Text(viewModel.equity.asCurrency)
                        .font(.system(size: 34, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                        .contentTransition(.numericText())
                        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: viewModel.equity)
                }

                HStack(spacing: 16) {
                    pnlIndicator(label: "Total P&L", value: viewModel.pnl.total, percent: viewModel.pnl.percent)

                    Divider().frame(height: 20)

                    VStack(alignment: .leading) {
                        Text("Realized")
                            .font(.caption2)
                            .foregroundColor(.gray)
                        Text(viewModel.pnl.realized.asPnL)
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                            .foregroundColor(viewModel.pnl.realized >= 0 ? Color(hex: "22C55E") : Color(hex: "EF4444"))
                    }

                    VStack(alignment: .leading) {
                        Text("Unrealized")
                            .font(.caption2)
                            .foregroundColor(.gray)
                        Text(viewModel.pnl.unrealized.asPnL)
                            .font(.system(size: 13, weight: .semibold, design: .monospaced))
                            .foregroundColor(viewModel.pnl.unrealized >= 0 ? Color(hex: "22C55E") : Color(hex: "EF4444"))
                    }
                }
            }
            .padding()
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Portfolio equity \(viewModel.equity.asCurrency)")
    }

    private func pnlIndicator(label: String, value: Double, percent: Double) -> some View {
        VStack(alignment: .leading) {
            Text(label).font(.caption2).foregroundColor(.gray)
            HStack(spacing: 4) {
                Image(systemName: value >= 0 ? "arrow.up.right" : "arrow.down.right")
                Text("\(value.asPnL) (\(percent.asPercent))")
            }
            .font(.system(size: 14, weight: .semibold))
            .foregroundColor(value >= 0 ? Color(hex: "22C55E") : Color(hex: "EF4444"))
        }
    }

    // MARK: - Quick Metrics

    private var quickMetrics: some View {
        HStack(spacing: 12) {
            MetricCard(title: "Positions", value: "\(viewModel.positions.count)", icon: "chart.bar.doc.horizontal", color: "3B82F6")
            MetricCard(title: "Orders", value: "\(viewModel.orders.count)", icon: "list.clipboard", color: "7B61FF")
            MetricCard(title: "Fees", value: (viewModel.portfolio?.totalFees ?? viewModel.pnl.fees).asCurrency, icon: "dollarsign.circle", color: "F59E0B")
        }
    }

    // MARK: - Risk State

    private var riskStateCard: some View {
        GlassCard {
            HStack(spacing: 12) {
                Image(systemName: viewModel.riskState.icon)
                    .font(.title2)
                    .foregroundColor(Color(hex: viewModel.riskState.colorHex))
                    .symbolEffect(.pulse, isActive: viewModel.riskState != .normal)

                VStack(alignment: .leading, spacing: 2) {
                    Text(viewModel.riskState.displayName)
                        .font(.headline)
                        .foregroundColor(Color(hex: viewModel.riskState.colorHex))
                    Text(viewModel.riskState.description)
                        .font(.caption)
                        .foregroundColor(.gray)
                }

                Spacer()

                Circle()
                    .fill(Color(hex: viewModel.riskState.colorHex))
                    .frame(width: 10, height: 10)
            }
            .padding()
        }
        .accessibilityLabel("Risk state: \(viewModel.riskState.displayName)")
    }

    // MARK: - Trading Funnel

    private func tradingFunnelCard(_ metrics: TradingMetrics) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Trading Funnel")
                        .font(.headline)
                        .foregroundColor(.white)
                    Spacer()
                    if let uptime = metrics.uptimeSec {
                        Text(formatUptime(uptime))
                            .font(.caption2.bold())
                            .foregroundColor(Color(hex: "22C55E"))
                            .padding(.horizontal, 8)
                            .padding(.vertical, 3)
                            .background(Color(hex: "22C55E").opacity(0.15))
                            .cornerRadius(6)
                    }
                }

                HStack(spacing: 0) {
                    funnelStep(label: "Signals", value: metrics.signalsDetected, color: "3B82F6")
                    Image(systemName: "chevron.right").font(.caption).foregroundColor(.gray)
                    funnelStep(label: "Orders", value: metrics.ordersCreated, color: "7B61FF")
                    Image(systemName: "chevron.right").font(.caption).foregroundColor(.gray)
                    funnelStep(label: "Filled", value: metrics.ordersFilled, color: "22C55E")
                }

                // Decision breakdown
                if metrics.decisionsCreate > 0 || metrics.decisionsSkip > 0 || metrics.signalsBlocked > 0 {
                    HStack(spacing: 12) {
                        if metrics.decisionsCreate > 0 {
                            Label("\(metrics.decisionsCreate) created", systemImage: "plus.circle.fill")
                                .font(.caption2).foregroundColor(Color(hex: "22C55E"))
                        }
                        if metrics.decisionsSkip > 0 {
                            Label("\(metrics.decisionsSkip) skipped", systemImage: "forward.fill")
                                .font(.caption2).foregroundColor(Color(hex: "F59E0B"))
                        }
                        if metrics.signalsBlocked > 0 {
                            Label("\(metrics.signalsBlocked) blocked", systemImage: "xmark.circle.fill")
                                .font(.caption2).foregroundColor(Color(hex: "EF4444"))
                        }
                    }
                }

                // Top reasons for skip/block
                if let reasons = metrics.topReasonsForSkipBlock, !reasons.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(reasons.prefix(3)) { reason in
                            HStack {
                                Text(reason.code.replacingOccurrences(of: "_", with: " "))
                                    .font(.caption2)
                                    .foregroundColor(.gray)
                                Spacer()
                                Text("\(reason.count)")
                                    .font(.caption2.bold())
                                    .foregroundColor(Color(hex: "F59E0B"))
                            }
                        }
                    }
                }
            }
            .padding()
        }
    }

    private func formatUptime(_ seconds: Double) -> String {
        let h = Int(seconds) / 3600
        let m = (Int(seconds) % 3600) / 60
        if h > 0 { return "\(h)h \(m)m" }
        return "\(m)m"
    }

    private func funnelStep(label: String, value: Int, color: String) -> some View {
        VStack(spacing: 4) {
            Text("\(value)")
                .font(.system(size: 20, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: color))
            Text(label)
                .font(.caption2)
                .foregroundColor(.gray)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Positions

    private var positionsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Active Positions")
                    .font(.headline)
                    .foregroundColor(.white)
                Spacer()
                if !viewModel.positions.isEmpty {
                    Button("Close All") {
                        Task { await viewModel.closeAllPositions() }
                    }
                    .font(.caption).foregroundColor(Color(hex: "EF4444"))
                }
            }

            if viewModel.positions.isEmpty {
                emptyCard(icon: "tray", text: "No active positions")
            } else {
                ForEach(Array(viewModel.positions.prefix(20))) { position in
                    PositionCardView(position: position) {
                        Task { await viewModel.closePosition(symbol: position.symbol) }
                    }
                }
                if viewModel.positions.count > 20 {
                    Text("+ \(viewModel.positions.count - 20) more positions")
                        .font(.caption).foregroundColor(.gray)
                        .frame(maxWidth: .infinity).padding(.vertical, 4)
                }
            }
        }
    }

    // MARK: - Orders

    private var ordersSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Active Orders")
                .font(.headline)
                .foregroundColor(.white)

            if viewModel.orders.isEmpty {
                emptyCard(icon: "doc.text", text: "No active orders")
            } else {
                ForEach(Array(viewModel.orders.prefix(20))) { order in
                    OrderCardView(order: order)
                }
                if viewModel.orders.count > 20 {
                    Text("+ \(viewModel.orders.count - 20) more orders")
                        .font(.caption).foregroundColor(.gray)
                        .frame(maxWidth: .infinity).padding(.vertical, 4)
                }
            }
        }
    }

    // MARK: - Performance Card (real equity history)

    private var performanceCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                Text("Performance").font(.headline).foregroundColor(.white)

                if viewModel.equityHistory.isEmpty {
                    HStack {
                        Spacer()
                        Text("Waiting for data...")
                            .font(.caption).foregroundColor(.gray)
                        Spacer()
                    }
                    .frame(height: 60)
                } else {
                    Sparkline(data: viewModel.equityHistory, lineWidth: 2)
                        .frame(height: 60)
                }

                HStack {
                    Text("\(viewModel.equityHistory.count) snapshots")
                        .font(.caption2).foregroundColor(.gray)
                    Spacer()
                    PnLBadge(value: viewModel.pnl.total, percentage: viewModel.pnl.percent)
                }
            }
            .padding()
        }
    }

    // MARK: - Killswitch Banner

    private func killswitchBanner(_ ks: KillswitchStatus) -> some View {
        GlassCard {
            HStack(spacing: 10) {
                Image(systemName: "exclamationmark.octagon.fill")
                    .font(.title2)
                    .foregroundColor(Color(hex: "EF4444"))

                VStack(alignment: .leading, spacing: 2) {
                    Text("KILLSWITCH TRIGGERED")
                        .font(.caption.bold())
                        .foregroundColor(Color(hex: "EF4444"))
                    if let reason = ks.reason {
                        Text(reason)
                            .font(.caption2)
                            .foregroundColor(.gray)
                    }
                    if ks.currentDrawdown > 0 {
                        Text("Drawdown: \(String(format: "%.1f", ks.currentDrawdown))%")
                            .font(.caption2.bold())
                            .foregroundColor(Color(hex: "EF4444"))
                    }
                }

                Spacer()
            }
            .padding()
            .background(Color(hex: "EF4444").opacity(0.1))
        }
    }

    // MARK: - Signal Rejection Card

    private func signalRejectionCard(_ stats: SignalRejectionStats) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: "xmark.seal.fill")
                        .foregroundColor(Color(hex: "F59E0B"))
                    Text("Signal Rejections")
                        .font(.headline)
                        .foregroundColor(.white)
                    Spacer()
                    Text("\(String(format: "%.1f", stats.rejectionRate))%")
                        .font(.system(.subheadline, design: .monospaced).bold())
                        .foregroundColor(stats.rejectionRate > 50 ? Color(hex: "EF4444") : Color(hex: "F59E0B"))
                }

                HStack(spacing: 16) {
                    VStack(alignment: .leading) {
                        Text("Rejected").font(.caption2).foregroundColor(.gray)
                        Text("\(stats.totalRejections)")
                            .font(.system(.body, design: .monospaced).bold())
                            .foregroundColor(Color(hex: "EF4444"))
                    }
                    VStack(alignment: .leading) {
                        Text("Total Signals").font(.caption2).foregroundColor(.gray)
                        Text("\(stats.totalSignals)")
                            .font(.system(.body, design: .monospaced).bold())
                            .foregroundColor(.white)
                    }
                    VStack(alignment: .leading) {
                        Text("Window").font(.caption2).foregroundColor(.gray)
                        Text("\(stats.timeWindowMinutes)m")
                            .font(.system(.body, design: .monospaced).bold())
                            .foregroundColor(.gray)
                    }
                }

                // By Reason
                if !stats.byReason.isEmpty {
                    Text("By Reason").font(.caption2.bold()).foregroundColor(.gray)
                    let topReasons = stats.byReason.sorted { $0.value > $1.value }.prefix(3)
                    ForEach(Array(topReasons), id: \.key) { reason, count in
                        HStack {
                            Text(reason.replacingOccurrences(of: "_", with: " ").capitalized)
                                .font(.caption2).foregroundColor(.gray)
                            Spacer()
                            Text("\(count)").font(.caption.bold()).foregroundColor(Color(hex: "F59E0B"))
                        }
                    }
                }

                // By Symbol
                if !stats.bySymbol.isEmpty {
                    Text("By Symbol").font(.caption2.bold()).foregroundColor(.gray)
                    let topSymbols = stats.bySymbol.sorted { $0.value > $1.value }.prefix(3)
                    ForEach(Array(topSymbols), id: \.key) { symbol, count in
                        HStack {
                            Text(symbol).font(.caption2).foregroundColor(.gray)
                            Spacer()
                            Text("\(count)").font(.caption.bold()).foregroundColor(Color(hex: "3B82F6"))
                        }
                    }
                }

                // By Strategy
                if !stats.byStrategy.isEmpty {
                    Text("By Strategy").font(.caption2.bold()).foregroundColor(.gray)
                    let topStrats = stats.byStrategy.sorted { $0.value > $1.value }.prefix(3)
                    ForEach(Array(topStrats), id: \.key) { strat, count in
                        HStack {
                            Text(strat.replacingOccurrences(of: "_", with: " ").capitalized)
                                .font(.caption2).foregroundColor(.gray)
                            Spacer()
                            Text("\(count)").font(.caption.bold()).foregroundColor(Color(hex: "7B61FF"))
                        }
                    }
                }
            }
            .padding()
        }
    }

    private func emptyCard(icon: String, text: String) -> some View {
        GlassCard {
            HStack {
                Image(systemName: icon).foregroundColor(.gray)
                Text(text).font(.subheadline).foregroundColor(.gray)
                Spacer()
            }
            .padding()
        }
    }
}

// MARK: - Supporting Views

struct PositionCardView: View {
    let position: Position
    let onClose: () -> Void

    var body: some View {
        GlassCard {
            HStack(spacing: 12) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(position.side == .long ? Color(hex: "22C55E") : Color(hex: "EF4444"))
                    .frame(width: 4, height: 50)

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(position.symbol)
                            .font(.system(.headline, design: .monospaced))
                            .foregroundColor(.white)
                        Text(position.side.displayName)
                            .font(.caption2.bold())
                            .foregroundColor(position.side == .long ? Color(hex: "22C55E") : Color(hex: "EF4444"))
                            .padding(.horizontal, 6).padding(.vertical, 2)
                            .background((position.side == .long ? Color(hex: "22C55E") : Color(hex: "EF4444")).opacity(0.15))
                            .cornerRadius(4)
                    }
                    HStack(spacing: 12) {
                        Text("Entry: \(position.formattedEntryPrice)")
                        Text("Size: \(position.formattedSize)")
                    }
                    .font(.caption).foregroundColor(.gray)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    PnLBadge(value: position.unrealizedPnL)
                    Button("Close") { onClose() }
                        .font(.caption.bold())
                        .foregroundColor(Color(hex: "EF4444"))
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background(Color(hex: "EF4444").opacity(0.15))
                        .cornerRadius(6)
                }
            }
            .padding()
        }
    }
}

struct OrderCardView: View {
    let order: Order

    var body: some View {
        GlassCard {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(order.symbol).font(.system(.subheadline, design: .monospaced)).foregroundColor(.white)
                        Text(order.side.displayName).font(.caption2.bold())
                            .foregroundColor(order.side == .buy ? Color(hex: "22C55E") : Color(hex: "EF4444"))
                        Text(order.type?.displayName ?? "").font(.caption2).foregroundColor(.gray)
                    }
                    HStack(spacing: 8) {
                        Text("Qty: \(String(format: "%.4f", order.quantity)) @ \(order.price != nil ? "$\(String(format: "%.2f", order.price!))" : "Market")")
                            .font(.caption).foregroundColor(.gray)
                        if let strat = order.strategyId {
                            Text(strat)
                                .font(.caption2)
                                .foregroundColor(Color(hex: "7B61FF"))
                                .padding(.horizontal, 4).padding(.vertical, 1)
                                .background(Color(hex: "7B61FF").opacity(0.15))
                                .cornerRadius(3)
                        }
                    }
                }
                Spacer()
                Text(order.status.displayName)
                    .font(.caption2.bold()).foregroundColor(Color(hex: "F59E0B"))
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .background(Color(hex: "F59E0B").opacity(0.15)).cornerRadius(6)
            }
            .padding()
        }
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: String

    var body: some View {
        GlassCard {
            VStack(spacing: 6) {
                Image(systemName: icon).font(.caption).foregroundColor(Color(hex: color))
                Text(value).font(.system(size: 18, weight: .bold, design: .monospaced)).foregroundColor(.white)
                Text(title).font(.caption2).foregroundColor(.gray)
            }
            .padding(.vertical, 10).padding(.horizontal, 4)
            .frame(maxWidth: .infinity)
        }
    }
}

struct QuickActionsRow: View {
    var onBuy: () -> Void = {}
    var onSell: () -> Void = {}
    var onCloseAll: () -> Void = {}

    var body: some View {
        HStack(spacing: 12) {
            QuickActionButton(title: "Buy", icon: "arrow.up.circle.fill", color: Color(hex: "22C55E"), action: onBuy)
            QuickActionButton(title: "Sell", icon: "arrow.down.circle.fill", color: Color(hex: "EF4444"), action: onSell)
            QuickActionButton(title: "Close All", icon: "xmark.circle.fill", color: Color(hex: "F59E0B"), action: onCloseAll)
        }
    }
}

struct QuickActionButton: View {
    let title: String
    let icon: String
    let color: Color
    let action: () -> Void

    var body: some View {
        Button(action: { Haptics.light(); action() }) {
            VStack(spacing: 8) {
                Image(systemName: icon).font(.system(size: 24))
                Text(title).font(.caption.bold())
            }
            .foregroundColor(color)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(color.opacity(0.15))
            .cornerRadius(12)
            .overlay(RoundedRectangle(cornerRadius: 12).stroke(color.opacity(0.3), lineWidth: 0.5))
        }
    }
}

#Preview {
    DashboardView()
        .environmentObject(DIContainer.shared)
        .preferredColorScheme(.dark)
}

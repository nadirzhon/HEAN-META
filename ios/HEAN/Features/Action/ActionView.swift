//
//  ActionView.swift
//  HEAN
//
//  Tab 3: Trading control — signal proposals, positions, stats
//

import SwiftUI

struct ActionView: View {
    @EnvironmentObject var container: DIContainer
    @StateObject private var viewModel = ActionViewModel()
    @State private var refreshTimer: Timer?
    @State private var showConfirmClose = false
    @State private var closeSymbol: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if viewModel.isLoading && viewModel.positions.isEmpty {
                        ProgressView("Loading trading data...")
                            .padding(.top, 40)
                    } else {
                        // Status card
                        statusCard

                        // Signal proposal
                        if let signal = viewModel.latestSignal, !signal.isNeutral, !viewModel.signalSkipped {
                            signalProposalCard(signal)
                        }

                        // Active positions
                        positionsSection

                        // Active orders
                        ordersSection

                        // Weekly stats
                        weeklyStatsSection

                        // Trading funnel
                        if let metrics = viewModel.tradingMetrics {
                            tradingFunnelMini(metrics)
                        }
                    }
                }
                .padding()
            }
            .background(Theme.Colors.background.ignoresSafeArea())
            .navigationTitle("Action")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink {
                        StrategiesView(viewModel: StrategiesViewModel(
                            strategyService: container.strategyService
                        ))
                    } label: {
                        Label("Strategies", systemImage: "brain.head.profile")
                            .labelStyle(.iconOnly)
                            .foregroundColor(Theme.Colors.accent)
                    }
                }
            }
            .onAppear { injectServices(); startRefresh() }
            .onDisappear { refreshTimer?.invalidate() }
            .refreshable { await viewModel.refresh() }
            .alert("Error", isPresented: .constant(viewModel.error != nil)) {
                Button("OK") { viewModel.error = nil }
            } message: {
                Text(viewModel.error ?? "")
            }
            .alert("Close Position", isPresented: $showConfirmClose) {
                Button("Cancel", role: .cancel) { }
                Button("Close", role: .destructive) {
                    if let symbol = closeSymbol {
                        Task { await viewModel.closePosition(symbol: symbol) }
                    }
                }
            } message: {
                Text("Close position for \(closeSymbol ?? "")?")
            }
        }
    }

    // MARK: - Setup

    private func injectServices() {
        viewModel.configure(
            tradingService: container.tradingService,
            apiClient: container.apiClient
        )
    }

    private func startRefresh() {
        Task { await viewModel.refresh() }
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { _ in
            Task { @MainActor in await viewModel.refresh() }
        }
    }

    // MARK: - Status Card

    private var statusCard: some View {
        GlassCard(padding: 16) {
            HStack {
                Image(systemName: viewModel.positions.isEmpty ? "checkmark.circle" : "chart.bar.doc.horizontal")
                    .font(.title2)
                    .foregroundColor(viewModel.positions.isEmpty ? Theme.Colors.success : Theme.Colors.accent)

                VStack(alignment: .leading, spacing: 2) {
                    Text(viewModel.positions.isEmpty ? "No Active Positions" : "\(viewModel.positions.count) Position\(viewModel.positions.count == 1 ? "" : "s") Active")
                        .font(.headline)
                        .foregroundColor(Theme.Colors.textPrimary)
                    Text(viewModel.orders.isEmpty ? "No pending orders" : "\(viewModel.orders.count) order\(viewModel.orders.count == 1 ? "" : "s") pending")
                        .font(.caption)
                        .foregroundColor(Theme.Colors.textSecondary)
                }

                Spacer()

                if !viewModel.positions.isEmpty {
                    Button {
                        Task { await viewModel.closeAllPositions() }
                    } label: {
                        Text("Close All")
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(Theme.Colors.error)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .background(Theme.Colors.error.opacity(0.15))
                            .cornerRadius(8)
                    }
                }
            }
        }
    }

    // MARK: - Signal Proposal

    private func signalProposalCard(_ signal: BrainSignal) -> some View {
        GlassCardAccent {
            VStack(spacing: 14) {
                // Header
                HStack {
                    Image(systemName: "lightbulb.fill")
                        .foregroundColor(Theme.Colors.warning)
                    Text("AI Signal Proposal")
                        .font(.headline)
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    Text("\(Int(signal.confidence * 100))%")
                        .font(.system(.title3, design: .monospaced))
                        .fontWeight(.bold)
                        .foregroundColor(Color(hex: signal.directionColor))
                }

                // Direction
                HStack {
                    Image(systemName: signal.isLong ? "arrow.up.circle.fill" : "arrow.down.circle.fill")
                        .font(.system(size: 32))
                        .foregroundColor(Color(hex: signal.directionColor))
                    VStack(alignment: .leading, spacing: 2) {
                        Text(signal.direction.uppercased())
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(Color(hex: signal.directionColor))
                        if let explanation = signal.explanation {
                            Text(explanation)
                                .font(.caption)
                                .foregroundColor(Theme.Colors.textSecondary)
                                .lineLimit(2)
                        }
                    }
                    Spacer()
                }

                // Price levels
                if signal.entryPrice != nil || signal.targetPrice != nil || signal.stopPrice != nil {
                    HStack(spacing: 16) {
                        if let entry = signal.entryPrice {
                            priceLevelMini("Entry", value: entry, color: Theme.Colors.accent)
                        }
                        if let target = signal.targetPrice {
                            priceLevelMini("Target", value: target, color: Theme.Colors.success)
                        }
                        if let stop = signal.stopPrice {
                            priceLevelMini("Stop", value: stop, color: Theme.Colors.error)
                        }
                        if let rr = signal.riskReward {
                            VStack(spacing: 2) {
                                Text("R:R")
                                    .font(.caption2)
                                    .foregroundColor(Theme.Colors.textSecondary)
                                Text("1:\(String(format: "%.1f", rr))")
                                    .font(.system(.caption, design: .monospaced))
                                    .fontWeight(.bold)
                                    .foregroundColor(Theme.Colors.accent)
                            }
                        }
                    }
                }

                // Action buttons
                HStack(spacing: 12) {
                    Button {
                        Task { await viewModel.confirmSignal() }
                    } label: {
                        HStack {
                            Image(systemName: "checkmark.circle.fill")
                            Text("CONFIRM")
                        }
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Color(hex: signal.directionColor))
                        .cornerRadius(10)
                    }

                    Button {
                        viewModel.skipSignal()
                    } label: {
                        HStack {
                            Image(systemName: "forward.fill")
                            Text("SKIP")
                        }
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(Theme.Colors.textSecondary)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 12)
                        .background(Theme.Colors.textSecondary.opacity(0.15))
                        .cornerRadius(10)
                    }
                }
            }
            .padding()
        }
    }

    private func priceLevelMini(_ label: String, value: Double, color: Color) -> some View {
        VStack(spacing: 2) {
            Text(label)
                .font(.caption2)
                .foregroundColor(Theme.Colors.textSecondary)
            Text(String(format: "$%.2f", value))
                .font(.system(.caption, design: .monospaced))
                .fontWeight(.bold)
                .foregroundColor(color)
        }
    }

    // MARK: - Positions

    private var positionsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Active Positions")
                    .font(.headline)
                    .foregroundColor(Theme.Colors.textPrimary)
                Spacer()
            }

            if viewModel.positions.isEmpty {
                GlassCard(padding: 16) {
                    HStack {
                        Image(systemName: "tray")
                            .foregroundColor(Theme.Colors.textSecondary)
                        Text("No active positions")
                            .font(.subheadline)
                            .foregroundColor(Theme.Colors.textSecondary)
                        Spacer()
                    }
                }
            } else {
                ForEach(viewModel.positions) { position in
                    actionPositionCard(position)
                }
            }
        }
    }

    private func actionPositionCard(_ position: Position) -> some View {
        GlassCard(padding: 14) {
            HStack(spacing: 12) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(position.side == .long ? Theme.Colors.success : Theme.Colors.error)
                    .frame(width: 4, height: 50)

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(position.symbol)
                            .font(.system(.headline, design: .monospaced))
                            .foregroundColor(Theme.Colors.textPrimary)
                        Text(position.side.displayName)
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundColor(position.side == .long ? Theme.Colors.success : Theme.Colors.error)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background((position.side == .long ? Theme.Colors.success : Theme.Colors.error).opacity(0.15))
                            .cornerRadius(4)
                    }
                    HStack(spacing: 12) {
                        Text("Entry: \(position.formattedEntryPrice)")
                        Text("Size: \(position.formattedSize)")
                    }
                    .font(.caption)
                    .foregroundColor(Theme.Colors.textSecondary)
                }

                Spacer()

                VStack(alignment: .trailing, spacing: 4) {
                    PnLBadge(value: position.unrealizedPnL)
                    Button {
                        closeSymbol = position.symbol
                        showConfirmClose = true
                    } label: {
                        Text("Close")
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(Theme.Colors.error)
                            .padding(.horizontal, 8)
                            .padding(.vertical, 4)
                            .background(Theme.Colors.error.opacity(0.15))
                            .cornerRadius(6)
                    }
                }
            }
        }
    }

    // MARK: - Orders

    private var ordersSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !viewModel.orders.isEmpty {
                Text("Pending Orders")
                    .font(.headline)
                    .foregroundColor(Theme.Colors.textPrimary)

                ForEach(viewModel.orders.filter { $0.isActive }) { order in
                    GlassCard(padding: 12) {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                HStack {
                                    Text(order.symbol)
                                        .font(.system(.subheadline, design: .monospaced))
                                        .foregroundColor(Theme.Colors.textPrimary)
                                    Text(order.side.displayName)
                                        .font(.caption2)
                                        .fontWeight(.bold)
                                        .foregroundColor(order.side == .buy ? Theme.Colors.success : Theme.Colors.error)
                                    Text(order.type?.displayName ?? "")
                                        .font(.caption2)
                                        .foregroundColor(Theme.Colors.textSecondary)
                                }
                                Text("Qty: \(order.formattedQuantity) @ \(order.formattedPrice)")
                                    .font(.caption)
                                    .foregroundColor(Theme.Colors.textSecondary)
                            }
                            Spacer()
                            Text(order.status.displayName)
                                .font(.caption2)
                                .fontWeight(.bold)
                                .foregroundColor(Theme.Colors.warning)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Theme.Colors.warning.opacity(0.15))
                                .cornerRadius(6)
                        }
                    }
                }
            }
        }
    }

    // MARK: - Weekly Stats

    private var weeklyStatsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Session Stats")
                .font(.headline)
                .foregroundColor(Theme.Colors.textPrimary)

            LazyVGrid(columns: [
                GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())
            ], spacing: 12) {
                InfoTile(
                    icon: "number",
                    title: "Trades",
                    value: "\(viewModel.weeklyStats.totalTrades)",
                    subtitle: nil,
                    color: Theme.Colors.accent,
                    term: nil,
                    numericValue: nil
                )
                InfoTile(
                    icon: "percent",
                    title: "Win Rate",
                    value: viewModel.weeklyStats.formattedWinRate,
                    subtitle: nil,
                    color: Theme.Colors.success,
                    term: .winRate,
                    numericValue: viewModel.weeklyStats.winRate
                )
                InfoTile(
                    icon: "dollarsign.circle",
                    title: "P&L",
                    value: viewModel.weeklyStats.formattedPnL,
                    subtitle: nil,
                    color: viewModel.weeklyStats.isProfit ? Theme.Colors.success : Theme.Colors.error,
                    term: nil,
                    numericValue: nil
                )
            }
        }
    }

    // MARK: - Trading Funnel Mini

    private func tradingFunnelMini(_ metrics: TradingMetrics) -> some View {
        GlassCard(padding: 14) {
            VStack(alignment: .leading, spacing: 10) {
                Text("Trading Funnel")
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundColor(Theme.Colors.textPrimary)

                HStack(spacing: 0) {
                    funnelStep("Signals", metrics.signalsDetected, Theme.Colors.accent)
                    Image(systemName: "chevron.right")
                        .font(.caption2)
                        .foregroundColor(Theme.Colors.textSecondary)
                    funnelStep("Orders", metrics.ordersCreated, .purple)
                    Image(systemName: "chevron.right")
                        .font(.caption2)
                        .foregroundColor(Theme.Colors.textSecondary)
                    funnelStep("Filled", metrics.ordersFilled, Theme.Colors.success)
                }

                if metrics.signalsBlocked > 0 {
                    HStack {
                        Image(systemName: "xmark.circle.fill")
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.error)
                        Text("\(metrics.signalsBlocked) blocked by risk")
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textSecondary)
                    }
                }
            }
        }
    }

    private func funnelStep(_ label: String, _ value: Int, _ color: Color) -> some View {
        VStack(spacing: 4) {
            Text("\(value)")
                .font(.system(size: 18, weight: .bold, design: .monospaced))
                .foregroundColor(color)
            Text(label)
                .font(.caption2)
                .foregroundColor(Theme.Colors.textSecondary)
        }
        .frame(maxWidth: .infinity)
    }
}

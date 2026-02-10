//
//  LiveView.swift
//  HEAN
//
//  Tab 1: Live market overview — price, physics, AI, balance of forces
//

import SwiftUI

struct LiveView: View {
    @EnvironmentObject var container: DIContainer
    @StateObject private var viewModel = LiveViewModel()
    @State private var refreshTimer: Timer?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    // Error banner
                    if let error = viewModel.error {
                        let engineDown = viewModel.engineState == "ERROR" || viewModel.engineState == "STOPPED"
                        backendErrorBanner(
                            message: error,
                            allDown: !viewModel.backendReachable,
                            engineDown: engineDown
                        )
                    }

                    if viewModel.isLoading && viewModel.physics == nil {
                        ProgressView(L.loadingLiveData)
                            .padding(.top, 40)
                    } else {
                        // Equity hero (main focus)
                        equityHeroCard

                        // P&L breakdown
                        pnlBreakdown

                        // AI Explanation
                        if let analysis = viewModel.brainAnalysis {
                            AIExplanationCard(
                                summary: analysis.summary,
                                regime: analysis.marketRegime
                            )
                        }

                        // Physics gauges
                        if let physics = viewModel.physics {
                            PhysicsInlineSection(physics: physics)

                            // Szilard profit
                            SzilardProfitCard(profit: physics.szilardProfit)
                        }

                        // Balance of forces
                        if let participants = viewModel.participants {
                            BalanceOfForcesCard(participants: participants)
                        }

                        // Risk indicator
                        riskIndicator

                        // Anomaly sections
                        if !viewModel.anomalies.isEmpty {
                            WhaleTradesSection(anomalies: viewModel.anomalies)
                            LiquidationsSection(anomalies: viewModel.anomalies)
                        }
                    }
                }
                .padding()
            }
            .background(Theme.Colors.background.ignoresSafeArea())
            .navigationTitle(L.live)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    NavigationLink {
                        MarketsView()
                    } label: {
                        Label(L.markets, systemImage: "chart.bar.doc.horizontal")
                            .labelStyle(.iconOnly)
                            .foregroundColor(Theme.Colors.accent)
                    }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    HStack(spacing: 8) {
                        StatusIndicator(status: viewModel.connectionStatus, showLabel: false)
                        Button {
                            Task { await viewModel.refresh() }
                        } label: {
                            Image(systemName: "arrow.clockwise")
                                .foregroundColor(Theme.Colors.accent)
                        }
                    }
                }
            }
            .onAppear { injectServices(); startRefresh() }
            .onDisappear { refreshTimer?.invalidate() }
            .refreshable { await viewModel.refresh() }
        }
    }

    // MARK: - Setup

    private func injectServices() {
        viewModel.configure(container: container)
    }

    private func startRefresh() {
        Task { await viewModel.refresh() }
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { _ in
            Task { @MainActor in await viewModel.refresh() }
        }
    }

    // MARK: - Equity Hero Card

    private var equityHeroCard: some View {
        GlassCard(padding: 16) {
            VStack(spacing: 12) {
                // Equity — the main number
                VStack(spacing: 4) {
                    Text(L.equity)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(Theme.Colors.textSecondary)
                        .textCase(.uppercase)
                        .tracking(1.2)

                    Text(viewModel.equity.asCurrency)
                        .font(.system(size: 36, weight: .bold, design: .rounded))
                        .foregroundColor(Theme.Colors.textPrimary)
                        .contentTransition(.numericText())
                        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: viewModel.equity)
                }

                // Total P&L
                HStack(spacing: 6) {
                    Image(systemName: viewModel.pnl.total >= 0 ? "arrow.up.right" : "arrow.down.right")
                        .font(.system(size: 14, weight: .bold))
                    Text(viewModel.pnl.total.asPnL)
                        .font(.system(size: 18, weight: .bold, design: .monospaced))
                    Text("(\(viewModel.pnl.percent.asPercent))")
                        .font(.system(size: 14, weight: .medium, design: .monospaced))
                }
                .foregroundColor(viewModel.pnl.total >= 0 ? Theme.Colors.success : Theme.Colors.error)

                // Divider
                Rectangle()
                    .fill(Theme.Colors.divider)
                    .frame(height: 1)

                // Bottom row: Initial Capital | Positions | Market Price
                HStack(spacing: 0) {
                    // Initial Capital
                    VStack(spacing: 2) {
                        Text(L.totalCapital)
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textTertiary)
                        Text(viewModel.initialCapital.asCurrency)
                            .font(.system(.caption, design: .monospaced))
                            .fontWeight(.semibold)
                            .foregroundColor(Theme.Colors.textSecondary)
                    }
                    .frame(maxWidth: .infinity)

                    // Vertical divider
                    Rectangle()
                        .fill(Theme.Colors.divider)
                        .frame(width: 1, height: 28)

                    // Positions
                    VStack(spacing: 2) {
                        Text(L.positions)
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textTertiary)
                        Text("\(viewModel.positionCount)")
                            .font(.system(.caption, design: .monospaced))
                            .fontWeight(.semibold)
                            .foregroundColor(Theme.Colors.textSecondary)
                    }
                    .frame(maxWidth: .infinity)

                    // Vertical divider
                    Rectangle()
                        .fill(Theme.Colors.divider)
                        .frame(width: 1, height: 28)

                    // Market price (compact)
                    VStack(spacing: 2) {
                        Text(viewModel.symbol)
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textTertiary)
                        HStack(spacing: 3) {
                            Text(viewModel.marketPrice.asCompactCurrency)
                                .font(.system(.caption, design: .monospaced))
                                .fontWeight(.semibold)
                                .foregroundColor(Theme.Colors.textSecondary)
                            Text(viewModel.priceChange24h >= 0 ? "▲" : "▼")
                                .font(.system(size: 8))
                                .foregroundColor(viewModel.priceChange24h >= 0 ? Theme.Colors.success : Theme.Colors.error)
                        }
                    }
                    .frame(maxWidth: .infinity)
                }
            }
        }
    }

    // MARK: - P&L Breakdown

    private var pnlBreakdown: some View {
        HStack(spacing: 12) {
            // Unrealized P&L
            GlassCard(padding: 12) {
                VStack(spacing: 4) {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(Theme.Colors.warning)
                            .frame(width: 6, height: 6)
                        Text(L.unrealizedPnL)
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textTertiary)
                    }
                    Text(viewModel.pnl.unrealized.asPnL)
                        .font(.system(.subheadline, design: .monospaced))
                        .fontWeight(.bold)
                        .foregroundColor(viewModel.pnl.unrealized >= 0 ? Theme.Colors.success : Theme.Colors.error)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                }
                .frame(maxWidth: .infinity)
            }

            // Realized P&L
            GlassCard(padding: 12) {
                VStack(spacing: 4) {
                    HStack(spacing: 4) {
                        Circle()
                            .fill(Theme.Colors.accent)
                            .frame(width: 6, height: 6)
                        Text(L.realizedPnL)
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textTertiary)
                    }
                    Text(viewModel.pnl.realized.asPnL)
                        .font(.system(.subheadline, design: .monospaced))
                        .fontWeight(.bold)
                        .foregroundColor(viewModel.pnl.realized >= 0 ? Theme.Colors.success : Theme.Colors.error)
                        .lineLimit(1)
                        .minimumScaleFactor(0.7)
                }
                .frame(maxWidth: .infinity)
            }
        }
    }

    // MARK: - Risk Indicator

    private var riskIndicator: some View {
        GlassCard(padding: 14) {
            HStack(spacing: 12) {
                Image(systemName: viewModel.riskState.icon)
                    .font(.title2)
                    .foregroundColor(Color(hex: viewModel.riskState.colorHex))
                    .symbolEffect(.pulse, isActive: viewModel.riskState != .normal)

                VStack(alignment: .leading, spacing: 2) {
                    Text(viewModel.riskState.displayName)
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(Color(hex: viewModel.riskState.colorHex))
                    Text(viewModel.riskState.description)
                        .font(.caption)
                        .foregroundColor(Theme.Colors.textSecondary)
                }

                Spacer()

                Circle()
                    .fill(Color(hex: viewModel.riskState.colorHex))
                    .frame(width: 10, height: 10)
            }
        }
        .explainable(.riskState, value: viewModel.riskState == .normal ? 0.2 : 0.8)
    }

    // MARK: - Error Banner

    private func backendErrorBanner(message: String, allDown: Bool, engineDown: Bool = false) -> some View {
        let icon = allDown ? "wifi.slash" : engineDown ? "bolt.slash.fill" : "exclamationmark.triangle.fill"
        let title = allDown ? "Backend Unreachable" : engineDown ? "Engine Stopped" : "Partial Connectivity"
        let subtitle = allDown ? "Check that the API server is running" : message
        let color = allDown ? Theme.Colors.error : engineDown ? .orange : Theme.Colors.warning

        return HStack(spacing: 10) {
            Image(systemName: icon)
                .font(.subheadline)
                .foregroundColor(color)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(color)
                Text(subtitle)
                    .font(.caption2)
                    .foregroundColor(Theme.Colors.textSecondary)
            }

            Spacer()

            Button {
                Task { await viewModel.refresh() }
            } label: {
                Image(systemName: "arrow.clockwise")
                    .font(.caption)
                    .foregroundColor(Theme.Colors.accent)
            }
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 10)
                .fill(color.opacity(0.1))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .strokeBorder(color.opacity(0.3), lineWidth: 1)
        )
    }

}

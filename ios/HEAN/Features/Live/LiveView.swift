//
//  LiveView.swift
//  HEAN
//
//  Tab 1: Live trading overview — account balance, P&L, risk, physics, AI
//

import SwiftUI

struct LiveView: View {
    @EnvironmentObject var container: DIContainer
    @StateObject private var viewModel = LiveViewModel()
    @State private var refreshTimer: Timer?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {
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
                        // 1. Account balance — THE main card
                        accountHeroCard

                        // 2. P&L breakdown row
                        pnlRow

                        // 3. Risk state
                        riskIndicator

                        // 4. AI Explanation
                        if let analysis = viewModel.brainAnalysis {
                            AIExplanationCard(
                                summary: analysis.summary,
                                regime: analysis.marketRegime
                            )
                        }

                        // 5. Physics (compact single card)
                        if let physics = viewModel.physics {
                            PhysicsCompactCard(physics: physics)

                            SzilardProfitCard(profit: physics.szilardProfit)
                        }

                        // 6. Balance of forces
                        if let participants = viewModel.participants {
                            BalanceOfForcesCard(participants: participants)
                        }

                        // 7. Anomalies
                        if !viewModel.anomalies.isEmpty {
                            WhaleTradesSection(anomalies: viewModel.anomalies)
                            LiquidationsSection(anomalies: viewModel.anomalies)
                        }
                    }
                }
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
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

    // MARK: - Account Hero Card

    private var accountHeroCard: some View {
        GlassCard(padding: 0) {
            VStack(spacing: 0) {
                // Main equity section
                VStack(spacing: 8) {
                    Text(L.equity)
                        .font(.system(size: 13, weight: .medium))
                        .foregroundColor(Theme.Colors.textSecondary)
                        .textCase(.uppercase)
                        .tracking(1.5)

                    Text(viewModel.equity.asCurrency)
                        .font(.system(size: 44, weight: .bold, design: .rounded))
                        .foregroundColor(Theme.Colors.textPrimary)
                        .contentTransition(.numericText())
                        .animation(.spring(response: 0.3, dampingFraction: 0.7), value: viewModel.equity)
                        .lineLimit(1)
                        .minimumScaleFactor(0.5)

                    // P&L badge
                    HStack(spacing: 6) {
                        Image(systemName: viewModel.pnl.total >= 0 ? "arrow.up.right" : "arrow.down.right")
                            .font(.system(size: 14, weight: .bold))
                        Text(viewModel.pnl.total.asPnL)
                            .font(.system(size: 20, weight: .bold, design: .monospaced))
                        Text(viewModel.pnl.percent.asPercent)
                            .font(.system(size: 14, weight: .semibold, design: .monospaced))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(
                                (viewModel.pnl.total >= 0 ? Theme.Colors.success : Theme.Colors.error)
                                    .opacity(0.15)
                            )
                            .cornerRadius(4)
                    }
                    .foregroundColor(viewModel.pnl.total >= 0 ? Theme.Colors.success : Theme.Colors.error)
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
                }
                .padding(.horizontal, 20)
                .padding(.top, 20)
                .padding(.bottom, 16)

                // Divider
                Rectangle()
                    .fill(Color.white.opacity(0.08))
                    .frame(height: 1)

                // Stats row
                HStack(spacing: 0) {
                    compactStat(label: L.totalCapital, value: viewModel.initialCapital.asCurrency)

                    Rectangle().fill(Color.white.opacity(0.08)).frame(width: 1, height: 32)

                    compactStat(label: L.positions, value: "\(viewModel.positionCount)")

                    Rectangle().fill(Color.white.opacity(0.08)).frame(width: 1, height: 32)

                    compactStat(
                        label: viewModel.symbol,
                        value: viewModel.marketPrice.asCompactCurrency,
                        valueColor: viewModel.priceChange24h >= 0 ? Theme.Colors.success : Theme.Colors.error
                    )
                }
                .padding(.vertical, 12)
            }
        }
    }

    private func compactStat(
        label: String,
        value: String,
        valueColor: Color = Theme.Colors.textPrimary
    ) -> some View {
        VStack(spacing: 3) {
            Text(label)
                .font(.system(size: 10, weight: .medium))
                .foregroundColor(Theme.Colors.textTertiary)
                .lineLimit(1)
                .minimumScaleFactor(0.7)
            Text(value)
                .font(.system(size: 13, weight: .bold, design: .monospaced))
                .foregroundColor(valueColor)
                .lineLimit(1)
                .minimumScaleFactor(0.6)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - P&L Row

    private var pnlRow: some View {
        HStack(spacing: 10) {
            pnlCard(
                label: L.unrealizedPnL,
                value: viewModel.pnl.unrealized,
                dotColor: Theme.Colors.warning
            )
            pnlCard(
                label: L.realizedPnL,
                value: viewModel.pnl.realized,
                dotColor: Theme.Colors.accent
            )
        }
    }

    private func pnlCard(label: String, value: Double, dotColor: Color) -> some View {
        GlassCard(padding: 12) {
            VStack(spacing: 6) {
                HStack(spacing: 4) {
                    Circle().fill(dotColor).frame(width: 5, height: 5)
                    Text(label)
                        .font(.system(size: 11, weight: .medium))
                        .foregroundColor(Theme.Colors.textTertiary)
                }
                Text(value.asPnL)
                    .font(.system(size: 17, weight: .bold, design: .monospaced))
                    .foregroundColor(value >= 0 ? Theme.Colors.success : Theme.Colors.error)
                    .lineLimit(1)
                    .minimumScaleFactor(0.6)
            }
            .frame(maxWidth: .infinity)
        }
    }

    // MARK: - Risk Indicator

    private var riskIndicator: some View {
        GlassCard(padding: 14) {
            HStack(spacing: 12) {
                Image(systemName: viewModel.riskState.icon)
                    .font(.title3)
                    .foregroundColor(Color(hex: viewModel.riskState.colorHex))
                    .symbolEffect(.pulse, isActive: viewModel.riskState != .normal)

                VStack(alignment: .leading, spacing: 2) {
                    Text(viewModel.riskState.displayName)
                        .font(.system(size: 15, weight: .bold))
                        .foregroundColor(Color(hex: viewModel.riskState.colorHex))
                    Text(viewModel.riskState.description)
                        .font(.system(size: 12))
                        .foregroundColor(Theme.Colors.textSecondary)
                        .lineLimit(1)
                }

                Spacer()

                Circle()
                    .fill(Color(hex: viewModel.riskState.colorHex))
                    .frame(width: 8, height: 8)
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

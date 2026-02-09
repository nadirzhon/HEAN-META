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
                    if viewModel.isLoading && viewModel.physics == nil {
                        ProgressView("Loading live data...")
                            .padding(.top, 40)
                    } else {
                        // Price header
                        priceHeader

                        // Portfolio summary
                        portfolioSummary

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
            .navigationTitle("Live")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    NavigationLink {
                        MarketsView()
                    } label: {
                        Label("Markets", systemImage: "chart.bar.doc.horizontal")
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
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { _ in
            Task { @MainActor in await viewModel.refresh() }
        }
    }

    // MARK: - Price Header

    private var priceHeader: some View {
        GlassCard(padding: 16) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("BTCUSDT")
                        .font(.system(.headline, design: .monospaced))
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    Text(viewModel.priceChange24h >= 0 ? "▲" : "▼")
                        .foregroundColor(viewModel.priceChange24h >= 0 ? Theme.Colors.success : Theme.Colors.error)
                    Text(String(format: "%.2f%%", viewModel.priceChange24h))
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .foregroundColor(viewModel.priceChange24h >= 0 ? Theme.Colors.success : Theme.Colors.error)
                        .monospacedDigit()
                }

                Text(formatPrice(viewModel.marketPrice))
                    .font(.system(size: 34, weight: .bold, design: .rounded))
                    .foregroundColor(Theme.Colors.textPrimary)
                    .contentTransition(.numericText())
                    .animation(.spring(response: 0.3, dampingFraction: 0.7), value: viewModel.marketPrice)
            }
        }
    }

    // MARK: - Portfolio Summary

    private var portfolioSummary: some View {
        HStack(spacing: 12) {
            InfoTile(
                icon: "banknote",
                title: "Equity",
                value: viewModel.equity.asCurrency,
                subtitle: nil,
                color: Theme.Colors.accent,
                term: nil,
                numericValue: nil
            )

            InfoTile(
                icon: "chart.line.uptrend.xyaxis",
                title: "P&L",
                value: viewModel.pnl.total.asPnL,
                subtitle: viewModel.pnl.percent.asPercent,
                color: viewModel.pnl.total >= 0 ? Theme.Colors.success : Theme.Colors.error,
                term: nil,
                numericValue: nil
            )

            InfoTile(
                icon: "chart.bar.doc.horizontal",
                title: "Positions",
                value: "\(viewModel.positionCount)",
                subtitle: nil,
                color: .blue,
                term: nil,
                numericValue: nil
            )
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

    // MARK: - Helpers

    private func formatPrice(_ price: Double) -> String {
        if price >= 1000 {
            return String(format: "$%.2f", price)
        } else if price >= 1 {
            return String(format: "$%.3f", price)
        } else {
            return String(format: "$%.5f", price)
        }
    }
}

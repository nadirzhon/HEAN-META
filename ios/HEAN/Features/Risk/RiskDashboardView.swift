//
//  RiskDashboardView.swift
//  HEAN
//
//  Risk state machine visualization with killswitch controls
//

import SwiftUI

struct RiskDashboardView: View {
    @StateObject var viewModel: RiskViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    // Risk State Hero
                    riskStateHero

                    // State Machine Visualization
                    stateMachineView

                    // Drawdown Progress
                    drawdownCard

                    // Metrics
                    metricsGrid

                    // Quarantined Symbols
                    if let metrics = viewModel.metrics, !metrics.quarantinedSymbols.isEmpty {
                        quarantineCard(metrics.quarantinedSymbols)
                    }

                    // Killswitch Control
                    killswitchCard

                    // Actions
                    actionsSection
                }
                .padding()
            }
            .background(Theme.Colors.background.ignoresSafeArea())
            .navigationTitle("Risk")
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.refresh() }
        }
    }

    // MARK: - Risk State Hero

    private var riskStateHero: some View {
        GlassCard {
            VStack(spacing: 16) {
                Image(systemName: viewModel.riskState.icon)
                    .font(.system(size: 50))
                    .foregroundColor(Color(hex: viewModel.riskState.colorHex))
                    .symbolEffect(.pulse, isActive: viewModel.riskState != .normal)

                Text(viewModel.riskState.displayName)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundColor(Color(hex: viewModel.riskState.colorHex))

                Text(viewModel.riskState.description)
                    .font(.subheadline)
                    .foregroundColor(.gray)
            }
            .frame(maxWidth: .infinity)
            .padding(24)
        }
    }

    // MARK: - State Machine

    private var stateMachineView: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                Text(L.riskProgression).font(.headline).foregroundColor(.white)

                HStack(spacing: 4) {
                    ForEach(RiskState.allCases, id: \.self) { state in
                        VStack(spacing: 4) {
                            Circle()
                                .fill(state.severity <= viewModel.riskState.severity
                                      ? Color(hex: state.colorHex)
                                      : Color.gray.opacity(0.3))
                                .frame(width: 20, height: 20)
                                .overlay(
                                    state == viewModel.riskState
                                        ? Circle().stroke(Color.white, lineWidth: 2).frame(width: 26, height: 26)
                                        : nil
                                )

                            Text(state.rawValue.replacingOccurrences(of: "_", with: "\n"))
                                .font(.system(size: 8, weight: .bold))
                                .foregroundColor(state.severity <= viewModel.riskState.severity
                                                 ? Color(hex: state.colorHex)
                                                 : .gray)
                                .multilineTextAlignment(.center)
                        }
                        .frame(maxWidth: .infinity)

                        if state != .hardStop {
                            Image(systemName: "chevron.right")
                                .font(.caption2)
                                .foregroundColor(.gray.opacity(0.5))
                        }
                    }
                }
            }
            .padding()
        }
    }

    // MARK: - Drawdown

    private var drawdownCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                Text(L.drawdown).font(.headline).foregroundColor(.white)

                if let metrics = viewModel.metrics {
                    // Progress bar
                    GeometryReader { geo in
                        ZStack(alignment: .leading) {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.gray.opacity(0.2))

                            RoundedRectangle(cornerRadius: 6)
                                .fill(drawdownColor(metrics.drawdownPercent))
                                .frame(width: geo.size.width * min(metrics.drawdownPercent / 100, 1))

                            // Warning threshold marker
                            Rectangle()
                                .fill(Theme.Colors.warning)
                                .frame(width: 2)
                                .offset(x: geo.size.width * (metrics.warningThreshold / 100))

                            // Critical threshold marker
                            Rectangle()
                                .fill(Theme.Colors.error)
                                .frame(width: 2)
                                .offset(x: geo.size.width * (metrics.criticalThreshold / 100))
                        }
                    }
                    .frame(height: 12)

                    HStack {
                        Text("Current: \(String(format: "%.2f%%", metrics.drawdownPercent))")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(drawdownColor(metrics.drawdownPercent))
                        Spacer()
                        Text("Max: \(String(format: "%.2f%%", metrics.maxDrawdown))")
                            .font(.system(.caption, design: .monospaced))
                            .foregroundColor(.gray)
                    }

                    HStack {
                        Label("Warning: \(String(format: "%.0f%%", metrics.warningThreshold))", systemImage: "exclamationmark.triangle")
                            .font(.caption2).foregroundColor(Theme.Colors.warning)
                        Spacer()
                        Label("Critical: \(String(format: "%.0f%%", metrics.criticalThreshold))", systemImage: "xmark.octagon")
                            .font(.caption2).foregroundColor(Theme.Colors.error)
                    }
                }
            }
            .padding()
        }
    }

    private func drawdownColor(_ percent: Double) -> Color {
        if percent >= 20 { return Theme.Colors.error }
        if percent >= 10 { return Theme.Colors.warning }
        return Theme.Colors.success
    }

    // MARK: - Metrics Grid

    private var metricsGrid: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
            riskMetric("Active Positions", "\(viewModel.metrics?.activePositions ?? 0)", "chart.bar.doc.horizontal", Theme.Colors.info)
            riskMetric("Exposure", viewModel.metrics?.totalExposure.asCurrency ?? "$0.00", "banknote", Theme.Colors.purple)
            riskMetric("Available Margin", viewModel.metrics?.availableMargin.asCurrency ?? "$0.00", "creditcard", Theme.Colors.success)
            riskMetric("Drawdown", viewModel.metrics?.drawdown.asCurrency ?? "$0.00", "arrow.down.circle", Theme.Colors.error)
        }
    }

    private func riskMetric(_ label: String, _ value: String, _ icon: String, _ color: Color) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundColor(color)
                Text(value)
                    .font(.system(size: 16, weight: .bold, design: .monospaced))
                    .foregroundColor(.white)
                Text(label)
                    .font(.caption2)
                    .foregroundColor(.gray)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(12)
        }
    }

    // MARK: - Quarantine

    private func quarantineCard(_ symbols: [String]) -> some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 8) {
                Label("Quarantined Symbols", systemImage: "lock.shield")
                    .font(.headline)
                    .foregroundColor(Theme.Colors.orange)

                FlowLayout(spacing: 8) {
                    ForEach(symbols, id: \.self) { symbol in
                        Text(symbol)
                            .font(.caption.bold())
                            .foregroundColor(Theme.Colors.orange)
                            .padding(.horizontal, 10).padding(.vertical, 6)
                            .background(Theme.Colors.orange.opacity(0.15))
                            .cornerRadius(8)
                    }
                }
            }
            .padding()
        }
    }

    // MARK: - Killswitch

    private var killswitchCard: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Image(systemName: "power")
                        .foregroundColor(viewModel.killswitchTriggered ? Theme.Colors.error : Theme.Colors.success)
                    Text(L.killSwitch)
                        .font(.headline)
                        .foregroundColor(.white)
                    Spacer()
                    Text(viewModel.killswitchTriggered ? "TRIGGERED" : "ARMED")
                        .font(.caption.bold())
                        .foregroundColor(viewModel.killswitchTriggered ? Theme.Colors.error : Theme.Colors.success)
                        .padding(.horizontal, 8).padding(.vertical, 4)
                        .background((viewModel.killswitchTriggered ? Theme.Colors.error : Theme.Colors.success).opacity(0.15))
                        .cornerRadius(6)
                }

                if viewModel.killswitchTriggered {
                    Text(L.tradingHalted)
                        .font(.caption)
                        .foregroundColor(Theme.Colors.error.opacity(0.8))
                }
            }
            .padding()
        }
    }

    // MARK: - Actions

    private var actionsSection: some View {
        VStack(spacing: 12) {
            if viewModel.killswitchTriggered {
                Button {
                    Task { await viewModel.resetKillswitch() }
                } label: {
                    Label("Reset Kill Switch", systemImage: "arrow.counterclockwise")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Theme.Colors.warning)
                        .cornerRadius(12)
                }
            }

            Button {
                Task { await viewModel.clearRiskBlocks() }
            } label: {
                Label("Clear Risk Blocks", systemImage: "xmark.circle")
                    .font(.subheadline)
                    .foregroundColor(Theme.Colors.accent)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(Theme.Colors.accent.opacity(0.1))
                    .cornerRadius(12)
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Theme.Colors.accent.opacity(0.3), lineWidth: 0.5)
                    )
            }
        }
    }
}

// MARK: - Risk ViewModel

@MainActor
final class RiskViewModel: ObservableObject {
    @Published var riskState: RiskState = .normal
    @Published var metrics: RiskMetrics?
    @Published var killswitchTriggered = false
    @Published var isLoading = false
    @Published var error: String?

    private let riskService: RiskServiceProtocol

    init(riskService: RiskServiceProtocol) {
        self.riskService = riskService
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }

        do {
            let status = try await riskService.fetchRiskStatus()
            riskState = status.state
            metrics = status.metrics
            killswitchTriggered = status.killswitchTriggered
        } catch {
            self.error = error.localizedDescription
        }
    }

    func resetKillswitch() async {
        do {
            try await riskService.resetKillswitch()
            killswitchTriggered = false
            await refresh()
        } catch {
            self.error = error.localizedDescription
        }
    }

    func clearRiskBlocks() async {
        do {
            try await riskService.clearRiskBlocks()
            await refresh()
        } catch {
            self.error = error.localizedDescription
        }
    }
}

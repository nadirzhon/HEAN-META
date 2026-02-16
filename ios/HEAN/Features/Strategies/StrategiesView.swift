//
//  StrategiesView.swift
//  HEAN
//
//  Strategy control panel with live enable/disable and performance metrics
//

import SwiftUI

struct StrategiesView: View {
    @StateObject var viewModel: StrategiesViewModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    summaryCard

                    ForEach(viewModel.strategies) { strategy in
                        StrategyCardView(strategy: strategy) { enabled in
                            Task { await viewModel.toggleStrategy(id: strategy.id, enabled: enabled) }
                        }
                    }

                    if viewModel.strategies.isEmpty && !viewModel.isLoading {
                        emptyState
                    }
                }
                .padding()
            }
            .background(Theme.Colors.background.ignoresSafeArea())
            .navigationTitle("Strategies")
            .refreshable { await viewModel.refresh() }
            .task { await viewModel.refresh() }
        }
    }

    private var summaryCard: some View {
        GlassCard {
            HStack(spacing: 20) {
                VStack {
                    Text("\(viewModel.strategies.filter(\.enabled).count)")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(Theme.Colors.success)
                    Text(L.active).font(.caption).foregroundColor(.gray)
                }
                VStack {
                    Text("\(viewModel.strategies.count)")
                        .font(.system(size: 28, weight: .bold, design: .rounded))
                        .foregroundColor(.white)
                    Text(L.total).font(.caption).foregroundColor(.gray)
                }
                Spacer()
                VStack(alignment: .trailing) {
                    let totalPnl = viewModel.strategies.reduce(0) { $0 + $1.totalPnl }
                    Text(totalPnl.asPnL)
                        .font(.system(size: 20, weight: .bold, design: .monospaced))
                        .foregroundColor(totalPnl >= 0 ? Theme.Colors.success : Theme.Colors.error)
                    Text(L.combinedPnL).font(.caption).foregroundColor(.gray)
                }
            }
            .padding()
        }
    }

    private var emptyState: some View {
        GlassCard {
            VStack(spacing: 12) {
                Image(systemName: "brain.head.profile").font(.system(size: 40)).foregroundColor(.gray)
                Text(L.noStrategiesLoaded).font(.headline).foregroundColor(.gray)
                Text(L.startEngineToLoad).font(.caption).foregroundColor(.gray.opacity(0.7))
            }
            .padding(30).frame(maxWidth: .infinity)
        }
    }
}

// MARK: - Strategy Card

struct StrategyCardView: View {
    let strategy: Strategy
    let onToggle: (Bool) -> Void

    var body: some View {
        GlassCard {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(strategy.name).font(.headline).foregroundColor(.white)
                        if !strategy.description.isEmpty {
                            Text(strategy.description).font(.caption).foregroundColor(.gray)
                        }
                    }
                    Spacer()
                    Toggle("", isOn: Binding(get: { strategy.enabled }, set: { onToggle($0) }))
                        .labelsHidden().tint(Theme.Colors.accent)
                }

                HStack(spacing: 16) {
                    stratMetric("P&L", strategy.totalPnl.asPnL, strategy.totalPnl >= 0 ? Theme.Colors.success : Theme.Colors.error)
                    stratMetric("Win Rate", strategy.winRate.asPercent, Theme.Colors.info)
                    stratMetric("Trades", "\(strategy.totalTrades)", Theme.Colors.warning)
                    stratMetric("PF", String(format: "%.2f", strategy.profitFactor), Theme.Colors.purple)
                }

                HStack(spacing: 12) {
                    HStack(spacing: 4) {
                        Text("W: \(strategy.wins)").font(.caption2).foregroundColor(Theme.Colors.success)
                        Text("L: \(strategy.losses)").font(.caption2).foregroundColor(Theme.Colors.error)
                    }
                }
            }
            .padding()
        }
    }

    private func stratMetric(_ label: String, _ value: String, _ color: Color) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundColor(.gray)
            Text(value).font(.subheadline.weight(.semibold)).monospacedDigit()
                .foregroundColor(color)
        }
    }
}

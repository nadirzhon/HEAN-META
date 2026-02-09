//
//  MindView.swift
//  HEAN
//
//  AI Mind — rich thought feed with explanations
//

import SwiftUI

struct MindView: View {
    @EnvironmentObject var container: DIContainer
    @StateObject private var viewModel = MindViewModel()
    @State private var refreshTimer: Timer?

    var body: some View {
        NavigationStack {
            ScrollView {
                LazyVStack(spacing: 16) {
                    if viewModel.isLoading && viewModel.analysis == nil {
                        ProgressView("Thinking...")
                            .padding(.top, 40)
                    } else {
                        // Latest analysis summary
                        if let analysis = viewModel.analysis {
                            latestAnalysisCard(analysis)
                        }

                        // Thought feed
                        ForEach(viewModel.thoughtHistory) { thought in
                            thoughtCard(thought)
                        }

                        if viewModel.thoughtHistory.isEmpty && !viewModel.isLoading {
                            VStack(spacing: 16) {
                                Image(systemName: "brain")
                                    .font(.system(size: 48))
                                    .foregroundColor(.gray)
                                Text("No thoughts yet")
                                    .foregroundColor(.secondary)
                                Text("AI will start analyzing when market data arrives")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                            .padding(.top, 20)
                        }
                    }
                }
                .padding()
            }
            .background(Theme.Colors.background.ignoresSafeArea())
            .navigationTitle("Mind")
            .onAppear { injectServices(); startRefresh() }
            .onDisappear { refreshTimer?.invalidate() }
            .refreshable { await viewModel.refresh() }
        }
    }

    // MARK: - Setup

    private func injectServices() {
        viewModel.configure(apiClient: container.apiClient)
    }

    private func startRefresh() {
        Task { await viewModel.refresh() }
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { _ in
            Task { @MainActor in await viewModel.refresh() }
        }
    }

    // MARK: - Latest Analysis

    private func latestAnalysisCard(_ analysis: BrainAnalysis) -> some View {
        GlassCardAccent {
            VStack(alignment: .leading, spacing: 12) {
                // Header
                HStack {
                    Image(systemName: "brain.head.profile")
                        .foregroundColor(Theme.Colors.accent)
                    Text("AI Analysis")
                        .font(.headline)
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    regimeBadge(analysis.marketRegime)
                }

                // Summary
                if !analysis.summary.isEmpty {
                    Text(analysis.summary)
                        .font(.subheadline)
                        .foregroundColor(Theme.Colors.textPrimary)
                        .fixedSize(horizontal: false, vertical: true)
                }

                // Forces
                if !analysis.forces.isEmpty {
                    VStack(spacing: 6) {
                        HStack {
                            Text("FORCES")
                                .font(.caption2)
                                .fontWeight(.bold)
                                .foregroundColor(Theme.Colors.textSecondary)
                            Spacer()
                        }
                        ForEach(analysis.forces) { force in
                            forceBar(force)
                        }
                    }
                }

                // Signal
                if let signal = analysis.signal, !signal.isNeutral {
                    signalBadge(signal)
                }
            }
            .padding()
        }
    }

    // MARK: - Thought Card

    private func thoughtCard(_ thought: BrainThought) -> some View {
        GlassCard(padding: 14) {
            VStack(alignment: .leading, spacing: 8) {
                // Header: timestamp + stage
                HStack {
                    Image(systemName: thought.stageIcon)
                        .font(.system(size: 14))
                        .foregroundColor(stageColor(thought.stage))
                    Text(thought.stageDisplayName)
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(stageColor(thought.stage))
                    Spacer()
                    Text(formatTimestamp(thought.timestamp))
                        .font(.caption2)
                        .foregroundColor(Theme.Colors.textSecondary)
                        .monospacedDigit()
                }

                // Content
                Text(thought.content)
                    .font(.subheadline)
                    .foregroundColor(Theme.Colors.textPrimary)
                    .fixedSize(horizontal: false, vertical: true)

                // Confidence
                if let confidence = thought.confidence {
                    HStack {
                        Text("Confidence")
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textSecondary)
                        ProgressView(value: confidence)
                            .tint(confidenceColor(confidence))
                        Text("\(Int(confidence * 100))%")
                            .font(.caption2)
                            .fontWeight(.bold)
                            .foregroundColor(confidenceColor(confidence))
                            .monospacedDigit()
                    }
                }
            }
        }
    }

    // MARK: - Helpers

    private func forceBar(_ force: BrainForce) -> some View {
        HStack(spacing: 8) {
            Text(force.value >= 0 ? "▲" : "▼")
                .font(.caption2)
                .foregroundColor(force.value >= 0 ? Theme.Colors.success : Theme.Colors.error)

            Text(force.name)
                .font(.caption)
                .foregroundColor(Theme.Colors.textPrimary)
                .frame(width: 100, alignment: .leading)

            ProgressView(value: abs(force.value), total: 1.0)
                .tint(force.value >= 0 ? Theme.Colors.success : Theme.Colors.error)

            Text(force.label)
                .font(.caption2)
                .foregroundColor(Theme.Colors.textSecondary)
                .frame(width: 70, alignment: .trailing)
        }
    }

    private func signalBadge(_ signal: BrainSignal) -> some View {
        HStack {
            Image(systemName: signal.isLong ? "arrow.up.circle.fill" : "arrow.down.circle.fill")
                .foregroundColor(Color(hex: signal.directionColor))
            Text(signal.direction.uppercased())
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(Color(hex: signal.directionColor))
            Text("\(Int(signal.confidence * 100))%")
                .font(.caption)
                .foregroundColor(Theme.Colors.textSecondary)
            Spacer()
            if let explanation = signal.explanation {
                Text(explanation)
                    .font(.caption2)
                    .foregroundColor(Theme.Colors.textSecondary)
                    .lineLimit(1)
            }
        }
        .padding(8)
        .background(Color(hex: signal.directionColor).opacity(0.1))
        .cornerRadius(8)
    }

    private func regimeBadge(_ regime: String) -> some View {
        Text(regime.uppercased())
            .font(.caption2)
            .fontWeight(.bold)
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(regimeColor(regime).opacity(0.2))
            .foregroundColor(regimeColor(regime))
            .cornerRadius(6)
    }

    private func stageColor(_ stage: String) -> Color {
        switch stage.lowercased() {
        case "anomaly": return .red
        case "physics": return .orange
        case "xray": return .purple
        case "decision": return Theme.Colors.accent
        default: return .gray
        }
    }

    private func regimeColor(_ regime: String) -> Color {
        switch regime.lowercased() {
        case "trending", "bullish": return Theme.Colors.success
        case "ranging", "neutral": return Theme.Colors.warning
        case "volatile", "bearish": return Theme.Colors.error
        default: return .gray
        }
    }

    private func confidenceColor(_ value: Double) -> Color {
        if value < 0.4 { return Theme.Colors.error }
        if value < 0.7 { return Theme.Colors.warning }
        return Theme.Colors.success
    }

    private func formatTimestamp(_ ts: String) -> String {
        // Extract HH:MM:SS from ISO timestamp
        if ts.count > 18 {
            let start = ts.index(ts.startIndex, offsetBy: 11)
            let end = ts.index(start, offsetBy: 8, limitedBy: ts.endIndex) ?? ts.endIndex
            return String(ts[start..<end])
        }
        return ts
    }
}

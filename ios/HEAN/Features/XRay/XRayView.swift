//
//  XRayView.swift
//  HEAN
//
//  Market participant X-Ray with educational info buttons
//

import SwiftUI

struct XRayView: View {
    @EnvironmentObject var container: DIContainer
    @StateObject private var viewModel = XRayViewModel()
    @State private var refreshTimer: Timer?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    if viewModel.isLoading && viewModel.participants == nil {
                        ProgressView(L.loadingXRay)
                            .padding(.top, 40)
                    } else if let bd = viewModel.participants {
                        // Meta signal banner
                        metaSignalBanner(bd.metaSignal)

                        // Dominant player
                        dominantPlayerCard(bd.dominantPlayer)

                        // Section header
                        sectionHeader(L.participants)

                        // Participant cards with info buttons
                        participantCard(
                            icon: "building.2.fill", title: L.marketMakers,
                            value: bd.mmActivityPercent, color: .blue,
                            description: L.mmDesc,
                            term: .mmActivity
                        )
                        participantCard(
                            icon: "briefcase.fill", title: L.institutional,
                            value: bd.institutionalFlowPercent, color: .purple,
                            description: L.instDesc,
                            term: .institutionalFlow
                        )
                        participantCard(
                            icon: "person.3.fill", title: L.retail,
                            value: bd.retailSentimentPercent, color: .green,
                            description: L.retailDesc,
                            term: .retailSentiment
                        )
                        participantCard(
                            icon: "fish.fill", title: L.whales,
                            value: bd.whaleActivityPercent, color: .orange,
                            description: L.whaleDesc,
                            term: .whaleActivity
                        )
                        participantCard(
                            icon: "cpu", title: L.arbBots,
                            value: bd.arbPressurePercent, color: .cyan,
                            description: L.arbDesc,
                            term: .arbPressure
                        )

                        // Anomalies section
                        if !viewModel.anomalies.isEmpty {
                            sectionHeader(L.anomalies)
                            ForEach(viewModel.anomalies) { anomaly in
                                anomalyCard(anomaly)
                            }
                        }
                    } else {
                        VStack(spacing: 16) {
                            Image(systemName: "eye.fill")
                                .font(.system(size: 48))
                                .foregroundColor(.gray)
                            Text(L.noParticipantData)
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 40)
                    }
                }
                .padding()
            }
            .background(Theme.Colors.background.ignoresSafeArea())
            .navigationTitle(L.xray)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink {
                        SignalFeedView(viewModel: SignalFeedViewModel(
                            signalService: container.signalService
                        ))
                    } label: {
                        Label("Signals", systemImage: "waveform")
                            .labelStyle(.iconOnly)
                            .foregroundColor(Theme.Colors.accent)
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
        viewModel.configure(apiClient: container.apiClient)
    }

    private func startRefresh() {
        Task { await viewModel.refresh() }
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: 5, repeats: true) { _ in
            Task { @MainActor in await viewModel.refresh() }
        }
    }

    // MARK: - Sub-Views

    private func metaSignalBanner(_ signal: String) -> some View {
        GlassCardAccent {
            HStack {
                Image(systemName: "brain.head.profile")
                    .foregroundColor(Theme.Colors.warning)
                Text(signal)
                    .font(.subheadline)
                    .fontWeight(.medium)
                    .foregroundColor(Theme.Colors.textPrimary)
                Spacer()
            }
            .padding()
        }
    }

    private func dominantPlayerCard(_ player: String) -> some View {
        GlassCard(padding: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(L.dominantPlayer)
                        .font(.caption)
                        .foregroundColor(Theme.Colors.textSecondary)
                    Text(player.replacingOccurrences(of: "_", with: " ").capitalized)
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundColor(Theme.Colors.textPrimary)
                }
                Spacer()
                Image(systemName: "crown.fill")
                    .font(.system(size: 28))
                    .foregroundColor(.yellow)
            }
        }
    }

    private func sectionHeader(_ title: String) -> some View {
        HStack {
            Text(title.uppercased())
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(Theme.Colors.textSecondary)
            Spacer()
        }
        .padding(.top, 8)
    }

    private func participantCard(icon: String, title: String, value: Double, color: Color, description: String, term: ExplanationTerm) -> some View {
        GlassCard(padding: 16) {
            VStack(spacing: 10) {
                HStack {
                    Image(systemName: icon)
                        .foregroundColor(color)
                        .frame(width: 24)
                    Text(title)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(Theme.Colors.textPrimary)
                    Spacer()
                    Text("\(Int(value * 100))%")
                        .font(.subheadline)
                        .fontWeight(.bold)
                        .monospacedDigit()
                        .foregroundColor(color)
                }

                ProgressView(value: value)
                    .tint(color)

                HStack {
                    Text(description)
                        .font(.caption)
                        .foregroundColor(Theme.Colors.textSecondary)
                        .lineLimit(2)
                    Spacer()
                }
            }
        }
        .explainable(term, value: value)
    }

    private func anomalyCard(_ anomaly: Anomaly) -> some View {
        let severityColor: Color = anomaly.severity < 0.3 ? .green : anomaly.severity < 0.7 ? .yellow : .red

        return GlassCard(padding: 12) {
            HStack(spacing: 12) {
                Image(systemName: anomaly.typeIcon)
                    .font(.system(size: 20))
                    .foregroundColor(severityColor)
                    .frame(width: 28)

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(anomaly.type.uppercased())
                            .font(.caption)
                            .fontWeight(.bold)
                            .foregroundColor(severityColor)
                        Spacer()
                        Text(anomaly.timestamp.suffix(9).prefix(8))
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textSecondary)
                            .monospacedDigit()
                    }
                    Text(anomaly.description)
                        .font(.caption)
                        .foregroundColor(Theme.Colors.textPrimary)
                        .lineLimit(2)
                    HStack {
                        Text(L.severity)
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textSecondary)
                        ProgressView(value: anomaly.severity)
                            .tint(severityColor)
                        Text("\(Int(anomaly.severity * 100))%")
                            .font(.caption2)
                            .foregroundColor(Theme.Colors.textSecondary)
                            .monospacedDigit()
                    }
                }
            }
        }
    }
}

//
//  PlayersView.swift
//  HEAN
//
//  Participant X-Ray breakdown view
//

import SwiftUI

struct PlayersView: View {
    @State private var breakdown: ParticipantBreakdown?
    @State private var isLoading = true

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 16) {
                    if let bd = breakdown {
                        // Meta signal banner
                        metaSignalBanner(bd.metaSignal)

                        // Dominant player
                        dominantPlayerCard(bd.dominantPlayer)

                        // Activity bars
                        activityBar(title: "Market Makers", value: bd.mmActivity, color: .blue, icon: "building.2.fill")
                        activityBar(title: "Institutional", value: bd.institutionalFlowPercent, color: .purple, icon: "briefcase.fill")
                        activityBar(title: "Retail", value: bd.retailSentimentPercent, color: .green, icon: "person.3.fill")
                        activityBar(title: "Whales", value: bd.whaleActivityPercent, color: .orange, icon: "fish.fill")
                        activityBar(title: "Arb Bots", value: bd.arbPressurePercent, color: .cyan, icon: "cpu")
                    } else if isLoading {
                        ProgressView("Loading X-Ray...")
                    } else {
                        VStack(spacing: 16) {
                            Image(systemName: "eye.fill")
                                .font(.system(size: 48))
                                .foregroundColor(.gray)
                            Text("No participant data")
                                .foregroundColor(.secondary)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Players X-Ray")
            .task { await loadData() }
            .refreshable { await loadData() }
        }
    }

    private func metaSignalBanner(_ signal: String) -> some View {
        HStack {
            Image(systemName: "brain.head.profile")
                .foregroundColor(.yellow)
            Text(signal)
                .font(.subheadline)
                .fontWeight(.medium)
            Spacer()
        }
        .padding()
        .background(Color.yellow.opacity(0.1))
        .cornerRadius(12)
    }

    private func dominantPlayerCard(_ player: String) -> some View {
        HStack {
            VStack(alignment: .leading) {
                Text("DOMINANT PLAYER")
                    .font(.caption)
                    .foregroundColor(.secondary)
                Text(player.replacingOccurrences(of: "_", with: " ").capitalized)
                    .font(.title2)
                    .fontWeight(.bold)
            }
            Spacer()
            Image(systemName: "crown.fill")
                .font(.system(size: 28))
                .foregroundColor(.yellow)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private func activityBar(title: String, value: Double, color: Color, icon: String) -> some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Text(title)
                    .font(.subheadline)
                    .fontWeight(.medium)
                Spacer()
                Text("\(Int(value * 100))%")
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .monospacedDigit()
                    .foregroundColor(color)
            }

            ProgressView(value: value)
                .tint(color)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }

    private func loadData() async {
        isLoading = true
        do {
            let result: ParticipantBreakdown = try await DIContainer.shared.apiClient.get("/api/v1/physics/participants?symbol=BTCUSDT")
            breakdown = result
        } catch {
            // Fallback to sample data
            breakdown = ParticipantBreakdown(
                mmActivity: 0.35, institutionalFlow: 250000,
                retailSentiment: 0.65, whaleActivity: 0.12,
                arbPressure: 0.08, dominantPlayer: "market_maker",
                metaSignal: "Institutional accumulating, trend continuation"
            )
        }
        isLoading = false
    }
}

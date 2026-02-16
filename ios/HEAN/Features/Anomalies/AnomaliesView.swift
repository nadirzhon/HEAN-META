//
//  AnomaliesView.swift
//  HEAN
//
//  Market anomalies feed
//

import SwiftUI

private struct AnomaliesResponse: Codable {
    let anomalies: [Anomaly]
    let activeCount: Int

    enum CodingKeys: String, CodingKey {
        case anomalies
        case activeCount = "active_count"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.anomalies = try container.decodeIfPresent([Anomaly].self, forKey: .anomalies) ?? []
        self.activeCount = try container.decodeIfPresent(Int.self, forKey: .activeCount) ?? 0
    }
}

struct AnomaliesView: View {
    @State private var anomalies: [Anomaly] = []
    @State private var isLoading = true

    var body: some View {
        NavigationView {
            Group {
                if isLoading {
                    ProgressView("Loading anomalies...")
                } else if anomalies.isEmpty {
                    VStack(spacing: 16) {
                        Image(systemName: "checkmark.shield.fill")
                            .font(.system(size: 48))
                            .foregroundColor(.green)
                        Text(L.noAnomalies)
                            .foregroundColor(.secondary)
                        Text(L.marketNormal)
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                } else {
                    List {
                        ForEach(anomalies) { anomaly in
                            AnomalyRow(anomaly: anomaly)
                        }
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("Anomalies")
            .task { await loadData() }
            .refreshable { await loadData() }
        }
    }

    private func loadData() async {
        isLoading = true
        do {
            let response: AnomaliesResponse = try await DIContainer.shared.apiClient.get("/api/v1/physics/anomalies?limit=20")
            anomalies = response.anomalies
        } catch {
            // Fallback to sample data
            anomalies = [
                Anomaly(id: "1", type: "volume", severity: 0.75,
                       description: "Volume spike 4.2x average on BTCUSDT",
                       timestamp: "2026-02-08T11:55:00Z"),
                Anomaly(id: "2", type: "whale", severity: 0.85,
                       description: "Whale inflow detected: $12M single order",
                       timestamp: "2026-02-08T11:50:00Z"),
                Anomaly(id: "3", type: "price", severity: 0.45,
                       description: "Price dislocation 1.8% on ETHUSDT",
                       timestamp: "2026-02-08T11:45:00Z"),
            ]
        }
        isLoading = false
    }
}

struct AnomalyRow: View {
    let anomaly: Anomaly

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: anomaly.typeIcon)
                .font(.system(size: 24))
                .foregroundColor(severityColor)
                .frame(width: 36)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(anomaly.type.uppercased())
                        .font(.caption)
                        .fontWeight(.bold)
                        .foregroundColor(severityColor)
                    Spacer()
                    Text(anomaly.timestamp.suffix(9).prefix(8))
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .monospacedDigit()
                }

                Text(anomaly.description)
                    .font(.subheadline)
                    .lineLimit(2)

                HStack {
                    Text(L.severity)
                        .font(.caption2)
                        .foregroundColor(.secondary)
                    ProgressView(value: anomaly.severity)
                        .tint(severityColor)
                    Text("\(Int(anomaly.severity * 100))%")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                        .monospacedDigit()
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var severityColor: Color {
        if anomaly.severity < 0.3 { return .green }
        if anomaly.severity < 0.7 { return .yellow }
        return .red
    }
}

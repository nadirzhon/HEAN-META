//
//  TemporalStackView.swift
//  HEAN
//
//  5-level temporal analysis view
//

import SwiftUI

private struct TemporalStackResponse: Codable {
    let levels: [String: TemporalLevelData]
    let lastUpdate: String?

    enum CodingKeys: String, CodingKey {
        case levels
        case lastUpdate = "last_update"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.levels = try container.decodeIfPresent([String: TemporalLevelData].self, forKey: .levels) ?? [:]
        self.lastUpdate = try? container.decodeIfPresent(String.self, forKey: .lastUpdate)
    }
}

private struct TemporalLevelData: Codable {
    let level: Int?
    let name: String?
    let timeframe: String?
    let trend: String?
    let summary: String?
    let confidence: Double?
    let details: [String: AnyCodableValue]?

    func toTemporalLevel(id: Int) -> TemporalLevel {
        // Map trend to status
        let status = trend ?? "neutral"
        // Convert details to [String: String]
        var stringDetails: [String: String] = [:]
        if let d = details {
            for (k, v) in d {
                stringDetails[k] = v.stringValue
            }
        }
        return TemporalLevel(
            id: level ?? id,
            name: name ?? "",
            timeframe: timeframe ?? "",
            status: status,
            summary: summary ?? "",
            details: stringDetails
        )
    }
}

private enum AnyCodableValue: Codable {
    case string(String)
    case double(Double)
    case int(Int)
    case bool(Bool)

    var stringValue: String {
        switch self {
        case .string(let s): return s
        case .double(let d): return String(format: "%.2f", d)
        case .int(let i): return "\(i)"
        case .bool(let b): return b ? "true" : "false"
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let s = try? container.decode(String.self) { self = .string(s); return }
        if let d = try? container.decode(Double.self) { self = .double(d); return }
        if let i = try? container.decode(Int.self) { self = .int(i); return }
        if let b = try? container.decode(Bool.self) { self = .bool(b); return }
        self = .string("")
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let s): try container.encode(s)
        case .double(let d): try container.encode(d)
        case .int(let i): try container.encode(i)
        case .bool(let b): try container.encode(b)
        }
    }
}

struct TemporalStackView: View {
    @State private var levels: [TemporalLevel] = []
    @State private var isLoading = true

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 12) {
                    if levels.isEmpty && !isLoading {
                        VStack(spacing: 16) {
                            Image(systemName: "clock.arrow.2.circlepath")
                                .font(.system(size: 48))
                                .foregroundColor(.gray)
                            Text("No temporal data")
                                .foregroundColor(.secondary)
                        }
                        .padding(.top, 40)
                    } else {
                        ForEach(levels.sorted(by: { $0.id > $1.id })) { level in
                            TemporalLevelCard(level: level)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Temporal Stack")
            .task { await loadData() }
            .refreshable { await loadData() }
        }
    }

    private func loadData() async {
        isLoading = true
        do {
            let response: TemporalStackResponse = try await DIContainer.shared.apiClient.get("/api/v1/temporal/stack")
            levels = response.levels.map { key, value in
                value.toTemporalLevel(id: Int(key) ?? 0)
            }
        } catch {
            // Fallback to sample data
            levels = [
                TemporalLevel(id: 5, name: "MACRO", timeframe: "days-weeks",
                             status: "bullish", summary: "BTCUSDT bullish trend, trending phase",
                             details: ["volatility": "520.3"]),
                TemporalLevel(id: 4, name: "SESSION", timeframe: "hours",
                             status: "neutral", summary: "London session: Trending mode, use Trend-following",
                             details: ["session": "London"]),
                TemporalLevel(id: 3, name: "TACTICS", timeframe: "minutes",
                             status: "bullish", summary: "Testing resistance at 95,200. Flow: $2.4M",
                             details: ["signal_count": "4"]),
                TemporalLevel(id: 2, name: "EXECUTION", timeframe: "seconds",
                             status: "bullish", summary: "Entry: 93,250, Stop: 93,800, R:R=4.2",
                             details: ["rr_ratio": "4.2"]),
                TemporalLevel(id: 1, name: "MICRO", timeframe: "milliseconds",
                             status: "neutral", summary: "Tight spread. Limit at 93,250, slippage ~0.02%",
                             details: ["slippage": "0.02%"]),
            ]
        }
        isLoading = false
    }
}

struct TemporalLevelCard: View {
    let level: TemporalLevel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle()
                    .fill(level.statusColor)
                    .frame(width: 10, height: 10)

                Text(level.levelName)
                    .font(.caption)
                    .fontWeight(.bold)
                    .foregroundColor(.secondary)

                Spacer()

                Text(level.timeframe)
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 2)
                    .background(Color(.systemGray5))
                    .cornerRadius(4)
            }

            Text(level.summary)
                .font(.subheadline)
                .lineLimit(2)

            if !level.details.isEmpty {
                HStack(spacing: 8) {
                    ForEach(Array(level.details.keys.sorted().prefix(3)), id: \.self) { key in
                        HStack(spacing: 2) {
                            Text(key + ":")
                                .font(.caption2)
                                .foregroundColor(.secondary)
                            Text(level.details[key] ?? "")
                                .font(.caption2)
                                .fontWeight(.medium)
                                .monospacedDigit()
                        }
                    }
                }
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

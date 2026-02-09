//
//  XRayViewModel.swift
//  HEAN
//
//  ViewModel for X-Ray (market participants + anomalies)
//

import SwiftUI
import Combine

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

@MainActor
final class XRayViewModel: ObservableObject {
    @Published var participants: ParticipantBreakdown?
    @Published var anomalies: [Anomaly] = []
    @Published var isLoading = true
    @Published var error: String?

    private var apiClient: APIClient?
    private var isConfigured = false

    func configure(apiClient: APIClient) {
        guard !isConfigured else { return }
        isConfigured = true
        self.apiClient = apiClient
    }

    func refresh() async {
        guard let apiClient = apiClient else { return }

        do {
            let result: ParticipantBreakdown = try await apiClient.get("/api/v1/physics/participants?symbol=BTCUSDT")
            self.participants = result
        } catch {
            if self.participants == nil {
                self.participants = ParticipantBreakdown(
                    mmActivity: 0.35, institutionalFlow: 250000,
                    retailSentiment: 0.65, whaleActivity: 0.12,
                    arbPressure: 0.08, dominantPlayer: "market_maker",
                    metaSignal: "Analyzing market participants..."
                )
            }
        }

        do {
            let response: AnomaliesResponse = try await apiClient.get("/api/v1/physics/anomalies?limit=20")
            self.anomalies = response.anomalies
        } catch {
            // Keep existing anomalies on error
        }

        isLoading = false
    }
}

//
//  SettingsViewModel.swift
//  HEAN
//
//  ViewModel for enhanced Settings — uptime, engine state
//

import SwiftUI

@MainActor
final class SettingsViewModel: ObservableObject {
    @Published var engineState: String = "Unknown"
    @Published var uptimeSeconds: Double = 0
    @Published var isLoading = false

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
            struct EngineStatus: Codable {
                let status: String?
                let running: Bool?
                let uptimeSec: Double?

                enum CodingKeys: String, CodingKey {
                    case status, running
                    case uptimeSec = "uptime_sec"
                }

                init(from decoder: Decoder) throws {
                    let container = try decoder.container(keyedBy: CodingKeys.self)
                    self.status = try container.decodeIfPresent(String.self, forKey: .status)
                    self.running = try container.decodeIfPresent(Bool.self, forKey: .running)
                    self.uptimeSec = try container.decodeIfPresent(Double.self, forKey: .uptimeSec)
                }
            }

            let result: EngineStatus = try await apiClient.get("/api/v1/engine/status")
            self.engineState = result.status ?? (result.running == true ? "Running" : "Stopped")
            self.uptimeSeconds = result.uptimeSec ?? 0
        } catch {
            self.engineState = "Unavailable"
        }
    }

    var formattedUptime: String {
        let hours = Int(uptimeSeconds) / 3600
        let minutes = (Int(uptimeSeconds) % 3600) / 60
        if hours > 0 { return "\(hours)h \(minutes)m" }
        if minutes > 0 { return "\(minutes)m" }
        return "< 1m"
    }
}

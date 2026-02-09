//
//  LiveRiskService.swift
//  HEAN
//

import Foundation
import Combine

@MainActor
final class LiveRiskService: RiskServiceProtocol {
    private let apiClient: APIClient
    private let websocket: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    private nonisolated(unsafe) let riskStateSubject = CurrentValueSubject<RiskState, Never>(.normal)

    var riskStatePublisher: AnyPublisher<RiskState, Never> {
        riskStateSubject.eraseToAnyPublisher()
    }

    init(apiClient: APIClient, websocket: WebSocketManager) {
        self.apiClient = apiClient
        self.websocket = websocket
        subscribeToWebSocket()
    }

    private func subscribeToWebSocket() {
        websocket.subscribe(topic: "risk_events")

        websocket.messagePublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] message in
                guard message.topic == "risk_events",
                      let data = message.data,
                      let stateStr = data["risk_state"] as? String,
                      let state = RiskState(rawValue: stateStr) else { return }
                self?.riskStateSubject.send(state)
            }
            .store(in: &cancellables)
    }

    func fetchRiskStatus() async throws -> RiskGovernorStatus {
        let status: RiskGovernorStatus = try await apiClient.get("/api/v1/risk/governor/status")
        riskStateSubject.send(status.state)
        return status
    }

    func fetchKillswitchStatus() async throws -> KillswitchStatus {
        try await apiClient.get("/api/v1/risk/killswitch/status")
    }

    func resetKillswitch() async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/risk/killswitch/reset", body: [:])
    }

    func quarantineSymbol(_ symbol: String) async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post(
            "/api/v1/risk/governor/quarantine/\(symbol)",
            body: [:]
        )
    }

    func clearRiskBlocks() async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post("/api/v1/risk/governor/clear", body: [:])
    }
}

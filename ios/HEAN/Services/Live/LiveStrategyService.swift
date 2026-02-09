//
//  LiveStrategyService.swift
//  HEAN
//

import Foundation
import Combine

@MainActor
final class LiveStrategyService: StrategyServiceProtocol {
    private let apiClient: APIClient
    private var cancellables = Set<AnyCancellable>()

    private let strategiesSubject = CurrentValueSubject<[Strategy], Never>([])

    var strategiesPublisher: AnyPublisher<[Strategy], Never> {
        strategiesSubject.eraseToAnyPublisher()
    }

    init(apiClient: APIClient) {
        self.apiClient = apiClient
    }

    func fetchStrategies() async throws -> [Strategy] {
        let strategies: [Strategy] = try await apiClient.get("/api/v1/strategies")
        strategiesSubject.send(strategies)
        return strategies
    }

    func toggleStrategy(id: String, enabled: Bool) async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post(
            "/api/v1/strategies/\(id)/enable",
            body: ["enabled": enabled]
        )
    }

    func updateParameters(id: String, parameters: [String: Any]) async throws {
        struct Empty: Codable {}
        let _: Empty = try await apiClient.post(
            "/api/v1/strategies/\(id)/params",
            body: parameters
        )
    }
}

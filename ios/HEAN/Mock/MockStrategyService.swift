//
//  MockStrategyService.swift
//  HEAN
//
//  Mock strategy service for development and preview
//

import Foundation
import Combine

@MainActor
final class MockStrategyService: StrategyServiceProtocol {
    private let strategiesSubject = CurrentValueSubject<[Strategy], Never>([])
    private var strategies: [Strategy] = []

    var strategiesPublisher: AnyPublisher<[Strategy], Never> {
        strategiesSubject.eraseToAnyPublisher()
    }

    init() {
        strategies = generateMockStrategies()
        strategiesSubject.send(strategies)
    }

    func fetchStrategies() async throws -> [Strategy] {
        // Simulate network delay
        try await Task.sleep(for: .milliseconds(300))
        return strategies
    }

    func toggleStrategy(id: String, enabled: Bool) async throws -> StrategyToggleResponse {
        try await Task.sleep(for: .milliseconds(200))

        if let index = strategies.firstIndex(where: { $0.id == id }) {
            let old = strategies[index]
            strategies[index] = Strategy(
                id: old.id,
                name: old.name,
                enabled: enabled,
                symbols: old.symbols,
                signalCount: old.signalCount,
                tradeCount: old.tradeCount,
                winRate: old.winRate,
                pnl: old.pnl,
                lastSignal: old.lastSignal,
                parameters: old.parameters
            )
            strategiesSubject.send(strategies)
        }

        return StrategyToggleResponse(
            strategyId: id,
            enabled: enabled,
            message: enabled ? "Strategy enabled" : "Strategy disabled"
        )
    }

    func updateParameters(id: String, params: [String: Double]) async throws {
        try await Task.sleep(for: .milliseconds(200))
        // In mock, we don't actually update parameters
    }

    private func generateMockStrategies() -> [Strategy] {
        [
            Strategy(
                id: "impulse-engine",
                name: "ImpulseEngine",
                enabled: true,
                symbols: ["BTCUSDT", "ETHUSDT"],
                signalCount: 145,
                tradeCount: 89,
                winRate: 0.62,
                pnl: 1234.56,
                lastSignal: Date().addingTimeInterval(-300),
                parameters: ["momentum_threshold": 0.015, "rsi_period": 14, "volume_multiplier": 1.5]
            ),
            Strategy(
                id: "funding-harvester",
                name: "FundingHarvester",
                enabled: true,
                symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
                signalCount: 67,
                tradeCount: 52,
                winRate: 0.78,
                pnl: 567.89,
                lastSignal: Date().addingTimeInterval(-1800),
                parameters: ["min_funding_rate": 0.0001, "max_position_size": 0.1]
            ),
            Strategy(
                id: "basis-arbitrage",
                name: "BasisArbitrage",
                enabled: false,
                symbols: ["BTCUSDT"],
                signalCount: 23,
                tradeCount: 18,
                winRate: 0.55,
                pnl: -45.67,
                lastSignal: Date().addingTimeInterval(-7200),
                parameters: ["min_spread": 0.002, "max_exposure": 5000]
            ),
            Strategy(
                id: "momentum-scalper",
                name: "MomentumScalper",
                enabled: true,
                symbols: ["SOLUSDT", "BNBUSDT", "XRPUSDT"],
                signalCount: 312,
                tradeCount: 198,
                winRate: 0.58,
                pnl: 890.12,
                lastSignal: Date().addingTimeInterval(-60),
                parameters: ["scalp_target": 0.003, "max_hold_time": 300]
            )
        ]
    }
}

//
//  StrategiesViewModel.swift
//  HEAN
//

import Foundation
import Combine

@MainActor
final class StrategiesViewModel: ObservableObject {
    @Published var strategies: [Strategy] = []
    @Published var isLoading = false
    @Published var error: String?

    private let strategyService: StrategyServiceProtocol

    init(strategyService: StrategyServiceProtocol) {
        self.strategyService = strategyService
    }

    func refresh() async {
        isLoading = true
        defer { isLoading = false }
        do {
            strategies = try await strategyService.fetchStrategies()
        } catch {
            self.error = error.localizedDescription
        }
    }

    func toggleStrategy(id: String, enabled: Bool) async {
        do {
            try await strategyService.toggleStrategy(id: id, enabled: enabled)
            if let i = strategies.firstIndex(where: { $0.id == id }) {
                let s = strategies[i]
                strategies[i] = Strategy(
                    id: s.id, name: s.name, enabled: enabled,
                    winRate: s.winRate, totalTrades: s.totalTrades,
                    profitFactor: s.profitFactor, description: s.description,
                    totalPnl: s.totalPnl, wins: s.wins, losses: s.losses
                )
            }
        } catch {
            self.error = error.localizedDescription
        }
    }
}

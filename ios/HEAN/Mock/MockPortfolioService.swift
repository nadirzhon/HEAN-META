//
//  MockPortfolioService.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation
import Combine

/// Mock implementation of PortfolioServiceProtocol with simulated PnL changes
@MainActor
final class MockPortfolioService: PortfolioServiceProtocol {

    // MARK: - Properties

    private nonisolated(unsafe) let equitySubject = CurrentValueSubject<Double, Never>(10000)
    private nonisolated(unsafe) let pnlSubject = CurrentValueSubject<PnLSnapshot, Never>(PnLSnapshot(realized: 0, unrealized: 0, total: 0, percent: 0, fees: 0))

    var equityPublisher: AnyPublisher<Double, Never> {
        equitySubject.eraseToAnyPublisher()
    }

    var pnlPublisher: AnyPublisher<PnLSnapshot, Never> {
        pnlSubject.eraseToAnyPublisher()
    }

    private var portfolio: Portfolio
    private let tradingService: MockTradingService
    private var cancellables = Set<AnyCancellable>()

    // MARK: - Initialization

    init(tradingService: MockTradingService) {
        self.tradingService = tradingService
        self.portfolio = MockDataProvider.generatePortfolio()

        equitySubject.send(portfolio.equity)
        updatePnL()

        subscribeToPositionUpdates()
    }

    // MARK: - PortfolioServiceProtocol

    func fetchPortfolio() async throws -> Portfolio {
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms
        return portfolio
    }

    func fetchEquityHistory(limit: Int = 50) async throws -> [EquityPoint] {
        // Generate mock equity curve
        let count = min(limit, 50)
        var equity = portfolio.equity
        return (0..<count).map { i in
            equity += Double.random(in: -50...60)
            return EquityPoint(timestamp: "2026-01-31T\(String(format: "%02d", i)):00:00", equity: equity)
        }
    }

    // MARK: - Private Methods

    private func subscribeToPositionUpdates() {
        tradingService.positionsPublisher
            .sink { [weak self] positions in
                self?.updatePortfolio(with: positions)
            }
            .store(in: &cancellables)
    }

    private func updatePortfolio(with positions: [Position]) {
        portfolio = MockDataProvider.updatePortfolio(portfolio, positions: positions)
        equitySubject.send(portfolio.equity)
        updatePnL()
    }

    private func updatePnL() {
        let pnl = PnLSnapshot(
            realized: portfolio.realizedPnL,
            unrealized: portfolio.unrealizedPnL,
            total: portfolio.totalPnL,
            percent: portfolio.equity > 0 ? (portfolio.totalPnL / portfolio.equity) * 100 : 0,
            fees: 0
        )
        pnlSubject.send(pnl)
    }
}

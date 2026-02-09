//
//  PortfolioServiceProtocol.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation
import Combine

@MainActor
protocol PortfolioServiceProtocol {
    var equityPublisher: AnyPublisher<Double, Never> { get }
    var pnlPublisher: AnyPublisher<PnLSnapshot, Never> { get }

    func fetchPortfolio() async throws -> Portfolio
    func fetchEquityHistory(limit: Int) async throws -> [EquityPoint]
}

struct PnLSnapshot: Codable {
    let realized: Double
    let unrealized: Double
    let total: Double
    let percent: Double
    let fees: Double
}

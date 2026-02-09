//
//  MarketServiceProtocol.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation
import Combine

@MainActor
protocol MarketServiceProtocol {
    /// Current markets list (for synchronous access)
    var markets: [Market] { get }

    /// Publisher for market updates
    var marketUpdates: AnyPublisher<[Market], Never> { get }

    func fetchMarkets() async throws -> [Market]
    func fetchMarket(symbol: String) async throws -> Market
    func fetchCandles(symbol: String, interval: String, limit: Int) async throws -> [Candle]
}

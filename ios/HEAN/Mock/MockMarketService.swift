//
//  MockMarketService.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation
import Combine

/// Mock implementation of MarketServiceProtocol with simulated price updates
@MainActor
final class MockMarketService: MarketServiceProtocol {

    // MARK: - Properties

    private nonisolated(unsafe) let marketUpdatesSubject = PassthroughSubject<[Market], Never>()
    var marketUpdates: AnyPublisher<[Market], Never> {
        marketUpdatesSubject.eraseToAnyPublisher()
    }

    /// Current markets list (protocol conformance)
    private(set) var markets: [Market] = []
    private nonisolated(unsafe) var updateTimer: Timer?
    private let updateInterval: TimeInterval = 2.0 // Update every 2 seconds

    // MARK: - Initialization

    init() {
        markets = MockDataProvider.generateMarkets()
        startPriceUpdates()
    }

    deinit {
        updateTimer?.invalidate()
    }

    // MARK: - MarketServiceProtocol

    func fetchMarkets() async throws -> [Market] {
        // Simulate network delay
        try await Task.sleep(nanoseconds: 200_000_000) // 200ms
        return markets
    }

    func fetchMarket(symbol: String) async throws -> Market {
        // Simulate network delay
        try await Task.sleep(nanoseconds: 150_000_000) // 150ms

        guard let market = markets.first(where: { $0.symbol == symbol }) else {
            throw MockServiceError.notFound
        }

        return market
    }

    func fetchCandles(symbol: String, interval: String, limit: Int) async throws -> [Candle] {
        // Simulate network delay
        try await Task.sleep(nanoseconds: 300_000_000) // 300ms

        guard let market = markets.first(where: { $0.symbol == symbol }) else {
            throw MockServiceError.notFound
        }

        return MockDataProvider.generateCandles(count: limit, basePrice: market.price)
    }

    // MARK: - Private Methods

    private func startPriceUpdates() {
        updateTimer = Timer.scheduledTimer(withTimeInterval: updateInterval, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.updatePrices()
            }
        }
    }

    private func updatePrices() {
        // Update all markets with smooth random walk
        markets = markets.map { market in
            // Higher volatility for smaller cap coins
            let volatility: Double
            switch market.symbol {
            case "BTCUSDT", "ETHUSDT":
                volatility = 0.0008 // 0.08% per update
            case "BNBUSDT", "SOLUSDT":
                volatility = 0.0012 // 0.12% per update
            default:
                volatility = 0.0015 // 0.15% per update
            }

            return MockDataProvider.updateMarketPrice(market, volatility: volatility)
        }

        // Publish updated markets
        marketUpdatesSubject.send(markets)
    }
}

// MARK: - Errors

enum MockServiceError: LocalizedError {
    case notFound
    case invalidRequest
    case simulatedFailure

    var errorDescription: String? {
        switch self {
        case .notFound:
            return "Resource not found"
        case .invalidRequest:
            return "Invalid request parameters"
        case .simulatedFailure:
            return "Simulated service failure"
        }
    }
}

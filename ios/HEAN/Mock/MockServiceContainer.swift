//
//  MockServiceContainer.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation

/// Container for all mock services with proper dependency injection
@MainActor
final class MockServiceContainer {

    // MARK: - Singleton

    static let shared = MockServiceContainer()

    // MARK: - Services

    let marketService: MarketServiceProtocol
    let tradingService: TradingServiceProtocol
    let portfolioService: PortfolioServiceProtocol
    let eventService: EventServiceProtocol

    // MARK: - Initialization

    private init() {
        // Create services with proper dependencies
        let mockMarketService = MockMarketService()
        let mockTradingService = MockTradingService(marketService: mockMarketService)
        let mockPortfolioService = MockPortfolioService(tradingService: mockTradingService)
        let mockEventService = MockEventService()

        // Assign to protocol properties
        self.marketService = mockMarketService
        self.tradingService = mockTradingService
        self.portfolioService = mockPortfolioService
        self.eventService = mockEventService
    }
}

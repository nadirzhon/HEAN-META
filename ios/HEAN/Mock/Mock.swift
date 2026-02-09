//
//  Mock.swift
//  HEAN
//
//  Central export file for mock data system
//  Created on 2026-01-31.
//

// Re-export all mock services for convenient importing
// Usage: import Mock

// Mock Services
// - MockMarketService: Simulated market data with live price updates
// - MockTradingService: Simulated order execution and position tracking
// - MockPortfolioService: Simulated portfolio with PnL updates
// - MockEventService: Simulated event stream and WebSocket health

// Mock Data Provider
// - MockDataProvider: Static generators for all mock data types

// Service Container
// - MockServiceContainer: Singleton DI container with all mock services

// Quick Start:
// let services = MockServiceContainer.shared
// let markets = try await services.marketService.fetchMarkets()

// See ExampleUsage.swift for complete integration example
// See README.md for comprehensive documentation

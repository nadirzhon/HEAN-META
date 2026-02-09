//
//  APIEndpoints.swift
//  HEAN
//
//  Created by HEAN Team on 2026-01-31.
//

import Foundation

enum HEANEndpoint: APIEndpoint {
    // System
    case health
    case engineState

    // Engine Control
    case engineStart
    case engineStop
    case enginePause
    case engineResume

    // Trading
    case activePositions
    case activeOrders
    case placeOrder(symbol: String, side: String, orderType: String, qty: Double, price: Double?)
    case cancelOrder(orderId: String)
    case closePosition(symbol: String)
    case cancelAllOrders
    case closeAllPositions

    // Trading Diagnostics
    case tradingWhy
    case tradingMetrics

    // Portfolio
    case portfolio
    case equity
    case pnl

    // Markets
    case markets
    case ticker(symbol: String)
    case orderBook(symbol: String)
    case klines(symbol: String, interval: String, limit: Int)

    // Strategies
    case strategies
    case strategyState(id: String)

    // Physics
    case physicsState(symbol: String)
    case physicsParticipants(symbol: String)
    case physicsAnomalies(limit: Int)

    // Temporal
    case temporalStack

    // Brain
    case brainThoughts
    case brainAnalysis

    // Telemetry
    case telemetryEvents

    var path: String {
        switch self {
        case .health:
            return "/health"
        case .engineState:
            return "/api/v1/engine/status"
        case .engineStart:
            return "/api/v1/engine/start"
        case .engineStop:
            return "/api/v1/engine/stop"
        case .enginePause:
            return "/api/v1/engine/pause"
        case .engineResume:
            return "/api/v1/engine/resume"
        case .activePositions:
            return "/api/v1/orders/positions"
        case .activeOrders:
            return "/api/v1/orders"
        case .placeOrder:
            return "/api/v1/orders/test"
        case .cancelOrder:
            return "/api/v1/orders/cancel"
        case .closePosition:
            return "/api/v1/orders/close-position"
        case .cancelAllOrders:
            return "/api/v1/orders/cancel-all"
        case .closeAllPositions:
            return "/api/v1/orders/close-all-positions"
        case .tradingWhy:
            return "/api/v1/trading/why"
        case .tradingMetrics:
            return "/api/v1/trading/metrics"
        case .portfolio:
            return "/api/v1/engine/status"
        case .equity:
            return "/api/v1/engine/status"
        case .pnl:
            return "/api/v1/engine/status"
        case .markets:
            return "/api/v1/market/tickers"
        case .ticker(let symbol):
            return "/api/v1/market/ticker?symbol=\(symbol)"
        case .orderBook(let symbol):
            return "/api/v1/market/orderbook?symbol=\(symbol)"
        case .klines(let symbol, let interval, let limit):
            return "/api/v1/market/candles?symbol=\(symbol)&timeframe=\(interval)&limit=\(limit)"
        case .strategies:
            return "/api/v1/strategies"
        case .strategyState(let id):
            return "/api/v1/strategies/\(id)"
        case .physicsState(let symbol):
            return "/api/v1/physics/state?symbol=\(symbol)"
        case .physicsParticipants(let symbol):
            return "/api/v1/physics/participants?symbol=\(symbol)"
        case .physicsAnomalies(let limit):
            return "/api/v1/physics/anomalies?limit=\(limit)"
        case .temporalStack:
            return "/api/v1/temporal/stack"
        case .brainThoughts:
            return "/api/v1/brain/thoughts"
        case .brainAnalysis:
            return "/api/v1/brain/analysis"
        case .telemetryEvents:
            return "/api/v1/telemetry/summary"
        }
    }

    var method: HTTPMethod {
        switch self {
        case .health, .engineState, .activePositions, .activeOrders,
             .portfolio, .equity, .pnl, .markets, .ticker, .orderBook, .klines,
             .strategies, .strategyState,
             .physicsState, .physicsParticipants, .physicsAnomalies,
             .temporalStack, .brainThoughts, .brainAnalysis,
             .tradingWhy, .tradingMetrics, .telemetryEvents:
            return .GET
        case .engineStart, .engineStop, .enginePause, .engineResume,
             .placeOrder, .cancelOrder, .closePosition,
             .cancelAllOrders, .closeAllPositions:
            return .POST
        }
    }

    var body: [String: Any]? {
        switch self {
        case .engineStart:
            return ["confirm_phrase": "I_UNDERSTAND_LIVE_TRADING"]
        case .engineStop, .enginePause, .engineResume:
            return [:]
        case .placeOrder(let symbol, let side, let orderType, let qty, let price):
            var body: [String: Any] = [
                "symbol": symbol,
                "side": side,
                "order_type": orderType,
                "size": qty
            ]
            if let price = price {
                body["price"] = price
            }
            return body
        case .cancelOrder(let orderId):
            return ["order_id": orderId]
        case .closePosition(let symbol):
            return ["symbol": symbol]
        case .cancelAllOrders, .closeAllPositions:
            return [:]
        default:
            return nil
        }
    }
}

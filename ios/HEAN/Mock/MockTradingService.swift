//
//  MockTradingService.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation
import Combine

/// Mock implementation of TradingServiceProtocol with simulated order execution
@MainActor
final class MockTradingService: TradingServiceProtocol {

    // MARK: - Properties

    private nonisolated(unsafe) let positionsSubject = CurrentValueSubject<[Position], Never>([])
    var positionsPublisher: AnyPublisher<[Position], Never> {
        positionsSubject.eraseToAnyPublisher()
    }

    private nonisolated(unsafe) let ordersSubject = CurrentValueSubject<[Order], Never>([])
    var ordersPublisher: AnyPublisher<[Order], Never> {
        ordersSubject.eraseToAnyPublisher()
    }

    private var positions: [Position] = []
    private var orders: [Order] = []
    private nonisolated(unsafe) var updateTimer: Timer?
    private let marketService: MockMarketService

    // MARK: - Initialization

    init(marketService: MockMarketService) {
        self.marketService = marketService
        positions = MockDataProvider.generatePositions()
        orders = MockDataProvider.generateOrders()

        positionsSubject.send(positions)
        ordersSubject.send(orders)

        startPositionUpdates()
    }

    deinit {
        updateTimer?.invalidate()
    }

    // MARK: - TradingServiceProtocol

    func fetchPositions() async throws -> [Position] {
        try await Task.sleep(for: .milliseconds(150))
        return positions
    }

    func fetchOrders(status: String? = nil) async throws -> [Order] {
        try await Task.sleep(for: .milliseconds(150))
        if let status = status {
            return orders.filter { $0.status.rawValue.lowercased() == status.lowercased() }
        }
        return orders
    }

    func placeOrder(symbol: String, side: String, type: String, qty: Double, price: Double?) async throws -> Order {
        try await Task.sleep(for: .milliseconds(250))

        let orderSide = side.uppercased() == "BUY" ? OrderSide.buy : OrderSide.sell
        let orderType = type.uppercased() == "MARKET" ? OrderType.market : OrderType.limit

        let order = Order(
            id: UUID().uuidString,
            symbol: symbol,
            side: orderSide,
            type: orderType,
            status: orderType == .market ? .filled : .new,
            price: price,
            quantity: qty,
            filledQuantity: orderType == .market ? qty : 0,
            createdAt: Date(),
            updatedAt: Date()
        )

        orders.insert(order, at: 0)
        ordersSubject.send(orders)

        return order
    }

    func cancelOrder(id: String) async throws {
        try await Task.sleep(for: .milliseconds(200))

        guard let index = orders.firstIndex(where: { $0.id == id }) else {
            throw MockServiceError.notFound
        }

        var order = orders[index]
        guard order.isActive else {
            throw MockServiceError.invalidRequest
        }

        order = Order(
            id: order.id,
            symbol: order.symbol,
            side: order.side,
            type: order.type,
            status: .cancelled,
            price: order.price,
            quantity: order.quantity,
            filledQuantity: order.filledQuantity,
            createdAt: order.createdAt,
            updatedAt: Date()
        )

        orders[index] = order
        ordersSubject.send(orders)
    }

    func cancelAllOrders() async throws {
        try await Task.sleep(for: .milliseconds(300))

        for i in 0..<orders.count {
            if orders[i].isActive {
                orders[i] = Order(
                    id: orders[i].id,
                    symbol: orders[i].symbol,
                    side: orders[i].side,
                    type: orders[i].type,
                    status: .cancelled,
                    price: orders[i].price,
                    quantity: orders[i].quantity,
                    filledQuantity: orders[i].filledQuantity,
                    createdAt: orders[i].createdAt,
                    updatedAt: Date()
                )
            }
        }
        ordersSubject.send(orders)
    }

    func closePosition(symbol: String) async throws {
        try await Task.sleep(for: .milliseconds(300))

        guard let index = positions.firstIndex(where: { $0.symbol == symbol }) else {
            throw MockServiceError.notFound
        }

        let position = positions[index]

        // Create closing order
        let closingOrder = Order(
            id: UUID().uuidString,
            symbol: symbol,
            side: position.side == .long ? .sell : .buy,
            type: .market,
            status: .filled,
            price: nil,
            quantity: position.size,
            filledQuantity: position.size,
            createdAt: Date(),
            updatedAt: Date()
        )

        orders.insert(closingOrder, at: 0)
        positions.remove(at: index)

        positionsSubject.send(positions)
        ordersSubject.send(orders)
    }

    func closeAllPositions() async throws {
        try await Task.sleep(for: .milliseconds(400))

        for position in positions {
            let closingOrder = Order(
                id: UUID().uuidString,
                symbol: position.symbol,
                side: position.side == .long ? .sell : .buy,
                type: .market,
                status: .filled,
                price: nil,
                quantity: position.size,
                filledQuantity: position.size,
                createdAt: Date(),
                updatedAt: Date()
            )
            orders.insert(closingOrder, at: 0)
        }

        positions.removeAll()
        positionsSubject.send(positions)
        ordersSubject.send(orders)
    }

    func fetchWhyDiagnostics() async throws -> WhyDiagnostics {
        try await Task.sleep(for: .milliseconds(200))

        // Return mock WhyDiagnostics JSON and decode it
        let json: [String: Any] = [
            "engine_state": "RUNNING",
            "killswitch_state": ["triggered": false, "reasons": [] as [String]],
            "last_tick_age_sec": 1.5,
            "active_orders_count": orders.filter { $0.isActive }.count,
            "active_positions_count": positions.count,
            "equity": 10000.0,
            "top_reason_codes_last_5m": [] as [[String: Any]]
        ]
        let data = try JSONSerialization.data(withJSONObject: json)
        return try JSONDecoder.hean.decode(WhyDiagnostics.self, from: data)
    }

    func fetchTradingMetrics() async throws -> TradingMetrics {
        try await Task.sleep(for: .milliseconds(200))

        return TradingMetrics(
            status: "ok",
            counters: nil,
            activeOrdersCount: orders.filter { $0.isActive }.count,
            activePositionsCount: positions.count,
            engineState: "RUNNING"
        )
    }

    // MARK: - Private Methods

    private func startPositionUpdates() {
        // Subscribe to market updates to update position mark prices
        updateTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                await self?.updatePositionPrices()
            }
        }
    }

    private func updatePositionPrices() async {
        var updated = false

        for i in 0..<positions.count {
            let position = positions[i]

            do {
                let market = try await marketService.fetchMarket(symbol: position.symbol)
                let updatedPosition = MockDataProvider.updatePosition(position, newMarkPrice: market.price)

                if updatedPosition.markPrice != position.markPrice {
                    positions[i] = updatedPosition
                    updated = true
                }
            } catch {
                continue
            }
        }

        if updated {
            positionsSubject.send(positions)
        }
    }
}

// Note: MockServiceError is defined in MockMarketService.swift

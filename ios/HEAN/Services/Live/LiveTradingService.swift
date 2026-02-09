//
//  LiveTradingService.swift
//  HEAN
//

import Foundation
import Combine

@MainActor
final class LiveTradingService: TradingServiceProtocol {
    private let apiClient: APIClient
    private let websocket: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    private nonisolated(unsafe) let positionsSubject = CurrentValueSubject<[Position], Never>([])
    private nonisolated(unsafe) let ordersSubject = CurrentValueSubject<[Order], Never>([])

    var positionsPublisher: AnyPublisher<[Position], Never> {
        positionsSubject.eraseToAnyPublisher()
    }

    var ordersPublisher: AnyPublisher<[Order], Never> {
        ordersSubject.eraseToAnyPublisher()
    }

    init(apiClient: APIClient, websocket: WebSocketManager) {
        self.apiClient = apiClient
        self.websocket = websocket
        subscribeToWebSocket()
    }

    private func subscribeToWebSocket() {
        websocket.subscribe(topic: "positions")
        websocket.subscribe(topic: "orders")

        websocket.messagePublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] message in
                self?.handleMessage(message)
            }
            .store(in: &cancellables)
    }

    private func handleMessage(_ message: WebSocketMessage) {
        switch message.topic {
        case "positions":
            if let data = message.data,
               let jsonData = try? JSONSerialization.data(withJSONObject: data),
               let positions = try? JSONDecoder.hean.decode([Position].self, from: jsonData) {
                positionsSubject.send(positions)
            }
        case "orders":
            if let data = message.data,
               let jsonData = try? JSONSerialization.data(withJSONObject: data),
               let orders = try? JSONDecoder.hean.decode([Order].self, from: jsonData) {
                ordersSubject.send(orders)
            }
        default:
            break
        }
    }

    func fetchPositions() async throws -> [Position] {
        let positions: [Position] = try await apiClient.get("/api/v1/orders/positions")
        positionsSubject.send(positions)
        return positions
    }

    func fetchOrders(status: String? = nil) async throws -> [Order] {
        var path = "/api/v1/orders"
        if let status = status {
            path += "?status=\(status)"
        }
        let orders: [Order] = try await apiClient.get(path)
        ordersSubject.send(orders)
        return orders
    }

    func placeOrder(symbol: String, side: String, type: String, qty: Double, price: Double?) async throws -> Order {
        var body: [String: Any] = [
            "symbol": symbol,
            "side": side,
            "type": type,
            "size": qty
        ]
        if let price = price {
            body["price"] = price
        }
        let order: Order = try await apiClient.post("/api/v1/orders/test", body: body)
        return order
    }

    func cancelOrder(id: String) async throws {
        let _: EmptyResponse = try await apiClient.post("/api/v1/orders/cancel", body: ["order_id": id])
    }

    func cancelAllOrders() async throws {
        let _: EmptyResponse = try await apiClient.post("/api/v1/orders/cancel-all", body: [:])
    }

    func closePosition(symbol: String) async throws {
        let _: EmptyResponse = try await apiClient.post("/api/v1/orders/close-position", body: ["symbol": symbol])
    }

    func closeAllPositions() async throws {
        let _: EmptyResponse = try await apiClient.post("/api/v1/orders/paper/close_all", body: [:])
    }

    func fetchWhyDiagnostics() async throws -> WhyDiagnostics {
        try await apiClient.get("/api/v1/trading/why")
    }

    func fetchTradingMetrics() async throws -> TradingMetrics {
        try await apiClient.get("/api/v1/trading/metrics")
    }
}

struct EmptyResponse: Codable {}

//
//  LivePortfolioService.swift
//  HEAN
//

import Foundation
import Combine

@MainActor
final class LivePortfolioService: PortfolioServiceProtocol {
    private let apiClient: APIClient
    private let websocket: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    private nonisolated(unsafe) let equitySubject = CurrentValueSubject<Double, Never>(0)
    private nonisolated(unsafe) let pnlSubject = CurrentValueSubject<PnLSnapshot, Never>(
        PnLSnapshot(realized: 0, unrealized: 0, total: 0, percent: 0, fees: 0)
    )

    var equityPublisher: AnyPublisher<Double, Never> {
        equitySubject.eraseToAnyPublisher()
    }

    var pnlPublisher: AnyPublisher<PnLSnapshot, Never> {
        pnlSubject.eraseToAnyPublisher()
    }

    init(apiClient: APIClient, websocket: WebSocketManager) {
        self.apiClient = apiClient
        self.websocket = websocket
        subscribeToWebSocket()
    }

    private func subscribeToWebSocket() {
        websocket.subscribe(topic: "account_state")
        websocket.subscribe(topic: "system_status")

        websocket.messagePublisher
            .receive(on: DispatchQueue.main)
            .sink { [weak self] message in
                self?.handleMessage(message)
            }
            .store(in: &cancellables)
    }

    private func handleMessage(_ message: WebSocketMessage) {
        guard message.topic == "account_state" || message.topic == "system_status",
              let data = message.data else { return }

        if let equity = data["equity"] as? Double {
            equitySubject.send(equity)
        }

        if let realized = data["realized_pnl"] as? Double,
           let unrealized = data["unrealized_pnl"] as? Double {
            let fees = data["fees"] as? Double ?? 0
            let total = realized + unrealized
            let equity = equitySubject.value
            let percent = equity > 0 ? (total / equity) * 100 : 0
            pnlSubject.send(PnLSnapshot(
                realized: realized,
                unrealized: unrealized,
                total: total,
                percent: percent,
                fees: fees
            ))
        }
    }

    func fetchPortfolio() async throws -> Portfolio {
        // Backend /engine/status returns fields that Portfolio can decode directly
        let portfolio: Portfolio = try await apiClient.get("/api/v1/engine/status")
        equitySubject.send(portfolio.equity)
        return portfolio
    }
}

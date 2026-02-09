//
//  LiveSignalService.swift
//  HEAN
//

import Foundation
import Combine

@MainActor
final class LiveSignalService: SignalServiceProtocol {
    private let websocket: WebSocketManager
    private var cancellables = Set<AnyCancellable>()

    private nonisolated(unsafe) let signalSubject = PassthroughSubject<Signal, Never>()
    private(set) var recentSignals: [Signal] = []
    private let maxSignals = 100

    var signalPublisher: AnyPublisher<Signal, Never> {
        signalSubject.eraseToAnyPublisher()
    }

    init(websocket: WebSocketManager) {
        self.websocket = websocket
    }

    func subscribeToSignals() {
        websocket.subscribe(topic: "signals")
        websocket.subscribe(topic: "strategy_events")

        websocket.messagePublisher
            .receive(on: DispatchQueue.main)
            .filter { $0.topic == "signals" || $0.topic == "strategy_events" }
            .sink { [weak self] message in
                self?.handleSignalMessage(message)
            }
            .store(in: &cancellables)
    }

    func unsubscribeFromSignals() {
        websocket.unsubscribe(topic: "signals")
        websocket.unsubscribe(topic: "strategy_events")
        cancellables.removeAll()
    }

    private func handleSignalMessage(_ message: WebSocketMessage) {
        guard let data = message.data,
              let jsonData = try? JSONSerialization.data(withJSONObject: data),
              let signal = try? JSONDecoder.hean.decode(Signal.self, from: jsonData) else { return }

        recentSignals.insert(signal, at: 0)
        if recentSignals.count > maxSignals {
            recentSignals.removeLast()
        }
        signalSubject.send(signal)
    }
}

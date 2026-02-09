//
//  MockSignalService.swift
//  HEAN
//
//  Mock signal service for development and preview
//

import Foundation
import Combine

@MainActor
final class MockSignalService: SignalServiceProtocol {
    private nonisolated(unsafe) let signalSubject = PassthroughSubject<Signal, Never>()
    private(set) var recentSignals: [Signal] = []
    private var timer: Timer?

    var signalPublisher: AnyPublisher<Signal, Never> {
        signalSubject.eraseToAnyPublisher()
    }

    init() {
        // Generate some initial signals
        for _ in 0..<5 {
            recentSignals.append(generateMockSignal())
        }
    }

    func subscribeToSignals() {
        // Start generating mock signals periodically
        timer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.emitMockSignal()
            }
        }
    }

    func unsubscribeFromSignals() {
        timer?.invalidate()
        timer = nil
    }

    private func emitMockSignal() {
        let signal = generateMockSignal()
        recentSignals.insert(signal, at: 0)
        if recentSignals.count > 50 {
            recentSignals.removeLast()
        }
        signalSubject.send(signal)
    }

    private func generateMockSignal() -> Signal {
        let symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
        let strategies = ["ImpulseEngine", "FundingHarvester", "BasisArbitrage", "MomentumScalper"]
        let sides = ["buy", "sell"]
        let reasonings = [
            "Strong momentum detected with RSI breakout above 70",
            "Funding rate arbitrage opportunity detected",
            "Volume spike indicates potential reversal",
            "Price broke key resistance level with high conviction",
            "Trend continuation signal after consolidation"
        ]

        let symbol = symbols.randomElement()!
        let basePrice = symbol == "BTCUSDT" ? 42000.0 : symbol == "ETHUSDT" ? 2200.0 : 100.0
        let price = basePrice * Double.random(in: 0.98...1.02)
        let side = sides.randomElement()!

        return Signal(
            id: UUID().uuidString,
            strategyId: UUID().uuidString,
            strategy: strategies.randomElement()!,
            symbol: symbol,
            side: side,
            entryPrice: price,
            size: Double.random(in: 0.01...0.5),
            stopLoss: side == "buy" ? price * 0.98 : price * 1.02,
            takeProfit: side == "buy" ? price * 1.03 : price * 0.97,
            confidence: Double.random(in: 0.4...0.95),
            urgency: Double.random(in: 0.3...0.9),
            reasoning: reasonings.randomElement()!,
            timestamp: Date(),
            metadata: nil
        )
    }
}

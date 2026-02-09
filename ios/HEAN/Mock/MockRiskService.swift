//
//  MockRiskService.swift
//  HEAN
//
//  Mock risk service for development and preview
//

import Foundation
import Combine

@MainActor
final class MockRiskService: RiskServiceProtocol {
    private nonisolated(unsafe) let riskStateSubject = CurrentValueSubject<RiskState, Never>(.normal)
    private nonisolated(unsafe) var riskMetricsSubject: CurrentValueSubject<RiskMetrics, Never>!

    private var killswitchTriggered = false

    var riskStatePublisher: AnyPublisher<RiskState, Never> {
        riskStateSubject.eraseToAnyPublisher()
    }

    var riskMetricsPublisher: AnyPublisher<RiskMetrics, Never> {
        riskMetricsSubject.eraseToAnyPublisher()
    }

    init() {
        riskMetricsSubject = CurrentValueSubject(RiskMetrics(
            drawdown: -123.45,
            drawdownPercent: 4.12,
            maxDrawdown: 8.5,
            activePositions: 3,
            totalExposure: 2500.0,
            availableMargin: 7500.0,
            quarantinedSymbols: [],
            warningThreshold: 10.0,
            criticalThreshold: 20.0
        ))
    }

    func fetchRiskStatus() async throws -> RiskGovernorStatus {
        try await Task.sleep(for: .milliseconds(300))

        return RiskGovernorStatus(
            state: riskStateSubject.value,
            metrics: riskMetricsSubject.value,
            killswitchTriggered: killswitchTriggered,
            killswitchReason: killswitchTriggered ? "Max drawdown exceeded" : nil
        )
    }

    func fetchKillswitchStatus() async throws -> KillswitchStatus {
        try await Task.sleep(for: .milliseconds(200))

        return KillswitchStatus(
            triggered: killswitchTriggered,
            reason: killswitchTriggered ? "Max drawdown exceeded" : nil,
            triggeredAt: killswitchTriggered ? Date().addingTimeInterval(-3600) : nil,
            currentDrawdown: riskMetricsSubject.value.drawdownPercent
        )
    }

    func resetKillswitch() async throws {
        try await Task.sleep(for: .milliseconds(300))
        killswitchTriggered = false
        riskStateSubject.send(.normal)
    }

    func quarantineSymbol(_ symbol: String) async throws {
        try await Task.sleep(for: .milliseconds(200))

        var metrics = riskMetricsSubject.value
        var quarantined = metrics.quarantinedSymbols
        if !quarantined.contains(symbol) {
            quarantined.append(symbol)
        }

        riskMetricsSubject.send(RiskMetrics(
            drawdown: metrics.drawdown,
            drawdownPercent: metrics.drawdownPercent,
            maxDrawdown: metrics.maxDrawdown,
            activePositions: metrics.activePositions,
            totalExposure: metrics.totalExposure,
            availableMargin: metrics.availableMargin,
            quarantinedSymbols: quarantined,
            warningThreshold: metrics.warningThreshold,
            criticalThreshold: metrics.criticalThreshold
        ))

        if riskStateSubject.value == .normal {
            riskStateSubject.send(.quarantine)
        }
    }

    func clearRiskBlocks() async throws {
        try await Task.sleep(for: .milliseconds(300))

        let metrics = riskMetricsSubject.value
        riskMetricsSubject.send(RiskMetrics(
            drawdown: metrics.drawdown,
            drawdownPercent: metrics.drawdownPercent,
            maxDrawdown: metrics.maxDrawdown,
            activePositions: metrics.activePositions,
            totalExposure: metrics.totalExposure,
            availableMargin: metrics.availableMargin,
            quarantinedSymbols: [],
            warningThreshold: metrics.warningThreshold,
            criticalThreshold: metrics.criticalThreshold
        ))

        riskStateSubject.send(.normal)
    }

    // MARK: - Mock State Simulation

    func simulateRiskEscalation() {
        let states: [RiskState] = [.normal, .softBrake, .quarantine, .hardStop]
        let currentIndex = states.firstIndex(of: riskStateSubject.value) ?? 0
        let nextIndex = min(currentIndex + 1, states.count - 1)
        riskStateSubject.send(states[nextIndex])

        if states[nextIndex] == .hardStop {
            killswitchTriggered = true
        }
    }
}

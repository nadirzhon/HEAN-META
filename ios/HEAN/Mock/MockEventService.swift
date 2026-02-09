//
//  MockEventService.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation
import Combine

/// Mock implementation of EventServiceProtocol with simulated event generation
@MainActor
final class MockEventService: EventServiceProtocol {

    // MARK: - Properties

    private nonisolated(unsafe) let eventStreamSubject = PassthroughSubject<TradingEvent, Never>()
    var eventStream: AnyPublisher<TradingEvent, Never> {
        eventStreamSubject.eraseToAnyPublisher()
    }

    private nonisolated(unsafe) let wsHealthUpdatesSubject = PassthroughSubject<WebSocketHealth, Never>()
    var wsHealthUpdates: AnyPublisher<WebSocketHealth, Never> {
        wsHealthUpdatesSubject.eraseToAnyPublisher()
    }

    private var recentEvents: [TradingEvent] = []
    private nonisolated(unsafe) var eventTimer: Timer?
    private nonisolated(unsafe) var healthTimer: Timer?

    private let connectionStartTime = Date()
    private var lastHeartbeat = Date()
    private var lastEventTimestamp = Date()
    private var eventCount = 0
    private var eventRateWindow: [Date] = []

    // MARK: - Initialization

    init() {
        recentEvents = MockDataProvider.generateEvents(count: 20)
        startEventGeneration()
        startHealthUpdates()
    }

    deinit {
        eventTimer?.invalidate()
        healthTimer?.invalidate()
    }

    // MARK: - EventServiceProtocol

    func fetchRecentEvents(limit: Int) async throws -> [TradingEvent] {
        try await Task.sleep(nanoseconds: 100_000_000) // 100ms
        return Array(recentEvents.prefix(limit))
    }

    // MARK: - Private Methods

    private func startEventGeneration() {
        // Generate events every 2-5 seconds
        scheduleNextEvent()
    }

    private func scheduleNextEvent() {
        let delay = Double.random(in: 2.0...5.0)

        eventTimer = Timer.scheduledTimer(withTimeInterval: delay, repeats: false) { [weak self] _ in
            Task { @MainActor in
                self?.generateEvent()
                self?.scheduleNextEvent()
            }
        }
    }

    private func generateEvent() {
        let events = MockDataProvider.generateEvents(count: 1)
        guard let event = events.first else { return }

        recentEvents.insert(event, at: 0)

        // Keep only last 100 events
        if recentEvents.count > 100 {
            recentEvents.removeLast()
        }

        lastEventTimestamp = event.timestamp
        eventCount += 1

        // Track for event rate calculation
        eventRateWindow.append(event.timestamp)
        let cutoff = Date().addingTimeInterval(-60) // Last 60 seconds
        eventRateWindow.removeAll { $0 < cutoff }

        eventStreamSubject.send(event)
    }

    private func startHealthUpdates() {
        healthTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            Task { @MainActor in
                self?.publishHealthUpdate()
            }
        }
    }

    private func publishHealthUpdate() {
        lastHeartbeat = Date()

        let eventsPerSecond = Double(eventRateWindow.count) / 60.0
        let connectionDuration = Date().timeIntervalSince(connectionStartTime)

        let health = WebSocketHealth(
            state: .connected,
            lastHeartbeat: lastHeartbeat,
            lastEventTimestamp: lastEventTimestamp,
            eventsPerSecond: eventsPerSecond,
            connectionDuration: connectionDuration
        )

        wsHealthUpdatesSubject.send(health)
    }
}

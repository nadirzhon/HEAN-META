//
//  EventServiceProtocol.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation
import Combine

@MainActor
protocol EventServiceProtocol {
    var eventStream: AnyPublisher<TradingEvent, Never> { get }
    var wsHealthUpdates: AnyPublisher<WebSocketHealth, Never> { get }

    func fetchRecentEvents(limit: Int) async throws -> [TradingEvent]
}

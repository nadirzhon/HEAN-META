//
//  SystemStatus.swift
//  HEAN
//

import Foundation

struct SystemStatus: Codable {
    let engineRunning: Bool
    let uptime: Double
    let wsClients: Int
    let eventsPerSec: Double
    let activeSymbols: [String]
    let redisConnected: Bool

    enum CodingKeys: String, CodingKey {
        case engineRunning = "engine_running"
        case uptime
        case wsClients = "ws_clients"
        case eventsPerSec = "events_per_sec"
        case activeSymbols = "active_symbols"
        case redisConnected = "redis_connected"
    }

    static let unknown = SystemStatus(
        engineRunning: false,
        uptime: 0,
        wsClients: 0,
        eventsPerSec: 0,
        activeSymbols: [],
        redisConnected: false
    )
}

struct HealthResponse: Codable {
    let status: String
    let version: String?
    let uptime: Double?
}

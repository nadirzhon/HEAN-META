//
//  WebSocketState.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation

enum WebSocketState: String, Codable {
    case disconnected = "DISCONNECTED"
    case connecting = "CONNECTING"
    case connected = "CONNECTED"
    case error = "ERROR"

    var displayName: String {
        rawValue
    }

    var isHealthy: Bool {
        self == .connected
    }

    var isConnected: Bool {
        self == .connected
    }

    var displayText: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .error: return "Connection Error"
        }
    }
}

struct WebSocketHealth {
    let state: WebSocketState
    let lastHeartbeat: Date?
    let lastEventTimestamp: Date?
    let eventsPerSecond: Double
    let connectionDuration: TimeInterval

    var heartbeatAge: TimeInterval? {
        guard let heartbeat = lastHeartbeat else { return nil }
        return Date().timeIntervalSince(heartbeat)
    }

    var lastEventAge: TimeInterval? {
        guard let event = lastEventTimestamp else { return nil }
        return Date().timeIntervalSince(event)
    }

    var isStale: Bool {
        guard let age = lastEventAge else { return true }
        return age > 30 // More than 30 seconds since last event
    }

    var isWarning: Bool {
        guard let age = lastEventAge else { return false }
        return age > 5 && age <= 30 // Between 5 and 30 seconds
    }

    var formattedHeartbeatAge: String {
        guard let age = heartbeatAge else { return "N/A" }
        if age < 60 {
            return String(format: "%.0fs ago", age)
        } else {
            return String(format: "%.1fm ago", age / 60)
        }
    }

    var formattedLastEventAge: String {
        guard let age = lastEventAge else { return "N/A" }
        if age < 60 {
            return String(format: "%.0fs ago", age)
        } else {
            return String(format: "%.1fm ago", age / 60)
        }
    }
}

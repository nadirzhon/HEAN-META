//
//  WebSocketEvent.swift
//  HEAN
//

import Foundation

struct WebSocketMessage {
    let topic: String
    let type: String?
    let data: [String: Any]?
    let timestamp: Date

    init(topic: String, type: String? = nil, data: [String: Any]? = nil) {
        self.topic = topic
        self.type = type
        self.data = data
        self.timestamp = Date()
    }
}

enum WebSocketConnectionState: Equatable {
    case disconnected
    case connecting
    case connected
    case reconnecting(attempt: Int)

    var isConnected: Bool {
        if case .connected = self { return true }
        return false
    }

    var displayText: String {
        switch self {
        case .disconnected: return "Disconnected"
        case .connecting: return "Connecting..."
        case .connected: return "Connected"
        case .reconnecting(let attempt): return "Reconnecting (\(attempt))..."
        }
    }
}

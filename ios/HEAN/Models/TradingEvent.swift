//
//  TradingEvent.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation

enum EventType: String, Codable {
    case signal = "SIGNAL"
    case orderPlaced = "ORDER_PLACED"
    case orderFilled = "ORDER_FILLED"
    case orderCancelled = "ORDER_CANCELLED"
    case positionOpened = "POSITION_OPENED"
    case positionClosed = "POSITION_CLOSED"
    case riskAlert = "RISK_ALERT"
    case systemInfo = "SYSTEM_INFO"
    case error = "ERROR"

    var displayName: String {
        switch self {
        case .signal: return "Signal"
        case .orderPlaced: return "Order Placed"
        case .orderFilled: return "Order Filled"
        case .orderCancelled: return "Order Cancelled"
        case .positionOpened: return "Position Opened"
        case .positionClosed: return "Position Closed"
        case .riskAlert: return "Risk Alert"
        case .systemInfo: return "System"
        case .error: return "Error"
        }
    }
}

struct TradingEvent: Identifiable, Codable {
    let id: String
    let type: EventType
    let symbol: String?
    let message: String
    let timestamp: Date
    let metadata: [String: String]?

    var formattedTime: String {
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm:ss"
        return formatter.string(from: timestamp)
    }

    var age: TimeInterval {
        Date().timeIntervalSince(timestamp)
    }

    var isRecent: Bool {
        age < 30 // Less than 30 seconds old
    }
}

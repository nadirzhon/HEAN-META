//
//  Anomaly.swift
//  HEAN
//
//  Market anomaly detection model
//

import Foundation

struct Anomaly: Codable, Identifiable {
    let id: String
    let type: String
    let severity: Double
    let description: String
    let timestamp: String

    var severityColor: String {
        if severity < 0.3 {
            return "green"
        } else if severity < 0.7 {
            return "yellow"
        } else {
            return "red"
        }
    }

    var typeIcon: String {
        switch type.lowercased() {
        case "whale":
            return "fish.fill"
        case "liquidation":
            return "bolt.fill"
        case "funding":
            return "dollarsign.circle.fill"
        case "volume":
            return "chart.bar.fill"
        case "price":
            return "chart.line.uptrend.xyaxis"
        default:
            return "exclamationmark.triangle.fill"
        }
    }

    enum CodingKeys: String, CodingKey {
        case id
        case type
        case severity
        case description
        case timestamp
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        self.type = try container.decodeIfPresent(String.self, forKey: .type) ?? "unknown"
        self.severity = try container.decodeIfPresent(Double.self, forKey: .severity) ?? 0
        self.description = try container.decodeIfPresent(String.self, forKey: .description) ?? "No description"
        self.timestamp = try container.decodeIfPresent(String.self, forKey: .timestamp) ?? ""
    }

    init(id: String, type: String, severity: Double, description: String, timestamp: String) {
        self.id = id
        self.type = type
        self.severity = severity
        self.description = description
        self.timestamp = timestamp
    }
}

//
//  BrainThought.swift
//  HEAN
//
//  AI brain thinking process timeline
//

import Foundation

struct BrainThought: Codable, Identifiable {
    let id: String
    let timestamp: String
    let stage: String
    let content: String
    let confidence: Double?

    var stageDisplayName: String {
        switch stage.lowercased() {
        case "anomaly":
            return "Anomaly Detection"
        case "physics":
            return "Physics Analysis"
        case "xray":
            return "Participant X-Ray"
        case "decision":
            return "Decision"
        default:
            return stage.capitalized
        }
    }

    var stageIcon: String {
        switch stage.lowercased() {
        case "anomaly":
            return "magnifyingglass"
        case "physics":
            return "thermometer.medium"
        case "xray":
            return "eye.fill"
        case "decision":
            return "target"
        default:
            return "brain"
        }
    }

    enum CodingKeys: String, CodingKey {
        case id
        case timestamp
        case stage
        case content
        case confidence
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        self.timestamp = try container.decodeIfPresent(String.self, forKey: .timestamp) ?? ""
        self.stage = try container.decodeIfPresent(String.self, forKey: .stage) ?? "unknown"
        self.content = try container.decodeIfPresent(String.self, forKey: .content) ?? ""
        self.confidence = try? container.decodeIfPresent(Double.self, forKey: .confidence)
    }

    init(id: String, timestamp: String, stage: String, content: String, confidence: Double?) {
        self.id = id
        self.timestamp = timestamp
        self.stage = stage
        self.content = content
        self.confidence = confidence
    }
}

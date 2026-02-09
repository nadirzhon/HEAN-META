//
//  PhysicsState.swift
//  HEAN
//
//  Physics state model for market thermodynamics
//

import Foundation

struct PhysicsState: Codable {
    let temperature: Double
    let entropy: Double
    let phase: String
    let szilardProfit: Double
    let timestamp: String

    var phaseDisplayName: String {
        switch phase.lowercased() {
        case "ice": return "Ice (Low Volatility)"
        case "water": return "Water (Normal)"
        case "vapor": return "Vapor (High Volatility)"
        default: return phase.capitalized
        }
    }

    var temperaturePercent: Double {
        min(max(temperature / 100.0, 0), 1)
    }

    var entropyPercent: Double {
        min(max(entropy, 0), 1)
    }

    enum CodingKeys: String, CodingKey {
        case temperature
        case entropy
        case phase
        case szilardProfit = "szilard_profit"
        case timestamp
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.temperature = try container.decodeIfPresent(Double.self, forKey: .temperature) ?? 0
        self.entropy = try container.decodeIfPresent(Double.self, forKey: .entropy) ?? 0
        self.phase = try container.decodeIfPresent(String.self, forKey: .phase) ?? "water"
        self.szilardProfit = try container.decodeIfPresent(Double.self, forKey: .szilardProfit) ?? 0
        // Backend may return timestamp as String or Double (Unix timestamp)
        if let ts = try? container.decode(String.self, forKey: .timestamp) {
            self.timestamp = ts
        } else if let ts = try? container.decode(Double.self, forKey: .timestamp) {
            self.timestamp = String(format: "%.0f", ts)
        } else {
            self.timestamp = ""
        }
    }

    init(temperature: Double, entropy: Double, phase: String, szilardProfit: Double, timestamp: String) {
        self.temperature = temperature
        self.entropy = entropy
        self.phase = phase
        self.szilardProfit = szilardProfit
        self.timestamp = timestamp
    }
}

//
//  BrainAnalysis.swift
//  HEAN
//
//  AI brain analysis response model
//

import Foundation

struct BrainAnalysis: Codable {
    let timestamp: String
    let thoughts: [BrainThought]
    let forces: [BrainForce]
    let signal: BrainSignal?
    let summary: String
    let marketRegime: String

    enum CodingKeys: String, CodingKey {
        case timestamp, thoughts, forces, signal, summary
        case marketRegime = "market_regime"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.timestamp = try container.decodeIfPresent(String.self, forKey: .timestamp) ?? ""
        self.thoughts = try container.decodeIfPresent([BrainThought].self, forKey: .thoughts) ?? []
        self.forces = try container.decodeIfPresent([BrainForce].self, forKey: .forces) ?? []
        self.signal = try? container.decodeIfPresent(BrainSignal.self, forKey: .signal)
        self.summary = try container.decodeIfPresent(String.self, forKey: .summary) ?? ""
        self.marketRegime = try container.decodeIfPresent(String.self, forKey: .marketRegime) ?? "unknown"
    }

    init(timestamp: String, thoughts: [BrainThought], forces: [BrainForce],
         signal: BrainSignal?, summary: String, marketRegime: String) {
        self.timestamp = timestamp
        self.thoughts = thoughts
        self.forces = forces
        self.signal = signal
        self.summary = summary
        self.marketRegime = marketRegime
    }
}

struct BrainForce: Codable, Identifiable {
    let id: String
    let name: String
    let value: Double
    let label: String

    enum CodingKeys: String, CodingKey {
        case id, name, value, label
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        self.name = try container.decodeIfPresent(String.self, forKey: .name) ?? ""
        self.value = try container.decodeIfPresent(Double.self, forKey: .value) ?? 0
        self.label = try container.decodeIfPresent(String.self, forKey: .label) ?? ""
    }

    init(id: String, name: String, value: Double, label: String) {
        self.id = id
        self.name = name
        self.value = value
        self.label = label
    }
}

struct BrainSignal: Codable {
    let direction: String
    let confidence: Double
    let entryPrice: Double?
    let targetPrice: Double?
    let stopPrice: Double?
    let riskReward: Double?
    let explanation: String?

    enum CodingKeys: String, CodingKey {
        case direction, confidence, explanation
        case entryPrice = "entry_price"
        case targetPrice = "target_price"
        case stopPrice = "stop_price"
        case riskReward = "risk_reward"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.direction = try container.decodeIfPresent(String.self, forKey: .direction) ?? "neutral"
        self.confidence = try container.decodeIfPresent(Double.self, forKey: .confidence) ?? 0
        self.entryPrice = try? container.decodeIfPresent(Double.self, forKey: .entryPrice)
        self.targetPrice = try? container.decodeIfPresent(Double.self, forKey: .targetPrice)
        self.stopPrice = try? container.decodeIfPresent(Double.self, forKey: .stopPrice)
        self.riskReward = try? container.decodeIfPresent(Double.self, forKey: .riskReward)
        self.explanation = try? container.decodeIfPresent(String.self, forKey: .explanation)
    }

    init(direction: String, confidence: Double, entryPrice: Double?, targetPrice: Double?,
         stopPrice: Double?, riskReward: Double?, explanation: String?) {
        self.direction = direction
        self.confidence = confidence
        self.entryPrice = entryPrice
        self.targetPrice = targetPrice
        self.stopPrice = stopPrice
        self.riskReward = riskReward
        self.explanation = explanation
    }

    var isLong: Bool { direction.lowercased() == "long" }
    var isShort: Bool { direction.lowercased() == "short" }
    var isNeutral: Bool { !isLong && !isShort }
    var directionColor: String { isLong ? "22C55E" : isShort ? "EF4444" : "8B92B0" }
}

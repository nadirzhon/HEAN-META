//
//  ParticipantBreakdown.swift
//  HEAN
//
//  Market participant breakdown (X-Ray)
//

import Foundation

struct ParticipantBreakdown: Codable {
    let mmActivity: Double
    let institutionalFlow: Double
    let retailSentiment: Double
    let whaleActivity: Double
    let arbPressure: Double
    let dominantPlayer: String
    let metaSignal: String

    var mmActivityPercent: Double {
        min(max(mmActivity, 0), 1)
    }

    var institutionalFlowPercent: Double {
        min(max(abs(institutionalFlow) / 1000000, 0), 1)
    }

    var retailSentimentPercent: Double {
        (retailSentiment + 1) / 2
    }

    var whaleActivityPercent: Double {
        min(max(whaleActivity, 0), 1)
    }

    var arbPressurePercent: Double {
        min(max(arbPressure, 0), 1)
    }

    enum CodingKeys: String, CodingKey {
        case mmActivity = "mm_activity"
        case institutionalFlow = "institutional_flow"
        case retailSentiment = "retail_sentiment"
        case whaleActivity = "whale_activity"
        case arbPressure = "arb_pressure"
        case dominantPlayer = "dominant_player"
        case metaSignal = "meta_signal"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.mmActivity = try container.decodeIfPresent(Double.self, forKey: .mmActivity) ?? 0
        self.institutionalFlow = try container.decodeIfPresent(Double.self, forKey: .institutionalFlow) ?? 0
        self.retailSentiment = try container.decodeIfPresent(Double.self, forKey: .retailSentiment) ?? 0
        self.whaleActivity = try container.decodeIfPresent(Double.self, forKey: .whaleActivity) ?? 0
        self.arbPressure = try container.decodeIfPresent(Double.self, forKey: .arbPressure) ?? 0
        self.dominantPlayer = try container.decodeIfPresent(String.self, forKey: .dominantPlayer) ?? "Unknown"
        self.metaSignal = try container.decodeIfPresent(String.self, forKey: .metaSignal) ?? ""
    }

    init(mmActivity: Double, institutionalFlow: Double, retailSentiment: Double,
         whaleActivity: Double, arbPressure: Double, dominantPlayer: String, metaSignal: String) {
        self.mmActivity = mmActivity
        self.institutionalFlow = institutionalFlow
        self.retailSentiment = retailSentiment
        self.whaleActivity = whaleActivity
        self.arbPressure = arbPressure
        self.dominantPlayer = dominantPlayer
        self.metaSignal = metaSignal
    }
}

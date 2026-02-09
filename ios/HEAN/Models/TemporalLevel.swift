//
//  TemporalLevel.swift
//  HEAN
//
//  Multi-timeframe temporal level model
//

import Foundation
import SwiftUI

struct TemporalLevel: Codable, Identifiable {
    let id: Int
    let name: String
    let timeframe: String
    let status: String
    let summary: String
    let details: [String: String]

    var statusColor: Color {
        switch status.lowercased() {
        case "bullish", "long", "green":
            return .green
        case "bearish", "short", "red":
            return .red
        case "neutral", "wait", "yellow":
            return .yellow
        default:
            return .gray
        }
    }

    var levelName: String {
        switch id {
        case 5:
            return "LEVEL 5: MACRO"
        case 4:
            return "LEVEL 4: SESSION"
        case 3:
            return "LEVEL 3: TACTICS"
        case 2:
            return "LEVEL 2: EXECUTION"
        case 1:
            return "LEVEL 1: MICRO"
        default:
            return "LEVEL \(id)"
        }
    }

    enum CodingKeys: String, CodingKey {
        case id
        case name
        case timeframe
        case status
        case summary
        case details
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decodeIfPresent(Int.self, forKey: .id) ?? 0
        self.name = try container.decodeIfPresent(String.self, forKey: .name) ?? ""
        self.timeframe = try container.decodeIfPresent(String.self, forKey: .timeframe) ?? ""
        self.status = try container.decodeIfPresent(String.self, forKey: .status) ?? "neutral"
        self.summary = try container.decodeIfPresent(String.self, forKey: .summary) ?? ""
        self.details = try container.decodeIfPresent([String: String].self, forKey: .details) ?? [:]
    }

    init(id: Int, name: String, timeframe: String, status: String, summary: String, details: [String: String]) {
        self.id = id
        self.name = name
        self.timeframe = timeframe
        self.status = status
        self.summary = summary
        self.details = details
    }
}

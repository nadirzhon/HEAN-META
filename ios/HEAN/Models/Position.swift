//
//  Position.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation

enum PositionSide: String, Codable {
    case long = "LONG"
    case short = "SHORT"

    var displayName: String {
        rawValue
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        let raw = try container.decode(String.self)
        switch raw.lowercased() {
        case "long", "buy":
            self = .long
        case "short", "sell":
            self = .short
        default:
            throw DecodingError.dataCorrupted(.init(
                codingPath: decoder.codingPath,
                debugDescription: "Unknown position side: \(raw)"
            ))
        }
    }
}

struct Position: Identifiable, Codable {
    let id: String
    let symbol: String
    let side: PositionSide
    let size: Double
    let entryPrice: Double
    var markPrice: Double
    var unrealizedPnL: Double
    var unrealizedPnLPercent: Double
    let leverage: Int
    let createdAt: Date?

    var isProfit: Bool {
        unrealizedPnL > 0
    }

    var notionalValue: Double {
        size * markPrice
    }

    var marginUsed: Double {
        guard leverage > 0 else { return notionalValue }
        return notionalValue / Double(leverage)
    }

    var formattedSize: String {
        String(format: "%.4f", size)
    }

    var formattedEntryPrice: String {
        if entryPrice >= 1000 {
            return String(format: "$%.2f", entryPrice)
        } else if entryPrice >= 1 {
            return String(format: "$%.3f", entryPrice)
        } else {
            return String(format: "$%.5f", entryPrice)
        }
    }

    var formattedMarkPrice: String {
        if markPrice >= 1000 {
            return String(format: "$%.2f", markPrice)
        } else if markPrice >= 1 {
            return String(format: "$%.3f", markPrice)
        } else {
            return String(format: "$%.5f", markPrice)
        }
    }

    var formattedPnL: String {
        let sign = unrealizedPnL >= 0 ? "+" : ""
        return String(format: "%@$%.2f", sign, unrealizedPnL)
    }

    var formattedPnLPercent: String {
        let sign = unrealizedPnLPercent >= 0 ? "+" : ""
        return String(format: "%@%.2f%%", sign, unrealizedPnLPercent)
    }

    enum CodingKeys: String, CodingKey {
        case id
        case symbol
        case side
        case size
        case entryPrice = "entry_price"
        case markPrice = "current_price"
        case unrealizedPnL = "unrealized_pnl"
        case unrealizedPnLPercent = "unrealized_pnl_percent"
        case leverage
        case createdAt = "created_at"
        // Backend alternative key names
        case positionId = "position_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        // id: try "id" first, then "position_id"
        if let id = try? container.decode(String.self, forKey: .id) {
            self.id = id
        } else if let posId = try? container.decode(String.self, forKey: .positionId) {
            self.id = posId
        } else {
            self.id = UUID().uuidString
        }

        self.symbol = try container.decode(String.self, forKey: .symbol)
        self.side = try container.decode(PositionSide.self, forKey: .side)
        self.size = try container.decode(Double.self, forKey: .size)
        self.entryPrice = try container.decodeIfPresent(Double.self, forKey: .entryPrice) ?? 0
        self.markPrice = try container.decodeIfPresent(Double.self, forKey: .markPrice) ?? 0
        self.unrealizedPnL = try container.decodeIfPresent(Double.self, forKey: .unrealizedPnL) ?? 0
        self.unrealizedPnLPercent = try container.decodeIfPresent(Double.self, forKey: .unrealizedPnLPercent) ?? 0
        self.leverage = try container.decodeIfPresent(Int.self, forKey: .leverage) ?? 1
        self.createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt)
    }

    init(id: String, symbol: String, side: PositionSide, size: Double, entryPrice: Double,
         markPrice: Double, unrealizedPnL: Double, unrealizedPnLPercent: Double = 0,
         leverage: Int = 1, createdAt: Date? = nil) {
        self.id = id
        self.symbol = symbol
        self.side = side
        self.size = size
        self.entryPrice = entryPrice
        self.markPrice = markPrice
        self.unrealizedPnL = unrealizedPnL
        self.unrealizedPnLPercent = unrealizedPnLPercent
        self.leverage = leverage
        self.createdAt = createdAt
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(symbol, forKey: .symbol)
        try container.encode(side, forKey: .side)
        try container.encode(size, forKey: .size)
        try container.encode(entryPrice, forKey: .entryPrice)
        try container.encode(markPrice, forKey: .markPrice)
        try container.encode(unrealizedPnL, forKey: .unrealizedPnL)
        try container.encode(unrealizedPnLPercent, forKey: .unrealizedPnLPercent)
        try container.encode(leverage, forKey: .leverage)
        try container.encodeIfPresent(createdAt, forKey: .createdAt)
    }
}

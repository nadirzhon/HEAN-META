//
//  Market.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation

struct Market: Identifiable, Decodable {
    let id: String
    let symbol: String
    let baseCurrency: String
    let quoteCurrency: String
    var name: String?
    var price: Double
    var change24h: Double
    var changePercent24h: Double
    var volume24h: Double
    var high24h: Double
    var low24h: Double
    var sparkline: [Double]?

    // Convenience aliases for view compatibility
    var lastPrice: Double { price }
    var priceChange24h: Double { changePercent24h }

    var isPositiveChange: Bool {
        change24h >= 0
    }

    var formattedPrice: String {
        if price >= 1000 {
            return String(format: "$%.2f", price)
        } else if price >= 1 {
            return String(format: "$%.3f", price)
        } else {
            return String(format: "$%.5f", price)
        }
    }

    var formattedChangePercent: String {
        let sign = changePercent24h >= 0 ? "+" : ""
        return String(format: "%@%.2f%%", sign, changePercent24h)
    }

    var formattedVolume: String {
        if volume24h >= 1_000_000_000 {
            return String(format: "$%.2fB", volume24h / 1_000_000_000)
        } else if volume24h >= 1_000_000 {
            return String(format: "$%.2fM", volume24h / 1_000_000)
        } else {
            return String(format: "$%.2fK", volume24h / 1_000)
        }
    }

    // MARK: - Memberwise Init

    init(
        id: String, symbol: String, baseCurrency: String, quoteCurrency: String,
        name: String? = nil, price: Double = 0, change24h: Double = 0,
        changePercent24h: Double = 0, volume24h: Double = 0,
        high24h: Double = 0, low24h: Double = 0, sparkline: [Double]? = nil
    ) {
        self.id = id; self.symbol = symbol; self.baseCurrency = baseCurrency
        self.quoteCurrency = quoteCurrency; self.name = name; self.price = price
        self.change24h = change24h; self.changePercent24h = changePercent24h
        self.volume24h = volume24h; self.high24h = high24h; self.low24h = low24h
        self.sparkline = sparkline
    }

    // MARK: - Custom Decoder (backend returns minimal fields)

    enum CodingKeys: String, CodingKey {
        case symbol, price, id, name, sparkline
        case baseCurrency = "base_currency"
        case quoteCurrency = "quote_currency"
        case change24h = "change_24h"
        case changePercent24h = "change_percent_24h"
        case volume24h = "volume_24h"
        case high24h = "high_24h"
        case low24h = "low_24h"
        case volume, bid, ask, timestamp
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        symbol = try c.decode(String.self, forKey: .symbol)
        price = try c.decodeIfPresent(Double.self, forKey: .price) ?? 0
        id = (try? c.decode(String.self, forKey: .id)) ?? symbol
        baseCurrency = (try? c.decode(String.self, forKey: .baseCurrency)) ?? String(symbol.dropLast(4))
        quoteCurrency = (try? c.decode(String.self, forKey: .quoteCurrency)) ?? "USDT"
        name = try c.decodeIfPresent(String.self, forKey: .name)
        change24h = try c.decodeIfPresent(Double.self, forKey: .change24h) ?? 0
        changePercent24h = try c.decodeIfPresent(Double.self, forKey: .changePercent24h) ?? 0
        volume24h = try c.decodeIfPresent(Double.self, forKey: .volume24h)
            ?? (try c.decodeIfPresent(Double.self, forKey: .volume)) ?? 0
        high24h = try c.decodeIfPresent(Double.self, forKey: .high24h) ?? 0
        low24h = try c.decodeIfPresent(Double.self, forKey: .low24h) ?? 0
        sparkline = try c.decodeIfPresent([Double].self, forKey: .sparkline)
    }
}

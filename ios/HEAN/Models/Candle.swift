//
//  Candle.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation

struct Candle: Identifiable, Codable {
    let id: String
    let timestamp: Date
    let open: Double
    let high: Double
    let low: Double
    let close: Double
    let volume: Double

    init(id: String = UUID().uuidString, timestamp: Date, open: Double, high: Double, low: Double, close: Double, volume: Double) {
        self.id = id
        self.timestamp = timestamp
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
    }

    var isGreen: Bool {
        close >= open
    }

    var bodyHeight: Double {
        abs(close - open)
    }

    var wickHigh: Double {
        high - max(open, close)
    }

    var wickLow: Double {
        min(open, close) - low
    }

    var priceRange: Double {
        high - low
    }
}

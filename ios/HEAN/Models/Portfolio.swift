//
//  Portfolio.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation

/// Matches the backend /engine/status JSON response
struct EngineStatusResponse: Codable {
    let status: String?
    let running: Bool?
    let engineState: String?
    let equity: Double?
    let dailyPnl: Double?
    let initialCapital: Double?
    let unrealizedPnl: Double?
    let realizedPnl: Double?
    let usedMargin: Double?
    let availableBalance: Double?
    let tradingMode: String?
    let isLive: Bool?
    let dryRun: Bool?
    let totalFees: Double?

    enum CodingKeys: String, CodingKey {
        case status
        case running
        case engineState = "engine_state"
        case equity
        case dailyPnl = "daily_pnl"
        case initialCapital = "initial_capital"
        case unrealizedPnl = "unrealized_pnl"
        case realizedPnl = "realized_pnl"
        case usedMargin = "used_margin"
        case availableBalance = "available_balance"
        case tradingMode = "trading_mode"
        case isLive = "is_live"
        case dryRun = "dry_run"
        case totalFees = "total_fees"
    }
}

struct Portfolio: Codable {
    var equity: Double
    var availableBalance: Double
    var usedMargin: Double
    var unrealizedPnL: Double
    var realizedPnL: Double
    let initialCapital: Double
    var lastUpdated: Date?
    let tradingMode: String?
    let isLive: Bool?
    let dryRun: Bool?
    let totalFees: Double?

    var totalPnL: Double {
        equity - initialCapital
    }

    var totalPnLPercent: Double {
        guard initialCapital > 0 else { return 0 }
        return (totalPnL / initialCapital) * 100
    }

    var isProfit: Bool {
        totalPnL > 0
    }

    var marginUsagePercent: Double {
        guard equity > 0 else { return 0 }
        return (usedMargin / equity) * 100
    }

    var formattedEquity: String {
        String(format: "$%.2f", equity)
    }

    var formattedAvailableBalance: String {
        String(format: "$%.2f", availableBalance)
    }

    var formattedTotalPnL: String {
        let sign = totalPnL >= 0 ? "+" : ""
        return String(format: "%@$%.2f", sign, totalPnL)
    }

    var formattedTotalPnLPercent: String {
        let sign = totalPnLPercent >= 0 ? "+" : ""
        return String(format: "%@%.2f%%", sign, totalPnLPercent)
    }

    var formattedUnrealizedPnL: String {
        let sign = unrealizedPnL >= 0 ? "+" : ""
        return String(format: "%@$%.2f", sign, unrealizedPnL)
    }

    var formattedRealizedPnL: String {
        let sign = realizedPnL >= 0 ? "+" : ""
        return String(format: "%@$%.2f", sign, realizedPnL)
    }

    // Keys for encoding (standard properties only)
    enum CodingKeys: String, CodingKey {
        case equity
        case availableBalance = "available_balance"
        case usedMargin = "used_margin"
        case unrealizedPnL = "unrealized_pnl"
        case realizedPnL = "realized_pnl"
        case initialCapital = "initial_capital"
        case lastUpdated = "last_updated"
    }

    // Additional keys for decoding from backend
    private enum DecodingKeys: String, CodingKey {
        case equity
        case availableBalance = "available_balance"
        case usedMargin = "used_margin"
        case unrealizedPnL = "unrealized_pnl"
        case realizedPnL = "realized_pnl"
        case initialCapital = "initial_capital"
        case lastUpdated = "last_updated"
        case dailyPnl = "daily_pnl"
        case tradingMode = "trading_mode"
        case isLive = "is_live"
        case dryRun = "dry_run"
        case totalFees = "total_fees"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DecodingKeys.self)

        self.equity = try container.decodeIfPresent(Double.self, forKey: .equity) ?? 0
        self.initialCapital = try container.decodeIfPresent(Double.self, forKey: .initialCapital) ?? 0
        self.availableBalance = try container.decodeIfPresent(Double.self, forKey: .availableBalance) ?? equity
        self.usedMargin = try container.decodeIfPresent(Double.self, forKey: .usedMargin) ?? 0
        self.unrealizedPnL = try container.decodeIfPresent(Double.self, forKey: .unrealizedPnL) ?? 0

        // realized_pnl or daily_pnl
        if let realized = try? container.decode(Double.self, forKey: .realizedPnL) {
            self.realizedPnL = realized
        } else if let daily = try? container.decode(Double.self, forKey: .dailyPnl) {
            self.realizedPnL = daily
        } else {
            self.realizedPnL = 0
        }

        self.lastUpdated = try container.decodeIfPresent(Date.self, forKey: .lastUpdated)
        self.tradingMode = try container.decodeIfPresent(String.self, forKey: .tradingMode)
        self.isLive = try container.decodeIfPresent(Bool.self, forKey: .isLive)
        self.dryRun = try container.decodeIfPresent(Bool.self, forKey: .dryRun)
        self.totalFees = try container.decodeIfPresent(Double.self, forKey: .totalFees)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(equity, forKey: .equity)
        try container.encode(availableBalance, forKey: .availableBalance)
        try container.encode(usedMargin, forKey: .usedMargin)
        try container.encode(unrealizedPnL, forKey: .unrealizedPnL)
        try container.encode(realizedPnL, forKey: .realizedPnL)
        try container.encode(initialCapital, forKey: .initialCapital)
        try container.encodeIfPresent(lastUpdated, forKey: .lastUpdated)
    }

    init(equity: Double = 0, availableBalance: Double = 0, usedMargin: Double = 0,
         unrealizedPnL: Double = 0, realizedPnL: Double = 0, initialCapital: Double = 0,
         lastUpdated: Date? = nil, tradingMode: String? = nil, isLive: Bool? = nil,
         dryRun: Bool? = nil, totalFees: Double? = nil) {
        self.equity = equity
        self.availableBalance = availableBalance
        self.usedMargin = usedMargin
        self.unrealizedPnL = unrealizedPnL
        self.realizedPnL = realizedPnL
        self.initialCapital = initialCapital
        self.lastUpdated = lastUpdated
        self.tradingMode = tradingMode
        self.isLive = isLive
        self.dryRun = dryRun
        self.totalFees = totalFees
    }

    /// Build Portfolio from EngineStatusResponse
    init(from response: EngineStatusResponse) {
        self.equity = response.equity ?? 0
        self.initialCapital = response.initialCapital ?? 0
        self.availableBalance = response.availableBalance ?? (response.equity ?? 0)
        self.usedMargin = response.usedMargin ?? 0
        self.unrealizedPnL = response.unrealizedPnl ?? 0
        self.realizedPnL = response.realizedPnl ?? response.dailyPnl ?? 0
        self.lastUpdated = Date()
        self.tradingMode = response.tradingMode
        self.isLive = response.isLive
        self.dryRun = response.dryRun
        self.totalFees = response.totalFees
    }
}

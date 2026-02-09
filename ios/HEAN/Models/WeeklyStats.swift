//
//  WeeklyStats.swift
//  HEAN
//
//  Weekly trading statistics computed model
//

import Foundation

struct WeeklyStats {
    let totalTrades: Int
    let winRate: Double
    let totalPnL: Double
    let bestTrade: Double
    let worstTrade: Double
    let avgRiskReward: Double

    static let empty = WeeklyStats(
        totalTrades: 0,
        winRate: 0,
        totalPnL: 0,
        bestTrade: 0,
        worstTrade: 0,
        avgRiskReward: 0
    )

    var isProfit: Bool { totalPnL >= 0 }

    var formattedPnL: String {
        let sign = totalPnL >= 0 ? "+" : ""
        return "\(sign)$\(String(format: "%.2f", totalPnL))"
    }

    var formattedWinRate: String {
        "\(Int(winRate * 100))%"
    }

    var formattedRR: String {
        "1:\(String(format: "%.1f", avgRiskReward))"
    }
}

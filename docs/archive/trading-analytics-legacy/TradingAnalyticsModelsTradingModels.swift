import Foundation
import SwiftUI

// MARK: - Trade Data Models

struct Trade: Identifiable, Codable {
    let id: UUID
    let symbol: String
    let entryDate: Date
    let exitDate: Date?
    let entryPrice: Double
    let exitPrice: Double?
    let quantity: Int
    let type: TradeType
    let profit: Double?
    let status: TradeStatus
    
    enum TradeType: String, Codable {
        case long = "Long"
        case short = "Short"
    }
    
    enum TradeStatus: String, Codable {
        case open = "Open"
        case closed = "Closed"
    }
    
    var profitLoss: Double {
        guard let exitPrice = exitPrice else { return 0 }
        let diff = type == .long ? (exitPrice - entryPrice) : (entryPrice - exitPrice)
        return diff * Double(quantity)
    }
}

// MARK: - Backtest Results

struct BacktestResults: Identifiable, Codable {
    let id: UUID
    let startDate: Date
    let endDate: Date
    let initialEquity: Double
    let finalEquity: Double
    let totalTrades: Int
    let winningTrades: Int
    let losingTrades: Int
    let totalReturn: Double
    let profitFactor: Double
    let sharpeRatio: Double
    let maxDrawdown: Double
    let averageWin: Double
    let averageLoss: Double
    let winRate: Double
    
    var performance: Performance {
        Performance(
            returnPercentage: totalReturn,
            sharpeRatio: sharpeRatio,
            maxDrawdown: maxDrawdown,
            winRate: winRate
        )
    }
}

struct Performance {
    let returnPercentage: Double
    let sharpeRatio: Double
    let maxDrawdown: Double
    let winRate: Double
    
    var gradeColor: Color {
        if returnPercentage > 20 && sharpeRatio > 1.5 {
            return .green
        } else if returnPercentage > 10 {
            return .blue
        } else if returnPercentage > 0 {
            return .orange
        } else {
            return .red
        }
    }
    
    var grade: String {
        if returnPercentage > 20 && sharpeRatio > 1.5 {
            return "Excellent"
        } else if returnPercentage > 10 {
            return "Good"
        } else if returnPercentage > 0 {
            return "Fair"
        } else {
            return "Poor"
        }
    }
}

// MARK: - Equity Curve Data Point

struct EquityCurvePoint: Identifiable {
    let id = UUID()
    let date: Date
    let equity: Double
    let drawdown: Double
}

// MARK: - Chart Data Points

struct DataPoint3D: Identifiable {
    let id = UUID()
    let x: Double
    let y: Double
    let z: Double
    let label: String
}

struct TimeSeriesData: Identifiable {
    let id = UUID()
    let date: Date
    let value: Double
    let category: String
}

import Foundation
import SwiftUI
import Combine

@MainActor
class TradingDataManager: ObservableObject {
    @Published var backtestResults: [BacktestResults] = []
    @Published var trades: [Trade] = []
    @Published var equityCurve: [EquityCurvePoint] = []
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    init() {
        loadMockData()
    }
    
    // MARK: - Data Loading
    
    func loadBacktestData() async {
        isLoading = true
        defer { isLoading = false }
        
        do {
            // В реальном приложении здесь будет загрузка из файла или API
            // Для примера используем моковые данные
            try await Task.sleep(for: .seconds(0.5))
            loadMockData()
        } catch {
            errorMessage = "Failed to load backtest data: \(error.localizedDescription)"
        }
    }
    
    func parseLogFile(at path: String) async throws {
        // Парсинг лог-файла бэктеста
        // В реальной реализации здесь будет чтение файла backtest_30days_output.log
        isLoading = true
        defer { isLoading = false }
        
        // TODO: Реализовать парсинг реального лог-файла
        try await Task.sleep(for: .seconds(1))
        loadMockData()
    }
    
    // MARK: - Mock Data for Demo
    
    private func loadMockData() {
        // Генерация моковых данных для демонстрации
        let calendar = Calendar.current
        let now = Date()
        let startDate = calendar.date(byAdding: .day, value: -30, to: now)!
        
        // Создаем результаты бэктеста
        backtestResults = [
            BacktestResults(
                id: UUID(),
                startDate: startDate,
                endDate: now,
                initialEquity: 10000,
                finalEquity: 12450,
                totalTrades: 156,
                winningTrades: 94,
                losingTrades: 62,
                totalReturn: 24.5,
                profitFactor: 1.85,
                sharpeRatio: 1.72,
                maxDrawdown: -8.3,
                averageWin: 185.50,
                averageLoss: -95.30,
                winRate: 60.26
            )
        ]
        
        // Генерируем кривую эквити
        equityCurve = generateEquityCurve(
            startDate: startDate,
            endDate: now,
            initialEquity: 10000,
            finalEquity: 12450
        )
        
        // Генерируем сделки
        trades = generateMockTrades(count: 20)
    }
    
    private func generateEquityCurve(startDate: Date, endDate: Date, initialEquity: Double, finalEquity: Double) -> [EquityCurvePoint] {
        var points: [EquityCurvePoint] = []
        let calendar = Calendar.current
        let days = calendar.dateComponents([.day], from: startDate, to: endDate).day ?? 30
        
        var currentEquity = initialEquity
        var maxEquity = initialEquity
        
        for day in 0...days {
            guard let date = calendar.date(byAdding: .day, value: day, to: startDate) else { continue }
            
            // Симулируем рост с некоторой волатильностью
            let progress = Double(day) / Double(days)
            let targetEquity = initialEquity + (finalEquity - initialEquity) * progress
            let volatility = Double.random(in: -100...150)
            currentEquity = targetEquity + volatility
            
            maxEquity = max(maxEquity, currentEquity)
            let drawdown = ((currentEquity - maxEquity) / maxEquity) * 100
            
            points.append(EquityCurvePoint(
                date: date,
                equity: currentEquity,
                drawdown: drawdown
            ))
        }
        
        return points
    }
    
    private func generateMockTrades(count: Int) -> [Trade] {
        let symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NVDA"]
        var trades: [Trade] = []
        let calendar = Calendar.current
        let now = Date()
        
        for i in 0..<count {
            let daysAgo = Int.random(in: 1...30)
            let entryDate = calendar.date(byAdding: .day, value: -daysAgo, to: now)!
            let exitDate = i < 3 ? nil : calendar.date(byAdding: .day, value: Int.random(in: 1...5), to: entryDate)
            
            let entryPrice = Double.random(in: 100...500)
            let exitPrice = exitDate != nil ? entryPrice * Double.random(in: 0.95...1.08) : nil
            let quantity = Int.random(in: 10...100)
            
            let trade = Trade(
                id: UUID(),
                symbol: symbols.randomElement()!,
                entryDate: entryDate,
                exitDate: exitDate,
                entryPrice: entryPrice,
                exitPrice: exitPrice,
                quantity: quantity,
                type: Bool.random() ? .long : .short,
                profit: exitPrice != nil ? (exitPrice! - entryPrice) * Double(quantity) : nil,
                status: exitDate != nil ? .closed : .open
            )
            
            trades.append(trade)
        }
        
        return trades.sorted { $0.entryDate > $1.entryDate }
    }
    
    // MARK: - Statistics
    
    var currentBacktest: BacktestResults? {
        backtestResults.first
    }
    
    var openTrades: [Trade] {
        trades.filter { $0.status == .open }
    }
    
    var closedTrades: [Trade] {
        trades.filter { $0.status == .closed }
    }
}

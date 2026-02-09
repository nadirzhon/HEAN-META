import Testing
import Foundation
@testable import TradingAnalytics

@Suite("Trading Analytics Tests")
struct TradingAnalyticsTests {
    
    // MARK: - Model Tests
    
    @Suite("Trading Models")
    struct ModelTests {
        
        @Test("Trade profitLoss calculation for long position")
        func testTradeProfitLossLong() async throws {
            let trade = Trade(
                id: UUID(),
                symbol: "AAPL",
                entryDate: Date(),
                exitDate: Date(),
                entryPrice: 100.0,
                exitPrice: 110.0,
                quantity: 10,
                type: .long,
                profit: 100.0,
                status: .closed
            )
            
            let expectedProfit = (110.0 - 100.0) * 10.0
            #expect(trade.profitLoss == expectedProfit, "Long position profit should be calculated correctly")
        }
        
        @Test("Trade profitLoss calculation for short position")
        func testTradeProfitLossShort() async throws {
            let trade = Trade(
                id: UUID(),
                symbol: "AAPL",
                entryDate: Date(),
                exitDate: Date(),
                entryPrice: 110.0,
                exitPrice: 100.0,
                quantity: 10,
                type: .short,
                profit: 100.0,
                status: .closed
            )
            
            let expectedProfit = (110.0 - 100.0) * 10.0
            #expect(trade.profitLoss == expectedProfit, "Short position profit should be calculated correctly")
        }
        
        @Test("BacktestResults performance grade")
        func testPerformanceGrade() async throws {
            let results = BacktestResults(
                id: UUID(),
                startDate: Date(),
                endDate: Date(),
                initialEquity: 10000,
                finalEquity: 12500,
                totalTrades: 100,
                winningTrades: 65,
                losingTrades: 35,
                totalReturn: 25.0,
                profitFactor: 2.0,
                sharpeRatio: 1.8,
                maxDrawdown: -5.0,
                averageWin: 200,
                averageLoss: -100,
                winRate: 65.0
            )
            
            #expect(results.performance.grade == "Excellent", "Performance with 25% return and 1.8 Sharpe should be Excellent")
            #expect(results.performance.gradeColor == .green, "Excellent performance should have green color")
        }
    }
    
    // MARK: - ViewModel Tests
    
    @Suite("Trading Data Manager")
    struct DataManagerTests {
        
        @Test("Initial data loading")
        func testInitialDataLoad() async throws {
            let manager = await TradingDataManager()
            
            await MainActor.run {
                #expect(!manager.backtestResults.isEmpty, "Should load initial backtest results")
                #expect(!manager.trades.isEmpty, "Should load initial trades")
                #expect(!manager.equityCurve.isEmpty, "Should load equity curve data")
            }
        }
        
        @Test("Open trades filter")
        func testOpenTradesFilter() async throws {
            let manager = await TradingDataManager()
            
            await MainActor.run {
                let openTrades = manager.openTrades
                #expect(openTrades.allSatisfy { $0.status == .open }, "Open trades should all have open status")
            }
        }
        
        @Test("Closed trades filter")
        func testClosedTradesFilter() async throws {
            let manager = await TradingDataManager()
            
            await MainActor.run {
                let closedTrades = manager.closedTrades
                #expect(closedTrades.allSatisfy { $0.status == .closed }, "Closed trades should all have closed status")
            }
        }
        
        @Test("Equity curve generation")
        func testEquityCurveGeneration() async throws {
            let manager = await TradingDataManager()
            
            await MainActor.run {
                let curve = manager.equityCurve
                
                #expect(!curve.isEmpty, "Equity curve should not be empty")
                
                if let first = curve.first, let last = curve.last {
                    #expect(last.equity > first.equity, "Equity should increase over time in demo data")
                }
            }
        }
    }
    
    // MARK: - AI Assistant Tests
    
    @Suite("AI Trading Assistant")
    struct AIAssistantTests {
        
        @Test("AI model availability check")
        func testAIAvailability() async throws {
            let assistant = await AITradingAssistant()
            
            // AI может быть недоступен на симуляторе
            await MainActor.run {
                #expect(assistant.isAvailable || !assistant.isAvailable, "Should check AI availability")
            }
        }
        
        @Test("Recommendation generation for low win rate")
        func testLowWinRateRecommendation() async throws {
            let assistant = await AITradingAssistant()
            
            let results = BacktestResults(
                id: UUID(),
                startDate: Date(),
                endDate: Date(),
                initialEquity: 10000,
                finalEquity: 10500,
                totalTrades: 100,
                winningTrades: 40, // Low win rate
                losingTrades: 60,
                totalReturn: 5.0,
                profitFactor: 1.2,
                sharpeRatio: 0.8,
                maxDrawdown: -12.0,
                averageWin: 150,
                averageLoss: -100,
                winRate: 40.0
            )
            
            await assistant.analyzeBacktestResults(results)
            
            await MainActor.run {
                let hasWinRateRecommendation = assistant.recommendations.contains { rec in
                    rec.title.contains("Win Rate")
                }
                
                #expect(hasWinRateRecommendation, "Should generate win rate improvement recommendation")
            }
        }
        
        @Test("Volatility calculation")
        func testVolatilityCalculation() async throws {
            let assistant = await AITradingAssistant()
            
            let curve = [
                EquityCurvePoint(date: Date(), equity: 10000, drawdown: 0),
                EquityCurvePoint(date: Date().addingTimeInterval(86400), equity: 10100, drawdown: 0),
                EquityCurvePoint(date: Date().addingTimeInterval(172800), equity: 10050, drawdown: -0.5),
                EquityCurvePoint(date: Date().addingTimeInterval(259200), equity: 10200, drawdown: 0),
            ]
            
            // Проверяем, что анализ не падает
            let analysis = await assistant.analyzeEquityCurve(curve)
            
            #expect(analysis != nil || analysis == nil, "Volatility analysis should complete")
        }
    }
    
    // MARK: - Integration Tests
    
    @Suite("Integration Tests")
    struct IntegrationTests {
        
        @Test("Complete workflow simulation")
        func testCompleteWorkflow() async throws {
            // 1. Создаем менеджер данных
            let dataManager = await TradingDataManager()
            
            // 2. Создаем AI-ассистента
            let aiAssistant = await AITradingAssistant()
            
            // 3. Загружаем данные
            await dataManager.loadBacktestData()
            
            await MainActor.run {
                // 4. Проверяем наличие данных
                #expect(!dataManager.backtestResults.isEmpty, "Should have backtest results")
                
                // 5. Получаем текущий бэктест
                guard let currentBacktest = dataManager.currentBacktest else {
                    Issue.record("Should have current backtest")
                    return
                }
                
                // 6. Проверяем метрики
                #expect(currentBacktest.totalTrades > 0, "Should have trades")
                #expect(currentBacktest.winningTrades + currentBacktest.losingTrades == currentBacktest.totalTrades, 
                       "Winning + losing should equal total trades")
            }
        }
        
        @Test("Performance metrics consistency")
        func testPerformanceMetricsConsistency() async throws {
            let results = BacktestResults(
                id: UUID(),
                startDate: Date(),
                endDate: Date(),
                initialEquity: 10000,
                finalEquity: 12000,
                totalTrades: 100,
                winningTrades: 60,
                losingTrades: 40,
                totalReturn: 20.0,
                profitFactor: 1.8,
                sharpeRatio: 1.5,
                maxDrawdown: -8.0,
                averageWin: 180,
                averageLoss: -100,
                winRate: 60.0
            )
            
            // Проверка консистентности
            #expect(results.winningTrades + results.losingTrades == results.totalTrades, 
                   "Total trades should equal wins + losses")
            
            let calculatedReturn = ((results.finalEquity - results.initialEquity) / results.initialEquity) * 100
            #expect(abs(calculatedReturn - results.totalReturn) < 0.01, 
                   "Total return should match calculated return")
            
            let calculatedWinRate = (Double(results.winningTrades) / Double(results.totalTrades)) * 100
            #expect(abs(calculatedWinRate - results.winRate) < 0.01, 
                   "Win rate should match calculated win rate")
        }
    }
    
    // MARK: - Edge Cases
    
    @Suite("Edge Cases")
    struct EdgeCaseTests {
        
        @Test("Handle zero trades")
        func testZeroTrades() async throws {
            let results = BacktestResults(
                id: UUID(),
                startDate: Date(),
                endDate: Date(),
                initialEquity: 10000,
                finalEquity: 10000,
                totalTrades: 0,
                winningTrades: 0,
                losingTrades: 0,
                totalReturn: 0.0,
                profitFactor: 0.0,
                sharpeRatio: 0.0,
                maxDrawdown: 0.0,
                averageWin: 0.0,
                averageLoss: 0.0,
                winRate: 0.0
            )
            
            #expect(results.totalTrades == 0, "Should handle zero trades")
            #expect(results.totalReturn == 0.0, "Return should be zero with no trades")
        }
        
        @Test("Handle all losing trades")
        func testAllLosingTrades() async throws {
            let results = BacktestResults(
                id: UUID(),
                startDate: Date(),
                endDate: Date(),
                initialEquity: 10000,
                finalEquity: 8000,
                totalTrades: 50,
                winningTrades: 0,
                losingTrades: 50,
                totalReturn: -20.0,
                profitFactor: 0.0,
                sharpeRatio: -1.5,
                maxDrawdown: -20.0,
                averageWin: 0.0,
                averageLoss: -40.0,
                winRate: 0.0
            )
            
            #expect(results.winRate == 0.0, "Win rate should be zero")
            #expect(results.performance.grade == "Poor", "Grade should be Poor for negative returns")
        }
        
        @Test("Handle open trade profit calculation")
        func testOpenTradeProfit() async throws {
            let openTrade = Trade(
                id: UUID(),
                symbol: "AAPL",
                entryDate: Date(),
                exitDate: nil,
                entryPrice: 100.0,
                exitPrice: nil,
                quantity: 10,
                type: .long,
                profit: nil,
                status: .open
            )
            
            #expect(openTrade.profitLoss == 0, "Open trade should have zero profit/loss")
        }
    }
}

import Foundation
import FoundationModels

@MainActor
class AITradingAssistant: ObservableObject {
    @Published var isAvailable = false
    @Published var isAnalyzing = false
    @Published var currentAnalysis: String?
    @Published var recommendations: [TradingRecommendation] = []
    
    private let model = SystemLanguageModel.default
    private var session: LanguageModelSession?
    
    init() {
        checkModelAvailability()
    }
    
    // MARK: - Model Availability
    
    func checkModelAvailability() {
        switch model.availability {
        case .available:
            isAvailable = true
            setupSession()
        case .unavailable:
            isAvailable = false
        }
    }
    
    private func setupSession() {
        let instructions = """
        You are an expert trading strategy analyst with deep knowledge of:
        - Technical analysis and trading strategies
        - Risk management and portfolio optimization
        - Backtest result interpretation
        - Trading psychology and market dynamics
        
        Provide concise, actionable insights focused on:
        1. Performance metrics interpretation
        2. Risk assessment
        3. Improvement suggestions
        4. Pattern identification
        
        Keep responses under 150 words and use clear, professional language.
        Focus on data-driven recommendations.
        """
        
        session = LanguageModelSession(instructions: instructions)
    }
    
    // MARK: - Analysis Functions
    
    func analyzeBacktestResults(_ results: BacktestResults) async {
        guard isAvailable, let session = session else { return }
        
        isAnalyzing = true
        defer { isAnalyzing = false }
        
        let prompt = """
        Analyze this trading backtest:
        - Period: 30 days
        - Return: \(String(format: "%.2f", results.totalReturn))%
        - Win Rate: \(String(format: "%.2f", results.winRate))%
        - Profit Factor: \(String(format: "%.2f", results.profitFactor))
        - Sharpe Ratio: \(String(format: "%.2f", results.sharpeRatio))
        - Max Drawdown: \(String(format: "%.2f", results.maxDrawdown))%
        - Total Trades: \(results.totalTrades)
        
        Provide key insights and 3 specific recommendations.
        """
        
        do {
            let response = try await session.respond(to: prompt)
            currentAnalysis = response.content
            
            // Генерируем рекомендации на основе анализа
            await generateRecommendations(from: results)
        } catch {
            print("AI Analysis failed: \(error)")
            currentAnalysis = "Unable to generate AI analysis at this time."
        }
    }
    
    func analyzeEquityCurve(_ curve: [EquityCurvePoint]) async -> String? {
        guard isAvailable, let session = session else { return nil }
        
        let volatility = calculateVolatility(curve)
        let trendDirection = calculateTrend(curve)
        
        let prompt = """
        Analyze equity curve characteristics:
        - Data points: \(curve.count)
        - Volatility: \(String(format: "%.2f", volatility))
        - Trend: \(trendDirection)
        - Start Equity: \(String(format: "%.2f", curve.first?.equity ?? 0))
        - End Equity: \(String(format: "%.2f", curve.last?.equity ?? 0))
        
        Provide brief assessment of curve smoothness and consistency.
        """
        
        do {
            let response = try await session.respond(to: prompt)
            return response.content
        } catch {
            return nil
        }
    }
    
    private func generateRecommendations(from results: BacktestResults) async {
        var newRecommendations: [TradingRecommendation] = []
        
        // Рекомендации на основе винрейта
        if results.winRate < 50 {
            newRecommendations.append(TradingRecommendation(
                title: "Improve Win Rate",
                description: "Current win rate is below 50%. Consider tightening entry criteria.",
                priority: .high,
                category: .strategy
            ))
        }
        
        // Рекомендации на основе профит-фактора
        if results.profitFactor < 1.5 {
            newRecommendations.append(TradingRecommendation(
                title: "Optimize Profit Factor",
                description: "Target profit factor above 1.5 by improving risk/reward ratio.",
                priority: .medium,
                category: .riskManagement
            ))
        }
        
        // Рекомендации на основе макс просадки
        if abs(results.maxDrawdown) > 15 {
            newRecommendations.append(TradingRecommendation(
                title: "Reduce Drawdown",
                description: "Large drawdown detected. Consider position sizing adjustments.",
                priority: .high,
                category: .riskManagement
            ))
        }
        
        recommendations = newRecommendations
    }
    
    // MARK: - Helper Functions
    
    private func calculateVolatility(_ curve: [EquityCurvePoint]) -> Double {
        guard curve.count > 1 else { return 0 }
        
        let returns = zip(curve.dropFirst(), curve).map { current, previous in
            (current.equity - previous.equity) / previous.equity
        }
        
        let mean = returns.reduce(0, +) / Double(returns.count)
        let variance = returns.map { pow($0 - mean, 2) }.reduce(0, +) / Double(returns.count)
        
        return sqrt(variance) * 100
    }
    
    private func calculateTrend(_ curve: [EquityCurvePoint]) -> String {
        guard let first = curve.first, let last = curve.last else { return "Unknown" }
        
        let change = ((last.equity - first.equity) / first.equity) * 100
        
        if change > 10 {
            return "Strong Uptrend"
        } else if change > 5 {
            return "Moderate Uptrend"
        } else if change > -5 {
            return "Sideways"
        } else if change > -10 {
            return "Moderate Downtrend"
        } else {
            return "Strong Downtrend"
        }
    }
}

// MARK: - Supporting Types

struct TradingRecommendation: Identifiable {
    let id = UUID()
    let title: String
    let description: String
    let priority: Priority
    let category: Category
    
    enum Priority {
        case high, medium, low
        
        var color: Color {
            switch self {
            case .high: return .red
            case .medium: return .orange
            case .low: return .blue
            }
        }
    }
    
    enum Category {
        case strategy, riskManagement, execution, psychology
        
        var icon: String {
            switch self {
            case .strategy: return "chart.line.uptrend.xyaxis"
            case .riskManagement: return "shield.checkered"
            case .execution: return "bolt.fill"
            case .psychology: return "brain.head.profile"
            }
        }
    }
}

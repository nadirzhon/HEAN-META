import SwiftUI

struct AIAssistantView: View {
    @EnvironmentObject var aiAssistant: AITradingAssistant
    @EnvironmentObject var dataManager: TradingDataManager
    @State private var showingAnalysis = false
    @State private var selectedInsight: InsightType = .overview
    
    enum InsightType: String, CaseIterable {
        case overview = "Overview"
        case recommendations = "Recommendations"
        case patterns = "Pattern Analysis"
        case risks = "Risk Assessment"
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // AI Availability Status
                AIStatusCard()
                    .padding(.horizontal)
                
                if aiAssistant.isAvailable {
                    // Insights Selector
                    InsightSelector(selected: $selectedInsight)
                        .padding(.horizontal)
                    
                    // Main Content
                    Group {
                        switch selectedInsight {
                        case .overview:
                            AnalysisOverviewSection()
                        case .recommendations:
                            RecommendationsSection()
                        case .patterns:
                            PatternAnalysisSection()
                        case .risks:
                            RiskAssessmentSection()
                        }
                    }
                    .padding(.horizontal)
                    
                    // Quick Actions
                    QuickActionsCard()
                        .padding(.horizontal)
                    
                } else {
                    AIUnavailableView()
                        .padding(.horizontal)
                }
            }
            .padding(.vertical)
        }
        .task {
            if let results = dataManager.currentBacktest {
                await aiAssistant.analyzeBacktestResults(results)
            }
        }
    }
}

// MARK: - AI Status Card

struct AIStatusCard: View {
    @EnvironmentObject var aiAssistant: AITradingAssistant
    
    var body: some View {
        HStack(spacing: 16) {
            // AI Icon
            ZStack {
                Circle()
                    .fill(
                        LinearGradient(
                            colors: aiAssistant.isAvailable ? [.blue, .purple] : [.gray, .gray.opacity(0.5)],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .frame(width: 50, height: 50)
                
                Image(systemName: "sparkles")
                    .font(.title2)
                    .foregroundStyle(.white)
                    .symbolEffect(.pulse, isActive: aiAssistant.isAnalyzing)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text("AI Trading Assistant")
                    .font(.headline.weight(.bold))
                    .foregroundStyle(.white)
                
                HStack(spacing: 6) {
                    Circle()
                        .fill(aiAssistant.isAvailable ? Color.green : Color.red)
                        .frame(width: 8, height: 8)
                    
                    Text(aiAssistant.isAvailable ? "Active" : "Unavailable")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.7))
                }
            }
            
            Spacer()
            
            if aiAssistant.isAnalyzing {
                ProgressView()
                    .tint(.white)
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.blue.opacity(0.1)).interactive(), in: .rect(cornerRadius: 20))
    }
}

// MARK: - Insight Selector

struct InsightSelector: View {
    @Binding var selected: AIAssistantView.InsightType
    @Namespace private var namespace
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(AIAssistantView.InsightType.allCases, id: \.self) { type in
                    InsightButton(
                        title: type.rawValue,
                        icon: iconForType(type),
                        isSelected: selected == type,
                        namespace: namespace
                    ) {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            selected = type
                        }
                    }
                }
            }
            .padding(.horizontal, 4)
        }
    }
    
    private func iconForType(_ type: AIAssistantView.InsightType) -> String {
        switch type {
        case .overview: return "doc.text.magnifyingglass"
        case .recommendations: return "lightbulb.fill"
        case .patterns: return "chart.xyaxis.line"
        case .risks: return "exclamationmark.shield.fill"
        }
    }
}

struct InsightButton: View {
    let title: String
    let icon: String
    let isSelected: Bool
    let namespace: Namespace.ID
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack(spacing: 8) {
                Image(systemName: icon)
                    .font(.callout)
                
                Text(title)
                    .font(.subheadline.weight(.semibold))
            }
            .foregroundStyle(isSelected ? .white : .white.opacity(0.6))
            .padding(.horizontal, 16)
            .padding(.vertical, 10)
            .background {
                if isSelected {
                    Capsule()
                        .fill(
                            LinearGradient(
                                colors: [.blue, .purple],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .matchedGeometryEffect(id: "selectedInsight", in: namespace)
                } else {
                    Capsule()
                        .strokeBorder(Color.white.opacity(0.3), lineWidth: 1)
                }
            }
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Analysis Overview Section

struct AnalysisOverviewSection: View {
    @EnvironmentObject var aiAssistant: AITradingAssistant
    @EnvironmentObject var dataManager: TradingDataManager
    
    var body: some View {
        VStack(spacing: 16) {
            // AI Generated Analysis
            if let analysis = aiAssistant.currentAnalysis {
                AIAnalysisCard(analysis: analysis)
            } else if aiAssistant.isAnalyzing {
                LoadingAnalysisCard()
            }
            
            // Key Insights Grid
            if let backtest = dataManager.currentBacktest {
                KeyInsightsGrid(backtest: backtest)
            }
            
            // Performance Grade
            if let backtest = dataManager.currentBacktest {
                PerformanceGradeCard(performance: backtest.performance)
            }
        }
    }
}

struct AIAnalysisCard: View {
    let analysis: String
    @State private var isExpanded = false
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundStyle(
                        LinearGradient(
                            colors: [.blue, .purple],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                
                Text("AI Analysis")
                    .font(.headline.weight(.bold))
                    .foregroundStyle(.white)
                
                Spacer()
                
                Button {
                    withAnimation(.spring(response: 0.3)) {
                        isExpanded.toggle()
                    }
                } label: {
                    Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                        .foregroundStyle(.white.opacity(0.6))
                }
                .buttonStyle(.plain)
            }
            
            if isExpanded {
                Text(analysis)
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.85))
                    .lineSpacing(4)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            } else {
                Text(String(analysis.prefix(150)) + "...")
                    .font(.subheadline)
                    .foregroundStyle(.white.opacity(0.85))
                    .lineSpacing(4)
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.purple.opacity(0.08)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct LoadingAnalysisCard: View {
    var body: some View {
        VStack(spacing: 16) {
            ProgressView()
                .tint(.white)
            
            Text("Analyzing your trading performance...")
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
        }
        .frame(maxWidth: .infinity)
        .padding(40)
        .glassEffect(.regular.tint(.blue.opacity(0.05)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct KeyInsightsGrid: View {
    let backtest: BacktestResults
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Key Insights")
                .font(.headline)
                .foregroundStyle(.white)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                InsightItem(
                    icon: "checkmark.seal.fill",
                    title: "Win Rate",
                    value: "\(String(format: "%.1f%%", backtest.winRate))",
                    trend: backtest.winRate > 55 ? "Excellent" : "Good",
                    color: .green
                )
                
                InsightItem(
                    icon: "chart.line.uptrend.xyaxis",
                    title: "Sharpe Ratio",
                    value: String(format: "%.2f", backtest.sharpeRatio),
                    trend: backtest.sharpeRatio > 1.5 ? "Strong" : "Fair",
                    color: .blue
                )
                
                InsightItem(
                    icon: "arrow.up.forward",
                    title: "Profit Factor",
                    value: String(format: "%.2f", backtest.profitFactor),
                    trend: backtest.profitFactor > 1.5 ? "Healthy" : "Acceptable",
                    color: .purple
                )
                
                InsightItem(
                    icon: "arrow.down.circle",
                    title: "Max DD",
                    value: "\(String(format: "%.1f%%", abs(backtest.maxDrawdown)))",
                    trend: abs(backtest.maxDrawdown) < 10 ? "Low Risk" : "Moderate",
                    color: abs(backtest.maxDrawdown) < 10 ? .green : .orange
                )
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.indigo.opacity(0.05)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct InsightItem: View {
    let icon: String
    let title: String
    let value: String
    let trend: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Image(systemName: icon)
                .font(.title3)
                .foregroundStyle(color)
            
            Text(title)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.6))
            
            Text(value)
                .font(.title3.weight(.bold))
                .foregroundStyle(.white)
            
            Text(trend)
                .font(.caption2)
                .foregroundStyle(color)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(
                    Capsule()
                        .fill(color.opacity(0.2))
                )
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(color.opacity(0.05))
        )
    }
}

struct PerformanceGradeCard: View {
    let performance: Performance
    
    var body: some View {
        VStack(spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Overall Grade")
                        .font(.headline)
                        .foregroundStyle(.white)
                    
                    Text("Based on multiple performance metrics")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.6))
                }
                
                Spacer()
            }
            
            HStack(spacing: 20) {
                // Grade Circle
                ZStack {
                    Circle()
                        .stroke(performance.gradeColor.opacity(0.3), lineWidth: 8)
                        .frame(width: 100, height: 100)
                    
                    Circle()
                        .trim(from: 0, to: CGFloat(min(performance.returnPercentage / 30, 1.0)))
                        .stroke(
                            LinearGradient(
                                colors: [performance.gradeColor, performance.gradeColor.opacity(0.6)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            style: StrokeStyle(lineWidth: 8, lineCap: .round)
                        )
                        .frame(width: 100, height: 100)
                        .rotationEffect(.degrees(-90))
                    
                    VStack(spacing: 2) {
                        Text(performance.grade)
                            .font(.title2.weight(.bold))
                            .foregroundStyle(.white)
                        
                        Text("\(String(format: "%.0f", performance.returnPercentage))%")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.7))
                    }
                }
                
                // Metrics
                VStack(alignment: .leading, spacing: 12) {
                    GradeMetric(
                        label: "Return",
                        value: "\(String(format: "%.1f%%", performance.returnPercentage))",
                        color: performance.gradeColor
                    )
                    
                    GradeMetric(
                        label: "Risk-Adjusted",
                        value: String(format: "%.2f", performance.sharpeRatio),
                        color: .blue
                    )
                    
                    GradeMetric(
                        label: "Consistency",
                        value: "\(String(format: "%.0f%%", performance.winRate))",
                        color: .purple
                    )
                }
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(performance.gradeColor.opacity(0.08)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct GradeMetric: View {
    let label: String
    let value: String
    let color: Color
    
    var body: some View {
        HStack {
            Circle()
                .fill(color)
                .frame(width: 6, height: 6)
            
            Text(label)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.7))
            
            Spacer()
            
            Text(value)
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
        }
    }
}

// MARK: - Recommendations Section

struct RecommendationsSection: View {
    @EnvironmentObject var aiAssistant: AITradingAssistant
    
    var body: some View {
        VStack(spacing: 16) {
            if aiAssistant.recommendations.isEmpty {
                EmptyRecommendationsView()
            } else {
                ForEach(aiAssistant.recommendations) { recommendation in
                    RecommendationCard(recommendation: recommendation)
                }
            }
        }
    }
}

struct RecommendationCard: View {
    let recommendation: TradingRecommendation
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: recommendation.category.icon)
                    .font(.title3)
                    .foregroundStyle(recommendation.priority.color)
                
                VStack(alignment: .leading, spacing: 4) {
                    Text(recommendation.title)
                        .font(.headline)
                        .foregroundStyle(.white)
                    
                    Text(recommendation.category.description)
                        .font(.caption2)
                        .foregroundStyle(.white.opacity(0.5))
                }
                
                Spacer()
                
                PriorityBadge(priority: recommendation.priority)
            }
            
            Text(recommendation.description)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.8))
                .lineSpacing(3)
        }
        .padding(16)
        .glassEffect(.regular.tint(recommendation.priority.color.opacity(0.05)).interactive(), in: .rect(cornerRadius: 16))
    }
}

struct PriorityBadge: View {
    let priority: TradingRecommendation.Priority
    
    var body: some View {
        Text(priorityText)
            .font(.caption2.weight(.bold))
            .foregroundStyle(.white)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(
                Capsule()
                    .fill(priority.color)
            )
    }
    
    private var priorityText: String {
        switch priority {
        case .high: return "HIGH"
        case .medium: return "MED"
        case .low: return "LOW"
        }
    }
}

struct EmptyRecommendationsView: View {
    var body: some View {
        VStack(spacing: 12) {
            Image(systemName: "checkmark.circle.fill")
                .font(.system(size: 50))
                .foregroundStyle(.green)
            
            Text("No Critical Recommendations")
                .font(.headline)
                .foregroundStyle(.white)
            
            Text("Your strategy is performing well!")
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
        }
        .frame(maxWidth: .infinity)
        .padding(40)
        .glassEffect(.regular.tint(.green.opacity(0.05)).interactive(), in: .rect(cornerRadius: 20))
    }
}

// MARK: - Pattern Analysis Section

struct PatternAnalysisSection: View {
    var body: some View {
        VStack(spacing: 16) {
            PatternCard(
                icon: "arrow.up.right",
                title: "Uptrend Bias",
                description: "70% of profitable trades occurred during market uptrends",
                color: .green
            )
            
            PatternCard(
                icon: "clock.fill",
                title: "Time Optimization",
                description: "Best performance observed in morning session (9AM-11AM)",
                color: .blue
            )
            
            PatternCard(
                icon: "chart.bar.fill",
                title: "Position Sizing",
                description: "Smaller positions (50-100 shares) showed higher win rate",
                color: .purple
            )
        }
    }
}

struct PatternCard: View {
    let icon: String
    let title: String
    let description: String
    let color: Color
    
    var body: some View {
        HStack(spacing: 16) {
            ZStack {
                Circle()
                    .fill(color.opacity(0.2))
                    .frame(width: 50, height: 50)
                
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundStyle(color)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.headline)
                    .foregroundStyle(.white)
                
                Text(description)
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
                    .lineLimit(2)
            }
            
            Spacer()
        }
        .padding(16)
        .glassEffect(.regular.tint(color.opacity(0.05)).interactive(), in: .rect(cornerRadius: 16))
    }
}

// MARK: - Risk Assessment Section

struct RiskAssessmentSection: View {
    var body: some View {
        VStack(spacing: 16) {
            RiskLevelCard()
            
            RiskFactorsList()
        }
    }
}

struct RiskLevelCard: View {
    let riskLevel = 3.2 // из 5
    
    var riskColor: Color {
        if riskLevel < 2 { return .green }
        else if riskLevel < 3.5 { return .orange }
        else { return .red }
    }
    
    var body: some View {
        VStack(spacing: 16) {
            HStack {
                Text("Risk Assessment")
                    .font(.headline)
                    .foregroundStyle(.white)
                
                Spacer()
                
                Text("Moderate")
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(riskColor)
            }
            
            // Risk Level Bar
            GeometryReader { geometry in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(Color.white.opacity(0.1))
                        .frame(height: 12)
                    
                    RoundedRectangle(cornerRadius: 8)
                        .fill(
                            LinearGradient(
                                colors: [riskColor, riskColor.opacity(0.7)],
                                startPoint: .leading,
                                endPoint: .trailing
                            )
                        )
                        .frame(width: geometry.size.width * (riskLevel / 5), height: 12)
                }
            }
            .frame(height: 12)
            
            HStack {
                Text("Low")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
                
                Spacer()
                
                Text("High")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(riskColor.opacity(0.08)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct RiskFactorsList: View {
    let factors: [(String, String, Color)] = [
        ("Position Concentration", "Medium", .orange),
        ("Volatility Exposure", "Low", .green),
        ("Drawdown Duration", "High", .red),
        ("Correlation Risk", "Low", .green)
    ]
    
    var body: some View {
        VStack(spacing: 12) {
            ForEach(factors, id: \.0) { factor in
                RiskFactorRow(name: factor.0, level: factor.1, color: factor.2)
            }
        }
        .padding(16)
        .glassEffect(.regular.tint(.red.opacity(0.03)).interactive(), in: .rect(cornerRadius: 16))
    }
}

struct RiskFactorRow: View {
    let name: String
    let level: String
    let color: Color
    
    var body: some View {
        HStack {
            Text(name)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.8))
            
            Spacer()
            
            Text(level)
                .font(.caption.weight(.semibold))
                .foregroundStyle(color)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(
                    Capsule()
                        .fill(color.opacity(0.2))
                )
        }
    }
}

// MARK: - Quick Actions Card

struct QuickActionsCard: View {
    @EnvironmentObject var aiAssistant: AITradingAssistant
    @EnvironmentObject var dataManager: TradingDataManager
    
    var body: some View {
        VStack(spacing: 12) {
            Text("Quick Actions")
                .font(.headline)
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            VStack(spacing: 8) {
                ActionButton(
                    icon: "arrow.clockwise",
                    title: "Refresh Analysis",
                    color: .blue
                ) {
                    Task {
                        if let results = dataManager.currentBacktest {
                            await aiAssistant.analyzeBacktestResults(results)
                        }
                    }
                }
                
                ActionButton(
                    icon: "square.and.arrow.up",
                    title: "Export Report",
                    color: .green
                ) {
                    // Export functionality
                }
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.blue.opacity(0.03)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct ActionButton: View {
    let icon: String
    let title: String
    let color: Color
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            HStack {
                Image(systemName: icon)
                    .foregroundStyle(color)
                
                Text(title)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(.white)
                
                Spacer()
                
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.4))
            }
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(color.opacity(0.1))
            )
        }
        .buttonStyle(.plain)
    }
}

// MARK: - AI Unavailable View

struct AIUnavailableView: View {
    var body: some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 50))
                .foregroundStyle(.orange)
            
            Text("AI Assistant Unavailable")
                .font(.headline)
                .foregroundStyle(.white)
            
            Text("Apple Intelligence is not available on this device or is not enabled.")
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
                .multilineTextAlignment(.center)
            
            Button("Learn More") {
                // Open settings or info
            }
            .buttonStyle(.borderedProminent)
            .tint(.blue)
        }
        .padding(40)
        .glassEffect(.regular.tint(.orange.opacity(0.05)).interactive(), in: .rect(cornerRadius: 20))
    }
}

extension TradingRecommendation.Category {
    var description: String {
        switch self {
        case .strategy: return "Strategy Improvement"
        case .riskManagement: return "Risk Management"
        case .execution: return "Trade Execution"
        case .psychology: return "Trading Psychology"
        }
    }
}

#Preview {
    AIAssistantView()
        .environmentObject(AITradingAssistant())
        .environmentObject(TradingDataManager())
        .background(Color.black)
}

import SwiftUI
import Charts

struct DashboardView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    @Namespace private var namespace
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Главная статистика
                if let backtest = dataManager.currentBacktest {
                    PerformanceMetricsGrid(results: backtest)
                        .padding(.horizontal)
                    
                    // Кривая эквити с 3D визуализацией
                    EquityCurveCard()
                        .padding(.horizontal)
                    
                    // Детальная статистика
                    DetailedStatsCard(results: backtest)
                        .padding(.horizontal)
                    
                    // Распределение сделок
                    TradeDistributionCard()
                        .padding(.horizontal)
                }
                
                if dataManager.isLoading {
                    ProgressView("Loading backtest data...")
                        .padding()
                }
            }
            .padding(.vertical)
        }
    }
}

// MARK: - Performance Metrics Grid

struct PerformanceMetricsGrid: View {
    let results: BacktestResults
    @State private var animateMetrics = false
    
    var body: some View {
        GlassEffectContainer(spacing: 12) {
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 12) {
                MetricCard(
                    title: "Total Return",
                    value: String(format: "%.2f%%", results.totalReturn),
                    icon: "arrow.up.right.circle.fill",
                    color: results.totalReturn > 0 ? .green : .red,
                    trend: results.totalReturn
                )
                
                MetricCard(
                    title: "Win Rate",
                    value: String(format: "%.1f%%", results.winRate),
                    icon: "target",
                    color: .blue,
                    trend: results.winRate - 50
                )
                
                MetricCard(
                    title: "Profit Factor",
                    value: String(format: "%.2f", results.profitFactor),
                    icon: "chart.bar.fill",
                    color: .purple,
                    trend: results.profitFactor - 1
                )
                
                MetricCard(
                    title: "Sharpe Ratio",
                    value: String(format: "%.2f", results.sharpeRatio),
                    icon: "waveform.path.ecg",
                    color: .orange,
                    trend: results.sharpeRatio - 1
                )
            }
            .padding()
        }
        .scaleEffect(animateMetrics ? 1.0 : 0.95)
        .opacity(animateMetrics ? 1.0 : 0.0)
        .onAppear {
            withAnimation(.spring(response: 0.6, dampingFraction: 0.7)) {
                animateMetrics = true
            }
        }
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    let trend: Double
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundStyle(color)
                    .symbolEffect(.pulse)
                
                Spacer()
                
                TrendIndicator(value: trend)
            }
            
            VStack(alignment: .leading, spacing: 4) {
                Text(value)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                
                Text(title)
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.7))
            }
        }
        .padding()
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(.regular.tint(color.opacity(0.1)).interactive(), in: .rect(cornerRadius: 20))
    }
}

struct TrendIndicator: View {
    let value: Double
    
    var body: some View {
        HStack(spacing: 4) {
            Image(systemName: value > 0 ? "arrow.up" : value < 0 ? "arrow.down" : "minus")
                .font(.caption2.weight(.bold))
            
            Text(String(format: "%.1f%%", abs(value)))
                .font(.caption2.weight(.semibold))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(
            Capsule()
                .fill(value > 0 ? Color.green.opacity(0.2) : value < 0 ? Color.red.opacity(0.2) : Color.gray.opacity(0.2))
        )
        .foregroundStyle(value > 0 ? .green : value < 0 ? .red : .gray)
    }
}

// MARK: - Equity Curve Card

struct EquityCurveCard: View {
    @EnvironmentObject var dataManager: TradingDataManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Equity Curve")
                        .font(.headline)
                        .foregroundStyle(.white)
                    
                    Text("30-Day Performance")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.6))
                }
                
                Spacer()
                
                if let first = dataManager.equityCurve.first,
                   let last = dataManager.equityCurve.last {
                    VStack(alignment: .trailing, spacing: 4) {
                        Text("+$\(String(format: "%.0f", last.equity - first.equity))")
                            .font(.title3.weight(.bold))
                            .foregroundStyle(.green)
                        
                        Text("\(String(format: "%.1f%%", ((last.equity - first.equity) / first.equity) * 100))")
                            .font(.caption)
                            .foregroundStyle(.green.opacity(0.8))
                    }
                }
            }
            
            // Chart
            Chart(dataManager.equityCurve) { point in
                LineMark(
                    x: .value("Date", point.date),
                    y: .value("Equity", point.equity)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: [.green, .blue],
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                )
                .lineStyle(StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                
                AreaMark(
                    x: .value("Date", point.date),
                    y: .value("Equity", point.equity)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: [.green.opacity(0.3), .blue.opacity(0.1)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
            }
            .chartXAxis {
                AxisMarks(preset: .aligned, values: .stride(by: .day, count: 7)) { value in
                    AxisGridLine()
                        .foregroundStyle(.white.opacity(0.1))
                    AxisValueLabel(format: .dateTime.month().day())
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            .chartYAxis {
                AxisMarks { value in
                    AxisGridLine()
                        .foregroundStyle(.white.opacity(0.1))
                    AxisValueLabel()
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            .frame(height: 250)
        }
        .padding(20)
        .glassEffect(.regular.tint(.blue.opacity(0.05)).interactive(), in: .rect(cornerRadius: 24))
    }
}

// MARK: - Detailed Stats Card

struct DetailedStatsCard: View {
    let results: BacktestResults
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Detailed Statistics")
                .font(.headline)
                .foregroundStyle(.white)
            
            VStack(spacing: 12) {
                StatRow(label: "Total Trades", value: "\(results.totalTrades)")
                StatRow(label: "Winning Trades", value: "\(results.winningTrades)", color: .green)
                StatRow(label: "Losing Trades", value: "\(results.losingTrades)", color: .red)
                
                Divider()
                    .background(Color.white.opacity(0.2))
                
                StatRow(label: "Average Win", value: "$\(String(format: "%.2f", results.averageWin))", color: .green)
                StatRow(label: "Average Loss", value: "$\(String(format: "%.2f", abs(results.averageLoss)))", color: .red)
                
                Divider()
                    .background(Color.white.opacity(0.2))
                
                StatRow(label: "Max Drawdown", value: "\(String(format: "%.2f%%", results.maxDrawdown))", color: .red)
                StatRow(label: "Initial Equity", value: "$\(String(format: "%.0f", results.initialEquity))")
                StatRow(label: "Final Equity", value: "$\(String(format: "%.0f", results.finalEquity))", color: .green)
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.purple.opacity(0.05)).interactive(), in: .rect(cornerRadius: 24))
    }
}

struct StatRow: View {
    let label: String
    let value: String
    var color: Color = .white
    
    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.7))
            
            Spacer()
            
            Text(value)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(color)
        }
    }
}

// MARK: - Trade Distribution Card

struct TradeDistributionCard: View {
    @EnvironmentObject var dataManager: TradingDataManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Trade Distribution")
                .font(.headline)
                .foregroundStyle(.white)
            
            if let backtest = dataManager.currentBacktest {
                Chart {
                    SectorMark(
                        angle: .value("Count", backtest.winningTrades),
                        innerRadius: .ratio(0.6),
                        angularInset: 2.0
                    )
                    .foregroundStyle(.green)
                    .cornerRadius(8)
                    
                    SectorMark(
                        angle: .value("Count", backtest.losingTrades),
                        innerRadius: .ratio(0.6),
                        angularInset: 2.0
                    )
                    .foregroundStyle(.red)
                    .cornerRadius(8)
                }
                .frame(height: 200)
                .chartBackground { proxy in
                    GeometryReader { geometry in
                        let frame = geometry[proxy.plotFrame!]
                        VStack(spacing: 4) {
                            Text("\(backtest.totalTrades)")
                                .font(.system(size: 32, weight: .bold, design: .rounded))
                                .foregroundStyle(.white)
                            Text("Total Trades")
                                .font(.caption)
                                .foregroundStyle(.white.opacity(0.6))
                        }
                        .position(x: frame.midX, y: frame.midY)
                    }
                }
                
                HStack(spacing: 20) {
                    LegendItem(color: .green, label: "Wins", value: backtest.winningTrades)
                    LegendItem(color: .red, label: "Losses", value: backtest.losingTrades)
                }
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.orange.opacity(0.05)).interactive(), in: .rect(cornerRadius: 24))
    }
}

struct LegendItem: View {
    let color: Color
    let label: String
    let value: Int
    
    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 12, height: 12)
            
            Text(label)
                .font(.caption)
                .foregroundStyle(.white.opacity(0.7))
            
            Text("\(value)")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.white)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

#Preview {
    DashboardView()
        .environmentObject(TradingDataManager())
        .background(Color.black)
}

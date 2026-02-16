import SwiftUI
import Charts

struct DashboardView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                if let backtest = dataManager.currentBacktest {
                    PerformanceMetricsGrid(results: backtest)
                    
                    EquityCurveCard()
                    
                    // Group smaller cards horizontally
                    HStack(alignment: .top, spacing: 20) {
                        DetailedStatsCard(results: backtest)
                        TradeDistributionCard()
                    }
                }
                
                if dataManager.isLoading {
                    ProgressView("Loading backtest data...")
                        .padding()
                        .tint(Theme.accent)
                }
            }
            .padding()
        }
        .background(Theme.background)
    }
}

// MARK: - Performance Metrics Grid

struct PerformanceMetricsGrid: View {
    let results: BacktestResults
    
    var body: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
            MetricCard(
                title: "Total Return",
                value: String(format: "%.2f%%", results.totalReturn),
                icon: "arrow.up.right.circle.fill",
                color: results.totalReturn > 0 ? Theme.positive : Theme.negative,
                trend: results.totalReturn,
                isProminent: true
            )
            
            MetricCard(
                title: "Win Rate",
                value: String(format: "%.1f%%", results.winRate),
                icon: "target",
                color: Theme.accent,
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
    }
}

struct MetricCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color
    let trend: Double
    var isProminent: Bool = false

    var body: some View {
        let cardContent = VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: icon)
                    .font(.title3)
                    .foregroundStyle(isProminent ? .white : color)

                Spacer()

                TrendIndicator(value: trend)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(value)
                    .font(.system(size: 28, weight: .bold, design: .rounded))
                    .foregroundStyle(isProminent ? .white : Theme.text)

                Text(title)
                    .font(.caption)
                    .foregroundStyle(isProminent ? .white.opacity(0.8) : Theme.secondaryText)
            }
        }

        if isProminent {
            cardContent
                .padding()
                .background(
                    LinearGradient(gradient: Gradient(colors: [color.opacity(0.7), color]), startPoint: .topLeading, endPoint: .bottomTrailing)
                )
                .cornerRadius(15)
                .shadow(color: color.opacity(0.3), radius: 10, y: 5)
        } else {
            cardContent.card()
        }
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
                .fill(value > 0 ? Theme.positive.opacity(0.1) : value < 0 ? Theme.negative.opacity(0.1) : Color.gray.opacity(0.1))
        )
        .foregroundStyle(value > 0 ? Theme.positive : value < 0 ? Theme.negative : Theme.secondaryText)
    }
}

// MARK: - Equity Curve Card

enum Timeframe: String, CaseIterable, Identifiable {
    case week = "1W"
    case month = "1M"
    case threeMonths = "3M"
    case all = "All"
    
    var id: String { self.rawValue }
}

struct EquityCurveCard: View {
    @EnvironmentObject var dataManager: TradingDataManager
    @State private var selectedTimeframe: Timeframe = .month

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Equity Curve")
                        .font(.headline)
                        .foregroundStyle(Theme.text)
                    
                    Text(timeframeDescription)
                        .font(.caption)
                        .foregroundStyle(Theme.secondaryText)
                }
                
                Spacer()
                
                if let first = dataManager.equityCurve.first, let last = dataManager.equityCurve.last {
                    let profit = last.equity - first.equity
                    VStack(alignment: .trailing, spacing: 4) {
                        Text(String(format: "%@$%.0f", profit >= 0 ? "+" : "", profit))
                            .font(.title3.weight(.bold))
                            .foregroundStyle(profit >= 0 ? Theme.positive : Theme.negative)
                    }
                }
            }

            Picker("Timeframe", selection: $selectedTimeframe) {
                ForEach(Timeframe.allCases) { timeframe in
                    Text(timeframe.rawValue).tag(timeframe)
                }
            }
            .pickerStyle(.segmented)
            .onChange(of: selectedTimeframe) {
                // This is a placeholder for the action to update data.
                // In a real app, you would call a method on dataManager, like:
                // dataManager.updateEquityCurve(for: selectedTimeframe)
            }

            Chart(dataManager.equityCurve) { point in
                LineMark(x: .value("Date", point.date), y: .value("Equity", point.equity))
                    .foregroundStyle(Theme.accent)
                    .lineStyle(StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
                
                AreaMark(x: .value("Date", point.date), y: .value("Equity", point.equity))
                    .foregroundStyle(LinearGradient(colors: [Theme.accent.opacity(0.3), Theme.accent.opacity(0.0)], startPoint: .top, endPoint: .bottom))
                
                if let initialEquity = dataManager.equityCurve.first?.equity {
                    RuleMark(y: .value("Start", initialEquity))
                        .foregroundStyle(Theme.secondaryText)
                        .lineStyle(StrokeStyle(lineWidth: 1, dash: [5, 5]))
                        .annotation(position: .top, alignment: .leading) {
                            Text("Start")
                                .font(.caption)
                                .foregroundStyle(Theme.secondaryText)
                                .padding(.leading, 5)
                        }
                }
            }
            .chartXAxis {
                AxisMarks(preset: .aligned, values: .stride(by: .day, count: 7)) { value in
                    AxisGridLine().foregroundStyle(Theme.separator)
                    AxisValueLabel(format: .dateTime.month().day()).foregroundStyle(Theme.secondaryText)
                }
            }
            .chartYAxis {
                AxisMarks { value in
                    AxisGridLine().foregroundStyle(Theme.separator)
                    AxisValueLabel().foregroundStyle(Theme.secondaryText)
                }
            }
            .frame(height: 250)
        }
        .card()
    }
    
    private var timeframeDescription: String {
        switch selectedTimeframe {
        case .week: return "7-Day Performance"
        case .month: return "30-Day Performance"
        case .threeMonths: return "90-Day Performance"
        case .all: return "All-Time Performance"
        }
    }
}

// MARK: - Detailed Stats Card

struct DetailedStatsCard: View {
    let results: BacktestResults
    
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("Detailed Statistics")
                .font(.headline)
                .foregroundStyle(Theme.text)
            
            VStack(spacing: 12) {
                StatRow(label: "Total Trades", value: "\(results.totalTrades)")
                StatRow(label: "Winning Trades", value: "\(results.winningTrades)", color: Theme.positive)
                StatRow(label: "Losing Trades", value: "\(results.losingTrades)", color: Theme.negative)
                
                Divider().background(Theme.separator)
                
                StatRow(label: "Average Win", value: "$\(String(format: "%.2f", results.averageWin))", color: Theme.positive)
                StatRow(label: "Average Loss", value: "$\(String(format: "%.2f", abs(results.averageLoss)))", color: Theme.negative)
                
                Divider().background(Theme.separator)
                
                StatRow(label: "Max Drawdown", value: "\(String(format: "%.2f%%", results.maxDrawdown))", color: Theme.negative)
                StatRow(label: "Initial Equity", value: "$\(String(format: "%.0f", results.initialEquity))")
                StatRow(label: "Final Equity", value: "$\(String(format: "%.0f", results.finalEquity))", color: Theme.positive)
            }
        }
        .card()
    }
}

struct StatRow: View {
    let label: String
    let value: String
    var color: Color = Theme.text
    
    var body: some View {
        HStack {
            Text(label)
                .font(.subheadline)
                .foregroundStyle(Theme.secondaryText)
            
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
                .foregroundStyle(Theme.text)
            
            if let backtest = dataManager.currentBacktest {
                Chart {
                    SectorMark(angle: .value("Count", backtest.winningTrades), innerRadius: .ratio(0.6), angularInset: 2.0)
                        .foregroundStyle(Theme.positive)
                        .cornerRadius(8)
                    
                    SectorMark(angle: .value("Count", backtest.losingTrades), innerRadius: .ratio(0.6), angularInset: 2.0)
                        .foregroundStyle(Theme.negative)
                        .cornerRadius(8)
                }
                .frame(height: 200)
                .chartBackground { proxy in
                    GeometryReader { geometry in
                        let frame = geometry[proxy.plotFrame!]
                        VStack(spacing: 4) {
                            Text("\(backtest.totalTrades)")
                                .font(.system(size: 32, weight: .bold, design: .rounded))
                                .foregroundStyle(Theme.text)
                            Text("Total Trades")
                                .font(.caption)
                                .foregroundStyle(Theme.secondaryText)
                        }
                        .position(x: frame.midX, y: frame.midY)
                    }
                }
                
                HStack(spacing: 20) {
                    LegendItem(color: Theme.positive, label: "Wins", value: backtest.winningTrades)
                    LegendItem(color: Theme.negative, label: "Losses", value: backtest.losingTrades)
                }
            }
        }
        .card()
    }
}

struct LegendItem: View {
    let color: Color
    let label: String
    let value: Int
    
    var body: some View {
        HStack(spacing: 8) {
            Circle().fill(color).frame(width: 12, height: 12)
            Text(label).font(.caption).foregroundStyle(Theme.secondaryText)
            Text("\(value)").font(.caption.weight(.semibold)).foregroundStyle(Theme.text)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

#Preview {
    DashboardView()
        .environmentObject(TradingDataManager.preview)
        .background(Theme.background)
}

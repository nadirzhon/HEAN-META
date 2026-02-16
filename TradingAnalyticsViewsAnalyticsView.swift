import SwiftUI
import Charts

// Assuming Chart3D and related components are available and will adapt to the environment.

struct AnalyticsView: View {
    @State private var selectedVisualization: VisualizationType = .performance3D
    
    enum VisualizationType: String, CaseIterable, Identifiable {
        var id: String { self.rawValue }
        case performance3D = "3D Performance"
        case heatmap = "Trade Heatmap"
        case returns = "Returns"
        case drawdown = "Drawdown"
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 24) {
                // Visualization Selector
                Picker("Visualization", selection: $selectedVisualization) {
                    ForEach(VisualizationType.allCases) { type in
                        Text(type.rawValue).tag(type)
                    }
                }
                .pickerStyle(.segmented)
                .padding(.horizontal)
                
                // Main Visualization Area
                Group {
                    switch selectedVisualization {
                    case .performance3D:
                        Performance3DContainerView()
                    case .heatmap:
                        TradeHeatmapView()
                    case .returns:
                        ReturnsDistributionView()
                    case .drawdown:
                        DrawdownAnalysisView()
                    }
                }
                .padding(.horizontal)
                
                AdvancedMetricsCard()
                    .padding(.horizontal)
                
                CorrelationMatrixView()
                    .padding(.horizontal)
            }
            .padding(.vertical)
        }
        .background(Theme.background)
        .navigationTitle("Analytics")
    }
}

// MARK: - 3D Performance Container
struct Performance3DContainerView: View {
    @State private var chartPose: Chart3DPose = .default
    
    var body: some View {
        VStack(spacing: 16) {
            Performance3DView(chartPose: $chartPose)
            Chart3DControls(chartPose: $chartPose)
        }
    }
}

// MARK: - 3D Performance View
struct Performance3DView: View {
    @Binding var chartPose: Chart3DPose
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Header(title: "3D Performance Surface", subtitle: "Interactive visualization", icon: "cube.transparent")
            
            Chart3D {
                SurfacePlot(x: "Time", y: "Returns", z: "Risk", function: { x, y in sin(x * 0.5) * cos(y * 0.8) * 1.5 + 0.5 })
                    .roughness(0.2)
                    .material(Material.lit(color: .blue))
            }
            .chart3DPose($chartPose)
            .chart3DCameraProjection(.perspective)
            .frame(height: 350)
            
            Text("Drag to rotate • Pinch to zoom")
                .font(.caption2)
                .foregroundStyle(Theme.secondaryText)
                .frame(maxWidth: .infinity, alignment: .center)
        }
        .card()
    }
}

// MARK: - Chart 3D Controls
struct Chart3DControls: View {
    @Binding var chartPose: Chart3DPose
    
    var body: some View {
        HStack(spacing: 8) {
            ControlButton(title: "Front", icon: "arrow.forward") { withAnimation(.spring) { chartPose = .front } }
            ControlButton(title: "Top", icon: "arrow.up") { withAnimation(.spring) { chartPose = .top } }
            ControlButton(title: "Side", icon: "arrow.left.and.right") { withAnimation(.spring) { chartPose = .right } }
            ControlButton(title: "Default", icon: "cube") { withAnimation(.spring) { chartPose = .default } }
        }
        .padding()
        .background(Theme.cardBackground)
        .cornerRadius(16)
        .shadow(color: .black.opacity(0.05), radius: 5)
    }
}

struct ControlButton: View {
    let title: String
    let icon: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Image(systemName: icon).font(.headline)
                Text(title).font(.caption2)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .foregroundStyle(Theme.accent)
            .background(Theme.accent.opacity(0.1))
            .cornerRadius(10)
        }
        .buttonStyle(.plain)
    }
}


// MARK: - Trade Heatmap View
struct TradeHeatmapView: View {
    private var hourlyData: [(hour: Int, day: String, count: Int)] {
        let days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        return days.flatMap { day in (9...16).map { hour in (hour: hour, day: day, count: Int.random(in: 0...15)) } }
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Header(title: "Trade Activity Heatmap", subtitle: "Trades by time of day", icon: "flame")
            
            Chart(hourlyData, id: \.hour) { item in
                RectangleMark(x: .value("Hour", "\(item.hour):00"), y: .value("Day", item.day))
                    .foregroundStyle(by: .value("Activity", item.count))
                    .cornerRadius(4)
            }
            .chartForegroundStyleScale(range: Gradient(colors: [Theme.accent.opacity(0.1), Theme.accent]))
            .frame(height: 250)
            .standardChartAxes()
        }
        .card()
    }
}

// MARK: - Returns Distribution View
struct ReturnsDistributionView: View {
    private var distributionData: [(range: String, count: Int)] {
        [("-5%...", 5), ("-3%...", 12), ("-1%...", 25), ("0%...", 35), ("1%...", 28), ("3%...", 18), ("5%+", 8)]
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Header(title: "Returns Distribution", subtitle: "Trade outcome spread", icon: "chart.bar.xaxis")
            
            Chart(distributionData, id: \.range) { item in
                BarMark(x: .value("Range", item.range), y: .value("Count", item.count))
                    .foregroundStyle(item.range.contains("-") ? Theme.negative : Theme.positive)
                    .cornerRadius(6)
            }
            .frame(height: 250)
            .standardChartAxes()
        }
        .card()
    }
}

// MARK: - Drawdown Analysis View
struct DrawdownAnalysisView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Header(title: "Drawdown Analysis", subtitle: "Maximum portfolio decline", icon: "chart.line.flaw")
            
            Chart(dataManager.equityCurve) { point in
                LineMark(x: .value("Date", point.date), y: .value("Drawdown %", point.drawdown))
                    .foregroundStyle(Theme.negative)
                AreaMark(x: .value("Date", point.date), y: .value("Drawdown %", point.drawdown))
                    .foregroundStyle(LinearGradient(colors: [Theme.negative.opacity(0.4), .clear], startPoint: .top, endPoint: .bottom))
            }
            .frame(height: 200)
            .standardChartAxes()
            
            HStack(spacing: 20) {
                DrawdownStat(label: "Max DD", value: "\(String(format: "%.2f%%", dataManager.currentBacktest?.maxDrawdown ?? 0))")
                DrawdownStat(label: "Avg DD", value: "-4.2%") // Example data
                DrawdownStat(label: "Recovery", value: "5.2 days") // Example data
            }
        }
        .card()
    }
}

struct DrawdownStat: View {
    let label: String
    let value: String
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value).font(.title3.weight(.bold)).foregroundStyle(Theme.text)
            Text(label).font(.caption2).foregroundStyle(Theme.secondaryText)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Advanced Metrics Card
struct AdvancedMetricsCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Header(title: "Advanced Metrics", subtitle: "Calculated risk & return ratios", icon: "gauge.high")
            
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                AdvancedMetricItem(label: "Calmar Ratio", value: "2.95", description: "Return / Max Drawdown", color: Theme.accent)
                AdvancedMetricItem(label: "Sortino Ratio", value: "2.13", description: "Downside deviation", color: .purple)
                AdvancedMetricItem(label: "R-Squared", value: "0.87", description: "Trend consistency", color: Theme.positive)
                AdvancedMetricItem(label: "Kelly %", value: "18.5%", description: "Optimal position size", color: .orange)
            }
        }
        .card()
    }
}

struct AdvancedMetricItem: View {
    let label: String
    let value: String
    let description: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Label(label, systemImage: "circle.fill").font(.caption).foregroundStyle(color)
            Text(value).font(.title2.weight(.bold)).foregroundStyle(Theme.text)
            Text(description).font(.caption2).foregroundStyle(Theme.secondaryText)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(color.opacity(0.1))
        .cornerRadius(12)
    }
}

// MARK: - Correlation Matrix View
struct CorrelationMatrixView: View {
    private let metrics = ["Return", "Win Rate", "Sharpe", "Profit F."]
    private let correlations = [[1.0, 0.75, 0.82, 0.68], [0.75, 1.0, 0.65, 0.71], [0.82, 0.65, 1.0, 0.59], [0.68, 0.71, 0.59, 1.0]]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Header(title: "Correlation Matrix", subtitle: "Metric interdependencies", icon: "square.grid.3x3")
            
            VStack(spacing: 4) {
                ForEach(0..<metrics.count, id: \.self) { row in
                    HStack(spacing: 4) {
                        Text(metrics[row]).font(.caption2).foregroundStyle(Theme.secondaryText).frame(width: 70, alignment: .leading)
                        ForEach(0..<metrics.count, id: \.self) { col in
                            CorrelationCell(value: correlations[row][col])
                        }
                    }
                }
            }
        }
        .card()
    }
}

struct CorrelationCell: View {
    let value: Double
    var body: some View {
        Text(String(format: "%.2f", value))
            .font(.caption2.weight(.medium))
            .foregroundStyle(Theme.text)
            .frame(maxWidth: .infinity, minHeight: 32)
            .background(Color.blue.opacity(value * 0.4))
            .cornerRadius(4)
    }
}

// MARK: - Reusable Components
struct Header: View {
    let title: String
    let subtitle: String
    let icon: String
    
    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 4) {
                Text(title).font(.headline).foregroundStyle(Theme.text)
                Text(subtitle).font(.caption).foregroundStyle(Theme.secondaryText)
            }
            Spacer()
            Image(systemName: icon).font(.title2).foregroundStyle(Theme.accent).symbolEffect(.scale.up)
        }
    }
}

extension View {
    func standardChartAxes() -> some View {
        self.chartXAxis {
            AxisMarks { _ in
                AxisGridLine().foregroundStyle(Theme.separator)
                AxisValueLabel().foregroundStyle(Theme.secondaryText).font(.caption2)
            }
        }
        .chartYAxis {
            AxisMarks { _ in
                AxisGridLine().foregroundStyle(Theme.separator)
                AxisValueLabel().foregroundStyle(Theme.secondaryText).font(.caption2)
            }
        }
    }
}

#Preview {
    AnalyticsView()
        .environmentObject(TradingDataManager.preview)
        .background(Theme.background)
}

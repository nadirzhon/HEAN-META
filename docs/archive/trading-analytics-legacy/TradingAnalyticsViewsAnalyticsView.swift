import SwiftUI
import Charts

struct AnalyticsView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    @State private var selectedVisualization: VisualizationType = .performance3D
    @State private var chartPose: Chart3DPose = .default
    
    enum VisualizationType: String, CaseIterable {
        case performance3D = "3D Performance"
        case heatmap = "Trade Heatmap"
        case returns = "Returns Distribution"
        case drawdown = "Drawdown Analysis"
    }
    
    var body: some View {
        ScrollView {
            VStack(spacing: 20) {
                // Визуализация селектор
                VisualizationSelector(selected: $selectedVisualization)
                    .padding(.horizontal)
                
                // Главная визуализация
                Group {
                    switch selectedVisualization {
                    case .performance3D:
                        Performance3DView(chartPose: $chartPose)
                    case .heatmap:
                        TradeHeatmapView()
                    case .returns:
                        ReturnsDistributionView()
                    case .drawdown:
                        DrawdownAnalysisView()
                    }
                }
                .padding(.horizontal)
                
                // Контролы для 3D графика
                if selectedVisualization == .performance3D {
                    Chart3DControls(chartPose: $chartPose)
                        .padding(.horizontal)
                }
                
                // Дополнительные метрики
                AdvancedMetricsCard()
                    .padding(.horizontal)
                
                // Корреляционный анализ
                CorrelationMatrixView()
                    .padding(.horizontal)
            }
            .padding(.vertical)
        }
    }
}

// MARK: - Visualization Selector

struct VisualizationSelector: View {
    @Binding var selected: AnalyticsView.VisualizationType
    @Namespace private var namespace
    
    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(AnalyticsView.VisualizationType.allCases, id: \.self) { type in
                    Button {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                            selected = type
                        }
                    } label: {
                        Text(type.rawValue)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(selected == type ? .white : .white.opacity(0.6))
                            .padding(.horizontal, 16)
                            .padding(.vertical, 10)
                            .background {
                                if selected == type {
                                    Capsule()
                                        .fill(Color.white.opacity(0.2))
                                        .matchedGeometryEffect(id: "selectedVis", in: namespace)
                                }
                            }
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 4)
        }
        .glassEffect(.regular.interactive(), in: .capsule)
    }
}

// MARK: - 3D Performance View

struct Performance3DView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    @Binding var chartPose: Chart3DPose
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("3D Performance Surface")
                        .font(.headline)
                        .foregroundStyle(.white)
                    
                    Text("Interactive visualization")
                        .font(.caption)
                        .foregroundStyle(.white.opacity(0.6))
                }
                
                Spacer()
                
                Image(systemName: "cube.transparent")
                    .font(.title2)
                    .foregroundStyle(.blue)
                    .symbolEffect(.rotate)
            }
            
            // 3D Chart
            Chart3D {
                SurfacePlot(
                    x: "Time",
                    y: "Returns",
                    z: "Risk",
                    function: { x, y in
                        // Функция производительности: сочетание времени и риска
                        let timeComponent = sin(x * 0.5)
                        let riskComponent = cos(y * 0.8)
                        return timeComponent * riskComponent * 1.5 + 0.5
                    }
                )
                .roughness(0.2)
            }
            .chart3DPose($chartPose)
            .chart3DCameraProjection(.perspective)
            .frame(height: 400)
            
            Text("Drag to rotate • Pinch to zoom")
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.5))
                .frame(maxWidth: .infinity, alignment: .center)
        }
        .padding(20)
        .glassEffect(.regular.tint(.blue.opacity(0.08)).interactive(), in: .rect(cornerRadius: 24))
    }
}

// MARK: - Chart 3D Controls

struct Chart3DControls: View {
    @Binding var chartPose: Chart3DPose
    
    var body: some View {
        VStack(spacing: 12) {
            Text("3D View Controls")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity, alignment: .leading)
            
            HStack(spacing: 8) {
                ControlButton(title: "Front", icon: "arrow.forward") {
                    withAnimation(.spring(response: 0.4)) {
                        chartPose = .front
                    }
                }
                
                ControlButton(title: "Top", icon: "arrow.up") {
                    withAnimation(.spring(response: 0.4)) {
                        chartPose = .top
                    }
                }
                
                ControlButton(title: "Side", icon: "arrow.left.and.right") {
                    withAnimation(.spring(response: 0.4)) {
                        chartPose = .right
                    }
                }
                
                ControlButton(title: "Default", icon: "cube") {
                    withAnimation(.spring(response: 0.4)) {
                        chartPose = .default
                    }
                }
            }
        }
        .padding(16)
        .glassEffect(.regular.interactive(), in: .rect(cornerRadius: 16))
    }
}

struct ControlButton: View {
    let title: String
    let icon: String
    let action: () -> Void
    
    var body: some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.title3)
                
                Text(title)
                    .font(.caption2)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .foregroundStyle(.white)
        }
        .buttonStyle(.plain)
        .glassEffect(.regular.tint(.white.opacity(0.05)).interactive(), in: .rect(cornerRadius: 12))
    }
}

// MARK: - Trade Heatmap View

struct TradeHeatmapView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    
    private var hourlyData: [(hour: Int, day: String, count: Int)] {
        // Генерируем данные для тепловой карты
        let days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        var data: [(hour: Int, day: String, count: Int)] = []
        
        for day in days {
            for hour in 9...16 { // Торговые часы
                let count = Int.random(in: 0...15)
                data.append((hour: hour, day: day, count: count))
            }
        }
        
        return data
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Trade Activity Heatmap")
                    .font(.headline)
                    .foregroundStyle(.white)
                
                Text("Trades by time of day")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.6))
            }
            
            Chart(hourlyData, id: \.hour) { item in
                RectangleMark(
                    x: .value("Hour", "\(item.hour):00"),
                    y: .value("Day", item.day),
                    width: 30,
                    height: 30
                )
                .foregroundStyle(by: .value("Activity", item.count))
                .cornerRadius(6)
            }
            .chartForegroundStyleScale(
                range: Gradient(colors: [
                    .blue.opacity(0.3),
                    .blue,
                    .purple,
                    .pink,
                    .red
                ])
            )
            .frame(height: 250)
            .chartXAxis {
                AxisMarks { _ in
                    AxisValueLabel()
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            .chartYAxis {
                AxisMarks { _ in
                    AxisValueLabel()
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.purple.opacity(0.08)).interactive(), in: .rect(cornerRadius: 24))
    }
}

// MARK: - Returns Distribution View

struct ReturnsDistributionView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    
    private var distributionData: [(range: String, count: Int)] {
        [
            ("-5% to -3%", 5),
            ("-3% to -1%", 12),
            ("-1% to 0%", 25),
            ("0% to 1%", 35),
            ("1% to 3%", 28),
            ("3% to 5%", 18),
            ("5% to 7%", 8)
        ]
    }
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Returns Distribution")
                    .font(.headline)
                    .foregroundStyle(.white)
                
                Text("Trade outcome spread")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.6))
            }
            
            Chart(distributionData, id: \.range) { item in
                BarMark(
                    x: .value("Range", item.range),
                    y: .value("Count", item.count)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: item.range.contains("-") ? [.red, .orange] : [.green, .blue],
                        startPoint: .bottom,
                        endPoint: .top
                    )
                )
                .cornerRadius(8)
            }
            .frame(height: 250)
            .chartXAxis {
                AxisMarks { _ in
                    AxisValueLabel()
                        .foregroundStyle(.white.opacity(0.6))
                        .font(.caption2)
                }
            }
            .chartYAxis {
                AxisMarks { _ in
                    AxisGridLine()
                        .foregroundStyle(.white.opacity(0.1))
                    AxisValueLabel()
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.green.opacity(0.08)).interactive(), in: .rect(cornerRadius: 24))
    }
}

// MARK: - Drawdown Analysis View

struct DrawdownAnalysisView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Drawdown Analysis")
                    .font(.headline)
                    .foregroundStyle(.white)
                
                Text("Maximum portfolio decline")
                    .font(.caption)
                    .foregroundStyle(.white.opacity(0.6))
            }
            
            Chart(dataManager.equityCurve) { point in
                LineMark(
                    x: .value("Date", point.date),
                    y: .value("Drawdown %", point.drawdown)
                )
                .foregroundStyle(.red)
                .lineStyle(StrokeStyle(lineWidth: 2))
                
                AreaMark(
                    x: .value("Date", point.date),
                    y: .value("Drawdown %", point.drawdown)
                )
                .foregroundStyle(
                    LinearGradient(
                        colors: [.red.opacity(0.4), .red.opacity(0.05)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )
            }
            .frame(height: 200)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day, count: 7)) { _ in
                    AxisGridLine()
                        .foregroundStyle(.white.opacity(0.1))
                    AxisValueLabel(format: .dateTime.month().day())
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            .chartYAxis {
                AxisMarks { _ in
                    AxisGridLine()
                        .foregroundStyle(.white.opacity(0.1))
                    AxisValueLabel()
                        .foregroundStyle(.white.opacity(0.6))
                }
            }
            
            // Статистика просадок
            HStack(spacing: 20) {
                DrawdownStat(label: "Max DD", value: "\(String(format: "%.2f%%", dataManager.currentBacktest?.maxDrawdown ?? 0))")
                DrawdownStat(label: "Avg DD", value: "-4.2%")
                DrawdownStat(label: "Recovery", value: "5.2 days")
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.red.opacity(0.08)).interactive(), in: .rect(cornerRadius: 24))
    }
}

struct DrawdownStat: View {
    let label: String
    let value: String
    
    var body: some View {
        VStack(spacing: 4) {
            Text(value)
                .font(.title3.weight(.bold))
                .foregroundStyle(.white)
            
            Text(label)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.6))
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Advanced Metrics Card

struct AdvancedMetricsCard: View {
    @EnvironmentObject var dataManager: TradingDataManager
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Advanced Metrics")
                .font(.headline)
                .foregroundStyle(.white)
            
            LazyVGrid(columns: [
                GridItem(.flexible()),
                GridItem(.flexible())
            ], spacing: 16) {
                AdvancedMetricItem(
                    label: "Calmar Ratio",
                    value: "2.95",
                    description: "Return / Max Drawdown",
                    color: .blue
                )
                
                AdvancedMetricItem(
                    label: "Sortino Ratio",
                    value: "2.13",
                    description: "Downside deviation",
                    color: .purple
                )
                
                AdvancedMetricItem(
                    label: "R-Squared",
                    value: "0.87",
                    description: "Trend consistency",
                    color: .green
                )
                
                AdvancedMetricItem(
                    label: "Kelly %",
                    value: "18.5%",
                    description: "Optimal position size",
                    color: .orange
                )
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.indigo.opacity(0.08)).interactive(), in: .rect(cornerRadius: 24))
    }
}

struct AdvancedMetricItem: View {
    let label: String
    let value: String
    let description: String
    let color: Color
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Circle()
                    .fill(color)
                    .frame(width: 8, height: 8)
                
                Text(label)
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.white.opacity(0.7))
            }
            
            Text(value)
                .font(.title3.weight(.bold))
                .foregroundStyle(.white)
            
            Text(description)
                .font(.caption2)
                .foregroundStyle(.white.opacity(0.5))
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: 12)
                .fill(color.opacity(0.1))
        )
    }
}

// MARK: - Correlation Matrix View

struct CorrelationMatrixView: View {
    private let metrics = ["Return", "Win Rate", "Sharpe", "Profit F."]
    private let correlations: [[Double]] = [
        [1.00, 0.75, 0.82, 0.68],
        [0.75, 1.00, 0.65, 0.71],
        [0.82, 0.65, 1.00, 0.59],
        [0.68, 0.71, 0.59, 1.00]
    ]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Correlation Matrix")
                .font(.headline)
                .foregroundStyle(.white)
            
            VStack(spacing: 4) {
                ForEach(0..<metrics.count, id: \.self) { row in
                    HStack(spacing: 4) {
                        Text(metrics[row])
                            .font(.caption2)
                            .foregroundStyle(.white.opacity(0.6))
                            .frame(width: 60, alignment: .leading)
                        
                        ForEach(0..<metrics.count, id: \.self) { col in
                            CorrelationCell(value: correlations[row][col])
                        }
                    }
                }
            }
        }
        .padding(20)
        .glassEffect(.regular.tint(.teal.opacity(0.08)).interactive(), in: .rect(cornerRadius: 24))
    }
}

struct CorrelationCell: View {
    let value: Double
    
    var body: some View {
        Text(String(format: "%.2f", value))
            .font(.caption2.weight(.medium))
            .foregroundStyle(value > 0.7 ? .white : .white.opacity(0.6))
            .frame(maxWidth: .infinity)
            .frame(height: 32)
            .background(
                RoundedRectangle(cornerRadius: 4)
                    .fill(correlationColor.opacity(abs(value) * 0.8))
            )
    }
    
    private var correlationColor: Color {
        if value > 0.7 {
            return .green
        } else if value > 0.4 {
            return .blue
        } else if value > 0 {
            return .orange
        } else {
            return .red
        }
    }
}

#Preview {
    AnalyticsView()
        .environmentObject(TradingDataManager())
        .background(Color.black)
}

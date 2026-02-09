# API Documentation - Trading Analytics

## 📚 Полная документация по API

### Table of Contents
1. [Models](#models)
2. [View Models](#view-models)
3. [Views](#views)
4. [Utilities](#utilities)
5. [Extensions](#extensions)

---

## Models

### Trade

Представляет одну торговую сделку.

```swift
struct Trade: Identifiable, Codable {
    let id: UUID
    let symbol: String              // Тикер (например, "AAPL")
    let entryDate: Date            // Дата входа
    let exitDate: Date?            // Дата выхода (nil для открытых)
    let entryPrice: Double         // Цена входа
    let exitPrice: Double?         // Цена выхода
    let quantity: Int              // Количество акций
    let type: TradeType           // .long или .short
    let profit: Double?           // Прибыль/убыток
    let status: TradeStatus       // .open или .closed
    
    var profitLoss: Double        // Вычисляемое свойство P&L
}
```

**Примеры использования:**

```swift
// Создание длинной позиции
let longTrade = Trade(
    id: UUID(),
    symbol: "AAPL",
    entryDate: Date(),
    exitDate: Date().addingTimeInterval(86400),
    entryPrice: 150.0,
    exitPrice: 155.0,
    quantity: 100,
    type: .long,
    profit: 500.0,
    status: .closed
)

// Открытая короткая позиция
let shortTrade = Trade(
    id: UUID(),
    symbol: "TSLA",
    entryDate: Date(),
    exitDate: nil,
    entryPrice: 200.0,
    exitPrice: nil,
    quantity: 50,
    type: .short,
    profit: nil,
    status: .open
)

// Получение прибыли/убытка
let pnl = longTrade.profitLoss // 500.0
```

---

### BacktestResults

Результаты бэктестинга торговой стратегии.

```swift
struct BacktestResults: Identifiable, Codable {
    let id: UUID
    let startDate: Date            // Начало периода
    let endDate: Date              // Конец периода
    let initialEquity: Double      // Начальный капитал
    let finalEquity: Double        // Конечный капитал
    let totalTrades: Int           // Всего сделок
    let winningTrades: Int         // Прибыльных сделок
    let losingTrades: Int          // Убыточных сделок
    let totalReturn: Double        // Доходность %
    let profitFactor: Double       // Фактор прибыли
    let sharpeRatio: Double        // Коэффициент Шарпа
    let maxDrawdown: Double        // Макс. просадка %
    let averageWin: Double         // Средняя прибыль
    let averageLoss: Double        // Средний убыток
    let winRate: Double            // Процент выигрышей
    
    var performance: Performance   // Оценка производительности
}
```

**Примеры использования:**

```swift
// Создание результатов
let results = BacktestResults(
    id: UUID(),
    startDate: Calendar.current.date(byAdding: .day, value: -30, to: Date())!,
    endDate: Date(),
    initialEquity: 10000,
    finalEquity: 12500,
    totalTrades: 150,
    winningTrades: 95,
    losingTrades: 55,
    totalReturn: 25.0,
    profitFactor: 1.85,
    sharpeRatio: 1.72,
    maxDrawdown: -8.3,
    averageWin: 185.50,
    averageLoss: -95.30,
    winRate: 63.33
)

// Получение оценки производительности
let grade = results.performance.grade // "Excellent"
let color = results.performance.gradeColor // .green
```

---

### EquityCurvePoint

Точка на кривой эквити (капитала).

```swift
struct EquityCurvePoint: Identifiable {
    let id: UUID
    let date: Date          // Дата
    let equity: Double      // Размер капитала
    let drawdown: Double    // Просадка %
}
```

**Примеры использования:**

```swift
// Создание точки
let point = EquityCurvePoint(
    date: Date(),
    equity: 11250.0,
    drawdown: -3.5
)

// Массив точек для графика
let curve: [EquityCurvePoint] = [
    EquityCurvePoint(date: day1, equity: 10000, drawdown: 0),
    EquityCurvePoint(date: day2, equity: 10200, drawdown: 0),
    EquityCurvePoint(date: day3, equity: 10150, drawdown: -0.49),
]
```

---

### Performance

Оценка производительности стратегии.

```swift
struct Performance {
    let returnPercentage: Double
    let sharpeRatio: Double
    let maxDrawdown: Double
    let winRate: Double
    
    var gradeColor: Color      // Цвет оценки
    var grade: String          // Текстовая оценка
}
```

**Оценки:**
- **Excellent**: Return > 20% и Sharpe > 1.5
- **Good**: Return > 10%
- **Fair**: Return > 0%
- **Poor**: Return ≤ 0%

---

### TradingRecommendation

Рекомендация от AI-ассистента.

```swift
struct TradingRecommendation: Identifiable {
    let id: UUID
    let title: String
    let description: String
    let priority: Priority          // .high, .medium, .low
    let category: Category          // Категория рекомендации
}
```

**Категории:**
- `.strategy` - Улучшение стратегии
- `.riskManagement` - Управление рисками
- `.execution` - Исполнение сделок
- `.psychology` - Торговая психология

**Примеры использования:**

```swift
let recommendation = TradingRecommendation(
    title: "Improve Win Rate",
    description: "Consider tightening entry criteria",
    priority: .high,
    category: .strategy
)

// Цвет по приоритету
let color = recommendation.priority.color // .red для .high

// Иконка по категории
let icon = recommendation.category.icon // "chart.line.uptrend.xyaxis"
```

---

## View Models

### TradingDataManager

Главный менеджер данных приложения.

```swift
@MainActor
class TradingDataManager: ObservableObject {
    @Published var backtestResults: [BacktestResults]
    @Published var trades: [Trade]
    @Published var equityCurve: [EquityCurvePoint]
    @Published var isLoading: Bool
    @Published var errorMessage: String?
    
    // Вычисляемые свойства
    var currentBacktest: BacktestResults?
    var openTrades: [Trade]
    var closedTrades: [Trade]
    
    // Методы
    func loadBacktestData() async
    func parseLogFile(at path: String) async throws
}
```

**Примеры использования:**

```swift
// В SwiftUI View
@EnvironmentObject var dataManager: TradingDataManager

var body: some View {
    VStack {
        if dataManager.isLoading {
            ProgressView()
        }
        
        if let backtest = dataManager.currentBacktest {
            Text("Return: \(backtest.totalReturn)%")
        }
        
        List(dataManager.openTrades) { trade in
            TradeRow(trade: trade)
        }
    }
    .task {
        await dataManager.loadBacktestData()
    }
}

// Парсинг лог-файла
Task {
    do {
        try await dataManager.parseLogFile(at: "/path/to/backtest.log")
    } catch {
        print("Error: \(error)")
    }
}
```

---

### AITradingAssistant

AI-ассистент для анализа торговли.

```swift
@MainActor
class AITradingAssistant: ObservableObject {
    @Published var isAvailable: Bool
    @Published var isAnalyzing: Bool
    @Published var currentAnalysis: String?
    @Published var recommendations: [TradingRecommendation]
    
    // Методы
    func checkModelAvailability()
    func analyzeBacktestResults(_ results: BacktestResults) async
    func analyzeEquityCurve(_ curve: [EquityCurvePoint]) async -> String?
}
```

**Примеры использования:**

```swift
@EnvironmentObject var aiAssistant: AITradingAssistant

var body: some View {
    VStack {
        if aiAssistant.isAvailable {
            if let analysis = aiAssistant.currentAnalysis {
                Text(analysis)
            }
            
            ForEach(aiAssistant.recommendations) { rec in
                RecommendationCard(recommendation: rec)
            }
        }
    }
    .task {
        if let results = dataManager.currentBacktest {
            await aiAssistant.analyzeBacktestResults(results)
        }
    }
}

// Анализ кривой эквити
if let curveAnalysis = await aiAssistant.analyzeEquityCurve(equityCurve) {
    print("Curve analysis: \(curveAnalysis)")
}
```

---

## Views

### DashboardView

Главный дашборд с метриками.

**Компоненты:**
- `PerformanceMetricsGrid` - Сетка основных метрик
- `EquityCurveCard` - График кривой эквити
- `DetailedStatsCard` - Детальная статистика
- `TradeDistributionCard` - Распределение сделок

**Использование:**

```swift
DashboardView()
    .environmentObject(dataManager)
```

---

### AnalyticsView

Продвинутая аналитика с 3D графиками.

**Типы визуализации:**
- `.performance3D` - 3D поверхность производительности
- `.heatmap` - Тепловая карта активности
- `.returns` - Распределение доходности
- `.drawdown` - Анализ просадок

**Использование:**

```swift
AnalyticsView()
    .environmentObject(dataManager)
```

**3D контролы:**

```swift
@State private var chartPose: Chart3DPose = .default

// Изменение вида
chartPose = .front    // Вид спереди
chartPose = .top      // Вид сверху
chartPose = .right    // Вид сбоку
chartPose = .default  // Стандартный вид
```

---

### TradesListView

Список сделок с фильтрацией.

**Фильтры:**
- `.all` - Все сделки
- `.open` - Открытые
- `.closed` - Закрытые
- `.profitable` - Прибыльные
- `.losses` - Убыточные

**Использование:**

```swift
TradesListView()
    .environmentObject(dataManager)
```

---

### AIAssistantView

AI-ассистент с рекомендациями.

**Разделы:**
- `.overview` - Общий анализ
- `.recommendations` - Рекомендации
- `.patterns` - Анализ паттернов
- `.risks` - Оценка рисков

**Использование:**

```swift
AIAssistantView()
    .environmentObject(aiAssistant)
    .environmentObject(dataManager)
```

---

## Utilities

### Liquid Glass Effects

Применение эффектов Liquid Glass.

```swift
// Базовый эффект
.glassEffect()

// С настройками
.glassEffect(.regular.tint(.blue.opacity(0.1)).interactive())

// Контейнер
GlassEffectContainer(spacing: 12) {
    // Контент
}

// С ID для морфинга
@Namespace private var namespace

.glassEffect()
.glassEffectID("uniqueID", in: namespace)

// Объединение эффектов
.glassEffectUnion(id: "groupID", namespace: namespace)
```

---

### Chart Customization

Настройка графиков.

```swift
// Базовый график
Chart(data) { item in
    LineMark(
        x: .value("Date", item.date),
        y: .value("Value", item.value)
    )
}

// С градиентом
.foregroundStyle(
    LinearGradient(
        colors: [.green, .blue],
        startPoint: .leading,
        endPoint: .trailing
    )
)

// Стилизация осей
.chartXAxis {
    AxisMarks { value in
        AxisGridLine().foregroundStyle(.white.opacity(0.1))
        AxisValueLabel().foregroundStyle(.white.opacity(0.6))
    }
}

// 3D график
Chart3D {
    SurfacePlot(
        x: "X",
        y: "Y",
        z: "Z",
        function: { x, y in
            sin(x) * cos(y)
        }
    )
    .roughness(0.2)
}
.chart3DPose($pose)
.chart3DCameraProjection(.perspective)
```

---

### Animations

Анимации в приложении.

```swift
// Spring анимация
withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
    isSelected.toggle()
}

// Matched Geometry Effect
@Namespace private var namespace

if isSelected {
    Circle()
        .matchedGeometryEffect(id: "indicator", in: namespace)
}

// Symbol Effects
Image(systemName: "sparkles")
    .symbolEffect(.pulse, isActive: isActive)
    .symbolEffect(.bounce, value: value)
    .symbolEffect(.rotate)
```

---

## Extensions

### Color Extensions

```swift
// Цвета производительности
extension Color {
    static func performanceColor(for value: Double) -> Color {
        value >= 0 ? .green : .red
    }
    
    static func gradeColor(for grade: String) -> Color {
        switch grade {
        case "Excellent": return .green
        case "Good": return .blue
        case "Fair": return .orange
        default: return .red
        }
    }
}
```

### Date Extensions

```swift
// Форматирование дат
extension Date {
    var shortFormat: String {
        formatted(date: .abbreviated, time: .omitted)
    }
    
    var fullFormat: String {
        formatted(date: .long, time: .shortened)
    }
}

// Использование
Text(trade.entryDate.shortFormat)
```

### Double Extensions

```swift
// Форматирование чисел
extension Double {
    var asPercent: String {
        String(format: "%.2f%%", self)
    }
    
    var asCurrency: String {
        String(format: "$%.2f", self)
    }
    
    var withSign: String {
        self >= 0 ? "+\(asPercent)" : asPercent
    }
}

// Использование
Text(results.totalReturn.asPercent)      // "25.00%"
Text(trade.profitLoss.asCurrency)        // "$500.00"
Text(change.withSign)                    // "+5.50%"
```

---

## Best Practices

### Performance

```swift
// ✅ Используйте LazyVStack для больших списков
ScrollView {
    LazyVStack {
        ForEach(trades) { trade in
            TradeCard(trade: trade)
        }
    }
}

// ✅ Асинхронная загрузка данных
.task {
    await dataManager.loadBacktestData()
}

// ✅ Оптимизация графиков
Chart(equityCurve.suffix(100)) { // Ограничение точек
    LineMark(...)
}
```

### State Management

```swift
// ✅ @EnvironmentObject для глобального состояния
@EnvironmentObject var dataManager: TradingDataManager

// ✅ @State для локального состояния
@State private var isExpanded = false

// ✅ @Binding для двустороннего связывания
struct ChildView: View {
    @Binding var selection: Int
}
```

### Accessibility

```swift
// ✅ Добавляйте метки для accessibility
Image(systemName: "chart.line.uptrend.xyaxis")
    .accessibilityLabel("Performance chart")

// ✅ Группируйте связанные элементы
VStack {
    Text("Win Rate")
    Text("65%")
}
.accessibilityElement(children: .combine)
.accessibilityLabel("Win Rate: 65%")
```

---

## Error Handling

```swift
// Обработка ошибок загрузки
func loadData() async {
    do {
        try await dataManager.parseLogFile(at: path)
    } catch {
        errorMessage = "Failed to load data: \(error.localizedDescription)"
        showError = true
    }
}

// AI ошибки
do {
    let response = try await session.respond(to: prompt)
} catch let error as LanguageModelSession.GenerationError {
    switch error {
    case .exceededContextWindowSize:
        print("Context too large")
    default:
        print("Generation error: \(error)")
    }
}
```

---

## Examples

### Complete Dashboard Example

```swift
struct MyDashboard: View {
    @StateObject private var dataManager = TradingDataManager()
    @StateObject private var aiAssistant = AITradingAssistant()
    
    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    // Метрики
                    if let backtest = dataManager.currentBacktest {
                        MetricsGrid(results: backtest)
                    }
                    
                    // График
                    EquityChart(curve: dataManager.equityCurve)
                    
                    // AI рекомендации
                    if !aiAssistant.recommendations.isEmpty {
                        RecommendationsSection(
                            recommendations: aiAssistant.recommendations
                        )
                    }
                }
                .padding()
            }
            .navigationTitle("Trading Dashboard")
        }
        .task {
            await dataManager.loadBacktestData()
            if let results = dataManager.currentBacktest {
                await aiAssistant.analyzeBacktestResults(results)
            }
        }
    }
}
```

### Custom Glass Card

```swift
struct CustomGlassCard<Content: View>: View {
    let title: String
    let icon: String
    let color: Color
    @ViewBuilder let content: () -> Content
    
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Image(systemName: icon)
                    .foregroundStyle(color)
                
                Text(title)
                    .font(.headline)
                    .foregroundStyle(.white)
                
                Spacer()
            }
            
            content()
        }
        .padding(20)
        .glassEffect(.regular.tint(color.opacity(0.08)).interactive())
    }
}

// Использование
CustomGlassCard(
    title: "Performance",
    icon: "chart.line.uptrend.xyaxis",
    color: .blue
) {
    Text("Your content here")
}
```

---

**Документация актуальна на 31 января 2026**

Для дополнительной информации см. README.md и QUICKSTART.md

# Интеграция с вашим проектом бэктестинга

## 🔗 Как интегрировать Trading Analytics с вашими данными

Этот гайд поможет вам подключить приложение Trading Analytics к вашей системе бэктестинга.

---

## 📋 Шаг 1: Понимание формата данных

Ваш скрипт `wait_and_show_results.sh` читает файл `backtest_30days_output.log`. Нужно создать парсер для этого файла.

### Типичный формат лог-файла:

```
BACKTEST REPORT
================
Period: 2026-01-01 to 2026-01-31
Initial Equity: $10,000.00
Final Equity: $12,450.00
Total Return: 24.50%

TRADE STATISTICS
Total Trades: 156
Winning Trades: 94
Losing Trades: 62
Win Rate: 60.26%

PERFORMANCE METRICS
Profit Factor: 1.85
Sharpe Ratio: 1.72
Max Drawdown: -8.30%
Average Win: $185.50
Average Loss: -$95.30
```

---

## 🛠 Шаг 2: Создание парсера

### Добавьте в `TradingDataManager.swift`:

```swift
import Foundation

extension TradingDataManager {
    
    /// Парсинг файла логов бэктеста
    func parseBacktestLog(at path: String) async throws {
        isLoading = true
        defer { isLoading = false }
        
        // Читаем файл
        guard let contents = try? String(contentsOfFile: path, encoding: .utf8) else {
            throw ParsingError.fileNotFound
        }
        
        // Парсим результаты
        let results = try parseBacktestResults(from: contents)
        let trades = try parseTrades(from: contents)
        let curve = try parseEquityCurve(from: contents)
        
        // Обновляем данные
        await MainActor.run {
            self.backtestResults = [results]
            self.trades = trades
            self.equityCurve = curve
        }
    }
    
    private func parseBacktestResults(from content: String) throws -> BacktestResults {
        let lines = content.components(separatedBy: .newlines)
        
        var initialEquity: Double = 0
        var finalEquity: Double = 0
        var totalTrades: Int = 0
        var winningTrades: Int = 0
        var losingTrades: Int = 0
        var totalReturn: Double = 0
        var profitFactor: Double = 0
        var sharpeRatio: Double = 0
        var maxDrawdown: Double = 0
        var averageWin: Double = 0
        var averageLoss: Double = 0
        var winRate: Double = 0
        
        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            
            if trimmed.contains("Initial Equity:") {
                initialEquity = extractNumber(from: trimmed)
            } else if trimmed.contains("Final Equity:") {
                finalEquity = extractNumber(from: trimmed)
            } else if trimmed.contains("Total Trades:") {
                totalTrades = Int(extractNumber(from: trimmed))
            } else if trimmed.contains("Winning Trades:") {
                winningTrades = Int(extractNumber(from: trimmed))
            } else if trimmed.contains("Losing Trades:") {
                losingTrades = Int(extractNumber(from: trimmed))
            } else if trimmed.contains("Total Return:") {
                totalReturn = extractNumber(from: trimmed)
            } else if trimmed.contains("Profit Factor:") {
                profitFactor = extractNumber(from: trimmed)
            } else if trimmed.contains("Sharpe Ratio:") {
                sharpeRatio = extractNumber(from: trimmed)
            } else if trimmed.contains("Max Drawdown:") {
                maxDrawdown = extractNumber(from: trimmed)
            } else if trimmed.contains("Average Win:") {
                averageWin = extractNumber(from: trimmed)
            } else if trimmed.contains("Average Loss:") {
                averageLoss = extractNumber(from: trimmed)
            } else if trimmed.contains("Win Rate:") {
                winRate = extractNumber(from: trimmed)
            }
        }
        
        return BacktestResults(
            id: UUID(),
            startDate: Date().addingTimeInterval(-30 * 86400),
            endDate: Date(),
            initialEquity: initialEquity,
            finalEquity: finalEquity,
            totalTrades: totalTrades,
            winningTrades: winningTrades,
            losingTrades: losingTrades,
            totalReturn: totalReturn,
            profitFactor: profitFactor,
            sharpeRatio: sharpeRatio,
            maxDrawdown: maxDrawdown,
            averageWin: averageWin,
            averageLoss: averageLoss,
            winRate: winRate
        )
    }
    
    private func parseTrades(from content: String) throws -> [Trade] {
        var trades: [Trade] = []
        
        // Пример парсинга секции TRADES
        // Адаптируйте под ваш формат
        let tradePattern = #"(\w+)\s+(\w+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+([-\d.]+)"#
        
        if let regex = try? NSRegularExpression(pattern: tradePattern) {
            let nsString = content as NSString
            let results = regex.matches(in: content, range: NSRange(content.startIndex..., in: content))
            
            for match in results {
                if match.numberOfRanges == 7 {
                    let symbol = nsString.substring(with: match.range(at: 1))
                    let type = nsString.substring(with: match.range(at: 2))
                    let entry = Double(nsString.substring(with: match.range(at: 3))) ?? 0
                    let exit = Double(nsString.substring(with: match.range(at: 4))) ?? 0
                    let quantity = Int(nsString.substring(with: match.range(at: 5))) ?? 0
                    let profit = Double(nsString.substring(with: match.range(at: 6))) ?? 0
                    
                    let trade = Trade(
                        id: UUID(),
                        symbol: symbol,
                        entryDate: Date().addingTimeInterval(-Double.random(in: 0...2592000)),
                        exitDate: Date(),
                        entryPrice: entry,
                        exitPrice: exit,
                        quantity: quantity,
                        type: type.lowercased() == "long" ? .long : .short,
                        profit: profit,
                        status: .closed
                    )
                    
                    trades.append(trade)
                }
            }
        }
        
        return trades
    }
    
    private func parseEquityCurve(from content: String) throws -> [EquityCurvePoint] {
        var points: [EquityCurvePoint] = []
        
        // Ищем секцию EQUITY CURVE
        let lines = content.components(separatedBy: .newlines)
        var inEquitySection = false
        
        for line in lines {
            if line.contains("EQUITY CURVE") {
                inEquitySection = true
                continue
            }
            
            if inEquitySection && line.isEmpty {
                break
            }
            
            if inEquitySection {
                // Парсим строку: Date, Equity, Drawdown
                let components = line.components(separatedBy: ",")
                if components.count >= 3 {
                    if let equity = Double(components[1].trimmingCharacters(in: .whitespaces)),
                       let drawdown = Double(components[2].trimmingCharacters(in: .whitespaces)) {
                        
                        let dateFormatter = DateFormatter()
                        dateFormatter.dateFormat = "yyyy-MM-dd"
                        let date = dateFormatter.date(from: components[0].trimmingCharacters(in: .whitespaces)) ?? Date()
                        
                        points.append(EquityCurvePoint(
                            date: date,
                            equity: equity,
                            drawdown: drawdown
                        ))
                    }
                }
            }
        }
        
        return points
    }
    
    private func extractNumber(from text: String) -> Double {
        let cleaned = text.replacingOccurrences(of: "[^0-9.-]", with: "", options: .regularExpression)
        return Double(cleaned) ?? 0
    }
    
    enum ParsingError: Error {
        case fileNotFound
        case invalidFormat
        case missingData
    }
}
```

---

## 📁 Шаг 3: Вызов парсера

### В `ContentView.swift`:

```swift
import SwiftUI

struct ContentView: View {
    @EnvironmentObject var dataManager: TradingDataManager
    @EnvironmentObject var aiAssistant: AITradingAssistant
    @State private var selectedTab: Tab = .dashboard
    
    // Путь к вашему лог-файлу
    let logFilePath = "/path/to/backtest_30days_output.log"
    
    var body: some View {
        NavigationStack {
            // ... ваш UI ...
        }
        .task {
            // Загружаем данные из лог-файла
            do {
                try await dataManager.parseBacktestLog(at: logFilePath)
                
                // После загрузки запускаем AI-анализ
                if let results = dataManager.currentBacktest {
                    await aiAssistant.analyzeBacktestResults(results)
                }
            } catch {
                print("Error loading backtest data: \(error)")
                // Fallback на демо-данные
                await dataManager.loadBacktestData()
            }
        }
    }
}
```

---

## 🔄 Шаг 4: Автоматическое обновление

### Создайте файловый мониторинг:

```swift
import Foundation
import Combine

class BacktestFileMonitor: ObservableObject {
    @Published var lastUpdate: Date?
    
    private var fileSystemWatcher: DispatchSourceFileSystemObject?
    private let fileURL: URL
    
    init(filePath: String) {
        self.fileURL = URL(fileURLWithPath: filePath)
        startMonitoring()
    }
    
    private func startMonitoring() {
        let fileDescriptor = open(fileURL.path, O_EVTONLY)
        
        fileSystemWatcher = DispatchSource.makeFileSystemObjectSource(
            fileDescriptor: fileDescriptor,
            eventMask: [.write, .extend],
            queue: DispatchQueue.global()
        )
        
        fileSystemWatcher?.setEventHandler { [weak self] in
            DispatchQueue.main.async {
                self?.lastUpdate = Date()
            }
        }
        
        fileSystemWatcher?.setCancelHandler {
            close(fileDescriptor)
        }
        
        fileSystemWatcher?.resume()
    }
    
    deinit {
        fileSystemWatcher?.cancel()
    }
}

// Использование:
@StateObject private var fileMonitor = BacktestFileMonitor(
    filePath: "/path/to/backtest_30days_output.log"
)

.onChange(of: fileMonitor.lastUpdate) { _, _ in
    Task {
        try? await dataManager.parseBacktestLog(at: logFilePath)
    }
}
```

---

## 🌐 Шаг 5: Альтернатива - JSON API

Если вы хотите использовать API вместо файлов:

```swift
extension TradingDataManager {
    
    func fetchBacktestFromAPI(url: String) async throws {
        guard let apiURL = URL(string: url) else {
            throw APIError.invalidURL
        }
        
        let (data, _) = try await URLSession.shared.data(from: apiURL)
        
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        
        let apiResponse = try decoder.decode(BacktestAPIResponse.self, from: data)
        
        await MainActor.run {
            self.backtestResults = [apiResponse.results]
            self.trades = apiResponse.trades
            self.equityCurve = apiResponse.equityCurve
        }
    }
}

struct BacktestAPIResponse: Codable {
    let results: BacktestResults
    let trades: [Trade]
    let equityCurve: [EquityCurvePoint]
}

enum APIError: Error {
    case invalidURL
    case networkError
    case decodingError
}
```

---

## 📊 Шаг 6: Форматы экспорта

### Экспорт в JSON:

```swift
extension TradingDataManager {
    
    func exportToJSON() throws -> Data {
        let export = ExportData(
            results: backtestResults.first!,
            trades: trades,
            equityCurve: equityCurve
        )
        
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        
        return try encoder.encode(export)
    }
    
    func saveToFile(path: String) throws {
        let data = try exportToJSON()
        try data.write(to: URL(fileURLWithPath: path))
    }
}

struct ExportData: Codable {
    let results: BacktestResults
    let trades: [Trade]
    let equityCurve: [EquityCurvePoint]
    let exportDate: Date = Date()
    let version: String = "1.0"
}
```

---

## 🔧 Шаг 7: Конфигурация

### Создайте файл конфигурации:

```swift
// Config.swift
struct AppConfig {
    // Пути к файлам
    static let backtestLogPath = "/path/to/backtest_30days_output.log"
    static let exportDirectory = "/path/to/exports"
    
    // API endpoints (если используете)
    static let apiBaseURL = "https://your-api.com"
    static let backtestEndpoint = "/api/backtest/latest"
    
    // Настройки обновления
    static let autoRefreshInterval: TimeInterval = 60 // секунд
    static let enableFileMonitoring = true
    
    // AI настройки
    static let enableAIAnalysis = true
    static let aiContextWindow = 4096
}

// Использование:
try await dataManager.parseBacktestLog(at: AppConfig.backtestLogPath)
```

---

## ✅ Чек-лист интеграции

- [ ] Определен формат вашего лог-файла
- [ ] Создан парсер для основных метрик
- [ ] Реализован парсер сделок
- [ ] Настроен парсинг кривой эквити
- [ ] Добавлен путь к файлу в конфигурацию
- [ ] Протестирован парсинг на реальных данных
- [ ] Настроена обработка ошибок
- [ ] (Опционально) Добавлен мониторинг файла
- [ ] (Опционально) Реализован API endpoint
- [ ] (Опционально) Настроен экспорт данных

---

## 🐛 Отладка

### Проверка парсинга:

```swift
// Добавьте в ваш код для отладки
do {
    try await dataManager.parseBacktestLog(at: logFilePath)
    print("✅ Parsing successful!")
    print("Results: \(dataManager.backtestResults.count)")
    print("Trades: \(dataManager.trades.count)")
    print("Equity points: \(dataManager.equityCurve.count)")
} catch {
    print("❌ Parsing failed: \(error)")
}
```

### Вывод первых строк:

```swift
let content = try String(contentsOfFile: logFilePath)
print("First 20 lines:")
content.components(separatedBy: .newlines)
    .prefix(20)
    .forEach { print($0) }
```

---

## 📞 Помощь

Если у вас возникли проблемы с интеграцией:

1. Проверьте формат вашего лог-файла
2. Убедитесь, что путь к файлу правильный
3. Добавьте print-отладку в парсер
4. Проверьте права доступа к файлу
5. Используйте демо-данные для тестирования UI

---

**Успешной интеграции! 🚀**

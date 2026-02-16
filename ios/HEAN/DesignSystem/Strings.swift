//
//  Strings.swift
//  HEAN
//
//  Bilingual string constants (English / Russian)
//  Language is toggled via @AppStorage("appLanguage") in Settings
//

import Foundation

struct L {
    static var isRussian: Bool {
        UserDefaults.standard.string(forKey: "appLanguage") == "ru"
    }

    // MARK: - Tabs

    static var live: String { isRussian ? "Трейдинг" : "Live" }
    static var mind: String { isRussian ? "Разум" : "Mind" }
    static var action: String { isRussian ? "Действия" : "Action" }
    static var xray: String { isRussian ? "Анализ" : "X-Ray" }
    static var genesis: String { isRussian ? "Генезис" : "Genesis" }
    static var settings: String { isRussian ? "Настройки" : "Settings" }

    // MARK: - Portfolio

    static var totalCapital: String { isRussian ? "Капитал" : "Capital" }
    static var equity: String { isRussian ? "Баланс" : "Equity" }
    static var profit: String { isRussian ? "Прибыль" : "Profit" }
    static var positions: String { isRussian ? "Позиции" : "Positions" }
    static var unrealizedPnL: String { isRussian ? "Нереализ." : "Unrealized" }
    static var realizedPnL: String { isRussian ? "Реализ." : "Realized" }

    // MARK: - Loading

    static var loadingLiveData: String { isRussian ? "Загрузка данных..." : "Loading live data..." }
    static var loadingTradingData: String { isRussian ? "Загрузка торговых данных..." : "Loading trading data..." }
    static var loadingXRay: String { isRussian ? "Загрузка анализа..." : "Loading X-Ray..." }
    static var thinking: String { isRussian ? "Анализирую..." : "Thinking..." }

    // MARK: - Live View

    static var markets: String { isRussian ? "Рынки" : "Markets" }

    // MARK: - Physics

    static var temperature: String { isRussian ? "Температура" : "Temperature" }
    static var entropy: String { isRussian ? "Энтропия" : "Entropy" }
    static var phase: String { isRussian ? "Фаза" : "Phase" }
    static var cold: String { isRussian ? "Холодно" : "Cold" }
    static var warm: String { isRussian ? "Тепло" : "Warm" }
    static var hot: String { isRussian ? "Горячо" : "Hot" }
    static var ordered: String { isRussian ? "Порядок" : "Ordered" }
    static var mixed: String { isRussian ? "Смешанно" : "Mixed" }
    static var chaotic: String { isRussian ? "Хаос" : "Chaotic" }
    static var szilardProfit: String { isRussian ? "Прибыль Сциларда" : "Szilard Profit" }
    static var edgePresent: String { isRussian ? "Есть грань" : "Edge Present" }
    static var noEdge: String { isRussian ? "Нет грани" : "No Edge" }

    // MARK: - AI

    static var aiExplanation: String { isRussian ? "ИИ Анализ" : "AI Explanation" }
    static var aiAnalysis: String { isRussian ? "ИИ Анализ" : "AI Analysis" }
    static var aiSignalProposal: String { isRussian ? "ИИ Сигнал" : "AI Signal Proposal" }
    static var forces: String { isRussian ? "СИЛЫ" : "FORCES" }
    static var confidence: String { isRussian ? "Уверенность" : "Confidence" }
    static var noThoughtsYet: String { isRussian ? "Мыслей пока нет" : "No thoughts yet" }
    static var aiWillStart: String { isRussian ? "ИИ начнет анализ при получении рыночных данных" : "AI will start analyzing when market data arrives" }

    // MARK: - Balance of Forces

    static var balanceOfForces: String { isRussian ? "Баланс сил" : "Balance of Forces" }
    static var buy: String { isRussian ? "ПОКУПКА" : "BUY" }
    static var sell: String { isRussian ? "ПРОДАЖА" : "SELL" }

    // MARK: - Anomalies

    static var whaleTrades: String { isRussian ? "Сделки китов" : "Whale Trades" }
    static var liquidations: String { isRussian ? "Ликвидации" : "Liquidations" }
    static var anomalies: String { isRussian ? "Аномалии" : "Anomalies" }
    static var severity: String { isRussian ? "Серьезность" : "Severity" }

    // MARK: - Risk

    static var riskNormal: String { isRussian ? "Нормальный" : "Normal" }
    static var riskSoftBrake: String { isRussian ? "Мягкий тормоз" : "Soft Brake" }
    static var riskQuarantine: String { isRussian ? "Карантин" : "Quarantine" }
    static var riskHardStop: String { isRussian ? "Полная остановка" : "Hard Stop" }
    static var riskDescNormal: String { isRussian ? "Все системы работают нормально" : "All systems operating normally" }
    static var riskDescSoftBrake: String { isRussian ? "Уменьшен размер позиций" : "Reduced position sizing active" }
    static var riskDescQuarantine: String { isRussian ? "Новые сделки приостановлены" : "New trades paused" }
    static var riskDescHardStop: String { isRussian ? "Вся торговля остановлена" : "All trading halted" }

    // MARK: - Action

    static var noActivePositions: String { isRussian ? "Нет активных позиций" : "No Active Positions" }
    static var activePositions: String { isRussian ? "Активные позиции" : "Active Positions" }
    static var noPendingOrders: String { isRussian ? "Нет ожидающих ордеров" : "No pending orders" }
    static var pendingOrders: String { isRussian ? "Ожидающие ордера" : "Pending Orders" }
    static var closeAll: String { isRussian ? "Закрыть все" : "Close All" }
    static var close: String { isRussian ? "Закрыть" : "Close" }
    static var closePosition: String { isRussian ? "Закрыть позицию" : "Close Position" }
    static var closePositionFor: String { isRussian ? "Закрыть позицию для" : "Close position for" }
    static var confirm: String { isRussian ? "ПОДТВЕРДИТЬ" : "CONFIRM" }
    static var skip: String { isRussian ? "ПРОПУСТИТЬ" : "SKIP" }
    static var cancel: String { isRussian ? "Отмена" : "Cancel" }
    static var sessionStats: String { isRussian ? "Статистика сессии" : "Session Stats" }
    static var trades: String { isRussian ? "Сделки" : "Trades" }
    static var winRate: String { isRussian ? "Винрейт" : "Win Rate" }
    static var tradingFunnel: String { isRussian ? "Воронка торговли" : "Trading Funnel" }
    static var signals: String { isRussian ? "Сигналы" : "Signals" }
    static var orders: String { isRussian ? "Ордера" : "Orders" }
    static var filled: String { isRussian ? "Исполнено" : "Filled" }
    static var blockedByRisk: String { isRussian ? "заблокировано риском" : "blocked by risk" }
    static var error: String { isRussian ? "Ошибка" : "Error" }

    // MARK: - X-Ray

    static var participants: String { isRussian ? "Участники" : "Participants" }
    static var dominantPlayer: String { isRussian ? "ДОМИНИРУЮЩИЙ ИГРОК" : "DOMINANT PLAYER" }
    static var noParticipantData: String { isRussian ? "Нет данных об участниках" : "No participant data" }
    static var marketMakers: String { isRussian ? "Маркет-мейкеры" : "Market Makers" }
    static var institutional: String { isRussian ? "Институционалы" : "Institutional" }
    static var retail: String { isRussian ? "Ретейл" : "Retail" }
    static var whales: String { isRussian ? "Киты" : "Whales" }
    static var arbBots: String { isRussian ? "Арбитражные боты" : "Arb Bots" }
    static var mmDesc: String { isRussian ? "Поддерживают ликвидность, зарабатывают на спреде." : "Provide liquidity, profit from the spread." }
    static var instDesc: String { isRussian ? "Крупные фонды. Торгуют iceberg-ордерами." : "Large funds. Trade with iceberg orders." }
    static var retailDesc: String { isRussian ? "Обычные трейдеры. Хороший контр-индикатор." : "Regular traders. Good contrarian indicator." }
    static var whaleDesc: String { isRussian ? "Крупные держатели. Их движения двигают рынок." : "Large holders. Their moves impact the market." }
    static var arbDesc: String { isRussian ? "Выравнивают цены между биржами." : "Equalize prices across exchanges." }

    // MARK: - Settings

    static var language: String { isRussian ? "Язык" : "Language" }
    static var connection: String { isRussian ? "Подключение" : "Connection" }
    static var preferences: String { isRussian ? "Настройки" : "Preferences" }
    static var hapticFeedback: String { isRussian ? "Тактильный отклик" : "Haptic Feedback" }
    static var pushNotifications: String { isRussian ? "Push-уведомления" : "Push Notifications" }
    static var showConfidenceScores: String { isRussian ? "Показывать уверенность" : "Show Confidence Scores" }
    static var engineControl: String { isRussian ? "Управление движком" : "Engine Control" }
    static var startEngine: String { isRussian ? "Запустить движок" : "Start Engine" }
    static var pauseEngine: String { isRussian ? "Пауза движка" : "Pause Engine" }
    static var resumeEngine: String { isRussian ? "Возобновить движок" : "Resume Engine" }
    static var stopEngine: String { isRussian ? "Остановить движок" : "Stop Engine" }
    static var emergencyKill: String { isRussian ? "Аварийная остановка" : "Emergency Kill Switch" }
    static var tradingParameters: String { isRussian ? "Параметры торговли" : "Trading Parameters" }
    static var maxRiskPerTrade: String { isRussian ? "Макс. риск на сделку" : "Max Risk per Trade" }
    static var maxLeverage: String { isRussian ? "Макс. кредитное плечо" : "Max Leverage" }
    static var minRiskReward: String { isRussian ? "Мин. риск/прибыль" : "Min Risk/Reward" }
    static var brainSettings: String { isRussian ? "Настройки ИИ" : "Brain Settings" }
    static var analysisInterval: String { isRussian ? "Интервал анализа" : "Analysis Interval" }
    static var confidenceThreshold: String { isRussian ? "Порог уверенности" : "Confidence Threshold" }
    static var about: String { isRussian ? "О приложении" : "About" }
    static var version: String { isRussian ? "Версия" : "Version" }
    static var build: String { isRussian ? "Сборка" : "Build" }
    static var target: String { isRussian ? "Цель" : "Target" }
    static var engine: String { isRussian ? "Движок" : "Engine" }
    static var uptime: String { isRussian ? "Время работы" : "Uptime" }
    static var save: String { isRussian ? "Сохранить" : "Save" }
    static var settingsSaved: String { isRussian ? "Настройки сохранены" : "Settings Saved" }
    static var startEngineMsg: String { isRussian ? "Это запустит торговый движок на Bybit Testnet." : "This will start the trading engine on Bybit Testnet." }
    static var pauseEngineMsg: String { isRussian ? "Исполнение ордеров будет приостановлено, позиции останутся открытыми." : "This will pause order execution while keeping positions open." }
    static var resumeEngineMsg: String { isRussian ? "Исполнение ордеров будет возобновлено." : "This will resume order execution." }
    static var stopEngineMsg: String { isRussian ? "Торговый движок будет остановлен." : "This will stop the trading engine gracefully." }
    static var killEngineMsg: String { isRussian ? "Все позиции будут закрыты, все ордера отменены, движок остановлен." : "This will close all positions, cancel all orders, and stop the engine." }

    // MARK: - Position

    static var entry: String { isRussian ? "Вход" : "Entry" }
    static var targetPrice: String { isRussian ? "Цель" : "Target" }
    static var stopPrice: String { isRussian ? "Стоп" : "Stop" }
    static var size: String { isRussian ? "Объем" : "Size" }
    static var long: String { isRussian ? "ЛОНГ" : "LONG" }
    static var short: String { isRussian ? "ШОРТ" : "SHORT" }

    // MARK: - Strategies

    static var active: String { isRussian ? "Активных" : "Active" }
    static var total: String { isRussian ? "Всего" : "Total" }
    static var combinedPnL: String { isRussian ? "Общий P&L" : "Combined P&L" }
    static var noStrategiesLoaded: String { isRussian ? "Стратегии не загружены" : "No strategies loaded" }
    static var startEngineToLoad: String { isRussian ? "Запустите движок для загрузки стратегий" : "Start the engine to load trading strategies" }
    static var profitFactor: String { isRussian ? "PF" : "PF" }

    // MARK: - Signals

    static var waitingForSignals: String { isRussian ? "Ожидание сигналов..." : "Waiting for signals..." }
    static var signalsWillAppear: String { isRussian ? "Сигналы появятся здесь в реальном времени\nкогда стратегии обнаружат торговые возможности" : "Signals will appear here in real-time\nas strategies detect trading opportunities" }
    static var strategy: String { isRussian ? "Стратегия" : "Strategy" }

    // MARK: - Risk Dashboard

    static var riskProgression: String { isRussian ? "Прогресс риска" : "Risk Progression" }
    static var drawdown: String { isRussian ? "Просадка" : "Drawdown" }
    static var killSwitch: String { isRussian ? "Аварийный стоп" : "Kill Switch" }
    static var tradingHalted: String { isRussian ? "Вся торговля остановлена. Проверьте позиции и сбросьте когда будете готовы." : "All trading has been halted. Review your positions and reset when ready." }
    static var current: String { isRussian ? "Текущая" : "Current" }
    static var maxDrawdown: String { isRussian ? "Макс." : "Max" }
    static var activePositionsCount: String { isRussian ? "Активные позиции" : "Active Positions" }
    static var exposure: String { isRussian ? "Экспозиция" : "Exposure" }
    static var availableMargin: String { isRussian ? "Доступная маржа" : "Available Margin" }
    static var quarantinedSymbols: String { isRussian ? "Символы в карантине" : "Quarantined Symbols" }

    // MARK: - AI Assistant

    static var analyzing: String { isRussian ? "Анализирую..." : "Analyzing..." }
    static var askAnything: String { isRussian ? "Спрашивайте что угодно о вашей торговой системе.\nЯ анализирую данные в реальном времени для полезных рекомендаций." : "Ask me anything about your trading system.\nI analyze real-time data to give you actionable insights." }
    static var quickQuestions: String { isRussian ? "Быстрые вопросы" : "Quick Questions" }

    // MARK: - Physics UI

    static var marketPhysics: String { isRussian ? "Физика рынка" : "Market Physics" }
    static var noBrainActivity: String { isRussian ? "Нет активности мозга" : "No brain activity yet" }
    static var noAnomalies: String { isRussian ? "Аномалий не обнаружено" : "No anomalies detected" }
    static var marketNormal: String { isRussian ? "Рыночные условия нормальные" : "Market conditions are normal" }
    static var noTemporalData: String { isRussian ? "Нет временных данных" : "No temporal data" }
    static var noParticipants: String { isRussian ? "Нет данных участников" : "No participant data" }

    // MARK: - Positions active count
    static func positionsActive(_ count: Int) -> String {
        isRussian ? "\(count) \(count == 1 ? "позиция активна" : "позиций активно")" : "\(count) Position\(count == 1 ? "" : "s") Active"
    }
    static func ordersPending(_ count: Int) -> String {
        isRussian ? "\(count) \(count == 1 ? "ордер ожидает" : "ордеров ожидают")" : "\(count) order\(count == 1 ? "" : "s") pending"
    }
}

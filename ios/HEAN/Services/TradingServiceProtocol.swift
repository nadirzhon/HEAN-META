//
//  TradingServiceProtocol.swift
//  HEAN
//
//  Created on 2026-01-31.
//

import Foundation
import Combine
import SwiftUI

// MARK: - Risk Types (for protocol compatibility)

enum RiskState: String, Codable, CaseIterable {
    case normal = "NORMAL"
    case softBrake = "SOFT_BRAKE"
    case quarantine = "QUARANTINE"
    case hardStop = "HARD_STOP"

    var displayName: String {
        switch self {
        case .normal: return "Normal"
        case .softBrake: return "Soft Brake"
        case .quarantine: return "Quarantine"
        case .hardStop: return "Hard Stop"
        }
    }

    var description: String {
        switch self {
        case .normal: return "All systems operational"
        case .softBrake: return "Reduced position sizing"
        case .quarantine: return "Close-only mode"
        case .hardStop: return "All trading halted"
        }
    }

    var icon: String {
        switch self {
        case .normal: return "checkmark.shield.fill"
        case .softBrake: return "exclamationmark.shield.fill"
        case .quarantine: return "xmark.shield.fill"
        case .hardStop: return "hand.raised.fill"
        }
    }

    var colorHex: String {
        switch self {
        case .normal: return "22C55E"
        case .softBrake: return "F59E0B"
        case .quarantine: return "F97316"
        case .hardStop: return "EF4444"
        }
    }

    var severity: Int {
        switch self {
        case .normal: return 0
        case .softBrake: return 1
        case .quarantine: return 2
        case .hardStop: return 3
        }
    }
}

/// Matches backend /risk/governor/status response
struct RiskGovernorStatus: Codable {
    let riskState: String
    let level: Int
    let reasonCodes: [String]
    let quarantinedSymbols: [String]
    let canClear: Bool
    let metric: String?
    let value: Double?
    let threshold: Double?
    let recommendedAction: String?
    let clearRule: String?
    let blockedAt: String?

    enum CodingKeys: String, CodingKey {
        case riskState = "risk_state"
        case level
        case reasonCodes = "reason_codes"
        case quarantinedSymbols = "quarantined_symbols"
        case canClear = "can_clear"
        case metric
        case value
        case threshold
        case recommendedAction = "recommended_action"
        case clearRule = "clear_rule"
        case blockedAt = "blocked_at"
    }

    var state: RiskState {
        RiskState(rawValue: riskState) ?? .normal
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.riskState = try container.decodeIfPresent(String.self, forKey: .riskState) ?? "NORMAL"
        self.level = try container.decodeIfPresent(Int.self, forKey: .level) ?? 0
        self.reasonCodes = try container.decodeIfPresent([String].self, forKey: .reasonCodes) ?? []
        self.quarantinedSymbols = try container.decodeIfPresent([String].self, forKey: .quarantinedSymbols) ?? []
        self.canClear = try container.decodeIfPresent(Bool.self, forKey: .canClear) ?? true
        self.metric = try container.decodeIfPresent(String.self, forKey: .metric)
        self.value = try container.decodeIfPresent(Double.self, forKey: .value)
        self.threshold = try container.decodeIfPresent(Double.self, forKey: .threshold)
        self.recommendedAction = try container.decodeIfPresent(String.self, forKey: .recommendedAction)
        self.clearRule = try container.decodeIfPresent(String.self, forKey: .clearRule)
        self.blockedAt = try container.decodeIfPresent(String.self, forKey: .blockedAt)
    }

    init(riskState: String = "NORMAL", level: Int = 0, reasonCodes: [String] = [],
         quarantinedSymbols: [String] = [], canClear: Bool = true, metric: String? = nil,
         value: Double? = nil, threshold: Double? = nil, recommendedAction: String? = nil,
         clearRule: String? = nil, blockedAt: String? = nil) {
        self.riskState = riskState
        self.level = level
        self.reasonCodes = reasonCodes
        self.quarantinedSymbols = quarantinedSymbols
        self.canClear = canClear
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.recommendedAction = recommendedAction
        self.clearRule = clearRule
        self.blockedAt = blockedAt
    }
}

/// Matches backend /risk/killswitch/status response
struct KillswitchStatus: Codable {
    let triggered: Bool
    let reasons: [String]
    let triggeredAt: String?
    let thresholds: [String: Double]?
    let currentMetrics: [String: Double]?

    enum CodingKeys: String, CodingKey {
        case triggered
        case reasons
        case triggeredAt = "triggered_at"
        case thresholds
        case currentMetrics = "current_metrics"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.triggered = try container.decodeIfPresent(Bool.self, forKey: .triggered) ?? false
        self.reasons = try container.decodeIfPresent([String].self, forKey: .reasons) ?? []
        self.triggeredAt = try container.decodeIfPresent(String.self, forKey: .triggeredAt)
        self.thresholds = try container.decodeIfPresent([String: Double].self, forKey: .thresholds)
        self.currentMetrics = try container.decodeIfPresent([String: Double].self, forKey: .currentMetrics)
    }

    init(triggered: Bool = false, reasons: [String] = [], triggeredAt: String? = nil,
         thresholds: [String: Double]? = nil, currentMetrics: [String: Double]? = nil) {
        self.triggered = triggered
        self.reasons = reasons
        self.triggeredAt = triggeredAt
        self.thresholds = thresholds
        self.currentMetrics = currentMetrics
    }

    var currentDrawdown: Double {
        currentMetrics?["current_drawdown_pct"] ?? 0
    }

    var reason: String? {
        reasons.first
    }
}

// MARK: - Trading Service Protocol

struct OrderRequest: Codable {
    let symbol: String
    let side: OrderSide
    let type: OrderType
    let quantity: Double
    let price: Double?
}

@MainActor
protocol TradingServiceProtocol {
    var positionsPublisher: AnyPublisher<[Position], Never> { get }
    var ordersPublisher: AnyPublisher<[Order], Never> { get }

    func fetchPositions() async throws -> [Position]
    func fetchOrders(status: String?) async throws -> [Order]
    func placeOrder(symbol: String, side: String, type: String, qty: Double, price: Double?) async throws -> Order
    func cancelOrder(id: String) async throws
    func cancelAllOrders() async throws
    func closePosition(symbol: String) async throws
    func closeAllPositions() async throws
    func fetchWhyDiagnostics() async throws -> WhyDiagnostics
    func fetchTradingMetrics() async throws -> TradingMetrics
}

// MARK: - Trading Types

/// Matches backend /trading/metrics response
struct TradingMetrics: Codable {
    let status: String?
    let counters: MetricsCounters?
    let activeOrdersCount: Int
    let activePositionsCount: Int
    let engineState: String?
    let lastSignalTs: String?
    let lastOrderTs: String?
    let lastFillTs: String?
    let mode: String?
    let topReasonsForSkipBlock: [TopReason]?
    let topSymbols: [TopSymbol]?
    let topStrategies: [TopStrategy]?
    let uptimeSec: Double?

    enum CodingKeys: String, CodingKey {
        case status
        case counters
        case activeOrdersCount = "active_orders_count"
        case activePositionsCount = "active_positions_count"
        case engineState = "engine_state"
        case lastSignalTs = "last_signal_ts"
        case lastOrderTs = "last_order_ts"
        case lastFillTs = "last_fill_ts"
        case mode
        case topReasonsForSkipBlock = "top_reasons_for_skip_block"
        case topSymbols = "top_symbols"
        case topStrategies = "top_strategies"
        case uptimeSec = "uptime_sec"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.status = try container.decodeIfPresent(String.self, forKey: .status)
        self.counters = try container.decodeIfPresent(MetricsCounters.self, forKey: .counters)
        self.activeOrdersCount = try container.decodeIfPresent(Int.self, forKey: .activeOrdersCount) ?? 0
        self.activePositionsCount = try container.decodeIfPresent(Int.self, forKey: .activePositionsCount) ?? 0
        self.engineState = try container.decodeIfPresent(String.self, forKey: .engineState)
        self.lastSignalTs = try container.decodeIfPresent(String.self, forKey: .lastSignalTs)
        self.lastOrderTs = try container.decodeIfPresent(String.self, forKey: .lastOrderTs)
        self.lastFillTs = try container.decodeIfPresent(String.self, forKey: .lastFillTs)
        self.mode = try container.decodeIfPresent(String.self, forKey: .mode)
        self.topReasonsForSkipBlock = try container.decodeIfPresent([TopReason].self, forKey: .topReasonsForSkipBlock)
        self.topSymbols = try container.decodeIfPresent([TopSymbol].self, forKey: .topSymbols)
        self.topStrategies = try container.decodeIfPresent([TopStrategy].self, forKey: .topStrategies)
        self.uptimeSec = try container.decodeIfPresent(Double.self, forKey: .uptimeSec)
    }

    init(status: String? = nil, counters: MetricsCounters? = nil, activeOrdersCount: Int = 0,
         activePositionsCount: Int = 0, engineState: String? = nil, lastSignalTs: String? = nil,
         lastOrderTs: String? = nil, lastFillTs: String? = nil, mode: String? = nil,
         topReasonsForSkipBlock: [TopReason]? = nil, topSymbols: [TopSymbol]? = nil,
         topStrategies: [TopStrategy]? = nil, uptimeSec: Double? = nil) {
        self.status = status
        self.counters = counters
        self.activeOrdersCount = activeOrdersCount
        self.activePositionsCount = activePositionsCount
        self.engineState = engineState
        self.lastSignalTs = lastSignalTs
        self.lastOrderTs = lastOrderTs
        self.lastFillTs = lastFillTs
        self.mode = mode
        self.topReasonsForSkipBlock = topReasonsForSkipBlock
        self.topSymbols = topSymbols
        self.topStrategies = topStrategies
        self.uptimeSec = uptimeSec
    }

    // Convenience accessors for session counters (used by DashboardView)
    var signalsDetected: Int { counters?.session?.signalsDetected ?? 0 }
    var ordersCreated: Int { counters?.session?.ordersCreated ?? 0 }
    var ordersFilled: Int { counters?.session?.ordersFilled ?? 0 }
    var signalsBlocked: Int { counters?.session?.signalsBlocked ?? 0 }
    var decisionsCreate: Int { counters?.session?.decisionsCreate ?? 0 }
    var decisionsSkip: Int { counters?.session?.decisionsSkip ?? 0 }
}

struct TopReason: Codable, Identifiable {
    let code: String
    let count: Int
    let pct: Double?

    var id: String { code }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.code = (try? container.decode(String.self, forKey: .code)) ?? "UNKNOWN"
        self.count = (try? container.decode(Int.self, forKey: .count)) ?? 0
        self.pct = try container.decodeIfPresent(Double.self, forKey: .pct)
    }

    enum CodingKeys: String, CodingKey {
        case code, count, pct
    }
}

struct TopSymbol: Codable, Identifiable {
    let symbol: String
    let count: Int

    var id: String { symbol }
}

struct TopStrategy: Codable, Identifiable {
    let strategyId: String
    let count: Int

    var id: String { strategyId }

    enum CodingKeys: String, CodingKey {
        case strategyId = "strategy_id"
        case count
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.strategyId = (try? container.decode(String.self, forKey: .strategyId)) ?? "unknown"
        self.count = (try? container.decode(Int.self, forKey: .count)) ?? 0
    }
}

struct MetricsCounters: Codable {
    let last1m: MetricsBucket?
    let last5m: MetricsBucket?
    let session: MetricsBucket?

    enum CodingKeys: String, CodingKey {
        case last1m = "last_1m"
        case last5m = "last_5m"
        case session
    }
}

struct MetricsBucket: Codable {
    let signalsDetected: Int?
    let decisionsCreate: Int?
    let decisionsSkip: Int?
    let signalsBlocked: Int?
    let ordersCreated: Int?
    let ordersFilled: Int?
    let ordersCanceled: Int?
    let ordersRejected: Int?
    let ordersOpen: Int?
    let positionsOpen: Int?
    let positionsClosed: Int?
    let pnlUnrealized: Double?
    let pnlRealized: Double?
    let equity: Double?

    enum CodingKeys: String, CodingKey {
        case signalsDetected = "signals_total"
        case decisionsCreate = "decisions_create"
        case decisionsSkip = "decisions_skip"
        case signalsBlocked = "decisions_block"
        case ordersCreated = "orders_created"
        case ordersFilled = "orders_filled"
        case ordersCanceled = "orders_canceled"
        case ordersRejected = "orders_rejected"
        case ordersOpen = "orders_open"
        case positionsOpen = "positions_open"
        case positionsClosed = "positions_closed"
        case pnlUnrealized = "pnl_unrealized"
        case pnlRealized = "pnl_realized"
        case equity
    }
}

/// Matches backend /trading/why response
struct WhyDiagnostics: Codable {
    let engineState: String
    let killswitchState: KillswitchState?
    let lastTickAgeSec: Double?
    let lastSignalTs: String?
    let lastDecisionTs: String?
    let lastOrderTs: String?
    let lastFillTs: String?
    let activeOrdersCount: Int
    let activePositionsCount: Int
    let topReasonCodesLast5m: [ReasonCode]?
    let equity: Double?
    let balance: Double?
    let unrealPnl: Double?
    let realPnl: Double?
    let marginUsed: Double?
    let marginFree: Double?
    let profitCaptureState: ProfitCaptureState?
    let executionQuality: ExecutionQuality?
    let multiSymbol: MultiSymbolState?

    enum CodingKeys: String, CodingKey {
        case engineState = "engine_state"
        case killswitchState = "killswitch_state"
        case lastTickAgeSec = "last_tick_age_sec"
        case lastSignalTs = "last_signal_ts"
        case lastDecisionTs = "last_decision_ts"
        case lastOrderTs = "last_order_ts"
        case lastFillTs = "last_fill_ts"
        case activeOrdersCount = "active_orders_count"
        case activePositionsCount = "active_positions_count"
        case topReasonCodesLast5m = "top_reason_codes_last_5m"
        case equity
        case balance
        case unrealPnl = "unreal_pnl"
        case realPnl = "real_pnl"
        case marginUsed = "margin_used"
        case marginFree = "margin_free"
        case profitCaptureState = "profit_capture_state"
        case executionQuality = "execution_quality"
        case multiSymbol = "multi_symbol"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.engineState = try container.decodeIfPresent(String.self, forKey: .engineState) ?? "UNKNOWN"
        self.killswitchState = try container.decodeIfPresent(KillswitchState.self, forKey: .killswitchState)
        self.lastTickAgeSec = try container.decodeIfPresent(Double.self, forKey: .lastTickAgeSec)
        self.lastSignalTs = try container.decodeIfPresent(String.self, forKey: .lastSignalTs)
        self.lastDecisionTs = try container.decodeIfPresent(String.self, forKey: .lastDecisionTs)
        self.lastOrderTs = try container.decodeIfPresent(String.self, forKey: .lastOrderTs)
        self.lastFillTs = try container.decodeIfPresent(String.self, forKey: .lastFillTs)
        self.activeOrdersCount = try container.decodeIfPresent(Int.self, forKey: .activeOrdersCount) ?? 0
        self.activePositionsCount = try container.decodeIfPresent(Int.self, forKey: .activePositionsCount) ?? 0
        self.topReasonCodesLast5m = try container.decodeIfPresent([ReasonCode].self, forKey: .topReasonCodesLast5m)
        self.equity = try container.decodeIfPresent(Double.self, forKey: .equity)
        self.balance = try container.decodeIfPresent(Double.self, forKey: .balance)
        self.unrealPnl = try container.decodeIfPresent(Double.self, forKey: .unrealPnl)
        self.realPnl = try container.decodeIfPresent(Double.self, forKey: .realPnl)
        self.marginUsed = try container.decodeIfPresent(Double.self, forKey: .marginUsed)
        self.marginFree = try container.decodeIfPresent(Double.self, forKey: .marginFree)
        self.profitCaptureState = try container.decodeIfPresent(ProfitCaptureState.self, forKey: .profitCaptureState)
        self.executionQuality = try container.decodeIfPresent(ExecutionQuality.self, forKey: .executionQuality)
        self.multiSymbol = try container.decodeIfPresent(MultiSymbolState.self, forKey: .multiSymbol)
    }
}

struct ProfitCaptureState: Codable {
    let enabled: Bool
    let armed: Bool
    let triggered: Bool
    let cleared: Bool
    let mode: String?
    let startEquity: Double?
    let peakEquity: Double?
    let targetPct: Double?
    let trailPct: Double?
    let afterAction: String?
    let continueRiskMult: Double?

    enum CodingKeys: String, CodingKey {
        case enabled, armed, triggered, cleared, mode
        case startEquity = "start_equity"
        case peakEquity = "peak_equity"
        case targetPct = "target_pct"
        case trailPct = "trail_pct"
        case afterAction = "after_action"
        case continueRiskMult = "continue_risk_mult"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.enabled = (try? container.decode(Bool.self, forKey: .enabled)) ?? false
        self.armed = (try? container.decode(Bool.self, forKey: .armed)) ?? false
        self.triggered = (try? container.decode(Bool.self, forKey: .triggered)) ?? false
        self.cleared = (try? container.decode(Bool.self, forKey: .cleared)) ?? false
        self.mode = try container.decodeIfPresent(String.self, forKey: .mode)
        self.startEquity = try container.decodeIfPresent(Double.self, forKey: .startEquity)
        self.peakEquity = try container.decodeIfPresent(Double.self, forKey: .peakEquity)
        self.targetPct = try container.decodeIfPresent(Double.self, forKey: .targetPct)
        self.trailPct = try container.decodeIfPresent(Double.self, forKey: .trailPct)
        self.afterAction = try container.decodeIfPresent(String.self, forKey: .afterAction)
        self.continueRiskMult = try container.decodeIfPresent(Double.self, forKey: .continueRiskMult)
    }
}

struct ExecutionQuality: Codable {
    let wsOk: Bool
    let restOk: Bool
    let avgLatencyMs: Double?
    let rejectRate5m: Double?
    let slippageEst5m: Double?

    enum CodingKeys: String, CodingKey {
        case wsOk = "ws_ok"
        case restOk = "rest_ok"
        case avgLatencyMs = "avg_latency_ms"
        case rejectRate5m = "reject_rate_5m"
        case slippageEst5m = "slippage_est_5m"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.wsOk = (try? container.decode(Bool.self, forKey: .wsOk)) ?? false
        self.restOk = (try? container.decode(Bool.self, forKey: .restOk)) ?? false
        self.avgLatencyMs = try container.decodeIfPresent(Double.self, forKey: .avgLatencyMs)
        self.rejectRate5m = try container.decodeIfPresent(Double.self, forKey: .rejectRate5m)
        self.slippageEst5m = try container.decodeIfPresent(Double.self, forKey: .slippageEst5m)
    }
}

struct MultiSymbolState: Codable {
    let enabled: Bool
    let symbolsCount: Int
    let lastScannedSymbol: String?
    let scanCursor: Int?
    let scanCycleTs: String?

    enum CodingKeys: String, CodingKey {
        case enabled
        case symbolsCount = "symbols_count"
        case lastScannedSymbol = "last_scanned_symbol"
        case scanCursor = "scan_cursor"
        case scanCycleTs = "scan_cycle_ts"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.enabled = (try? container.decode(Bool.self, forKey: .enabled)) ?? false
        self.symbolsCount = (try? container.decode(Int.self, forKey: .symbolsCount)) ?? 0
        self.lastScannedSymbol = try container.decodeIfPresent(String.self, forKey: .lastScannedSymbol)
        self.scanCursor = try container.decodeIfPresent(Int.self, forKey: .scanCursor)
        self.scanCycleTs = try container.decodeIfPresent(String.self, forKey: .scanCycleTs)
    }
}

struct KillswitchState: Codable {
    let triggered: Bool
    let reasons: [String]
    let triggeredAt: String?

    enum CodingKeys: String, CodingKey {
        case triggered
        case reasons
        case triggeredAt = "triggered_at"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.triggered = try container.decodeIfPresent(Bool.self, forKey: .triggered) ?? false
        self.reasons = try container.decodeIfPresent([String].self, forKey: .reasons) ?? []
        self.triggeredAt = try container.decodeIfPresent(String.self, forKey: .triggeredAt)
    }
}

struct ReasonCode: Codable {
    let code: String
    let count: Int
}

// MARK: - Signal Service Protocol

@MainActor
protocol SignalServiceProtocol {
    var signalPublisher: AnyPublisher<Signal, Never> { get }
    func subscribeToSignals()
    func unsubscribeFromSignals()
}

/// Signal from backend WebSocket
struct Signal: Identifiable, Codable {
    let symbol: String
    let side: String
    let confidence: Double?
    let strategy: String?
    let timestamp: Date?
    let reason: String?

    var id: String { "\(symbol)-\(side)-\(timestamp?.timeIntervalSince1970 ?? 0)" }

    // Keys for encoding
    enum CodingKeys: String, CodingKey {
        case symbol, side, confidence, strategy, timestamp, reason
    }

    // Additional keys for decoding
    private enum DecodingKeys: String, CodingKey {
        case symbol, side, confidence, strategy, timestamp, reason
        case strategyId = "strategy_id"
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DecodingKeys.self)
        self.symbol = try container.decode(String.self, forKey: .symbol)
        self.side = try container.decode(String.self, forKey: .side)
        self.confidence = try container.decodeIfPresent(Double.self, forKey: .confidence)
        // strategy or strategy_id
        if let strategy = try? container.decode(String.self, forKey: .strategy) {
            self.strategy = strategy
        } else if let strategyId = try? container.decode(String.self, forKey: .strategyId) {
            self.strategy = strategyId
        } else {
            self.strategy = nil
        }
        self.timestamp = try container.decodeIfPresent(Date.self, forKey: .timestamp)
        self.reason = try container.decodeIfPresent(String.self, forKey: .reason)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(symbol, forKey: .symbol)
        try container.encode(side, forKey: .side)
        try container.encodeIfPresent(confidence, forKey: .confidence)
        try container.encodeIfPresent(strategy, forKey: .strategy)
        try container.encodeIfPresent(timestamp, forKey: .timestamp)
        try container.encodeIfPresent(reason, forKey: .reason)
    }

    init(symbol: String, side: String, confidence: Double? = nil, strategy: String? = nil,
         timestamp: Date? = nil, reason: String? = nil) {
        self.symbol = symbol
        self.side = side
        self.confidence = confidence
        self.strategy = strategy
        self.timestamp = timestamp
        self.reason = reason
    }

    var isBuy: Bool { side.lowercased() == "buy" }
}

// MARK: - Strategy Service Protocol

@MainActor
protocol StrategyServiceProtocol {
    func fetchStrategies() async throws -> [Strategy]
    func toggleStrategy(id: String, enabled: Bool) async throws
    func updateParameters(id: String, parameters: sending [String: Any]) async throws
}

/// Matches backend /strategies response
/// Backend returns: {strategy_id, type, enabled, win_rate, total_trades, profit_factor, total_pnl, wins, losses}
struct Strategy: Identifiable, Codable {
    let id: String
    let name: String
    let enabled: Bool
    let winRate: Double
    let totalTrades: Int
    let profitFactor: Double
    let description: String
    let totalPnl: Double
    let wins: Int
    let losses: Int

    // Keys for encoding
    enum CodingKeys: String, CodingKey {
        case id, name, enabled
        case winRate = "win_rate"
        case totalTrades = "total_trades"
        case profitFactor = "profit_factor"
        case description
        case totalPnl = "total_pnl"
        case wins, losses
    }

    // Additional keys for decoding from backend
    private enum DecodingKeys: String, CodingKey {
        case id, name, enabled
        case winRate = "win_rate"
        case totalTrades = "total_trades"
        case profitFactor = "profit_factor"
        case description
        case strategyId = "strategy_id"
        case type
        case totalPnl = "total_pnl"
        case wins, losses
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: DecodingKeys.self)

        // id: try "id" first, then "strategy_id"
        if let id = try? container.decode(String.self, forKey: .id) {
            self.id = id
        } else if let stratId = try? container.decode(String.self, forKey: .strategyId) {
            self.id = stratId
        } else {
            self.id = UUID().uuidString
        }

        // name: try "name" first, then "type"
        if let name = try? container.decode(String.self, forKey: .name) {
            self.name = name
        } else if let type = try? container.decode(String.self, forKey: .type) {
            self.name = type
        } else {
            self.name = self.id
        }

        self.enabled = try container.decodeIfPresent(Bool.self, forKey: .enabled) ?? false
        self.winRate = try container.decodeIfPresent(Double.self, forKey: .winRate) ?? 0
        self.totalTrades = try container.decodeIfPresent(Int.self, forKey: .totalTrades) ?? 0
        self.profitFactor = try container.decodeIfPresent(Double.self, forKey: .profitFactor) ?? 0
        self.description = try container.decodeIfPresent(String.self, forKey: .description) ?? ""
        self.totalPnl = try container.decodeIfPresent(Double.self, forKey: .totalPnl) ?? 0
        self.wins = try container.decodeIfPresent(Int.self, forKey: .wins) ?? 0
        self.losses = try container.decodeIfPresent(Int.self, forKey: .losses) ?? 0
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(id, forKey: .id)
        try container.encode(name, forKey: .name)
        try container.encode(enabled, forKey: .enabled)
        try container.encode(winRate, forKey: .winRate)
        try container.encode(totalTrades, forKey: .totalTrades)
        try container.encode(profitFactor, forKey: .profitFactor)
        try container.encode(description, forKey: .description)
        try container.encode(totalPnl, forKey: .totalPnl)
        try container.encode(wins, forKey: .wins)
        try container.encode(losses, forKey: .losses)
    }

    init(id: String, name: String, enabled: Bool, winRate: Double = 0,
         totalTrades: Int = 0, profitFactor: Double = 0, description: String = "",
         totalPnl: Double = 0, wins: Int = 0, losses: Int = 0) {
        self.id = id
        self.name = name
        self.enabled = enabled
        self.winRate = winRate
        self.totalTrades = totalTrades
        self.profitFactor = profitFactor
        self.description = description
        self.totalPnl = totalPnl
        self.wins = wins
        self.losses = losses
    }
}

// MARK: - Risk Service Protocol

@MainActor
protocol RiskServiceProtocol {
    var riskStatePublisher: AnyPublisher<RiskState, Never> { get }
    func fetchRiskStatus() async throws -> RiskGovernorStatus
    func resetKillswitch() async throws
    func quarantineSymbol(_ symbol: String) async throws
    func clearRiskBlocks() async throws
    func fetchKillswitchStatus() async throws -> KillswitchStatus
    func fetchSignalRejectionStats() async throws -> SignalRejectionStats
}

/// Matches backend /telemetry/signal-rejections response
struct SignalRejectionStats: Codable {
    let totalRejections: Int
    let totalSignals: Int
    let rejectionRate: Double
    let byReason: [String: Int]
    let timeWindowMinutes: Int
    let byCategory: [String: Int]
    let bySymbol: [String: Int]
    let byStrategy: [String: Int]
    let rates: [String: Double]

    enum CodingKeys: String, CodingKey {
        case totalRejections = "total_rejections"
        case totalSignals = "total_signals"
        case rejectionRate = "rejection_rate"
        case byReason = "by_reason"
        case timeWindowMinutes = "time_window_minutes"
        case byCategory = "by_category"
        case bySymbol = "by_symbol"
        case byStrategy = "by_strategy"
        case rates
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.totalRejections = (try? container.decode(Int.self, forKey: .totalRejections)) ?? 0
        self.totalSignals = (try? container.decode(Int.self, forKey: .totalSignals)) ?? 0
        self.rejectionRate = (try? container.decode(Double.self, forKey: .rejectionRate)) ?? 0
        self.byReason = (try? container.decode([String: Int].self, forKey: .byReason)) ?? [:]
        self.timeWindowMinutes = (try? container.decode(Int.self, forKey: .timeWindowMinutes)) ?? 60
        self.byCategory = (try? container.decode([String: Int].self, forKey: .byCategory)) ?? [:]
        self.bySymbol = (try? container.decode([String: Int].self, forKey: .bySymbol)) ?? [:]
        self.byStrategy = (try? container.decode([String: Int].self, forKey: .byStrategy)) ?? [:]
        self.rates = (try? container.decode([String: Double].self, forKey: .rates)) ?? [:]
    }

    init(totalRejections: Int = 0, totalSignals: Int = 0, rejectionRate: Double = 0,
         byReason: [String: Int] = [:], timeWindowMinutes: Int = 60,
         byCategory: [String: Int] = [:], bySymbol: [String: Int] = [:],
         byStrategy: [String: Int] = [:], rates: [String: Double] = [:]) {
        self.totalRejections = totalRejections
        self.totalSignals = totalSignals
        self.rejectionRate = rejectionRate
        self.byReason = byReason
        self.timeWindowMinutes = timeWindowMinutes
        self.byCategory = byCategory
        self.bySymbol = bySymbol
        self.byStrategy = byStrategy
        self.rates = rates
    }
}

// NOTE: Mock implementations are in the Mock/ folder
// NOTE: Live service implementations are in the Services/Live/ folder

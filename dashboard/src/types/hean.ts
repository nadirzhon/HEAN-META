/**
 * HEAN Trading System - Comprehensive TypeScript Type Definitions
 *
 * Auto-generated from Python backend source code.
 * Covers all API endpoints, WebSocket messages, event types, and data models.
 *
 * Backend: FastAPI + async EventBus
 * API prefix: /api/v1
 * WebSocket: ws://<host>:8000/ws
 */

// =============================================================================
// Core Event Types (from src/hean/core/types.py)
// =============================================================================

/** All event types flowing through the HEAN EventBus. */
export enum EventType {
  // Market events
  TICK = "tick",
  FUNDING = "funding",
  FUNDING_UPDATE = "funding_update",
  ORDER_BOOK_UPDATE = "order_book_update",
  REGIME_UPDATE = "regime_update",

  // Strategy events
  SIGNAL = "signal",
  STRATEGY_PARAMS_UPDATED = "strategy_params_updated",

  // Risk events
  ORDER_REQUEST = "order_request",
  RISK_BLOCKED = "risk_blocked",
  RISK_ALERT = "risk_alert",

  // Execution events
  ORDER_PLACED = "order_placed",
  ORDER_FILLED = "order_filled",
  ORDER_CANCELLED = "order_cancelled",
  ORDER_REJECTED = "order_rejected",

  // Portfolio events
  POSITION_OPENED = "position_opened",
  POSITION_CLOSED = "position_closed",
  POSITION_UPDATE = "position_update",
  POSITION_CLOSE_REQUEST = "position_close_request",
  EQUITY_UPDATE = "equity_update",
  PNL_UPDATE = "pnl_update",
  ORDER_DECISION = "order_decision",
  ORDER_EXIT_DECISION = "order_exit_decision",

  // System events
  STOP_TRADING = "stop_trading",
  KILLSWITCH_TRIGGERED = "killswitch_triggered",
  KILLSWITCH_RESET = "killswitch_reset",
  ERROR = "error",
  STATUS = "status",
  HEARTBEAT = "heartbeat",

  // Market structure / context events
  CANDLE = "candle",
  CONTEXT_UPDATE = "context_update",

  // Meta-learning events
  META_LEARNING_PATCH = "meta_learning_patch",

  // Brain/AI analysis events
  BRAIN_ANALYSIS = "brain_analysis",

  // Integration events (ContextAggregator)
  CONTEXT_READY = "context_ready",
  PHYSICS_UPDATE = "physics_update",
  ORACLE_PREDICTION = "oracle_prediction",
  OFI_UPDATE = "ofi_update",
  CAUSAL_SIGNAL = "causal_signal",

  // Self-analysis telemetry
  SELF_ANALYTICS = "self_analytics",

  // Council events
  COUNCIL_REVIEW = "council_review",
  COUNCIL_RECOMMENDATION = "council_recommendation",

  // Digital Organism events
  MARKET_GENOME_UPDATE = "market_genome_update",
  RISK_SIMULATION_RESULT = "risk_simulation_result",
  META_STRATEGY_UPDATE = "meta_strategy_update",
}

/** Order status enumeration. */
export enum OrderStatus {
  PENDING = "pending",
  PLACED = "placed",
  PARTIALLY_FILLED = "partially_filled",
  FILLED = "filled",
  CANCELLED = "cancelled",
  REJECTED = "rejected",
}

/** Engine lifecycle states. */
export enum EngineState {
  STOPPED = "STOPPED",
  RUNNING = "RUNNING",
  PAUSED = "PAUSED",
  ERROR = "ERROR",
  BUSY = "BUSY",
}

/** Job status enumeration. */
export enum JobStatus {
  PENDING = "pending",
  RUNNING = "running",
  COMPLETED = "completed",
  FAILED = "failed",
  CANCELLED = "cancelled",
}

// =============================================================================
// Core Data Models (from src/hean/core/types.py)
// =============================================================================

/** Market tick data. */
export interface Tick {
  symbol: string;
  price: number;
  timestamp: string;
  volume: number;
  bid: number | null;
  ask: number | null;
}

/** Funding rate data. */
export interface FundingRate {
  symbol: string;
  rate: number;
  timestamp: string;
  next_funding_time: string | null;
}

/** Trading signal from a strategy. */
export interface Signal {
  strategy_id: string;
  symbol: string;
  side: "buy" | "sell";
  entry_price: number;
  stop_loss: number | null;
  take_profit: number | null;
  take_profit_1: number | null;
  size: number | null;
  confidence: number;
  urgency: number;
  metadata: Record<string, unknown>;
  prefer_maker: boolean;
  min_maker_edge_bps: number | null;
}

/** Order request from risk layer. */
export interface OrderRequest {
  signal_id: string;
  strategy_id: string;
  symbol: string;
  side: "buy" | "sell";
  size: number;
  price: number | null;
  order_type: "market" | "limit";
  stop_loss: number | null;
  take_profit: number | null;
  reduce_only: boolean;
  metadata: Record<string, unknown>;
}

/** Order representation. */
export interface Order {
  order_id: string;
  strategy_id: string;
  symbol: string;
  side: string;
  size: number;
  filled_size: number;
  price: number | null;
  avg_fill_price: number | null;
  order_type: string;
  status: OrderStatus;
  stop_loss: number | null;
  take_profit: number | null;
  timestamp: string;
  metadata: Record<string, unknown>;
  is_maker: boolean;
  placed_at: string | null;
}

/** Position representation. */
export interface Position {
  position_id: string;
  symbol: string;
  side: "long" | "short";
  size: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  opened_at: string | null;
  strategy_id: string;
  stop_loss: number | null;
  take_profit: number | null;
  take_profit_1: number | null;
  break_even_activated: boolean;
  max_time_sec: number | null;
  metadata: Record<string, unknown>;
}

/** Equity snapshot at a point in time. */
export interface EquitySnapshot {
  timestamp: string;
  equity: number;
  cash: number;
  positions_value: number;
  unrealized_pnl: number;
  realized_pnl: number;
  daily_pnl: number;
  drawdown: number;
  drawdown_pct: number;
}

// =============================================================================
// Engine Facade Response Types
// =============================================================================

/** Engine status response from GET /api/v1/engine/status */
export interface EngineStatusResponse {
  status: "running" | "stopped" | "busy";
  running: boolean;
  engine_state: string;
  trading_mode: string;
  is_live: boolean;
  dry_run: boolean;
  equity?: number;
  daily_pnl?: number;
  initial_capital?: number;
  unrealized_pnl?: number;
  realized_pnl?: number;
  available_balance?: number;
  used_margin?: number;
  total_fees?: number;
  message?: string;
}

/** Engine control response (start/stop/pause/resume/kill/restart). */
export interface EngineControlResponse {
  ok?: boolean;
  status: string;
  message: string;
  engine_state?: string;
  start_result?: Record<string, unknown>;
  cancelled_orders?: number;
  closed_positions?: number;
  trading_mode?: string;
  is_live?: boolean;
  dry_run?: boolean;
}

/** Lock profit response from POST /api/v1/engine/lock-profit */
export interface LockProfitResponse {
  ok: boolean;
  status: "profit_locked";
  message: string;
  profit_locked: number;
  equity: number;
  initial_capital: number;
}

// =============================================================================
// Trading / Orders Response Types
// =============================================================================

/** Positions response from GET /api/v1/orders/positions */
export interface PositionsResponse {
  positions: PositionDTO[];
}

/** Position as returned by the API. */
export interface PositionDTO {
  symbol: string;
  size: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  realized_pnl: number;
  side: string;
  position_id: string;
  take_profit: number | null;
  stop_loss: number | null;
  strategy_id: string;
  leverage: number;
  status: "open" | "closed";
  created_at: string | null;
}

/** Orders response from GET /api/v1/orders */
export interface OrdersResponse {
  orders: OrderDTO[];
}

/** Order as returned by the API. */
export interface OrderDTO {
  order_id: string;
  symbol: string;
  side: string;
  size: number;
  filled_size: number;
  price: number | null;
  type: string;
  status: string;
  strategy_id: string;
  timestamp: string | null;
  updated_at: string | null;
}

/** Test order response from POST /api/v1/orders/test */
export interface TestOrderResponse {
  status: "success";
  message: string;
  signal: Signal;
}

/** Close position response from POST /api/v1/orders/close-position */
export interface ClosePositionResponse {
  status: "closed" | "not_found" | "error";
  position_id?: string;
  price?: number;
  message?: string;
}

/** Cancel order response from POST /api/v1/orders/cancel */
export interface CancelOrderResponse {
  status: "success";
  message: string;
}

/** Cancel all orders response from POST /api/v1/orders/cancel-all */
export interface CancelAllOrdersResponse {
  status: "success";
  message: string;
  cancelled: number;
}

/** Close all positions response from POST /api/v1/orders/close-all-positions */
export interface CloseAllPositionsResponse {
  status: "closed_all";
  closed: number;
  [key: string]: unknown;
}

/** Reset paper state response from POST /api/v1/orders/paper/reset_state */
export interface ResetPaperStateResponse {
  status: string;
  message?: string;
}

/** Roundtrip test response from POST /api/v1/orders/test_roundtrip */
export interface TestRoundtripResponse {
  status: "ok";
  message: string;
  account_state: AccountState | null;
  positions: PositionDTO[];
  orders: OrderDTO[];
  exit_decisions: Record<string, unknown>[];
}

/** Orderbook presence from GET /api/v1/orders/orderbook-presence */
export interface OrderbookPresence {
  symbol?: string;
  orders?: Array<{
    price: number;
    size: number;
    distance_bps: number;
    side: string;
  }>;
}

// =============================================================================
// Trading Diagnostics (Why / Metrics / State)
// =============================================================================

/** Killswitch state sub-object. */
export interface KillswitchState {
  triggered: boolean;
  reasons: string[];
  triggered_at?: string | null;
}

/** Profit capture state sub-object. */
export interface ProfitCaptureState {
  enabled: boolean;
  armed: boolean;
  triggered: boolean;
  cleared: boolean;
  mode: string | null;
  start_equity: number | null;
  peak_equity: number | null;
  target_pct: number | null;
  trail_pct: number | null;
  after_action: string | null;
  continue_risk_mult: number | null;
  last_action: string | null;
  last_reason: string | null;
}

/** Execution quality sub-object. */
export interface ExecutionQuality {
  ws_ok: boolean | null;
  rest_ok: boolean | null;
  avg_latency_ms: number | null;
  reject_rate_5m: number | null;
  slippage_est_5m: number | null;
}

/** Multi-symbol state sub-object. */
export interface MultiSymbolState {
  enabled: boolean;
  symbols_count: number;
  last_scanned_symbol: string | null;
  scan_cursor: string | null;
  scan_cycle_ts: string | null;
}

/** Reason code entry in the "why" endpoint. */
export interface ReasonCodeEntry {
  code: string;
  count: number;
}

/** Response from GET /api/v1/trading/why */
export interface WhyNotTradingResponse {
  engine_state: string;
  killswitch_state: KillswitchState;
  last_tick_age_sec: number | null;
  last_signal_ts: string | null;
  last_decision_ts: string | null;
  last_order_ts: string | null;
  last_fill_ts: string | null;
  active_orders_count: number;
  active_positions_count: number;
  top_reason_codes_last_5m: ReasonCodeEntry[];
  equity: number | null;
  balance: number | null;
  unreal_pnl: number | null;
  real_pnl: number | null;
  margin_used: number | null;
  margin_free: number | null;
  profit_capture_state: ProfitCaptureState;
  execution_quality: ExecutionQuality;
  multi_symbol: MultiSymbolState;
}

/** Equity history snapshot from GET /api/v1/trading/equity-history */
export interface EquityHistoryResponse {
  snapshots: Array<{
    timestamp: string;
    equity: number;
  }>;
  count: number;
}

/** Counters bucket (1m, 5m, session). */
export interface TradingCountersBucket {
  signals_detected?: number;
  orders_created?: number;
  orders_filled?: number;
  orders_cancelled?: number;
  orders_rejected?: number;
  positions_opened?: number;
  positions_closed?: number;
  decisions_skip?: number;
  decisions_block?: number;
  decisions_allow?: number;
  [key: string]: number | undefined;
}

/** Response from GET /api/v1/trading/metrics */
export interface TradingMetricsResponse {
  status: "ok" | "error";
  counters: {
    last_1m: TradingCountersBucket;
    last_5m: TradingCountersBucket;
    session: TradingCountersBucket;
  };
  top_reasons_for_skip_block: Array<{ reason: string; count: number }>;
  active_orders_count: number;
  active_positions_count: number;
  last_signal_ts: string | null;
  last_order_ts: string | null;
  last_fill_ts: string | null;
  engine_state: string;
  mode: string;
  message?: string;
}

/** Response from GET /api/v1/trading/state */
export interface TradingStateResponse {
  status: "ok" | "error";
  open_positions: PositionDTO[];
  open_orders: OrderDTO[];
  recent_fills: OrderDTO[];
  recent_decisions: Record<string, unknown>[];
  timestamp: string;
  message?: string;
}

// =============================================================================
// Strategies
// =============================================================================

/** Strategy info from GET /api/v1/strategies */
export interface StrategyInfo {
  strategy_id: string;
  enabled: boolean;
  type: string;
  win_rate: number;
  total_trades: number;
  profit_factor: number;
  total_pnl: number;
  wins: number;
  losses: number;
  description: string;
}

/** Response from POST /api/v1/strategies/{id}/enable */
export interface StrategyEnableResponse {
  status: "success" | "error";
  message: string;
}

/** Response from POST /api/v1/strategies/{id}/params */
export interface StrategyParamsResponse {
  status: "success";
  message: string;
  strategy_id: string;
  updated_params: Record<string, unknown>;
}

// =============================================================================
// Risk Management
// =============================================================================

/** Response from GET /api/v1/risk/status */
export interface RiskStatusResponse {
  killswitch_triggered: boolean;
  stop_trading: boolean;
  equity: number;
  daily_pnl: number;
  drawdown: number;
  drawdown_pct: number;
  max_open_positions: number;
  current_positions: number;
  max_open_orders: number;
  current_orders: number;
}

/** Response from GET /api/v1/risk/limits */
export interface RiskLimitsResponse {
  max_open_positions: number;
  max_daily_attempts: number;
  max_exposure_usd: number;
  min_notional_usd: number;
  cooldown_seconds: number;
}

/** Response from POST /api/v1/risk/limits */
export interface UpdateRiskLimitsResponse {
  status: "success";
  message: string;
  updated: string[];
  current_limits: RiskLimitsResponse;
}

/** Decision memory block entry from GET /api/v1/risk/decision-memory/blocks */
export interface DecisionMemoryBlock {
  strategy_id: string;
  context: string;
  block_until: string;
  seconds_remaining: number;
  loss_streak: number;
  trades_count: number;
  max_drawdown_pct: number;
}

/** Response from GET /api/v1/risk/decision-memory/blocks */
export interface DecisionMemoryBlocksResponse {
  blocked: DecisionMemoryBlock[];
}

/** Response from GET /api/v1/risk/killswitch/status */
export interface KillswitchStatusResponse {
  triggered: boolean;
  reasons: string[];
  triggered_at: string | null;
  thresholds: {
    drawdown_pct: number;
    equity_drop: number;
    max_loss: number;
    risk_limit: number;
  };
  current_metrics: {
    current_drawdown_pct: number;
    current_equity: number;
    max_drawdown_pct: number;
    peak_equity: number;
  };
}

/** Response from POST /api/v1/risk/killswitch/reset */
export interface KillswitchResetResponse {
  status: "success";
  message: string;
}

// =============================================================================
// Risk Governor
// =============================================================================

/** Response from GET /api/v1/risk/governor/status */
export interface RiskGovernorStatusResponse {
  risk_state: "NORMAL" | "SOFT_BRAKE" | "QUARANTINE" | "HARD_STOP";
  level: number;
  reason_codes: string[];
  metric: string | null;
  value: number | null;
  threshold: number | null;
  recommended_action: string;
  clear_rule: string;
  quarantined_symbols: string[];
  blocked_at: string | null;
  can_clear: boolean;
}

/** Response from POST /api/v1/risk/governor/clear */
export interface RiskGovernorClearResponse {
  status: string;
  message?: string;
}

/** Response from POST /api/v1/risk/governor/quarantine/{symbol} */
export interface QuarantineSymbolResponse {
  status: "quarantined";
  symbol: string;
  reason: string;
}

// =============================================================================
// Physics (Market Thermodynamics)
// =============================================================================

/** Response from GET /api/v1/physics/state */
export interface PhysicsStateResponse {
  symbol: string;
  temperature: number;
  temperature_regime: string;
  entropy: number;
  entropy_state: string;
  phase: string;
  phase_confidence: number;
  szilard_profit: number;
  should_trade: boolean;
  trade_reason: string;
  size_multiplier: number;
}

/** Response from GET /api/v1/physics/history */
export interface PhysicsHistoryResponse {
  symbol: string;
  temperature: Array<{
    value: number;
    regime: string;
    timestamp: string;
  }>;
  entropy: Array<{
    value: number;
    state: string;
    timestamp: string;
  }>;
  phases: Array<{
    phase: string;
    confidence: number;
    timestamp: string;
  }>;
}

/** Response from GET /api/v1/physics/participants */
export interface ParticipantsResponse {
  symbol: string;
  mm_activity: number;
  institutional_flow: number;
  retail_sentiment: number;
  whale_activity: number;
  arb_pressure: number;
  dominant_player: string;
  meta_signal: string;
}

/** Response from GET /api/v1/physics/anomalies */
export interface AnomaliesResponse {
  anomalies: Record<string, unknown>[];
  active_count: number;
}

// =============================================================================
// Temporal Stack
// =============================================================================

/** Response from GET /api/v1/temporal/stack */
export interface TemporalStackResponse {
  levels: Record<string, unknown>;
  last_update: string;
}

/** Impulse entry from cross-market analysis. */
export interface CrossMarketImpulse {
  source: string;
  change_pct: number;
  timestamp: string;
  propagated_to: string[];
}

/** Propagation stat from cross-market analysis. */
export interface PropagationStat {
  source: string;
  target: string;
  avg_delay_ms: number;
  correlation: number;
  samples: number;
}

/** Response from GET /api/v1/temporal/impulse */
export interface CrossMarketImpulseResponse {
  impulses: CrossMarketImpulse[];
  propagation_stats: PropagationStat[];
}

/** Response from GET /api/v1/temporal/sessions */
export interface TradingSessionsResponse {
  current_session: "Asia" | "London" | "New York";
  next_session: "Asia" | "London" | "New York";
  hours_remaining: number;
  utc_hour: number;
  timestamp: string;
}

// =============================================================================
// Brain (AI Analysis)
// =============================================================================

/** Brain thought entry. */
export interface BrainThought {
  id: string;
  timestamp: string;
  stage: string;
  content: string;
  confidence: number;
}

/** Brain signal sub-object. */
export interface BrainSignal {
  symbol: string;
  action: string;
  confidence: number;
  reason: string;
}

/** Response from GET /api/v1/brain/analysis */
export interface BrainAnalysisResponse {
  timestamp: string;
  thoughts: BrainThought[];
  forces: unknown[];
  signal: BrainSignal | null;
  summary: string;
  market_regime: string;
}

/** Response from GET /api/v1/brain/thoughts */
export type BrainThoughtsResponse = BrainThought[];

/** Response from GET /api/v1/brain/history */
export type BrainHistoryResponse = BrainAnalysisResponse[];

// =============================================================================
// Council (Multi-Agent AI)
// =============================================================================

/** Response from GET /api/v1/council/status */
export interface CouncilStatusResponse {
  enabled: boolean;
  message?: string;
  [key: string]: unknown;
}

/** Response from GET /api/v1/council/reviews */
export type CouncilReviewsResponse = Record<string, unknown>[];

/** Response from GET /api/v1/council/recommendations */
export type CouncilRecommendationsResponse = Record<string, unknown>[];

/** Response from POST /api/v1/council/approve/{rec_id} */
export interface CouncilApproveResponse {
  status: "approved" | "not_found" | "error";
  recommendation?: Record<string, unknown>;
  apply_result?: Record<string, unknown>;
  message?: string;
}

/** Response from POST /api/v1/council/trigger */
export type CouncilTriggerResponse = Record<string, unknown>;

// =============================================================================
// Market Data
// =============================================================================

/** Kline (candle) data point. */
export interface Kline {
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
  [key: string]: unknown;
}

/** Response from GET /api/v1/market/snapshot */
export interface MarketSnapshotResponse {
  symbol: string;
  last_tick: Tick | null;
  klines: Kline[];
  last_seq: number;
  [key: string]: unknown;
}

/** Response from GET /api/v1/market/ticker */
export interface MarketTickerResponse {
  symbol: string;
  price: number | null;
  bid: number | null;
  ask: number | null;
  volume: number | null;
  timestamp: string | null;
}

/** Response from GET /api/v1/market/candles */
export interface MarketCandlesResponse {
  symbol: string;
  timeframe: string;
  klines: Kline[];
  count: number;
}

// =============================================================================
// Analytics
// =============================================================================

/** Response from GET /api/v1/analytics/summary */
export interface AnalyticsSummaryResponse {
  total_trades: number;
  win_rate: number;
  profit_factor: number;
  max_drawdown: number;
  max_drawdown_pct: number;
  avg_trade_duration_sec: number;
  trades_per_day: number;
  total_pnl: number;
  daily_pnl: number;
}

/** Reason code with measured values. */
export interface ReasonCode {
  code: string;
  message: string;
  measured: Record<string, unknown>;
  thresholds: Record<string, unknown>;
  timestamp: string;
  symbol: string | null;
}

/** Response from GET /api/v1/analytics/blocks */
export interface BlockedSignalsAnalyticsResponse {
  total_blocks: number;
  top_reasons: Array<Record<string, unknown>>;
  blocks_by_hour: Record<string, number>;
  recent_blocks: ReasonCode[];
}

/** Response from POST /api/v1/analytics/backtest */
export interface BacktestJobResponse {
  job_id: string;
  status: "pending";
}

/** Response from POST /api/v1/analytics/evaluate */
export interface EvaluateJobResponse {
  job_id: string;
  status: "pending";
}

/** Response from GET /api/v1/analytics/phase5/correlation-matrix */
export interface CorrelationMatrixResponse {
  correlation_matrix: Record<string, number>;
  symbols: string[];
  min_correlation: number;
  threshold: number;
}

/** Response from GET /api/v1/analytics/phase5/profit-probability-curve */
export interface ProfitProbabilityCurveResponse {
  strategies: Array<{
    strategy_id: string;
    win_rate: number;
    odds_ratio: number;
    kelly_fraction: number;
    fractional_kelly: number;
    total_trades: number;
    profit_factor: number;
  }>;
  curve_points: Array<{
    strategy_id: string;
    position_fraction: number;
    probability_profit: number;
  }>;
}

/** Response from GET /api/v1/analytics/phase5/safety-net-status */
export interface SafetyNetStatusResponse {
  active: boolean;
  entropy_metrics: Record<string, unknown>;
  hedge_positions: Record<string, unknown>;
  size_multiplier?: number;
}

/** Response from GET /api/v1/analytics/phase5/system-health */
export interface SystemHealthStatusResponse {
  status: string;
  metrics: Record<string, unknown>;
  healthy?: boolean;
}

// =============================================================================
// Storage (DuckDB)
// =============================================================================

/** Response from GET /api/v1/storage/ticks */
export type StoredTicksResponse = Record<string, unknown>[];

/** Response from GET /api/v1/storage/physics */
export type StoredPhysicsResponse = Record<string, unknown>[];

/** Response from GET /api/v1/storage/brain */
export type StoredBrainResponse = Record<string, unknown>[];

// =============================================================================
// System
// =============================================================================

/** Changelog entry. */
export interface ChangelogEntry {
  hash?: string;
  message: string;
  author?: string;
  date?: string;
  type?: string;
  commit_hash?: string;
  timestamp?: string;
  category?: string;
}

/** Response from GET /api/v1/system/changelog/today */
export interface ChangelogTodayResponse {
  available: boolean;
  source?: string;
  reason?: string;
  entries: ChangelogEntry[];
  count: number;
  status?: string;
  date?: string;
  items_count?: number;
  items?: ChangelogEntry[];
}

/** Response from GET /api/v1/system/v1/dashboard */
export interface DashboardResponse {
  account_state: AccountState | null;
  metrics: {
    equity: number;
    daily_pnl: number;
    return_pct: number;
    open_positions: number;
  };
  positions: PositionDTO[];
  orders: OrderDTO[];
  status: {
    engine_running: boolean;
    trading_mode: string;
  };
}

/** Account state object used in multiple responses. */
export interface AccountState {
  equity: number;
  wallet_balance?: number;
  balance?: number;
  available_balance?: number;
  unrealized_pnl: number;
  realized_pnl: number;
  used_margin: number;
  free_margin?: number;
  fees?: number;
  [key: string]: unknown;
}

/** Response from GET /api/v1/system/cpp/status */
export interface CppStatusResponse {
  indicators_cpp_available: boolean;
  order_router_cpp_available: boolean;
  performance_hint: string;
  build_instructions: string;
  error?: string;
}

/** Response from GET /api/v1/system/agents */
export interface AgentsResponse {
  status: "ok";
  agents_count: number;
  agents: Record<string, unknown>[];
  note?: string;
}

// =============================================================================
// Telemetry
// =============================================================================

/** Response from GET /api/v1/telemetry/ping */
export interface TelemetryPingResponse {
  status: "ok";
  ts: string;
}

/** Response from GET /api/v1/telemetry/summary */
export interface TelemetrySummaryResponse {
  engine_state: string;
  mode: string;
  ws_clients: number;
  events_per_sec: number;
  events_total: number;
  last_event_ts: string;
  last_heartbeat: Record<string, unknown> | null;
  available: boolean;
  duration_ms: number;
  note: string;
  error?: { name: string; message: string };
}

/** Response from GET /api/v1/portfolio/summary */
export interface PortfolioSummaryResponse {
  available: boolean;
  equity: number | null;
  balance: number | null;
  used_margin: number;
  free_margin: number | null;
  unrealized_pnl: number;
  realized_pnl: number;
  fees: number;
  note: string;
  duration_ms: number;
  error?: { name: string; message: string };
}

/** Response from GET /api/v1/telemetry/signal-rejections */
export interface SignalRejectionsResponse {
  time_window_minutes: number;
  total_rejections: number;
  total_signals: number;
  rejection_rate: number;
  by_category: Record<string, number>;
  by_reason: Record<string, number>;
  by_symbol: Record<string, number>;
  by_strategy: Record<string, number>;
  rates: {
    "1m": number;
    "5m": number;
    "15m": number;
    "1h": number;
  };
}

/** Response from GET /api/v1/telemetry/signal-rejections/recent */
export interface RecentRejectionsResponse {
  count: number;
  limit: number;
  events: Record<string, unknown>[];
}

/** Response from GET /api/v1/telemetry/signal-rejections/summary */
export type RejectionSummaryResponse = Record<string, unknown>;

/** Health component detail. */
export interface HealthComponent {
  score: number;
  status: string;
  weight: number;
  details: Record<string, unknown>;
  last_updated: string;
  stale: boolean;
}

/** Response from GET /api/v1/telemetry/health */
export interface HealthScoreResponse {
  overall_score: number;
  status: string;
  components: Record<string, HealthComponent>;
  recommendations?: string[];
  [key: string]: unknown;
}

/** Response from GET /api/v1/telemetry/health/components */
export interface HealthComponentsResponse {
  overall_score: number;
  status: string;
  components: Record<string, HealthComponent>;
}

/** Response from GET /api/v1/telemetry/health/recommendations */
export interface HealthRecommendationsResponse {
  overall_score: number;
  status: string;
  can_trade: boolean;
  recommendations: string[];
  recommendation_count: number;
}

/** Latency stats for a single histogram. */
export interface LatencyHistogramStats {
  name: string;
  count: number;
  window_seconds: number;
  latency_ms: {
    min: number;
    max: number;
    mean: number;
    p50: number;
    p90: number;
    p95: number;
    p99: number;
    p999: number;
  };
  thresholds: {
    p999_warning_ms: number;
    p999_critical_ms: number;
  };
  alert_level: string;
}

/** Response from GET /api/v1/telemetry/latency */
export type LatencySummaryResponse = Record<string, unknown>;

/** Response from GET /api/v1/telemetry/latency/{histogram_name} */
export type LatencyHistogramResponse = LatencyHistogramStats;

/** Latency alert entry. */
export interface LatencyAlert {
  histogram: string;
  p999_ms: number;
  threshold_ms: number;
  timestamp: string;
  level: string;
}

/** Response from GET /api/v1/telemetry/latency/alerts/recent */
export interface LatencyAlertsResponse {
  alerts: LatencyAlert[];
  count: number;
  limit: number;
}

/** Response from GET /api/v1/telemetry/money-log */
export interface MoneyLogSummaryResponse {
  total_entries: number;
  by_type: Record<string, number>;
  recent: Record<string, unknown>[];
  [key: string]: unknown;
}

/** Response from GET /api/v1/telemetry/money-log/entries */
export interface MoneyLogEntriesResponse {
  entries: Record<string, unknown>[];
  count: number;
  limit: number;
  filters: {
    event_type: string | null;
    symbol: string | null;
  };
}

/** Response from GET /api/v1/telemetry/money-log/chain/{correlation_id} */
export interface MoneyLogChainResponse {
  chain?: Record<string, unknown>;
  entries?: Record<string, unknown>[];
  correlation_id?: string;
  error?: string;
}

/** Response from GET /api/v1/telemetry/money-log/stats */
export type MoneyLogStatsResponse = Record<string, unknown>;

/** Response from GET /api/v1/telemetry/money-log/verify */
export interface MoneyLogVerifyResponse {
  is_valid: boolean;
  violation_count: number;
  violations: string[];
}

// =============================================================================
// Meta-Learning
// =============================================================================

/** Response from GET /api/v1/meta-learning/state */
export interface MetaLearningStateResponse {
  total_scenarios_simulated: number;
  scenarios_per_second: number;
  failures_detected: number;
  patches_applied: number;
  performance_improvement: number;
  last_simulation_time: string | null;
}

/** Code weight entry from GET /api/v1/meta-learning/weights */
export interface CodeWeightResponse {
  name: string;
  file_path: string;
  line_number: number;
  current_value: number;
  value_range: number[];
  impact_score: number;
}

/** Patch history entry from GET /api/v1/meta-learning/patches */
export interface PatchHistoryResponse {
  timestamp: string;
  weight: string;
  old_value: number;
  new_value: number;
  scenario_id: string;
}

// =============================================================================
// Causal Inference
// =============================================================================

/** Causal relationship from GET /api/v1/causal-inference/relationships */
export interface CausalRelationship {
  source_symbol: string;
  target_symbol: string;
  granger_causality: number;
  transfer_entropy: number;
  lag_period: number;
  p_value: number;
  confidence: number;
  last_updated: string;
}

/** Pre-echo signal from GET /api/v1/causal-inference/pre-echoes */
export interface PreEchoSignal {
  target_symbol: string;
  source_symbol: string;
  predicted_direction: string;
  predicted_magnitude: number;
  confidence: number;
  lag_ms: number;
  granger_score: number;
  transfer_entropy_score: number;
  timestamp: string;
}

/** Response from GET /api/v1/causal-inference/stats */
export interface CausalInferenceStatsResponse {
  relationships: Record<string, CausalRelationship>;
  pre_echo_signals: PreEchoSignal[];
}

// =============================================================================
// Multimodal Swarm
// =============================================================================

/** Multimodal tensor from GET /api/v1/multimodal-swarm/tensors/{symbol} */
export interface MultimodalTensor {
  timestamp: string;
  symbol: string;
  price_features: number[];
  sentiment_features: number[];
  onchain_features: number[];
  macro_features: number[];
  unified_tensor: number[];
  confidence: number;
  modality_weights: Record<string, number>;
}

/** Response from GET /api/v1/multimodal-swarm/stats */
export interface MultimodalSwarmStatsResponse {
  tensor_size: number;
  modality_weights: Record<string, number>;
  num_agents: number;
}

// =============================================================================
// Singularity
// =============================================================================

/** Response from GET /api/v1/singularity/metamorphic/sel */
export interface SystemEvolutionLevelResponse {
  sel: number;
  status: string;
}

/** Causal graph node. */
export interface CausalGraphNode {
  id: string;
  x: number;
  y: number;
  z: number;
}

/** Causal graph edge. */
export interface CausalGraphEdge {
  source: string;
  target: string;
  strength: number;
  lag_us: number;
}

/** Response from GET /api/v1/singularity/causal/graph */
export interface CausalGraphResponse {
  nodes: CausalGraphNode[];
  edges: CausalGraphEdge[];
  status: string;
}

/** Response from GET /api/v1/singularity/atomic/clusters */
export interface AtomicClustersResponse {
  clusters: unknown[];
  statistics: {
    total_clusters_created: number;
    active_clusters: number;
    total_orders_placed: number;
  };
  status: string;
}

// =============================================================================
// Graph Engine
// =============================================================================

/** Asset in graph state. */
export interface GraphAsset {
  symbol: string;
  price: number;
  correlation: Record<string, number>;
  leader_score: number;
}

/** Response from GET /api/v1/graph-engine/state */
export interface GraphEngineStateResponse {
  assets: GraphAsset[];
  correlations: Record<string, number>;
  asset_count: number;
}

/** Response from GET /api/v1/graph-engine/leader */
export interface GraphLeaderResponse {
  leader: string;
  leader_score: number;
}

/** Response from GET /api/v1/graph-engine/feature-vector */
export interface FeatureVectorResponse {
  feature_vector: number[];
  size: number;
}

/** Response from GET /api/v1/graph-engine/topology/score */
export interface TopologyScoreResponse {
  market_topology_score: number;
  is_disconnected: boolean;
  stability: "stable" | "unstable" | "collapsing";
}

/** Response from GET /api/v1/graph-engine/topology/manifold/{symbol} */
export interface ManifoldDataResponse {
  symbol: string;
  point_cloud: unknown[];
  persistence_barcodes: unknown[];
  num_holes: number;
  topology_score: number;
}

/** Response from GET /api/v1/graph-engine/topology/watchdog */
export interface WatchdogStatusResponse {
  halt_active: boolean;
  topology_score: number;
  is_disconnected: boolean;
}

// =============================================================================
// Meta-Brain (MetaStrategyBrain)
// =============================================================================

/** Response from GET /api/v1/meta-brain/status */
export interface MetaBrainStatusResponse {
  enabled: boolean;
  message?: string;
  strategies?: Record<string, unknown>;
  current_regime?: string;
  regime_confidence?: number;
}

/** Response from GET /api/v1/meta-brain/transitions */
export interface MetaBrainTransitionsResponse {
  transitions: Record<string, unknown>[];
}

/** Response from GET /api/v1/meta-brain/affinity-matrix */
export interface MetaBrainAffinityMatrixResponse {
  matrix: Record<string, unknown>;
}

/** Response from POST /api/v1/meta-brain/force-state */
export interface MetaBrainForceStateResponse {
  success: boolean;
  strategy_id: string;
  new_state: string;
}

/** Response from GET /api/v1/meta-brain/evolution */
export interface MetaBrainEvolutionResponse {
  enabled: boolean;
  pending: unknown[];
  evolved: unknown[];
}

// =============================================================================
// Emergency Killswitch
// =============================================================================

/** Response from POST /api/v1/emergency/killswitch */
export interface EmergencyKillswitchResponse {
  status: "success";
  message: string;
  reason: string;
  response_time_ms: number;
  timestamp: string;
}

// =============================================================================
// Health Check
// =============================================================================

/** Component health entry. */
export interface HealthCheckComponent {
  status: string;
  queue_depth?: number;
  active_clients?: number;
  triggered?: boolean;
}

/** Response from GET /health */
export interface HealthCheckResponse {
  status: "healthy" | "degraded";
  timestamp: string;
  components: {
    api: HealthCheckComponent;
    event_bus: HealthCheckComponent;
    redis: HealthCheckComponent;
    engine: HealthCheckComponent;
    killswitch: HealthCheckComponent;
    websocket: HealthCheckComponent;
  };
}

// =============================================================================
// Settings
// =============================================================================

/** Response from GET /settings */
export interface SettingsResponse {
  trading_mode: string;
  environment: string;
  is_live: boolean;
  bybit_testnet: boolean;
  bybit_api_key: string | null;
  bybit_api_secret: string | null;
  api_auth_enabled: boolean;
  api_auth_key: string | null;
  jwt_secret: string | null;
  gemini_api_key: string | null;
  initial_capital: number;
  max_trade_risk_pct: number;
  max_open_positions: number;
  max_daily_drawdown_pct: number;
  killswitch_drawdown_pct: number;
  trading_symbols: string[];
  total_symbols: number;
  debug_mode: boolean;
  log_level: string;
  impulse_engine_enabled: boolean;
  funding_harvester_enabled: boolean;
  basis_arbitrage_enabled: boolean;
}

// =============================================================================
// WebSocket Types
// =============================================================================

/** WebSocket action types (client -> server). */
export enum WebSocketAction {
  SUBSCRIBE = "subscribe",
  UNSUBSCRIBE = "unsubscribe",
  PING = "ping",
}

/** Valid WebSocket subscription topics. */
export enum WebSocketTopic {
  SYSTEM_STATUS = "system_status",
  SYSTEM_HEARTBEAT = "system_heartbeat",
  TELEMETRY = "telemetry",
  MARKET_DATA = "market_data",
  MARKET_TICKS = "market_ticks",
  TRADING_SIGNALS = "trading_signals",
  POSITIONS = "positions",
  ORDERS = "orders",
  ORDERS_SNAPSHOT = "orders_snapshot",
  ACCOUNT_STATE = "account_state",
  ORDER_DECISIONS = "order_decisions",
  ORDER_EXIT_DECISIONS = "order_exit_decisions",
  PERFORMANCE = "performance",
  RISK = "risk",
  RISK_EVENTS = "risk_events",
  STRATEGY_EVENTS = "strategy_events",
  SIGNALS = "signals",
  TRADING_EVENTS = "trading_events",
  TRADING_METRICS = "trading_metrics",
  METRICS = "metrics",
  SNAPSHOT = "snapshot",
  LOGS = "logs",
  PHYSICS_UPDATE = "physics_update",
  BRAIN_UPDATE = "brain_update",
  AI_REASONING = "ai_reasoning",
  AI_CATALYST = "ai_catalyst",
  TRIANGULAR_ARB = "triangular_arb",
  ORDER_FILLED = "order_filled",
  ORDER_CANCELLED = "order_cancelled",
}

/** Client -> server WebSocket message. */
export interface WebSocketClientMessage {
  action: WebSocketAction;
  topic?: WebSocketTopic | string;
  data?: Record<string, unknown>;
}

/** Server -> client subscription confirmation. */
export interface WebSocketSubscribedMessage {
  type: "subscribed";
  topic: string;
  connection_id: string;
}

/** Server -> client unsubscribed confirmation. */
export interface WebSocketUnsubscribedMessage {
  type: "unsubscribed";
  topic: string;
}

/** Server -> client pong (heartbeat response). */
export interface WebSocketPongMessage {
  type: "pong";
  timestamp: string;
}

/** Server -> client ping (keepalive). */
export interface WebSocketPingMessage {
  type: "ping";
  timestamp: string;
}

/** Server -> client error message. */
export interface WebSocketErrorMessage {
  type: "error";
  error?: string;
  message?: string;
  details?: unknown[];
  timestamp?: string;
}

/** Event envelope wrapping all topic messages from the server. */
export interface EventEnvelope {
  seq: number;
  type: string;
  payload: Record<string, unknown>;
  timestamp: string;
  source: string;
  severity: string;
  correlation_id: string | null;
  context: Record<string, unknown>;
  topic: string;
}

/** Topic message wrapper sent by the server. */
export interface WebSocketTopicMessage {
  topic: string;
  data: Record<string, unknown>;
  timestamp: string;
}

// --- Specific topic message payloads ---

/** system_status topic: status_update */
export interface SystemStatusUpdate {
  type: "status_update" | "engine_status" | "killswitch_triggered";
  engine: "running" | "stopped";
  redis: "connected" | "disconnected" | "not_configured";
  equity: number;
  timestamp: string;
}

/** system_heartbeat topic payload. */
export interface HeartbeatPayload {
  type: "HEARTBEAT";
  engine_state: string;
  mode: string;
  ws_clients: number;
  uptime_sec: number;
  events_per_sec: number;
  last_event_ts: string;
  timestamp: string;
}

/** metrics topic: metrics_update */
export interface MetricsUpdate {
  type: "metrics_update" | "equity";
  equity: number;
  daily_pnl: number;
  return_pct: number;
  open_positions: number;
  engine_running?: boolean;
  timestamp: string;
}

/** market_ticks / market_data topic: MARKET_TICK */
export interface MarketTickPayload {
  kind: "tick";
  symbol: string;
  price: number;
  bid: number | null;
  ask: number | null;
  volume: number;
  ts: string;
  ts_ms: number;
  source?: string;
}

/** market_data topic: kline */
export interface KlinePayload {
  kind: "kline";
  symbol: string;
  timeframe: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  timestamp: number;
}

/** signals topic payload. */
export interface SignalPayload {
  type: "signal";
  symbol: string;
  side: string;
  strategy_id: string;
  size?: number;
  entry_price?: number;
  metadata: Record<string, unknown>;
}

/** orders topic: order_event */
export interface OrderEventPayload {
  type: "order_event";
  event: string;
  order_id: string;
  symbol: string;
  side: string;
  quantity: number;
  price: number | null;
  status: string;
  created_at: string;
  updated_at: string;
  filled_qty: number;
  avg_fill_price: number | null;
  fee: number | null;
  strategy_id: string;
  signal_id: string | null;
  rationale: string | null;
  exit_plan: Record<string, unknown> | null;
  expected_pnl: number | null;
  status_timeline: Array<{ status: string; timestamp: string }>;
  leverage: number;
  margin_used: number;
}

/** positions topic payload. */
export interface PositionsPayload {
  positions: Array<{
    position_id: string;
    symbol: string;
    side: string;
    qty: number;
    entry_price: number;
    mark_price: number;
    unrealized_pnl: number;
    realized_pnl: number;
    opened_at: string | null;
    strategy_id: string;
    exit_plan: Record<string, unknown>;
    status: "open" | "closed";
    closed_at: string | null;
  }>;
  type: "snapshot" | "update" | "pnl_update" | string;
}

/** account_state topic payload. */
export interface AccountStatePayload {
  equity: number;
  wallet_balance?: number;
  balance?: number;
  available_balance?: number;
  unrealized_pnl: number;
  realized_pnl: number;
  used_margin?: number;
  fees?: number;
  [key: string]: unknown;
}

/** order_decisions topic payload. */
export interface OrderDecisionPayload {
  decision: Record<string, unknown>;
  type: "order_decision";
}

/** order_exit_decisions topic payload. */
export interface OrderExitDecisionPayload {
  decision: Record<string, unknown>;
  type: "order_exit_decision";
}

/** physics_update topic payload. */
export interface PhysicsUpdatePayload {
  temperature: number;
  entropy: number;
  phase: string;
  regime: string;
  liquidations: unknown[];
  players: {
    marketMaker: number;
    institutional: number;
    arbBot: number;
    retail: number;
    whale: number;
  };
}

/** brain_update topic payload. */
export interface BrainUpdatePayload {
  stage: string;
  content: string;
  confidence: number;
  analysis: Record<string, unknown>;
}

/** risk_events topic: risk_blocked */
export interface RiskBlockedPayload {
  type: "risk_blocked";
  reason: string;
  symbol: string;
  strategy_id: string;
  metadata: Record<string, unknown>;
  timestamp: string;
}

/** trading_events topic: SIGNAL_DETECTED */
export interface SignalDetectedPayload {
  signal_id: string;
  symbol: string;
  timeframe: string;
  strategy_id: string;
  score: number | null;
  confidence: number | null;
  signal_type: string | null;
  features: Record<string, unknown>;
}

/** trading_events topic: ORDER_CREATED */
export interface OrderCreatedPayload {
  order_id: string;
  symbol: string;
  side: string;
  qty: number;
  price: number | null;
  type: string;
  strategy_id: string;
  signal_id: string | null;
}

/** trading_events topic: ORDER_UPDATE */
export interface OrderUpdatePayload {
  order_id: string;
  status: string;
  filled_qty: number;
  avg_price: number | null;
  last_fill_ts: string;
  reject_reason: string | null;
}

/** trading_events topic: POSITION_UPDATE */
export interface PositionUpdatePayload {
  symbol: string;
  size: number;
  entry_price: number;
  mark_price: number;
  unrealized_pnl: number;
  realized_pnl_delta: number | null;
  position_id: string;
}

/** trading_events topic: PNL_SNAPSHOT */
export interface PnlSnapshotPayload {
  equity: number;
  balance: number;
  unrealized_pnl: number;
  realized_pnl: number;
  fees: number;
  used_margin: number;
  free_margin: number;
}

/** trading_events topic: ORDER_DECISION (enhanced) */
export interface OrderDecisionFunnelPayload {
  decision: string;
  signal_id: string | null;
  reason_codes: string[];
  confidence: number | null;
  score: number | null;
  intended_side: string | null;
  intended_size: number | null;
  gating: {
    risk_ok: boolean;
    liquidity_ok: boolean;
    cooldown_ok: boolean;
    max_orders_ok: boolean;
    paused: boolean;
  };
  [key: string]: unknown;
}

/** trading_metrics topic payload. */
export interface TradingMetricsPayload {
  type: "trading_metrics_update" | "trading_metrics_snapshot";
  counters: {
    last_1m: TradingCountersBucket;
    last_5m: TradingCountersBucket;
    session: TradingCountersBucket;
  };
  active_orders_count: number;
  active_positions_count: number;
  engine_state: string;
  mode: string;
  timestamp: string;
}

/** snapshot topic: full realtime snapshot */
export interface RealtimeSnapshot {
  type: "snapshot";
  account_state: AccountState | null;
  positions: PositionDTO[];
  orders: OrderDTO[];
  order_decisions: Record<string, unknown>[];
  order_exit_decisions: Record<string, unknown>[];
  last_seq: number;
  events: EventEnvelope[];
  market?: MarketSnapshotResponse;
}

/** Union type for all possible server WebSocket messages. */
export type WebSocketServerMessage =
  | WebSocketSubscribedMessage
  | WebSocketUnsubscribedMessage
  | WebSocketPongMessage
  | WebSocketPingMessage
  | WebSocketErrorMessage
  | WebSocketTopicMessage;

// =============================================================================
// API Request Body Types
// =============================================================================

/** POST /api/v1/engine/start request body. */
export interface EngineStartRequest {
  confirm_phrase?: string | null;
}

/** POST /api/v1/engine/stop request body. */
export type EngineStopRequest = Record<string, never>;

/** POST /api/v1/engine/pause request body. */
export type EnginePauseRequest = Record<string, never>;

/** POST /api/v1/orders/test request body. */
export interface TestOrderRequestBody {
  symbol?: string;
  side?: "buy" | "sell";
  size?: number;
  price?: number | null;
}

/** POST /api/v1/orders/test_roundtrip request body. */
export interface TestRoundtripRequestBody {
  symbol?: string;
  side?: "buy" | "sell";
  size?: number;
  take_profit_pct?: number;
  stop_loss_pct?: number;
  hold_timeout_sec?: number;
}

/** POST /api/v1/orders/close-position request body. */
export interface ClosePositionRequestBody {
  position_id: string;
  confirm_phrase?: string | null;
}

/** POST /api/v1/orders/cancel request body. */
export interface CancelOrderRequestBody {
  order_id: string;
}

/** POST /api/v1/orders/cancel-all request body. */
export interface CancelAllOrdersRequestBody {
  confirm_phrase?: string | null;
}

/** POST /api/v1/orders/close-position-by-symbol request body. */
export interface ClosePositionBySymbolRequestBody {
  symbol: string;
}

/** POST /api/v1/strategies/{id}/enable request body. */
export interface StrategyEnableRequestBody {
  enabled: boolean;
}

/** POST /api/v1/strategies/{id}/params request body. */
export interface StrategyParamsRequestBody {
  params: Record<string, unknown>;
}

/** POST /api/v1/risk/limits request body. */
export interface RiskLimitsRequestBody {
  max_open_positions?: number | null;
  max_daily_attempts?: number | null;
  max_exposure_usd?: number | null;
  min_notional_usd?: number | null;
  cooldown_seconds?: number | null;
}

/** POST /api/v1/analytics/backtest request body. */
export interface BacktestRequestBody {
  symbol?: string;
  start_date: string;
  end_date: string;
  initial_capital?: number;
  strategy_id?: string | null;
}

/** POST /api/v1/analytics/evaluate request body. */
export interface EvaluateRequestBody {
  symbol?: string;
  days?: number;
}

/** POST /api/v1/risk/governor/clear request body. */
export interface RiskGovernorClearRequestBody {
  confirm?: boolean;
  force?: boolean;
  symbol?: string | null;
}

/** POST /api/v1/multimodal-swarm/modality-weights request body. */
export type ModalityWeightsRequestBody = Record<string, number>;

/** POST /api/v1/meta-brain/force-state query params. */
export interface MetaBrainForceStateParams {
  strategy_id: string;
  state: string;
}

// =============================================================================
// API Endpoint Constants
// =============================================================================

/** Base API prefix for all endpoints. */
export const API_PREFIX = "/api/v1" as const;

/** All API endpoint paths. */
export const API_ENDPOINTS = {
  // Engine Control
  ENGINE_START: `${API_PREFIX}/engine/start`,
  ENGINE_STOP: `${API_PREFIX}/engine/stop`,
  ENGINE_PAUSE: `${API_PREFIX}/engine/pause`,
  ENGINE_RESUME: `${API_PREFIX}/engine/resume`,
  ENGINE_KILL: `${API_PREFIX}/engine/kill`,
  ENGINE_RESTART: `${API_PREFIX}/engine/restart`,
  ENGINE_STATUS: `${API_PREFIX}/engine/status`,
  ENGINE_LOCK_PROFIT: `${API_PREFIX}/engine/lock-profit`,

  // Trading / Orders
  ORDERS: `${API_PREFIX}/orders`,
  POSITIONS: `${API_PREFIX}/orders/positions`,
  POSITION_MONITOR_STATS: `${API_PREFIX}/orders/positions/monitor/stats`,
  TEST_ORDER: `${API_PREFIX}/orders/test`,
  TEST_ROUNDTRIP: `${API_PREFIX}/orders/test_roundtrip`,
  CLOSE_POSITION: `${API_PREFIX}/orders/close-position`,
  CLOSE_POSITION_BY_SYMBOL: `${API_PREFIX}/orders/close-position-by-symbol`,
  CLOSE_ALL_POSITIONS: `${API_PREFIX}/orders/close-all-positions`,
  CANCEL_ORDER: `${API_PREFIX}/orders/cancel`,
  CANCEL_ALL_ORDERS: `${API_PREFIX}/orders/cancel-all`,
  PANIC_CLOSE_ALL: `${API_PREFIX}/orders/paper/close_all`,
  RESET_PAPER_STATE: `${API_PREFIX}/orders/paper/reset_state`,
  ORDERBOOK_PRESENCE: `${API_PREFIX}/orders/orderbook-presence`,

  // Trading Diagnostics
  TRADING_WHY: `${API_PREFIX}/trading/why`,
  TRADING_METRICS: `${API_PREFIX}/trading/metrics`,
  TRADING_STATE: `${API_PREFIX}/trading/state`,
  EQUITY_HISTORY: `${API_PREFIX}/trading/equity-history`,

  // Strategies
  STRATEGIES: `${API_PREFIX}/strategies`,
  STRATEGY_ENABLE: (id: string) => `${API_PREFIX}/strategies/${id}/enable`,
  STRATEGY_PARAMS: (id: string) => `${API_PREFIX}/strategies/${id}/params`,

  // Risk
  RISK_STATUS: `${API_PREFIX}/risk/status`,
  RISK_LIMITS: `${API_PREFIX}/risk/limits`,
  RISK_DECISION_MEMORY_BLOCKS: `${API_PREFIX}/risk/decision-memory/blocks`,
  KILLSWITCH_STATUS: `${API_PREFIX}/risk/killswitch/status`,
  KILLSWITCH_RESET: `${API_PREFIX}/risk/killswitch/reset`,

  // Risk Governor
  RISK_GOVERNOR_STATUS: `${API_PREFIX}/risk/governor/status`,
  RISK_GOVERNOR_CLEAR: `${API_PREFIX}/risk/governor/clear`,
  RISK_GOVERNOR_QUARANTINE: (symbol: string) =>
    `${API_PREFIX}/risk/governor/quarantine/${symbol}`,

  // Physics
  PHYSICS_STATE: `${API_PREFIX}/physics/state`,
  PHYSICS_HISTORY: `${API_PREFIX}/physics/history`,
  PHYSICS_PARTICIPANTS: `${API_PREFIX}/physics/participants`,
  PHYSICS_ANOMALIES: `${API_PREFIX}/physics/anomalies`,

  // Temporal
  TEMPORAL_STACK: `${API_PREFIX}/temporal/stack`,
  TEMPORAL_IMPULSE: `${API_PREFIX}/temporal/impulse`,
  TEMPORAL_SESSIONS: `${API_PREFIX}/temporal/sessions`,

  // Brain
  BRAIN_ANALYSIS: `${API_PREFIX}/brain/analysis`,
  BRAIN_THOUGHTS: `${API_PREFIX}/brain/thoughts`,
  BRAIN_HISTORY: `${API_PREFIX}/brain/history`,

  // Council
  COUNCIL_STATUS: `${API_PREFIX}/council/status`,
  COUNCIL_REVIEWS: `${API_PREFIX}/council/reviews`,
  COUNCIL_RECOMMENDATIONS: `${API_PREFIX}/council/recommendations`,
  COUNCIL_APPROVE: (recId: string) => `${API_PREFIX}/council/approve/${recId}`,
  COUNCIL_TRIGGER: `${API_PREFIX}/council/trigger`,

  // Market Data
  MARKET_SNAPSHOT: `${API_PREFIX}/market/snapshot`,
  MARKET_TICKER: `${API_PREFIX}/market/ticker`,
  MARKET_CANDLES: `${API_PREFIX}/market/candles`,

  // Analytics
  ANALYTICS_SUMMARY: `${API_PREFIX}/analytics/summary`,
  ANALYTICS_BLOCKS: `${API_PREFIX}/analytics/blocks`,
  ANALYTICS_BACKTEST: `${API_PREFIX}/analytics/backtest`,
  ANALYTICS_EVALUATE: `${API_PREFIX}/analytics/evaluate`,
  ANALYTICS_CORRELATION_MATRIX: `${API_PREFIX}/analytics/phase5/correlation-matrix`,
  ANALYTICS_PROFIT_PROBABILITY: `${API_PREFIX}/analytics/phase5/profit-probability-curve`,
  ANALYTICS_SAFETY_NET: `${API_PREFIX}/analytics/phase5/safety-net-status`,
  ANALYTICS_SYSTEM_HEALTH: `${API_PREFIX}/analytics/phase5/system-health`,

  // Storage (DuckDB)
  STORAGE_TICKS: `${API_PREFIX}/storage/ticks`,
  STORAGE_PHYSICS: `${API_PREFIX}/storage/physics`,
  STORAGE_BRAIN: `${API_PREFIX}/storage/brain`,

  // System
  SYSTEM_CHANGELOG_TODAY: `${API_PREFIX}/system/changelog/today`,
  SYSTEM_DASHBOARD: `${API_PREFIX}/system/v1/dashboard`,
  SYSTEM_CPP_STATUS: `${API_PREFIX}/system/cpp/status`,
  SYSTEM_AGENTS: `${API_PREFIX}/system/agents`,

  // Telemetry
  TELEMETRY_PING: `${API_PREFIX}/telemetry/ping`,
  TELEMETRY_SUMMARY: `${API_PREFIX}/telemetry/summary`,
  PORTFOLIO_SUMMARY: `${API_PREFIX}/portfolio/summary`,
  TELEMETRY_SIGNAL_REJECTIONS: `${API_PREFIX}/telemetry/signal-rejections`,
  TELEMETRY_SIGNAL_REJECTIONS_RECENT: `${API_PREFIX}/telemetry/signal-rejections/recent`,
  TELEMETRY_SIGNAL_REJECTIONS_SUMMARY: `${API_PREFIX}/telemetry/signal-rejections/summary`,
  TELEMETRY_HEALTH: `${API_PREFIX}/telemetry/health`,
  TELEMETRY_HEALTH_COMPONENTS: `${API_PREFIX}/telemetry/health/components`,
  TELEMETRY_HEALTH_RECOMMENDATIONS: `${API_PREFIX}/telemetry/health/recommendations`,
  TELEMETRY_LATENCY: `${API_PREFIX}/telemetry/latency`,
  TELEMETRY_LATENCY_HISTOGRAM: (name: string) =>
    `${API_PREFIX}/telemetry/latency/${name}`,
  TELEMETRY_LATENCY_ALERTS: `${API_PREFIX}/telemetry/latency/alerts/recent`,
  TELEMETRY_LATENCY_PROMETHEUS: `${API_PREFIX}/telemetry/latency/prometheus`,
  TELEMETRY_MONEY_LOG: `${API_PREFIX}/telemetry/money-log`,
  TELEMETRY_MONEY_LOG_ENTRIES: `${API_PREFIX}/telemetry/money-log/entries`,
  TELEMETRY_MONEY_LOG_CHAIN: (id: string) =>
    `${API_PREFIX}/telemetry/money-log/chain/${id}`,
  TELEMETRY_MONEY_LOG_STATS: `${API_PREFIX}/telemetry/money-log/stats`,
  TELEMETRY_MONEY_LOG_VERIFY: `${API_PREFIX}/telemetry/money-log/verify`,

  // Meta-Learning
  META_LEARNING_STATE: `${API_PREFIX}/meta-learning/state`,
  META_LEARNING_WEIGHTS: `${API_PREFIX}/meta-learning/weights`,
  META_LEARNING_PATCHES: `${API_PREFIX}/meta-learning/patches`,

  // Causal Inference
  CAUSAL_INFERENCE_STATS: `${API_PREFIX}/causal-inference/stats`,
  CAUSAL_INFERENCE_RELATIONSHIPS: `${API_PREFIX}/causal-inference/relationships`,
  CAUSAL_INFERENCE_PRE_ECHOES: `${API_PREFIX}/causal-inference/pre-echoes`,

  // Multimodal Swarm
  MULTIMODAL_SWARM_STATS: `${API_PREFIX}/multimodal-swarm/stats`,
  MULTIMODAL_SWARM_TENSORS: (symbol: string) =>
    `${API_PREFIX}/multimodal-swarm/tensors/${symbol}`,
  MULTIMODAL_SWARM_WEIGHTS: `${API_PREFIX}/multimodal-swarm/modality-weights`,

  // Singularity
  SINGULARITY_SEL: `${API_PREFIX}/singularity/metamorphic/sel`,
  SINGULARITY_CAUSAL_GRAPH: `${API_PREFIX}/singularity/causal/graph`,
  SINGULARITY_ATOMIC_CLUSTERS: `${API_PREFIX}/singularity/atomic/clusters`,

  // Graph Engine
  GRAPH_ENGINE_STATE: `${API_PREFIX}/graph-engine/state`,
  GRAPH_ENGINE_LEADER: `${API_PREFIX}/graph-engine/leader`,
  GRAPH_ENGINE_FEATURE_VECTOR: `${API_PREFIX}/graph-engine/feature-vector`,
  GRAPH_ENGINE_TOPOLOGY_SCORE: `${API_PREFIX}/graph-engine/topology/score`,
  GRAPH_ENGINE_MANIFOLD: (symbol: string) =>
    `${API_PREFIX}/graph-engine/topology/manifold/${symbol}`,
  GRAPH_ENGINE_WATCHDOG: `${API_PREFIX}/graph-engine/topology/watchdog`,

  // Meta-Brain
  META_BRAIN_STATUS: `${API_PREFIX}/meta-brain/status`,
  META_BRAIN_TRANSITIONS: `${API_PREFIX}/meta-brain/transitions`,
  META_BRAIN_AFFINITY_MATRIX: `${API_PREFIX}/meta-brain/affinity-matrix`,
  META_BRAIN_FORCE_STATE: `${API_PREFIX}/meta-brain/force-state`,
  META_BRAIN_EVOLUTION: `${API_PREFIX}/meta-brain/evolution`,

  // Emergency
  EMERGENCY_KILLSWITCH: `${API_PREFIX}/emergency/killswitch`,

  // Non-prefixed endpoints
  HEALTH: "/health",
  SETTINGS: "/settings",
  METRICS_PROMETHEUS: "/metrics",

  // WebSocket
  WEBSOCKET: "/ws",
} as const;

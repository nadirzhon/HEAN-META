/**
 * Type definitions for HEAN backend API responses.
 * All fields match the actual backend contract.
 */

export interface EngineStatus {
  status: string;
  running: boolean;
  equity: number;
  daily_pnl: number;
  initial_capital: number;
}

export interface Position {
  position_id: string;
  symbol: string;
  side: "LONG" | "SHORT";
  size: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  leverage?: number;
  opened_at?: number;
}

export interface Order {
  order_id: string;
  symbol: string;
  side: "BUY" | "SELL";
  qty: number;
  price: number;
  status: string;
  order_type?: string;
  created_at?: number;
}

export interface RiskGovernorStatus {
  risk_state: "NORMAL" | "SOFT_BRAKE" | "QUARANTINE" | "HARD_STOP";
  level: number;
  reason_codes: string[];
  quarantined_symbols: string[];
  can_clear: boolean;
}

export interface KillswitchStatus {
  triggered: boolean;
  reasons: string[];
  thresholds?: Record<string, number>;
  current_metrics?: Record<string, number>;
}

export interface TradingMetrics {
  counters: {
    last_1m: MetricCounters;
    last_5m: MetricCounters;
    session: MetricCounters;
  };
}

export interface MetricCounters {
  signals_detected: number;
  orders_created: number;
  orders_filled: number;
  positions_opened: number;
  positions_closed: number;
}

export interface Strategy {
  strategy_id: string;
  type: string;
  enabled: boolean;
}

export interface WebSocketEvent {
  type: string;
  timestamp: number;
  data?: any;
}

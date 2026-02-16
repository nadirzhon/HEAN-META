/**
 * HEAN API Service
 *
 * Fetches data from the HEAN backend API at http://localhost:8000/api/v1/
 */

const BASE_URL = "http://localhost:8000/api/v1";

async function fetchJSON<T>(endpoint: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${endpoint}`, {
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    throw new Error(`API error ${res.status}: ${endpoint}`);
  }
  return res.json();
}

export const api = {
  /** Engine status — running state, equity, PnL */
  getEngineStatus: () =>
    fetchJSON<{
      status: string;
      running: boolean;
      equity: number;
      daily_pnl: number;
      initial_capital: number;
    }>("/engine/status"),

  /** Open positions */
  getPositions: () =>
    fetchJSON<
      Array<{
        position_id: string;
        symbol: string;
        side: string;
        size: number;
        entry_price: number;
        current_price: number;
        unrealized_pnl: number;
        leverage: number;
        strategy_id?: string;
      }>
    >("/orders/positions"),

  /** Active orders */
  getOrders: () =>
    fetchJSON<
      Array<{
        order_id: string;
        symbol: string;
        side: string;
        qty: number;
        price: number;
        status: string;
        order_type: string;
        created_at?: string;
      }>
    >("/orders"),

  /** Strategy states */
  getStrategies: () =>
    fetchJSON<
      Array<{
        strategy_id: string;
        type: string;
        enabled: boolean;
      }>
    >("/strategies"),

  /** Risk governor state */
  getRiskGovernor: () =>
    fetchJSON<{
      risk_state: string;
      level: number;
      reason_codes: string[];
      quarantined_symbols: string[];
      can_clear: boolean;
    }>("/risk/governor/status"),

  /** KillSwitch state */
  getKillSwitch: () =>
    fetchJSON<{
      triggered: boolean;
      reasons: string[];
      thresholds: Record<string, number>;
      current_metrics: Record<string, number>;
    }>("/risk/killswitch/status"),

  /** Trading metrics (signal/order counts) */
  getTradingMetrics: () =>
    fetchJSON<{
      counters: {
        last_1m: { signals_detected: number; orders_created: number; orders_filled: number };
        last_5m: { signals_detected: number; orders_created: number; orders_filled: number };
        session: { signals_detected: number; orders_created: number; orders_filled: number };
      };
    }>("/trading/metrics"),

  /** Physics state for a symbol */
  getPhysics: (symbol: string) =>
    fetchJSON<{
      temperature: number;
      entropy: number;
      phase: string;
      szilard_profit: number;
    }>(`/physics/state?symbol=${symbol}`),

  /** Brain AI analysis */
  getBrainAnalysis: () =>
    fetchJSON<{
      timestamp: string;
      summary: string;
      market_sentiment: string;
      recommendations: string[];
    }>("/brain/analysis"),

  /** Council status */
  getCouncilStatus: () =>
    fetchJSON<{
      active: boolean;
      last_decision?: string;
      consensus?: string;
    }>("/council/status"),

  /** Trading diagnostic */
  getTradingWhy: () =>
    fetchJSON<Record<string, unknown>>("/trading/why"),

  /** Meta-brain strategy lifecycle */
  getMetaBrain: () =>
    fetchJSON<Record<string, unknown>>("/meta-brain/status"),
};

/**
 * Test data generators for Cockpit tab components.
 * Use ONLY for development/testing - NEVER in production.
 */

import type {
  EngineStatus,
  Position,
  RiskGovernorStatus,
  TradingMetrics,
} from "../types/api";

/**
 * Generate mock engine status data.
 * For testing only - real app must use API.
 */
export function generateMockEngineStatus(): EngineStatus {
  const initialCapital = 1000;
  const equity = initialCapital + Math.random() * 500 - 100;
  const dailyPnl = equity - initialCapital;

  return {
    status: "running",
    running: true,
    equity,
    daily_pnl: dailyPnl,
    initial_capital: initialCapital,
  };
}

/**
 * Generate mock positions data.
 * For testing only - real app must use API.
 */
export function generateMockPositions(count: number = 3): Position[] {
  const symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT"];
  const positions: Position[] = [];

  for (let i = 0; i < count; i++) {
    const symbol = symbols[i % symbols.length];
    const side = Math.random() > 0.5 ? "LONG" : "SHORT";
    const entryPrice = Math.random() * 50000 + 10000;
    const size = Math.random() * 0.1 + 0.01;
    const currentPrice = entryPrice + (Math.random() * 2000 - 1000);
    const unrealizedPnl =
      side === "LONG"
        ? (currentPrice - entryPrice) * size
        : (entryPrice - currentPrice) * size;

    positions.push({
      position_id: `pos-${i + 1}`,
      symbol,
      side: side as "LONG" | "SHORT",
      size,
      entry_price: entryPrice,
      current_price: currentPrice,
      unrealized_pnl: unrealizedPnl,
      realized_pnl: 0,
      leverage: 10,
      opened_at: Date.now() - Math.random() * 3600000,
    });
  }

  return positions;
}

/**
 * Generate mock risk governor status.
 * For testing only - real app must use API.
 */
export function generateMockRiskStatus(): RiskGovernorStatus {
  const states = ["NORMAL", "SOFT_BRAKE", "QUARANTINE", "HARD_STOP"] as const;
  const state = states[Math.floor(Math.random() * states.length)];

  return {
    risk_state: state,
    level: Math.floor(Math.random() * 100),
    reason_codes: state === "NORMAL" ? [] : ["HIGH_VOLATILITY", "RAPID_DRAWDOWN"],
    quarantined_symbols: state === "QUARANTINE" ? ["BTCUSDT"] : [],
    can_clear: state !== "HARD_STOP",
  };
}

/**
 * Generate mock trading metrics.
 * For testing only - real app must use API.
 */
export function generateMockMetrics(): TradingMetrics {
  const generateCounters = () => ({
    signals_detected: Math.floor(Math.random() * 50),
    orders_created: Math.floor(Math.random() * 30),
    orders_filled: Math.floor(Math.random() * 20),
    positions_opened: Math.floor(Math.random() * 10),
    positions_closed: Math.floor(Math.random() * 8),
  });

  return {
    counters: {
      last_1m: generateCounters(),
      last_5m: generateCounters(),
      session: generateCounters(),
    },
  };
}

/**
 * Generate mock equity history data.
 * For testing only - real app must use API.
 */
export function generateMockEquityHistory(points: number = 20) {
  const history = [];
  const initialCapital = 1000;
  let currentEquity = initialCapital;
  const now = Date.now();

  for (let i = 0; i < points; i++) {
    const timestamp = now - (points - i) * 60000; // 1 minute intervals
    currentEquity += (Math.random() - 0.5) * 50; // Random walk
    const pnl = currentEquity - initialCapital;

    history.push({
      timestamp,
      equity: currentEquity,
      pnl,
    });
  }

  return history;
}

/**
 * Generate mock live events.
 * For testing only - real app must use API.
 */
export function generateMockEvents(count: number = 20) {
  const types = [
    "SIGNAL",
    "ORDER_FILLED",
    "POSITION_OPENED",
    "POSITION_CLOSED",
    "RISK_ALERT",
    "ERROR",
  ];
  const events = [];

  for (let i = 0; i < count; i++) {
    const type = types[Math.floor(Math.random() * types.length)];
    events.push({
      id: `event-${i}`,
      type,
      timestamp: Date.now() - i * 5000,
      summary: `Mock ${type.toLowerCase()} event ${i + 1}`,
    });
  }

  return events;
}

/**
 * WARNING: This is for testing ONLY.
 * Creates a mock data set for all Cockpit components.
 */
export function generateMockCockpitData() {
  return {
    engineStatus: generateMockEngineStatus(),
    positions: generateMockPositions(5),
    riskStatus: generateMockRiskStatus(),
    metrics: generateMockMetrics(),
    equityHistory: generateMockEquityHistory(30),
    liveEvents: generateMockEvents(50),
  };
}

/**
 * Simulates live data updates by modifying values slightly.
 * For testing animations and real-time updates.
 */
export function simulateLiveUpdate(data: ReturnType<typeof generateMockCockpitData>) {
  // Update equity slightly
  data.engineStatus.equity += (Math.random() - 0.5) * 10;
  data.engineStatus.daily_pnl = data.engineStatus.equity - data.engineStatus.initial_capital;

  // Add new equity point
  data.equityHistory.push({
    timestamp: Date.now(),
    equity: data.engineStatus.equity,
    pnl: data.engineStatus.daily_pnl,
  });

  // Keep only last 100 points
  if (data.equityHistory.length > 100) {
    data.equityHistory.shift();
  }

  // Update position PnLs
  data.positions.forEach((pos) => {
    pos.current_price += (Math.random() - 0.5) * 100;
    pos.unrealized_pnl =
      pos.side === "LONG"
        ? (pos.current_price - pos.entry_price) * pos.size
        : (pos.entry_price - pos.current_price) * pos.size;
  });

  // Maybe add a new event
  if (Math.random() > 0.7) {
    const types = ["SIGNAL", "ORDER_FILLED", "POSITION_UPDATE"];
    const type = types[Math.floor(Math.random() * types.length)];
    data.liveEvents.unshift({
      id: `event-${Date.now()}`,
      type,
      timestamp: Date.now(),
      summary: `Live ${type.toLowerCase()} event`,
    });

    // Keep only last 100 events
    if (data.liveEvents.length > 100) {
      data.liveEvents.pop();
    }
  }

  return data;
}

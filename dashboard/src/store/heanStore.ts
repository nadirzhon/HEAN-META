"use client";

import { create } from "zustand";

/* ── Type Definitions ── */

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
  leverage: number;
  strategy_id?: string;
}

export interface Order {
  order_id: string;
  symbol: string;
  side: "Buy" | "Sell";
  qty: number;
  price: number;
  status: string;
  order_type: string;
  created_at?: string;
}

export interface StrategyState {
  strategy_id: string;
  type: string;
  enabled: boolean;
  state?: string;
  pnl?: number;
  win_rate?: number;
  signals_count?: number;
  positions_count?: number;
}

export interface RiskState {
  risk_state: string;
  level: number;
  reason_codes: string[];
  quarantined_symbols: string[];
  can_clear: boolean;
}

export interface KillSwitchState {
  triggered: boolean;
  reasons: string[];
  thresholds: Record<string, number>;
  current_metrics: Record<string, number>;
}

export interface PhysicsState {
  symbol: string;
  temperature: number;
  entropy: number;
  phase: string;
  szilard_profit: number;
}

export interface BrainAnalysis {
  timestamp: string;
  summary: string;
  market_sentiment: string;
  recommendations: string[];
}

export interface CouncilStatus {
  active: boolean;
  last_decision?: string;
  consensus?: string;
}

export interface TradingMetrics {
  last_1m: { signals_detected: number; orders_created: number; orders_filled: number };
  last_5m: { signals_detected: number; orders_created: number; orders_filled: number };
  session: { signals_detected: number; orders_created: number; orders_filled: number };
}

export interface SystemEvent {
  id: string;
  timestamp: string;
  type: string;
  summary: string;
  data?: Record<string, unknown>;
}

export interface AgentThought {
  id: string;
  agent: "brain" | "council" | "risk" | "meta";
  timestamp: string;
  content: string;
}

export type ConnectionStatus = "connected" | "disconnected" | "reconnecting";

export interface EquityPoint {
  timestamp: number;
  equity: number;
  pnl: number;
}

/* ── Store Interface ── */

interface HeanStore {
  // Status slice
  engineStatus: EngineStatus | null;
  riskState: RiskState | null;
  killSwitch: KillSwitchState | null;
  tradingMetrics: TradingMetrics | null;

  // Positions & Orders
  positions: Position[];
  orders: Order[];

  // Strategies
  strategies: StrategyState[];

  // Intelligence
  physics: Record<string, PhysicsState>;
  brainAnalysis: BrainAnalysis | null;
  councilStatus: CouncilStatus | null;

  // Events & Agent thoughts
  systemEvents: SystemEvent[];
  agentThoughts: AgentThought[];

  // Equity history
  equityHistory: EquityPoint[];

  // Connection meta
  connectionStatus: ConnectionStatus;
  lastHeartbeat: number | null;

  // Actions
  setEngineStatus: (status: EngineStatus) => void;
  setRiskState: (state: RiskState) => void;
  setKillSwitch: (state: KillSwitchState) => void;
  setTradingMetrics: (metrics: TradingMetrics) => void;
  setPositions: (positions: Position[]) => void;
  setOrders: (orders: Order[]) => void;
  setStrategies: (strategies: StrategyState[]) => void;
  setPhysics: (symbol: string, state: PhysicsState) => void;
  setBrainAnalysis: (analysis: BrainAnalysis) => void;
  setCouncilStatus: (status: CouncilStatus) => void;
  addSystemEvent: (event: SystemEvent) => void;
  addAgentThought: (thought: AgentThought) => void;
  addEquityPoint: (point: EquityPoint) => void;
  setConnectionStatus: (status: ConnectionStatus) => void;
  setLastHeartbeat: (ts: number) => void;
}

const MAX_EVENTS = 500;
const MAX_THOUGHTS = 50;
const MAX_EQUITY_POINTS = 1000;

export const useHeanStore = create<HeanStore>((set) => ({
  // Initial state
  engineStatus: null,
  riskState: null,
  killSwitch: null,
  tradingMetrics: null,
  positions: [],
  orders: [],
  strategies: [],
  physics: {},
  brainAnalysis: null,
  councilStatus: null,
  systemEvents: [],
  agentThoughts: [],
  equityHistory: [],
  connectionStatus: "disconnected",
  lastHeartbeat: null,

  // Actions
  setEngineStatus: (status) => set({ engineStatus: status }),
  setRiskState: (state) => set({ riskState: state }),
  setKillSwitch: (state) => set({ killSwitch: state }),
  setTradingMetrics: (metrics) => set({ tradingMetrics: metrics }),
  setPositions: (positions) => set({ positions }),
  setOrders: (orders) => set({ orders }),
  setStrategies: (strategies) => set({ strategies }),

  setPhysics: (symbol, state) =>
    set((prev) => ({ physics: { ...prev.physics, [symbol]: state } })),

  setBrainAnalysis: (analysis) => set({ brainAnalysis: analysis }),
  setCouncilStatus: (status) => set({ councilStatus: status }),

  addSystemEvent: (event) =>
    set((prev) => ({
      systemEvents: [event, ...prev.systemEvents].slice(0, MAX_EVENTS),
    })),

  addAgentThought: (thought) =>
    set((prev) => ({
      agentThoughts: [thought, ...prev.agentThoughts].slice(0, MAX_THOUGHTS),
    })),

  addEquityPoint: (point) =>
    set((prev) => ({
      equityHistory: [...prev.equityHistory, point].slice(-MAX_EQUITY_POINTS),
    })),

  setConnectionStatus: (status) => set({ connectionStatus: status }),
  setLastHeartbeat: (ts) => set({ lastHeartbeat: ts }),
}));

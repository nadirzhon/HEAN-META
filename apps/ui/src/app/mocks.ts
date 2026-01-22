import { Position } from "@/app/components/trading/PositionsTable";
import { Decision } from "@/app/components/trading/DecisionsTimeline";
import { FlowStep } from "@/app/components/trading/DecisionFlow";
import { RiskMetric } from "@/app/components/trading/RiskMonitor";
import { Strategy } from "@/app/components/trading/StrategyCard";
import { TradeResult } from "@/app/components/trading/WhyResultPanel";
import { OrderRow, EventFeedItem } from "@/app/types/trading";
import { WalletState } from "@/app/components/trading/WalletSummary";

export const mockPositions: Position[] = [
  {
    id: "1",
    symbol: "BTC/USD",
    side: "LONG",
    entryPrice: 43250.5,
    lastPrice: 43890.25,
    unrealizedPnL: 639.75,
    takeProfit: 44500.0,
    stopLoss: 42800.0,
    ttl: 3420,
    status: "ACTIVE",
  },
  {
    id: "2",
    symbol: "ETH/USD",
    side: "SHORT",
    entryPrice: 2340.8,
    lastPrice: 2310.15,
    unrealizedPnL: 30.65,
    takeProfit: 2250.0,
    stopLoss: 2380.0,
    ttl: 1890,
    status: "ACTIVE",
  },
  {
    id: "3",
    symbol: "SOL/USD",
    side: "LONG",
    entryPrice: 98.45,
    lastPrice: 96.8,
    unrealizedPnL: -1.65,
    takeProfit: 102.0,
    stopLoss: 95.0,
    ttl: 2640,
    status: "ACTIVE",
  },
];

export const mockDecisions: Decision[] = [
  {
    id: "1",
    type: "ORDER_DECISION",
    symbol: "BTC/USD",
    reasonCode: "MOMENTUM_BREAKOUT",
    outcome: "ENTRY",
    timestamp: new Date(Date.now() - 120000),
  },
  {
    id: "2",
    type: "EXIT_DECISION",
    symbol: "AVAX/USD",
    reasonCode: "TP_REACHED",
    outcome: "TP_HIT",
    timestamp: new Date(Date.now() - 300000),
  },
  {
    id: "3",
    type: "ORDER_DECISION",
    symbol: "ETH/USD",
    reasonCode: "RSI_OVERSOLD",
    outcome: "ENTRY",
    timestamp: new Date(Date.now() - 480000),
  },
  {
    id: "4",
    type: "EXIT_DECISION",
    symbol: "MATIC/USD",
    reasonCode: "TTL_EXPIRED",
    outcome: "TIMEOUT_TTL",
    timestamp: new Date(Date.now() - 720000),
  },
  {
    id: "5",
    type: "ORDER_DECISION",
    symbol: "LINK/USD",
    reasonCode: "VOLATILITY_TOO_HIGH",
    outcome: "SUPPRESSED",
    timestamp: new Date(Date.now() - 900000),
  },
];

export const mockDecisionFlows: Array<{ symbol: string; steps: FlowStep[]; timestamp: Date }> = [
  {
    symbol: "BTC/USD",
    timestamp: new Date(Date.now() - 120000),
    steps: [
      { stage: "TICK", status: "passed", reasonCode: "Price data received" },
      { stage: "SIGNAL", status: "passed", reasonCode: "MOMENTUM_BREAKOUT detected" },
      { stage: "FILTER", status: "passed", reasonCode: "Trend alignment confirmed" },
      { stage: "RISK", status: "passed", reasonCode: "Position size approved" },
      { stage: "ORDER", status: "passed", reasonCode: "Order placed successfully" },
      { stage: "EXIT", status: "passed", reasonCode: "Exit monitors active" },
    ],
  },
  {
    symbol: "LINK/USD",
    timestamp: new Date(Date.now() - 900000),
    steps: [
      { stage: "TICK", status: "passed", reasonCode: "Price data received" },
      { stage: "SIGNAL", status: "passed", reasonCode: "RSI_OVERSOLD detected" },
      { stage: "FILTER", status: "blocked", reasonCode: "Volatility too high", details: "ATR > 2.5%" },
      { stage: "RISK", status: "suppressed" },
      { stage: "ORDER", status: "suppressed" },
      { stage: "EXIT", status: "suppressed" },
    ],
  },
  {
    symbol: "ETH/USD",
    timestamp: new Date(Date.now() - 480000),
    steps: [
      { stage: "TICK", status: "passed", reasonCode: "Price data received" },
      { stage: "SIGNAL", status: "passed", reasonCode: "Mean reversion signal" },
      { stage: "FILTER", status: "passed", reasonCode: "Volume confirmed" },
      { stage: "RISK", status: "suppressed", reasonCode: "Max positions reached", details: "3/3 slots used" },
      { stage: "ORDER", status: "suppressed" },
      { stage: "EXIT", status: "suppressed" },
    ],
  },
];

export const mockRiskMetrics: RiskMetric[] = [
  {
    label: "EventBus Pressure (P0)",
    value: 142,
    max: 1000,
    unit: "msg/s",
    status: "ok",
    isWired: true,
  },
  {
    label: "EventBus Pressure (P1)",
    value: 387,
    max: 500,
    unit: "msg/s",
    status: "warning",
    isWired: true,
  },
  {
    label: "Exit Latency",
    value: 23,
    max: 100,
    unit: "ms",
    status: "ok",
    isWired: true,
  },
  {
    label: "Dropped Ticks Rate",
    value: 0.3,
    max: 5,
    unit: "%",
    status: "ok",
    isWired: true,
  },
  {
    label: "Decision Memory Blocks",
    value: 1247,
    max: 10000,
    unit: "blocks",
    status: "ok",
    isWired: true,
  },
  {
    label: "WebSocket Lag",
    value: 0,
    max: 500,
    unit: "ms",
    status: "ok",
    isWired: false,
  },
];

export const mockStrategies: Strategy[] = [
  {
    id: "1",
    name: "Momentum Breakout",
    enabled: true,
    symbols: ["BTC/USD", "ETH/USD", "SOL/USD"],
    aggression: "HIGH",
    signalsPerMinute: 12,
    lastDecision: {
      outcome: "ENTRY",
      timestamp: new Date(Date.now() - 120000),
    },
    isWired: true,
  },
  {
    id: "2",
    name: "Mean Reversion",
    enabled: true,
    symbols: ["BTC/USD", "ETH/USD", "AVAX/USD", "MATIC/USD"],
    aggression: "MEDIUM",
    signalsPerMinute: 8,
    lastDecision: {
      outcome: "SUPPRESSED",
      timestamp: new Date(Date.now() - 480000),
    },
    isWired: true,
  },
  {
    id: "3",
    name: "Arbitrage Scanner",
    enabled: false,
    symbols: ["BTC/USD", "ETH/USD"],
    aggression: "LOW",
    signalsPerMinute: 3,
    isWired: false,
  },
  {
    id: "4",
    name: "Volatility Capture",
    enabled: true,
    symbols: ["SOL/USD", "AVAX/USD", "LINK/USD", "MATIC/USD", "DOT/USD"],
    aggression: "HIGH",
    signalsPerMinute: 15,
    lastDecision: {
      outcome: "HOLD",
      timestamp: new Date(Date.now() - 60000),
    },
    isWired: true,
  },
];

export const mockTradeResults: TradeResult[] = [
  {
    symbol: "AVAX/USD",
    side: "LONG",
    result: 12.45,
    entryPrice: 38.2,
    exitPrice: 39.65,
    duration: "23m 14s",
    timestamp: new Date(Date.now() - 300000),
    factors: [
      {
        stage: "Entry Signal",
        passed: true,
        explanation: "Entry aligned with impulse regime — strong momentum detected",
        impact: "positive",
      },
      {
        stage: "Position Sizing",
        passed: true,
        explanation: "Risk allowed full position size — portfolio exposure within limits",
        impact: "positive",
      },
      {
        stage: "Volatility Filter",
        passed: false,
        explanation: "Volatility filter partially suppressed exposure — ATR above threshold, position reduced by 40%",
        impact: "negative",
      },
      {
        stage: "Exit Execution",
        passed: true,
        explanation: "Exit triggered by TP_HIT — target price reached, clean execution",
        impact: "positive",
      },
    ],
  },
  {
    symbol: "SOL/USD",
    side: "SHORT",
    result: -3.2,
    entryPrice: 96.8,
    exitPrice: 97.12,
    duration: "14m 08s",
    timestamp: new Date(Date.now() - 1200000),
    factors: [
      {
        stage: "Entry Signal",
        passed: true,
        explanation: "Mean reversion signal triggered — RSI oversold condition met",
        impact: "positive",
      },
      {
        stage: "Trend Alignment",
        passed: false,
        explanation: "Trend filter showed conflicting signals — entered counter-trend position",
        impact: "negative",
      },
      {
        stage: "Position Sizing",
        passed: true,
        explanation: "Position size reduced to 50% due to trend uncertainty",
        impact: "neutral",
      },
      {
        stage: "Exit Execution",
        passed: false,
        explanation: "Stop loss hit — price moved against position before target reached",
        impact: "negative",
      },
    ],
  },
  {
    symbol: "BTC/USD",
    side: "LONG",
    result: 84.3,
    entryPrice: 43150.0,
    exitPrice: 43890.25,
    duration: "1h 34m",
    timestamp: new Date(Date.now() - 120000),
    factors: [
      {
        stage: "Entry Signal",
        passed: true,
        explanation: "Momentum breakout signal confirmed — volume surge detected",
        impact: "positive",
      },
      {
        stage: "Trend Alignment",
        passed: true,
        explanation: "Multiple timeframe trend alignment — all indicators bullish",
        impact: "positive",
      },
      {
        stage: "Risk Management",
        passed: true,
        explanation: "Optimal risk/reward ratio (1:3.2) — stop loss properly positioned",
        impact: "positive",
      },
      {
        stage: "Exit Execution",
        passed: true,
        explanation: "Partial TP hit, trailing stop secured remainder — maximized profit",
        impact: "positive",
      },
    ],
  },
];

export const mockOrders: OrderRow[] = [
  {
    id: "ord-1",
    symbol: "BTC/USD",
    side: "BUY",
    price: 43820.5,
    size: 0.25,
    filled: 0.1,
    status: "OPEN",
    strategyId: "momentum_breakout",
    createdAt: new Date(Date.now() - 90_000),
    type: "LIMIT",
  },
  {
    id: "ord-2",
    symbol: "ETH/USD",
    side: "SELL",
    price: 2315.1,
    size: 1.5,
    filled: 1.0,
    status: "PARTIAL",
    strategyId: "mean_rev",
    createdAt: new Date(Date.now() - 180_000),
    type: "LIMIT",
  },
  {
    id: "ord-3",
    symbol: "SOL/USD",
    side: "BUY",
    price: 97.8,
    size: 4.0,
    filled: 0,
    status: "OPEN",
    strategyId: "vol_capture",
    createdAt: new Date(Date.now() - 300_000),
    type: "MARKET",
  },
];

export const mockWallet: WalletState = {
  wallet_balance: 125000,
  available_balance: 88000,
  equity: 134500,
  used_margin: 31000,
  reserved_margin: 6000,
  unrealized_pnl: 8500,
  realized_pnl: 4200,
};

export const mockEvents: EventFeedItem[] = [
  {
    id: "evt-1",
    ts: Date.now() - 1500,
    type: "HEARTBEAT",
    severity: "INFO",
    source: "mock",
    payload: { engine_state: "RUNNING", events_per_sec: 4.2 },
    context: { topic: "system_heartbeat" },
    topic: "system_heartbeat",
    category: "system",
    message: "Heartbeat — engine RUNNING, eps 4.2",
  },
  {
    id: "evt-2",
    ts: Date.now() - 18_000,
    type: "ORDER_PLACED",
    severity: "INFO",
    source: "mock",
    payload: { symbol: "BTC/USD", side: "BUY", price: 43820, status: "OPEN" },
    context: { topic: "orders" },
    topic: "orders",
    category: "orders",
    message: "BTC/USD BUY OPEN",
  },
  {
    id: "evt-3",
    ts: Date.now() - 42_000,
    type: "RISK_EVENT",
    severity: "WARN",
    source: "mock",
    payload: { message: "Drawdown at 8%", drawdown_pct: 8.1 },
    context: { topic: "risk_events" },
    topic: "risk_events",
    category: "risk",
    message: "Drawdown at 8%",
  },
];

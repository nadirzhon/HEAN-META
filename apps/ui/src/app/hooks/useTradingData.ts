import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  AccountState,
  ApiError,
  ControlAction,
  ControlResponse,
  EventEnvelope,
  PortfolioSummary,
  RealtimeClient,
  TelemetrySummary,
  MarketSnapshot,
  WsStatus,
  getApiBase,
  getWsUrl,
  fetchDashboard,
  fetchEngineStatus,
  fetchOrders,
  fetchPortfolioSummary,
  fetchPositions,
  fetchRiskStatus,
  fetchStrategies,
  fetchTelemetrySummary,
  fetchMarketSnapshot,
  sendControlAction,
  fetchTradingMetrics,
} from "@/app/api/client";
import {
  mockDecisionFlows,
  mockEvents,
  mockOrders,
  mockPositions,
  mockRiskMetrics,
  mockStrategies,
  mockTradeResults,
  mockWallet,
} from "@/app/mocks";
import { Position } from "@/app/components/trading/PositionsTable";
import { Decision } from "@/app/components/trading/DecisionsTimeline";
import { Strategy } from "@/app/components/trading/StrategyCard";
import { RiskMetric } from "@/app/components/trading/RiskMonitor";
import { TradeResult } from "@/app/components/trading/WhyResultPanel";
import { FlowStep } from "@/app/components/trading/DecisionFlow";
import { WalletState } from "@/app/components/trading/WalletSummary";
import { CandleData } from "@/app/utils/chartData";
import { ChartMarker, EventCategory, EventFeedItem, OrderRow } from "@/app/types/trading";

export type SystemMode = "CALM" | "ACTIVE" | "DEFENSIVE" | "EMERGENCY" | "UNKNOWN";
export type HealthStatus = "OK" | "WARNING" | "CRITICAL";
type RestHealth = "ok" | "degraded" | "error";

export interface TelemetryState {
  engine_state: string;
  events_per_sec: number;
  last_event_ts?: number;
  ws_clients?: number;
  uptime?: number;
}

export interface WsMeta {
  status: WsStatus;
  lastMessageAt?: number;
  lastHeartbeatAt?: number;
  reconnectAttempts: number;
  lastError?: string | null;
  url: string;
}

export interface ControlStatus {
  action?: ControlAction;
  state: "idle" | "pending" | "waiting" | "ok" | "error";
  message?: string;
}
const COLORS = {
  cyan: "#06b6d4",
  amber: "#f59e0b",
  red: "#ef4444",
  green: "#10b981",
  purple: "#a855f7",
};

export interface ChartState {
  symbol: string;
  candles: CandleData[];
  priceLine: Array<{ time: number; value: number }>;
  markers: ChartMarker[];
  hasMarketData: boolean;
  positionLabel?: string;
}

export interface PulseState {
  engineState: string;
  mode: "paper" | "live" | "unknown";
  wsStatus: WsStatus;
  eventsPerSec: number;
  lastEventTs?: number;
  lastHeartbeatTs?: number;
  lastMessageAt?: number;
  wsClients?: number;
  uptime?: number;
  mockMode: boolean;
  restHealth: RestHealth;
}

export interface TradingFunnelMetrics {
  signals_total_1m: number;
  signals_total_session: number;
  decisions_create: number;
  decisions_skip: number;
  decisions_block: number;
  orders_created: number;
  orders_filled: number;
  orders_canceled: number;
  orders_rejected: number;
  orders_open: number;
  positions_open: number;
  positions_closed: number;
  pnl_unrealized: number;
  pnl_realized: number;
  equity: number;
  used_margin: number;
  free_margin: number;
  top_reasons: Array<{ code: string; count: number; pct: number }>;
  last_signal_ts?: string;
  last_order_ts?: string;
  last_fill_ts?: string;
  engine_state: string;
  mode: string;
}

export interface TradingDataState {
  positions: Position[];
  orders: OrderRow[];
  decisions: Decision[];
  decisionFlows: Array<{ symbol: string; steps: FlowStep[]; timestamp: Date }>;
  strategies: Strategy[];
  account: WalletState;
  riskMetrics: RiskMetric[];
  tradeResults: TradeResult[];
  metrics: {
    equity: number;
    dailyPnl: number;
    openPositions: number;
    openOrders: number;
    returnPct?: number;
    initialCapital?: number;
  };
  funnelMetrics: TradingFunnelMetrics | null;
  systemMode: SystemMode;
  health: HealthStatus;
  pulse: PulseState;
  telemetry: TelemetryState;
  portfolio: PortfolioSummary | null;
  ws: WsMeta;
  backendAvailable: boolean;
  lastBackendError: string | null;
  controlStatus: ControlStatus;
  apiBase: string;
  wsUrl: string;
  lastSync: Date | null;
  usingLiveData: boolean;
  loading: boolean;
  error: string | null;
  eventFeed: EventFeedItem[];
  chart: ChartState;
  controlSupport: Partial<Record<ControlAction, boolean>>;
  refresh: () => Promise<void>;
  runControl: (action: ControlAction) => Promise<ControlResponse>;
}

const RING_BUFFER = 200;
const DEFAULT_WALLET: WalletState = {
  wallet_balance: 0,
  available_balance: 0,
  equity: 0,
  used_margin: 0,
  reserved_margin: 0,
  unrealized_pnl: 0,
  realized_pnl: 0,
};

const OUTCOME_MAP: Record<string, Decision["outcome"]> = {
  CREATE: "ENTRY",
  ACCEPTED: "ENTRY",
  ENTRY: "ENTRY",
  ENTER: "ENTRY",
  REJECT: "SUPPRESSED",
  SKIP: "SUPPRESSED",
  SUPPRESSED: "SUPPRESSED",
  HOLD: "HOLD",
  CLOSE: "CLOSE",
  FORCE_CLOSE: "CLOSE",
  TP_HIT: "TP_HIT",
  SL_HIT: "SL_HIT",
  TIMEOUT_TTL: "TIMEOUT_TTL",
};

const mapRiskMetricStatus = (value: number, max: number) => {
  const pct = max === 0 ? 0 : (value / max) * 100;
  if (pct >= 90) return "critical";
  if (pct >= 70) return "warning";
  return "ok";
};

const toUnixMs = (value?: string | number | Date) => {
  if (!value) return Date.now();
  if (typeof value === "number") return value;
  if (value instanceof Date) return value.getTime();
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? Date.now() : parsed;
};

const mapKlineToCandle = (k: any): CandleData => {
  const timestamp = k?.open_time_ms ?? k?.close_time_ms ?? toUnixMs(k?.open_time ?? k?.close_time);
  return {
    timestamp,
    open: Number(k?.open ?? k?.o ?? 0),
    high: Number(k?.high ?? k?.h ?? k?.open ?? 0),
    low: Number(k?.low ?? k?.l ?? k?.open ?? 0),
    close: Number(k?.close ?? k?.c ?? k?.open ?? 0),
    volume: Number(k?.volume ?? k?.v ?? 0),
  };
};

const deriveCategory = (event: EventEnvelope): EventCategory => {
  const topic = (event.topic ?? event.context?.topic ?? "").toLowerCase();
  const type = (event.type ?? "").toLowerCase();
  if (type.includes("control")) return "system";
  if (topic.includes("system") || type.includes("heartbeat")) return "system";
  if (topic.includes("order") || type.includes("order")) return "orders";
  if (topic.includes("position") || type.includes("position")) return "positions";
  if (topic.includes("risk") || type.includes("risk")) return "risk";
  if (topic.includes("strategy") || type.includes("strategy")) return "strategies";
  return "other";
};

const summarizeEvent = (event: EventEnvelope, category: EventCategory): string => {
  const p = event.payload || {};
  const type = (event.type ?? "").toUpperCase();
  if (type.includes("CONTROL")) {
    const action = (p.action ?? p.command ?? event.type)?.toString().toUpperCase();
    const outcome = p.status ?? p.result ?? p.message ?? "pending";
    return `${action} — ${outcome}`;
  }
  if (category === "system") {
    return `Heartbeat — engine ${p.engine_state ?? p.engine ?? "?"}, eps ${p.events_per_sec ?? "?"}`;
  }
  if (category === "orders") {
    const status = p.status ?? p.decision ?? event.type;
    return `${p.symbol ?? p.ticker ?? "?"} ${p.side ?? ""} ${status}`.trim();
  }
  if (category === "positions") {
    const action = p.action ?? p.reason ?? event.type;
    return `${p.symbol ?? "?"} ${action}`;
  }
  if (category === "risk") {
    return p.message ?? p.reason ?? event.type;
  }
  return event.type;
};

const mapAccountState = (a?: AccountState | PortfolioSummary | null): WalletState => {
  if (!a) return DEFAULT_WALLET;
  const walletBalance =
    "wallet_balance" in a && typeof a.wallet_balance === "number"
      ? a.wallet_balance
      : (a as PortfolioSummary).balance ?? 0;
  const available =
    "available_balance" in a && typeof a.available_balance === "number"
      ? a.available_balance
      : (a as PortfolioSummary).free_margin ?? 0;
  return {
    wallet_balance: walletBalance ?? 0,
    available_balance: available ?? 0,
    equity: a.equity ?? walletBalance ?? 0,
    used_margin: a.used_margin ?? 0,
    reserved_margin: a.reserved_margin ?? 0,
    unrealized_pnl: a.unrealized_pnl ?? 0,
    realized_pnl: a.realized_pnl ?? 0,
    timestamp: (a as PortfolioSummary).timestamp,
  };
};

const mapPosition = (p: any): Position => {
  const lastPrice = p.current_price ?? p.last_price ?? p.entry_price ?? 0;
  const side = p.side?.toUpperCase() === "SHORT" ? "SHORT" : "LONG";
  const status = p.status?.toUpperCase() === "CLOSED" ? "PENDING" : "ACTIVE";
  return {
    id: p.position_id ?? `${p.symbol}-${side}`,
    symbol: p.symbol,
    side,
    entryPrice: p.entry_price ?? 0,
    lastPrice,
    unrealizedPnL: p.unrealized_pnl ?? 0,
    takeProfit: p.take_profit,
    stopLoss: p.stop_loss,
    status,
  };
};

const mapStrategy = (s: any): Strategy => ({
  id: s.strategy_id ?? s.id ?? s.name,
  name: s.name ?? s.strategy_id ?? "strategy",
  enabled: Boolean(s.enabled ?? true),
  symbols: s.symbols ?? [],
  aggression: (s.aggression ?? "MEDIUM").toUpperCase(),
  signalsPerMinute: s.signals_per_minute ?? s.frequency ?? 0,
  isWired: true,
});

const mapOrder = (o: any): OrderRow => ({
  id: o.order_id ?? o.id ?? crypto.randomUUID(),
  symbol: o.symbol ?? o.ticker ?? "—",
  side: (o.side ?? o.direction ?? "BUY").toUpperCase() === "SELL" ? "SELL" : "BUY",
  price: o.price ?? o.limit_price ?? undefined,
  size: o.size ?? o.qty ?? o.quantity,
  filled: o.filled_size ?? o.filled ?? o.executed_qty,
  status: (o.status ?? o.state ?? "OPEN").toString().toUpperCase(),
  strategyId: o.strategy_id,
  createdAt: o.timestamp ? new Date(o.timestamp) : undefined,
  type: o.type,
  takeProfit: o.take_profit,
  stopLoss: o.stop_loss,
});

const mapDecision = (d: any): Decision => {
  const ts = d?.timestamp ? new Date(d.timestamp) : new Date();
  const rawOutcome = d?.decision ?? d?.reason_code ?? "HOLD";
  const normalizedOutcome = OUTCOME_MAP[rawOutcome] ?? "HOLD";

  return {
    id:
      d?.signal_id ??
      d?.position_id ??
      d?.order_id ??
      `${d?.symbol ?? "UNKNOWN"}-${ts.getTime()}`,
    type: d?.type === "ORDER_EXIT_DECISION" ? "EXIT_DECISION" : "ORDER_DECISION",
    symbol: d?.symbol ?? d?.position_id ?? "UNKNOWN",
    reasonCode: d?.reason_code ?? d?.decision ?? "UNKNOWN",
    outcome: normalizedOutcome,
    timestamp: ts,
  };
};

const mapRisk = (risk: any, metricsEquity: number): RiskMetric[] => {
  if (!risk) return mockRiskMetrics;

  const drawdownPct = Math.abs(risk.drawdown_pct ?? 0);
  const openPositions = risk.current_positions ?? 0;
  const maxPositions = risk.max_open_positions ?? Math.max(openPositions, 1);
  const openOrders = risk.current_orders ?? 0;
  const maxOrders = risk.max_open_orders ?? Math.max(openOrders, 1);

  return [
    {
      label: "Drawdown %",
      value: Number(drawdownPct.toFixed(2)),
      max: 100,
      unit: "%",
      status: drawdownPct >= 20 ? "critical" : drawdownPct >= 10 ? "warning" : "ok",
      isWired: true,
    },
    {
      label: "Open Positions",
      value: openPositions,
      max: maxPositions,
      unit: "open",
      status: mapRiskMetricStatus(openPositions, maxPositions),
      isWired: true,
    },
    {
      label: "Open Orders",
      value: openOrders,
      max: maxOrders,
      unit: "open",
      status: mapRiskMetricStatus(openOrders, maxOrders),
      isWired: true,
    },
    {
      label: "Equity",
      value: Number((risk.equity ?? metricsEquity).toFixed(2)),
      max: Math.max(risk.equity ?? metricsEquity, 1),
      unit: "USD",
      status: "ok",
      isWired: true,
    },
  ];
};

const deriveMode = (engine: any): "paper" | "live" | "unknown" => {
  if (engine?.is_live) return "live";
  if (engine?.trading_mode) {
    return engine.trading_mode.toLowerCase() === "live" ? "live" : "paper";
  }
  return "paper";
};

export function useTradingData(): TradingDataState {
  const [positions, setPositions] = useState<Position[]>(mockPositions);
  const [orders, setOrders] = useState<OrderRow[]>(mockOrders);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>(mockStrategies);
  const [riskMetrics, setRiskMetrics] = useState<RiskMetric[]>(mockRiskMetrics);
  const [tradeResults] = useState<TradeResult[]>(mockTradeResults);
  const [account, setAccount] = useState<WalletState>(mockWallet ?? DEFAULT_WALLET);
  const [metrics, setMetrics] = useState({
    equity: mockWallet?.equity ?? 0,
    dailyPnl: 0,
    openPositions: mockPositions.length,
    openOrders: mockOrders.length,
    returnPct: 0,
    initialCapital: undefined,
  });
  const [systemMode, setSystemMode] = useState<SystemMode>("ACTIVE");
  const [health, setHealth] = useState<HealthStatus>("OK");
  const [lastSync, setLastSync] = useState<Date | null>(null);
  const [usingLiveData, setUsingLiveData] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [eventFeed, setEventFeed] = useState<EventFeedItem[]>(mockEvents);
  const [telemetry, setTelemetry] = useState<TelemetryState>({
    engine_state: "UNKNOWN",
    events_per_sec: 0,
    last_event_ts: undefined,
    ws_clients: undefined,
    uptime: undefined,
  });
  const [portfolioSummary, setPortfolioSummary] = useState<PortfolioSummary | null>(null);
  const [backendAvailable, setBackendAvailable] = useState<boolean>(true);
  const [lastBackendError, setLastBackendError] = useState<string | null>(null);
  const [wsMeta, setWsMeta] = useState<WsMeta>({
    status: "connecting",
    reconnectAttempts: 0,
    url: getWsUrl(),
    lastMessageAt: undefined,
    lastHeartbeatAt: undefined,
  });
  const [controlStatus, setControlStatus] = useState<ControlStatus>({ state: "idle" });
  const [pulse, setPulse] = useState<PulseState>({
    engineState: "UNKNOWN",
    mode: "paper",
    wsStatus: "connecting",
    eventsPerSec: 0,
    mockMode: false,
    restHealth: "ok",
  });
  const [chart, setChart] = useState<ChartState>({
    symbol: "BTCUSDT",
    candles: [],
    priceLine: [],
    markers: [],
    hasMarketData: false,
  });
  const [controlSupport, setControlSupport] = useState<Partial<Record<ControlAction, boolean>>>({
    pause: true,
    resume: true,
    restart: true,
    kill: true,
    start: true,
    stop: true,
  });
  const [funnelMetrics, setFunnelMetrics] = useState<TradingFunnelMetrics | null>(null);

  const apiBase = useMemo(() => getApiBase(), []);
  const wsUrl = useMemo(() => getWsUrl(), []);
  const wsRef = useRef<RealtimeClient | null>(null);
  const lastSeqRef = useRef<number>(0);
  const pendingSnapshot = useRef<boolean>(false);
  const candleBufferRef = useRef<CandleData[]>([]);
  const decisionMapRef = useRef<Record<string, string>>({});

  const appendEvent = useCallback((env: EventEnvelope) => {
    const category = deriveCategory(env);
    const item: EventFeedItem = {
      ...env,
      id: `${env.ts}-${Math.random().toString(16).slice(2, 8)}`,
      category,
      message: summarizeEvent(env, category),
    };
    setEventFeed((prev) => {
      const next = [item, ...prev];
      return next.slice(0, RING_BUFFER);
    });
  }, []);

  const pushControlEvent = useCallback(
    (action: ControlAction, status: string, message?: string, severity: EventEnvelope["severity"] = "INFO") => {
      const now = Date.now();
      appendEvent({
        ts: now,
        type: status === "error" ? "CONTROL_ERROR" : "CONTROL_COMMAND",
        severity,
        source: "ui",
        payload: { action, status, message },
        context: { topic: "control" },
        topic: "control",
        raw: { action, status, message },
      });
    },
    [appendEvent]
  );

  const pushPrice = useCallback((price?: number, ts?: number) => {
    if (typeof price !== "number") return;
    const time = ts ?? Date.now();
    setChart((prev) => {
      const priceLine = [...prev.priceLine, { time, value: price }].slice(-500);
      return { ...prev, priceLine, hasMarketData: true };
    });
  }, []);

  const commitCandles = useCallback((candles: CandleData[]) => {
    candleBufferRef.current = candles;
    setChart((prev) => ({
      ...prev,
      candles,
      hasMarketData: candles.length > 0 || prev.hasMarketData,
    }));
  }, []);

  const upsertCandleFromTick = useCallback(
    (price?: number, ts?: number, volume: number = 0) => {
      if (typeof price !== "number") return;
      const timestamp = ts ?? Date.now();
      const bucket = Math.floor(timestamp / 60000) * 60000;
      const buffer = [...candleBufferRef.current];
      const idx = buffer.findIndex((c) => c.timestamp === bucket);
      if (idx >= 0) {
        const c = { ...buffer[idx] };
        c.high = Math.max(c.high, price);
        c.low = Math.min(c.low, price);
        c.close = price;
        c.volume = (c.volume ?? 0) + volume;
        buffer[idx] = c;
      } else {
        buffer.push({
          timestamp: bucket,
          open: price,
          high: price,
          low: price,
          close: price,
          volume: volume ?? 0,
        });
      }
      buffer.sort((a, b) => a.timestamp - b.timestamp);
      commitCandles(buffer.slice(-220));
    },
    [commitCandles]
  );

  const applyKline = useCallback(
    (kline: any) => {
      const candle = mapKlineToCandle(kline);
      if (!candle.timestamp) return;
      const buffer = candleBufferRef.current.filter((c) => c.timestamp !== candle.timestamp);
      buffer.push(candle);
      buffer.sort((a, b) => a.timestamp - b.timestamp);
      commitCandles(buffer.slice(-220));
    },
    [commitCandles]
  );

  const pushMarker = useCallback((marker: ChartMarker) => {
    setChart((prev) => {
      const markers = [...prev.markers, marker].slice(-120);
      return { ...prev, markers };
    });
  }, []);

  const handleOrderMarker = useCallback(
    (env: EventEnvelope) => {
      const price = env.payload?.price ?? env.payload?.avg_price ?? env.payload?.limit_price;
      const symbol = env.payload?.symbol ?? env.payload?.ticker;
      if (!price || !symbol) return;
      const side = (env.payload?.side ?? env.payload?.direction ?? "BUY").toUpperCase();
      const isEntry = env.type.includes("ENTRY") || env.type.includes("CREATE") || env.type.includes("ORDER");
      const correlation = env.correlation_id ?? env.payload?.correlation_id ?? env.payload?.signal_id;
      const decisionTag = correlation ? decisionMapRef.current[correlation] : undefined;
      const labelParts = [symbol, side, isEntry ? "IN" : "OUT"];
      if (correlation) labelParts.push(`#${correlation.slice(0, 8)}`);
      if (decisionTag) labelParts.push(decisionTag);
      pushMarker({
        time: env.ts,
        position: isEntry ? "belowBar" : "aboveBar",
        color: isEntry ? COLORS.cyan : COLORS.amber,
        shape: side === "BUY" ? "arrowUp" : "arrowDown",
        text: labelParts.join(" "),
      });
    },
    [pushMarker]
  );

  const mergePositions = useCallback((incoming: Position[]) => {
    setPositions((prev) => {
      const map = new Map<string, Position>();
      prev.forEach((p) => map.set(p.id, p));
      incoming.forEach((p) => map.set(p.id, { ...map.get(p.id), ...p }));
      return Array.from(map.values());
    });
  }, []);

  const mergeOrders = useCallback((incoming: OrderRow[]) => {
    setOrders((prev) => {
      const map = new Map<string, OrderRow>();
      prev.forEach((o) => map.set(o.id, o));
      incoming.forEach((o) => map.set(o.id, { ...map.get(o.id), ...o }));
      return Array.from(map.values());
    });
  }, []);

  const requestSnapshot = useCallback((reason: string = "manual") => {
    pendingSnapshot.current = true;
    wsRef.current?.requestSnapshot();
  }, []);

  const handleMarketData = useCallback(
    (event: EventEnvelope) => {
      const kind = (event.payload?.kind ?? event.payload?.type ?? "").toString().toLowerCase();
      if (kind.includes("kline") || kind.includes("candle")) {
        applyKline(event.payload);
        if (typeof event.payload?.close === "number") {
          pushPrice(event.payload.close, event.payload?.close_time_ms ?? event.payload?.ts_ms ?? event.ts);
        }
      } else {
        const price = event.payload?.price ?? event.payload?.last_price ?? event.payload?.value;
        const ts = event.payload?.ts_ms ?? event.ts;
        if (typeof price === "number") {
          pushPrice(price, ts);
          upsertCandleFromTick(price, ts, event.payload?.volume ?? 0);
        }
      }
      setChart((prev) => ({ ...prev, hasMarketData: true }));
    },
    [applyKline, pushPrice, upsertCandleFromTick]
  );

  const processEvent = useCallback(
    (event: EventEnvelope) => {
      const seq = event.seq;
      if (seq) {
        if (lastSeqRef.current && seq <= lastSeqRef.current) {
          return;
        }
        if (lastSeqRef.current && seq > lastSeqRef.current + 1) {
          requestSnapshot("gap_detected");
        }
        lastSeqRef.current = Math.max(lastSeqRef.current, seq);
      }

      const status = wsRef.current?.getStatus();
      if (status) {
        setWsMeta((prev) => ({ ...prev, ...status }));
      }

      setTelemetry((prev) => ({
        engine_state: (event.payload?.engine_state ?? prev.engine_state ?? "UNKNOWN").toString().toUpperCase(),
        events_per_sec: event.payload?.events_per_sec ?? status?.eventsPerSec ?? prev.events_per_sec ?? 0,
        last_event_ts: event.ts ?? status?.lastEventTs ?? prev.last_event_ts ?? Date.now(),
        ws_clients: event.payload?.ws_clients ?? prev.ws_clients,
        uptime: event.payload?.uptime ?? prev.uptime,
      }));

      const isHeartbeat = event.type?.includes("HEARTBEAT") || event.topic === "system_heartbeat";
      const heartbeatTs = isHeartbeat ? (event.ts ?? status?.lastHeartbeatAt ?? Date.now()) : prev.lastHeartbeatTs;
      
      // Update REAL_MODE based on WS + heartbeat
      const wsConnected = status?.status === "connected";
      const heartbeatAge = heartbeatTs ? Date.now() - heartbeatTs : Infinity;
      const heartbeatOk = heartbeatAge <= 2000;
      const realMode = wsConnected && heartbeatOk;
      
      setPulse((prev) => ({
        ...prev,
        engineState: (event.payload?.engine_state ?? prev.engineState ?? "UNKNOWN").toString().toUpperCase(),
        eventsPerSec: status?.eventsPerSec ?? event.payload?.events_per_sec ?? prev.eventsPerSec,
        lastEventTs: event.ts ?? status?.lastEventTs ?? prev.lastEventTs,
        lastHeartbeatTs: heartbeatTs,
        lastMessageAt: status?.lastMessageAt ?? prev.lastMessageAt,
        wsStatus: status?.status ?? prev.wsStatus,
        wsClients: event.payload?.ws_clients ?? prev.wsClients,
        uptime: event.payload?.uptime ?? prev.uptime,
        mockMode: !realMode,
        restHealth: prev.restHealth,
      }));
      setUsingLiveData(realMode);

      if (event.topic === "market_data" || event.topic === "market_ticks") {
        handleMarketData(event);
        return;
      }

      appendEvent(event);
      handleOrderMarker(event);

      const category = deriveCategory(event);
      if (event.type.includes("HEARTBEAT") || category === "system") {
        const heartbeatTs = event.ts ?? Date.now();
        const wsStatus = wsRef.current?.getStatus();
        const wsConnected = wsStatus?.status === "connected";
        const heartbeatAge = Date.now() - heartbeatTs;
        const heartbeatOk = heartbeatAge <= 2000;
        const realMode = wsConnected && heartbeatOk;
        
        setPulse((prev) => ({
          ...prev,
          engineState: (event.payload?.engine_state ?? prev.engineState ?? "UNKNOWN").toString().toUpperCase(),
          mode:
            (event.payload?.mode ?? prev.mode) === "live"
              ? "live"
              : (event.payload?.mode ?? prev.mode) === "paper"
              ? "paper"
              : prev.mode,
          lastHeartbeatTs: heartbeatTs,
          mockMode: !realMode,
        }));
        setUsingLiveData(realMode);
        setPortfolioSummary((prev) =>
          event.payload?.equity || event.payload?.balance
            ? { ...(prev ?? {}), equity: event.payload.equity ?? prev?.equity, balance: event.payload.balance ?? prev?.balance }
            : prev
        );
        if (event.payload?.equity) {
          setMetrics((prev) => ({ ...prev, equity: event.payload.equity }));
        }
      }

      if (category === "orders") {
        const payloadOrders = Array.isArray(event.payload?.orders)
          ? event.payload.orders.map(mapOrder)
          : event.payload?.order_id || event.payload?.symbol
          ? [mapOrder(event.payload)]
          : [];
        if (payloadOrders.length) {
          mergeOrders(payloadOrders);
          setMetrics((prev) => ({ ...prev, openOrders: payloadOrders.length }));
        }
      }

      if (category === "positions" && Array.isArray(event.payload?.positions)) {
        const mapped = event.payload.positions.map(mapPosition);
        mergePositions(mapped);
        setMetrics((prev) => ({ ...prev, openPositions: mapped.length }));
      }

      const upperType = (event.type ?? "").toUpperCase();
      if (upperType.includes("ORDER_DECISION") || upperType.includes("EXIT_DECISION")) {
        const decisionPayload = event.payload?.decision ?? event.payload;
        const decision = mapDecision(decisionPayload);
        const correlation = event.correlation_id ?? decisionPayload?.signal_id ?? decisionPayload?.position_id;
        if (correlation) decisionMapRef.current[correlation] = decision.reasonCode ?? decision.outcome;
        setDecisions((prev) => [decision, ...prev].slice(0, 50));
      }

      if (upperType.includes("CONTROL_RESULT")) {
        setControlStatus({
          action: (event.payload?.action ?? event.payload?.command ?? event.payload?.name ?? "control") as ControlAction,
          state: event.severity === "ERROR" ? "error" : "ok",
          message: event.payload?.message ?? event.payload?.status ?? event.payload?.result ?? "Completed",
        });
      }

      if (typeof event.payload?.price === "number") {
        pushPrice(event.payload.price, event.ts);
      }
      
      // Handle trading_metrics updates
      if (event.topic === "trading_metrics" || event.type === "trading_metrics_update" || event.type === "TRADING_METRICS_UPDATE") {
        // Handle both formats: {topic: "trading_metrics", data: {...}} and {type: "trading_metrics_update", ...}
        const metrics = event.data || event.payload || event;
        const counters = metrics.counters || {};
        const session = counters.session || {};
        const last1m = counters.last_1m || {};
        setFunnelMetrics({
          signals_total_1m: last1m.signals_total || 0,
          signals_total_session: session.signals_total || 0,
          decisions_create: session.decisions_create || 0,
          decisions_skip: session.decisions_skip || 0,
          decisions_block: session.decisions_block || 0,
          orders_created: session.orders_created || 0,
          orders_filled: session.orders_filled || 0,
          orders_canceled: session.orders_canceled || 0,
          orders_rejected: session.orders_rejected || 0,
          orders_open: metrics.active_orders_count || session.orders_open || 0,
          positions_open: metrics.active_positions_count || session.positions_open || 0,
          positions_closed: session.positions_closed || 0,
          pnl_unrealized: session.pnl_unrealized || 0,
          pnl_realized: session.pnl_realized || 0,
          equity: session.equity || metrics.equity || 0,
          used_margin: 0, // Will be updated from account state
          free_margin: 0, // Will be updated from account state
          top_reasons: metrics.top_reasons_for_skip_block || [],
          last_signal_ts: metrics.last_signal_ts,
          last_order_ts: metrics.last_order_ts,
          last_fill_ts: metrics.last_fill_ts,
          engine_state: metrics.engine_state || "UNKNOWN",
          mode: metrics.mode || "unknown",
        });
      }
      
      // Handle trading_events (funnel events)
      if (event.topic === "trading_events") {
        const eventType = (event.type || "").toUpperCase();
        if (eventType === "SIGNAL_DETECTED") {
          // Add marker for signal
          const price = event.payload?.entry_price;
          if (price) {
            pushMarker({
              time: event.ts,
              position: "belowBar",
              color: COLORS.purple,
              shape: "circle",
              text: `SIGNAL ${event.payload?.symbol || ""}`,
            });
          }
        } else if (eventType === "ORDER_CREATED") {
          handleOrderMarker(event);
        } else if (eventType === "ORDER_UPDATE") {
          const status = (event.payload?.status || "").toUpperCase();
          if (status === "FILLED") {
            handleOrderMarker(event);
          }
        } else if (eventType === "POSITION_UPDATE") {
          // Position markers handled in handleOrderMarker
        }
      }
    },
    [appendEvent, handleMarketData, handleOrderMarker, mergeOrders, mergePositions, pushPrice, pushMarker, requestSnapshot]
  );

  const handleSnapshot = useCallback(
    (snapshot: any) => {
      if (!snapshot) return;
      if (typeof snapshot.last_seq === "number") {
        lastSeqRef.current = Math.max(lastSeqRef.current, snapshot.last_seq);
      }
      if (Array.isArray(snapshot.positions)) {
        mergePositions(snapshot.positions.map(mapPosition));
      }
      if (Array.isArray(snapshot.orders)) {
        mergeOrders(snapshot.orders.map(mapOrder));
      }
      if (snapshot.account_state) {
        const wallet = mapAccountState(snapshot.account_state);
        setAccount(wallet);
        setMetrics((prev) => ({ ...prev, equity: wallet.equity ?? prev.equity }));
      }
      if (Array.isArray(snapshot.order_decisions) || Array.isArray(snapshot.order_exit_decisions)) {
        const allDecisions = [
          ...(snapshot.order_decisions ?? []),
          ...(snapshot.order_exit_decisions ?? []),
        ];
        if (allDecisions.length) {
          const mapped = allDecisions.map((raw: any) => {
            const decisionPayload = raw.decision ?? raw;
            const decision = mapDecision(decisionPayload);
            const correlation =
              raw.correlation_id ??
              decisionPayload?.signal_id ??
              decisionPayload?.position_id ??
              decisionPayload?.order_id;
            if (correlation) {
              decisionMapRef.current[correlation] = decision.reasonCode ?? decision.outcome;
            }
            return decision;
          });
          setDecisions((prev) => [...mapped, ...prev].slice(0, 50));
        }
      }
      if (snapshot.market?.klines?.length) {
        commitCandles(snapshot.market.klines.map(mapKlineToCandle));
        setChart((prev) => ({
          ...prev,
          symbol: snapshot.market.symbol ?? prev.symbol,
          hasMarketData: true,
        }));
        if (snapshot.market.last_tick?.price) {
          const ts = snapshot.market.last_tick?.ts_ms ?? toUnixMs(snapshot.market.last_tick?.ts);
          pushPrice(snapshot.market.last_tick.price, ts);
        }
      }
      if (Array.isArray(snapshot.events)) {
        const ordered = [...snapshot.events].sort((a, b) => (a.seq ?? 0) - (b.seq ?? 0));
        ordered.forEach((raw) =>
          processEvent({
            seq: raw.seq ?? raw.id,
            ts: toUnixMs(raw.ts ?? raw.timestamp),
            type: (raw.type ?? raw.event ?? "UNKNOWN").toString().toUpperCase(),
            severity: (raw.severity ?? "INFO").toString().toUpperCase() as EventEnvelope["severity"],
            source: raw.source ?? "snapshot",
            payload: raw.payload ?? raw.data ?? {},
            correlation_id: raw.correlation_id ?? raw.correlationId,
            context: raw.context ?? {},
            topic: raw.topic ?? raw.context?.topic,
            raw,
          })
        );
      }
      setUsingLiveData(true);
      pendingSnapshot.current = false;
    },
    [commitCandles, mergeOrders, mergePositions, processEvent, pushPrice, setAccount, setMetrics]
  );

  const handleRealtimeEvent = useCallback(
    (event: EventEnvelope) => {
      if (event.topic === "snapshot" || event.type === "SNAPSHOT") {
        handleSnapshot(event.payload ?? event.raw ?? {});
        return;
      }
      processEvent(event);
    },
    [handleSnapshot, processEvent]
  );

    const setupWebSocket = useCallback(() => {
    if (wsRef.current || typeof window === "undefined") return;
    const client = new RealtimeClient({
      topics: [
        "system_heartbeat",
        "order_decisions",
        "order_exit_decisions",
        "orders",
        "positions",
        "risk_events",
        "strategy_events",
        "market_data",
        "market_ticks",
        "snapshot",
        "trading_metrics",
        "trading_events",
      ],
      onEvent: handleRealtimeEvent,
      onStatusChange: (status) => {
        const snapshot = client.getStatus();
        setWsMeta((prev) => ({ ...prev, ...snapshot, status }));
        setPulse((prev) => ({
          ...prev,
          wsStatus: status,
          lastMessageAt: snapshot.lastMessageAt ?? prev.lastMessageAt,
          eventsPerSec: snapshot.eventsPerSec ?? prev.eventsPerSec,
          lastEventTs: snapshot.lastEventTs ?? prev.lastEventTs,
        }));
        if (status === "connected") {
          requestSnapshot("reconnected");
        }
      },
    });
    wsRef.current = client;
    client.connect();
  }, [handleRealtimeEvent, requestSnapshot]);

  const pollSummaries = useCallback(async (): Promise<boolean> => {
    const [telemetryRes, portfolioRes] = await Promise.allSettled([fetchTelemetrySummary(), fetchPortfolioSummary()]);

    const telemetryOk = telemetryRes.status === "fulfilled";
    const portfolioOk = portfolioRes.status === "fulfilled";
    let portfolioData: PortfolioSummary | null = null;

    if (portfolioOk) {
      portfolioData = portfolioRes.value;
      setPortfolioSummary(portfolioData);
      const wallet = mapAccountState(portfolioData);
      setAccount(wallet);
      setMetrics((prev) => ({ ...prev, equity: wallet.equity ?? prev.equity }));
    } else {
      const dashboardRes = await fetchDashboard().catch(() => null);
      if (dashboardRes?.account_state) {
        portfolioData = dashboardRes.account_state as PortfolioSummary;
        const wallet = mapAccountState(portfolioData);
        setPortfolioSummary(portfolioData);
        setAccount(wallet);
        setMetrics((prev) => ({ ...prev, equity: wallet.equity ?? prev.equity }));
      }
    }

    if (telemetryOk) {
      const tele = telemetryRes.value as TelemetrySummary;
      setTelemetry((prev) => {
        const normalizedTs = tele.last_event_ts ? toUnixMs(tele.last_event_ts) : prev.last_event_ts;
        return {
          engine_state: (tele.engine_state ?? prev.engine_state ?? "UNKNOWN").toString().toUpperCase(),
          events_per_sec: tele.events_per_sec ?? prev.events_per_sec,
          last_event_ts: normalizedTs,
          ws_clients: tele.ws_clients ?? prev.ws_clients,
          uptime: tele.uptime ?? prev.uptime,
        };
      });
      setPulse((prev) => {
        const normalizedTs = tele.last_event_ts ? toUnixMs(tele.last_event_ts) : prev.lastEventTs;
        return {
          ...prev,
          engineState: (tele.engine_state ?? prev.engineState ?? "UNKNOWN").toString().toUpperCase(),
          eventsPerSec: tele.events_per_sec ?? prev.eventsPerSec,
          lastEventTs: normalizedTs ?? prev.lastEventTs,
          wsClients: tele.ws_clients ?? prev.wsClients,
          uptime: tele.uptime ?? prev.uptime,
        };
      });
    }

    // STRICT REAL_MODE logic: REST success AND WS connected AND heartbeat <= 2s
    const wsStatus = wsRef.current?.getStatus();
    const wsConnected = wsStatus?.status === "connected";
    const lastHeartbeatAge = wsStatus?.lastHeartbeatAt ? Date.now() - wsStatus.lastHeartbeatAt : Infinity;
    const heartbeatOk = lastHeartbeatAge <= 2000; // <= 2s
    
    // REST is OK if at least one endpoint works (telemetry OR portfolio)
    // This is more resilient - if one fails, we still have data from the other
    const restOk = telemetryOk || Boolean(portfolioData);
    const realMode = restOk && wsConnected && heartbeatOk;
    
    // Determine restHealth: "ok" if both work, "degraded" if only one works, "error" if both fail
    let restHealth: RestHealth = "error";
    if (telemetryOk && portfolioData) {
      restHealth = "ok";
    } else if (telemetryOk || portfolioData) {
      restHealth = "degraded";
    }
    
    setBackendAvailable(restOk);
    setUsingLiveData(realMode);
    setPulse((prev) => ({ 
      ...prev, 
      mockMode: !realMode, 
      restHealth: restHealth
    }));

    if (restOk) {
      setLastBackendError(null);
      setError(null);
      setLastSync(new Date());
    } else {
      const errMsg =
        telemetryRes.status === "rejected"
          ? telemetryRes.reason?.message ?? "Telemetry unavailable"
          : portfolioRes.status === "rejected"
          ? portfolioRes.reason?.message ?? "Portfolio unavailable"
          : "Backend unavailable";
      setLastBackendError(errMsg);
      setError(errMsg);
    }

    return restOk;
  }, []);

  const refresh = useCallback(async () => {
    // Дебаунсинг: не обновлять если уже идет обновление
    if (loading) return;
    
    const ok = await pollSummaries();
    if (!ok) {
      return;
    }

    // Загружаем только критичные данные, остальное через WebSocket
    const [engineRes, riskRes, positionsRes, ordersRes, strategiesRes, metricsRes] = await Promise.allSettled([
      fetchEngineStatus(),
      fetchRiskStatus(),
      fetchPositions(),
      fetchOrders("open"),
      fetchStrategies(),
      fetchTradingMetrics(),
    ]);
    
    // Load trading metrics from REST API if WebSocket hasn't provided them yet
    if (metricsRes.status === "fulfilled" && !funnelMetrics) {
      const metrics = metricsRes.value;
      const counters = metrics.counters || {};
      const session = counters.session || {};
      const last1m = counters.last_1m || {};
      setFunnelMetrics({
        signals_total_1m: last1m.signals_total || 0,
        signals_total_session: session.signals_total || 0,
        decisions_create: session.decisions_create || 0,
        decisions_skip: session.decisions_skip || 0,
        decisions_block: session.decisions_block || 0,
        orders_created: session.orders_created || 0,
        orders_filled: session.orders_filled || 0,
        orders_canceled: session.orders_canceled || 0,
        orders_rejected: session.orders_rejected || 0,
        orders_open: metrics.active_orders_count || session.orders_open || 0,
        positions_open: metrics.active_positions_count || session.positions_open || 0,
        positions_closed: session.positions_closed || 0,
        pnl_unrealized: session.pnl_unrealized || 0,
        pnl_realized: session.pnl_realized || 0,
        equity: session.equity || 0,
        used_margin: 0,
        free_margin: 0,
        top_reasons: metrics.top_reasons_for_skip_block || [],
        last_signal_ts: metrics.last_signal_ts,
        last_order_ts: metrics.last_order_ts,
        last_fill_ts: metrics.last_fill_ts,
        engine_state: metrics.engine_state || "UNKNOWN",
        mode: metrics.mode || "unknown",
      });
    }

    if (engineRes.status === "fulfilled") {
      const engine = engineRes.value;
      setSystemMode(engine.running ? "ACTIVE" : "DEFENSIVE");
      setMetrics((prev) => ({
        ...prev,
        equity: engine.equity ?? prev.equity,
        dailyPnl: engine.daily_pnl ?? prev.dailyPnl ?? 0,
        initialCapital: engine.initial_capital ?? prev.initialCapital,
      }));
      setPulse((prev) => ({
        ...prev,
        engineState: engine.status?.toUpperCase() ?? prev.engineState,
        mode: deriveMode(engine),
      }));
    }

    if (riskRes.status === "fulfilled") {
      const risk = riskRes.value;
      setHealth(risk.killswitch_triggered ? "CRITICAL" : "OK");
      setRiskMetrics(mapRisk(risk, metrics.equity));
      setMetrics((prev) => ({
        ...prev,
        openPositions: risk.current_positions ?? prev.openPositions,
        openOrders: risk.current_orders ?? prev.openOrders,
      }));
    }

    if (positionsRes.status === "fulfilled") {
      const livePositions = positionsRes.value.map(mapPosition);
      mergePositions(livePositions);
      setMetrics((prev) => ({ ...prev, openPositions: livePositions.length }));
      setChart((prev) => ({
        ...prev,
        positionLabel: livePositions.length ? `${livePositions[0].side} ${livePositions[0].symbol}` : undefined,
      }));
      const avgLast = livePositions.length
        ? livePositions.reduce((sum, p) => sum + p.lastPrice, 0) / livePositions.length
        : undefined;
      if (avgLast) pushPrice(avgLast);
    }

    if (ordersRes.status === "fulfilled") {
      const liveOrders = ordersRes.value.map(mapOrder).filter((o) => o.status !== "FILLED");
      mergeOrders(liveOrders);
      setMetrics((prev) => ({ ...prev, openOrders: liveOrders.length }));
    }

    if (strategiesRes.status === "fulfilled" && strategiesRes.value.length) {
      setStrategies(strategiesRes.value.map(mapStrategy));
    }
  }, [mergeOrders, mergePositions, metrics.equity, pollSummaries, pushPrice, funnelMetrics]);

  const loadMarketSnapshot = useCallback(async () => {
    try {
      const snap = await fetchMarketSnapshot().catch(() => null);
      if (snap?.klines?.length) {
        commitCandles(snap.klines.map(mapKlineToCandle));
        setChart((prev) => ({
          ...prev,
          symbol: snap.symbol ?? prev.symbol,
          hasMarketData: true,
        }));
        if (snap.last_tick?.price) {
          const ts = snap.last_tick?.ts_ms ?? toUnixMs(snap.last_tick?.ts);
          pushPrice(snap.last_tick.price, ts);
        }
        if (snap.last_seq && snap.last_seq > lastSeqRef.current) {
          lastSeqRef.current = snap.last_seq;
        }
      }
    } catch {
      // ignore snapshot errors to avoid blocking UI
    }
  }, [commitCandles, pushPrice]);

  useEffect(() => {
    let cancelled = false;
    const tick = async () => {
      if (cancelled) return;
      await pollSummaries();
    };
    tick();
    // Увеличено до 15 секунд для снижения нагрузки на сервер
    const id = window.setInterval(tick, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [pollSummaries]);

  // REAL_MODE watchdog: check REST + WS + heartbeat every 5s, switch to MOCK if dead > 5s
  useEffect(() => {
    let mockModeTimeout: number | null = null;
    
    const checkRealMode = async () => {
      // Check REST
      const restOk = await fetchTelemetrySummary()
        .then(() => true)
        .catch(() => false);
      
      // Check WS
      const wsStatus = wsRef.current?.getStatus();
      const wsConnected = wsStatus?.status === "connected";
      const lastHeartbeatAge = wsStatus?.lastHeartbeatAt ? Date.now() - wsStatus.lastHeartbeatAt : Infinity;
      const heartbeatOk = lastHeartbeatAge <= 2000;
      
      const realMode = restOk && wsConnected && heartbeatOk;
      
      if (realMode) {
        // Clear mock mode timeout if REAL_MODE is active
        if (mockModeTimeout) {
          clearTimeout(mockModeTimeout);
          mockModeTimeout = null;
        }
        setPulse((prev) => ({ ...prev, mockMode: false }));
        setUsingLiveData(true);
        setBackendAvailable(true);
        
        // Clear mocks when REAL_MODE is active
        if (positions.length === mockPositions.length && positions[0]?.id === mockPositions[0]?.id) {
          setPositions([]);
        }
        if (orders.length === mockOrders.length && orders[0]?.id === mockOrders[0]?.id) {
          setOrders([]);
        }
        if (account.equity === mockWallet?.equity && account.wallet_balance === mockWallet?.wallet_balance) {
          // Keep account if it's from real data, otherwise reset to DEFAULT_WALLET
          if (!usingLiveData) {
            setAccount(DEFAULT_WALLET);
          }
        }
      } else {
        // If not REAL_MODE, set timeout to switch to MOCK after 5s
        if (!mockModeTimeout) {
          mockModeTimeout = window.setTimeout(() => {
            setPulse((prev) => ({ ...prev, mockMode: true }));
            setUsingLiveData(false);
            if (!restOk) {
              setBackendAvailable(false);
            }
          }, 5000);
        }
      }
    };
    
    const id = window.setInterval(checkRealMode, 15000); // Увеличено с 5 до 15 секунд
    checkRealMode(); // Initial check
    
    return () => {
      clearInterval(id);
      if (mockModeTimeout) clearTimeout(mockModeTimeout);
    };
  }, [positions, orders, account, usingLiveData]);

  useEffect(() => {
    setupWebSocket();
    return () => {
      wsRef.current?.disconnect();
      wsRef.current = null;
    };
  }, [setupWebSocket]);

  useEffect(() => {
    loadMarketSnapshot();
  }, [loadMarketSnapshot]);

  useEffect(() => {
    const id = window.setInterval(() => {
      const snapshot = wsRef.current?.getStatus();
      if (!snapshot) return;
      setWsMeta((prev) => ({ ...prev, ...snapshot }));
      setPulse((prev) => ({
        ...prev,
        wsStatus: snapshot.status,
        lastMessageAt: snapshot.lastMessageAt ?? prev.lastMessageAt,
        eventsPerSec: snapshot.eventsPerSec ?? prev.eventsPerSec,
        lastEventTs: snapshot.lastEventTs ?? prev.lastEventTs,
      }));
    }, 10000); // Увеличено до 10 секунд для снижения нагрузки

    return () => {
      clearInterval(id);
    };
  }, []);

  useEffect(() => {
    refresh().finally(() => setLoading(false));
  }, [refresh]);

  const runControl = useCallback(
    async (action: ControlAction) => {
      try {
        setControlStatus({ action, state: "pending", message: "Sending…" });
        pushControlEvent(action, "sending");
        const res = await sendControlAction(action);
        setControlSupport((prev) => ({ ...prev, [action]: true }));
        setControlStatus({ action, state: "waiting", message: res.message ?? "Awaiting CONTROL_RESULT" });
        pushControlEvent(action, "sent", res.message ?? "sent");
        return res;
      } catch (err: any) {
        if (err instanceof ApiError && err.isNotSupported()) {
          setControlSupport((prev) => ({ ...prev, [action]: false }));
        }
        const message = err?.message ?? "Control action failed";
        setControlStatus({ action, state: "error", message });
        pushControlEvent(action, "error", message, "ERROR");
        throw err;
      }
    },
    [pushControlEvent]
  );

  return {
    positions,
    orders,
    decisions,
    decisionFlows: mockDecisionFlows,
    strategies,
    account,
    riskMetrics,
    tradeResults,
    metrics,
    funnelMetrics,
    systemMode,
    health,
    pulse,
    telemetry,
    portfolio: portfolioSummary,
    ws: wsMeta,
    backendAvailable,
    lastBackendError,
    controlStatus,
    apiBase,
    wsUrl,
    lastSync,
    usingLiveData,
    loading,
    error,
    eventFeed,
    chart,
    controlSupport,
    refresh,
    runControl,
  };
}

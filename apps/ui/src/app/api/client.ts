const API_BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/$/, "");
const RAW_WS_URL = import.meta.env.VITE_WS_URL ?? "/ws";

function resolveWsUrl(raw: string): string {
  const value = raw || "/ws";

  // Absolute ws/wss URL – use as-is
  if (value.startsWith("ws://") || value.startsWith("wss://")) {
    return value;
  }

  // In browser: build ws(s)://<current-host>/<path>
  if (typeof window !== "undefined") {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;

    if (value.startsWith("/")) {
      return `${protocol}//${host}${value}`;
    }

    if (value.startsWith("http://") || value.startsWith("https://")) {
      return value.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
    }

    return `${protocol}//${host}/${value.replace(/^\/+/, "")}`;
  }

  // Non-browser (SSR/tests): fall back to localhost
  if (value.startsWith("/")) {
    return `ws://localhost:8000${value}`;
  }
  if (value.startsWith("http://") || value.startsWith("https://")) {
    return value.replace(/^http:/, "ws:").replace(/^https:/, "wss:");
  }
  return `ws://localhost:8000/${value.replace(/^\/+/, "")}`;
}

export type WsTopic =
  | "system_heartbeat"
  | "order_decisions"
  | "orders"
  | "positions"
  | "risk_events"
  | "strategy_events"
  | "snapshot"
  | "market_data"
  | "market_ticks"
  | "trading_metrics";

export type WsStatus = "connecting" | "connected" | "reconnecting" | "disconnected";

export interface EventEnvelope {
  seq?: number;
  ts: number;
  type: string;
  severity: "INFO" | "WARN" | "ERROR";
  source?: string;
  payload: Record<string, any>;
  correlation_id?: string;
  context?: Record<string, any>;
  topic?: string;
  raw?: any;
}

export class ApiError extends Error {
  status?: number;
  body?: any;

  constructor(message: string, status?: number, body?: any) {
    super(message);
    this.status = status;
    this.body = body;
  }

  isNotSupported() {
    return this.status === 404 || this.status === 501;
  }
}

export interface EngineStatusResponse {
  status: string;
  running?: boolean;
  trading_mode?: string;
  is_live?: boolean;
  dry_run?: boolean;
  equity?: number;
  daily_pnl?: number;
  initial_capital?: number;
}

export interface RiskStatusResponse {
  killswitch_triggered?: boolean;
  stop_trading?: boolean;
  equity?: number;
  daily_pnl?: number;
  drawdown?: number;
  drawdown_pct?: number;
  max_open_positions?: number;
  current_positions?: number;
  max_open_orders?: number;
  current_orders?: number;
}

export interface PositionResponse {
  position_id?: string;
  symbol: string;
  side: string;
  size?: number;
  entry_price: number;
  unrealized_pnl?: number;
  realized_pnl?: number;
  current_price?: number;
  take_profit?: number;
  stop_loss?: number;
  status?: string;
}

export interface StrategyResponse {
  strategy_id: string;
  enabled: boolean;
  type?: string;
}

export interface OrderResponse {
  order_id: string;
  symbol: string;
  side: string;
  size?: number;
  filled_size?: number;
  price?: number;
  status?: string;
  strategy_id?: string;
  timestamp?: string;
  type?: string;
}

export interface AccountState {
  wallet_balance: number;
  available_balance: number;
  equity: number;
  used_margin: number;
  reserved_margin: number;
  unrealized_pnl: number;
  realized_pnl: number;
  fees?: number;
  fees_24h?: number;
  timestamp?: string;
}

export interface DashboardResponse {
  account_state: AccountState;
  metrics: {
    equity: number;
    daily_pnl: number;
    return_pct: number;
    open_positions: number;
  };
  positions: any[];
  orders: any[];
  status: {
    engine_running: boolean;
    trading_mode: string;
  };
}

export interface PortfolioSummary {
  equity?: number;
  wallet_balance?: number;
  balance?: number;
  available_balance?: number;
  used_margin?: number;
  free_margin?: number;
  unrealized_pnl?: number;
  realized_pnl?: number;
  timestamp?: string;
}

export interface TelemetrySummary {
  engine_state?: string;
  events_per_sec?: number;
  last_event_ts?: string | number;
  ws_clients?: number;
  mode?: string;
  events_total?: number;
  uptime?: number;
}

export interface MarketKline {
  open_time?: string;
  close_time?: string;
  open_time_ms?: number;
  close_time_ms?: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
  timeframe?: string;
  symbol?: string;
}

export interface MarketSnapshot {
  symbol: string;
  timeframe: string;
  klines: MarketKline[];
  last_tick?: any;
  last_seq?: number;
  count?: number;
}

export async function apiRequest<T>(path: string, options: RequestInit = {}): Promise<T> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), 8000);

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      signal: controller.signal,
    });

    if (!response.ok) {
      let body: any = null;
      try {
        body = await response.json();
      } catch {
        body = await response.text();
      }
      const message =
        (body && (body.detail || body.message)) ||
        (typeof body === "string" && body) ||
        `HTTP ${response.status}`;
      throw new ApiError(message, response.status, body);
    }

    if (response.status === 204) {
      return {} as T;
    }

    return (await response.json()) as T;
  } finally {
    clearTimeout(id);
  }
}

export async function fetchEngineStatus(): Promise<EngineStatusResponse> {
  return apiRequest<EngineStatusResponse>("/engine/status");
}

export async function fetchRiskStatus(): Promise<RiskStatusResponse> {
  return apiRequest<RiskStatusResponse>("/risk/status");
}

export async function fetchPositions(): Promise<PositionResponse[]> {
  return apiRequest<PositionResponse[]>("/orders/positions");
}

export async function fetchOrders(status: "all" | "open" | "filled" = "open"): Promise<OrderResponse[]> {
  const query = status ? `?status=${status}` : "";
  return apiRequest<OrderResponse[]>(`/orders${query}`);
}

export async function fetchStrategies(): Promise<StrategyResponse[]> {
  return apiRequest<StrategyResponse[]>("/strategies");
}

export async function fetchDashboard(): Promise<DashboardResponse> {
  return apiRequest<DashboardResponse>("/system/v1/dashboard");
}

export async function fetchPortfolioSummary(): Promise<PortfolioSummary> {
  return apiRequest<PortfolioSummary>("/portfolio/summary");
}

export async function fetchTelemetrySummary(): Promise<TelemetrySummary> {
  return apiRequest<TelemetrySummary>("/telemetry/summary");
}

export async function fetchMarketSnapshot(
    symbol?: string,
    timeframe: string = "1m",
    limit: number = 200
): Promise<MarketSnapshot> {
    const params = new URLSearchParams({ timeframe, limit: String(limit) });
    if (symbol) params.set("symbol", symbol);
    return apiRequest<MarketSnapshot>(`/market/snapshot?${params.toString()}`);
}

export interface WhyNotTradingResponse {
    status: string;
    engine_state: string;
    engine_running: boolean;
    top_reasons: Array<{
        code: string;
        message: string;
        severity: "error" | "warning" | "info";
        count?: number;
    }>;
    last_decisions: any[];
    risk_blocks: string[];
    strategy_state: {
        total: number;
        enabled: number;
    };
    risk_status: {
        killswitch_triggered: boolean;
        stop_trading: boolean;
        current_positions: number;
        max_positions: number;
        current_orders: number;
        max_orders: number;
    };
}

export async function fetchWhyNotTrading(): Promise<WhyNotTradingResponse> {
    return apiRequest<WhyNotTradingResponse>("/trading/why");
}

export interface TradingMetricsResponse {
    status: string;
    counters: {
        last_1m: Record<string, number>;
        last_5m: Record<string, number>;
        session: Record<string, number>;
    };
    top_reasons_for_skip_block: Array<{code: string; count: number; pct: number}>;
    active_orders_count: number;
    active_positions_count: number;
    last_signal_ts?: string;
    last_order_ts?: string;
    last_fill_ts?: string;
    engine_state: string;
    mode: string;
    top_symbols: Array<{symbol: string; count: number}>;
    top_strategies: Array<{strategy_id: string; count: number}>;
    uptime_sec: number;
}

export async function fetchTradingMetrics(): Promise<TradingMetricsResponse> {
    return apiRequest<TradingMetricsResponse>("/trading/metrics");
}

export async function triggerKillSwitch(reason = "ui_panic_button"): Promise<void> {
  await apiRequest("/api/v1/emergency/killswitch", {
    method: "POST",
    body: JSON.stringify({ reason }),
  });
}

export async function setStrategyEnabled(id: string, enabled: boolean): Promise<void> {
  await apiRequest(`/strategies/${id}/enable`, {
    method: "POST",
    body: JSON.stringify({ enabled }),
  });
}

export type ControlAction = "pause" | "resume" | "restart" | "kill" | "start" | "stop";

export interface ControlResponse {
  status: string;
  message?: string;
  code?: number;
}

export async function sendControlAction(action: ControlAction): Promise<ControlResponse> {
  const endpoint: Record<ControlAction, string> = {
    pause: "/engine/pause",
    resume: "/engine/resume",
    restart: "/engine/restart",
    kill: "/api/v1/emergency/killswitch",
    start: "/engine/start",
    stop: "/engine/stop",
  };

  if (!endpoint[action]) {
    throw new ApiError(`Unsupported control action ${action}`, 400);
  }

  const result = await apiRequest<ControlResponse>(endpoint[action], { method: "POST" });
  return { ...result, status: result.status ?? action };
}

export function getApiBase(): string {
  return API_BASE;
}

export function getWsUrl(): string {
  return resolveWsUrl(RAW_WS_URL);
}

export interface RealtimeClientOptions {
  topics?: WsTopic[];
  url?: string;
  onEvent: (event: EventEnvelope) => void;
  onStatusChange?: (status: WsStatus) => void;
}

export interface RealtimeStatus {
  status: WsStatus;
  lastMessageAt?: number;
  lastHeartbeatAt?: number;
  lastEventTs?: number;
  eventsPerSec: number;
  reconnectAttempts: number;
  url: string;
  lastError?: string | null;
}

export class RealtimeClient {
  private socket: WebSocket | null = null;
  private reconnectTimer: number | null = null;
  private attempts = 0;
  private destroyed = false;
  private readonly options: RealtimeClientOptions;
  private readonly backoffSteps = [1000, 2000, 5000, 10000]; // Увеличено для более стабильного переподключения
  private readonly windowMs = 60_000;
  private eventTimestamps: number[] = [];
  private _lastEventTs: number | undefined;
  private status: WsStatus = "disconnected";
  private lastMessageAt?: number;
  private lastHeartbeatAt?: number;
  private lastError: string | null = null;
  private subscriptions = new Set<WsTopic>();
  private eventHandlers: Array<(event: EventEnvelope) => void> = [];
  private pingInterval: number | null = null;
  private readonly PING_INTERVAL = 25000; // Отправлять ping каждые 25 секунд
  private readonly CONNECTION_TIMEOUT = 60000; // Таймаут соединения 60 секунд

  constructor(options: RealtimeClientOptions) {
    this.options = options;
    if (options.topics?.length) {
      options.topics.forEach((t) => this.subscriptions.add(t));
    }
    this.subscriptions.add("system_heartbeat");
    this.subscriptions.add("snapshot");
    this.eventHandlers.push(options.onEvent);
  }

  connect() {
    if (typeof window === "undefined" || this.socket || this.destroyed) return;
    this.setStatus(this.attempts > 0 ? "reconnecting" : "connecting");
    this.lastError = null;

    const socket = new WebSocket(this.options.url ?? getWsUrl());
    this.socket = socket;

    socket.onopen = () => {
      this.attempts = 0;
      this.setStatus("connected");
      this.flushSubscriptions();
      // Запустить периодические ping для поддержания соединения
      this.startPingInterval();
    };

    socket.onmessage = (evt) => {
      this.lastMessageAt = Date.now();
      let parsed: any = evt.data;
      try {
        parsed = typeof evt.data === "string" ? JSON.parse(evt.data) : evt.data;
      } catch (err: any) {
        this.lastError = err?.message ?? "Failed to parse WS message";
        this.options.onStatusChange?.(this.status);
        return;
      }

      // Обработка ping/pong от сервера
      if (parsed?.type === "ping") {
        try {
          this.socket?.send(JSON.stringify({ action: "ping", type: "pong" }));
        } catch (err) {
          console.warn("Failed to respond to ping:", err);
        }
        return;
      }

      try {
        const env = this.normalizeEvent(parsed);
        this.recordEventTimestamp(env.ts);
        this.handleHeartbeat(env);
        this.eventHandlers.forEach((fn) => fn(env));
      } catch (err) {
        console.warn("WS parse error", err);
      }
    };

    socket.onclose = (event) => {
      this.socket = null;
      this.stopPingInterval();
      if (this.destroyed) {
        this.setStatus("disconnected");
        return;
      }
      // Не переподключаться сразу при нормальном закрытии
      if (event.code === 1000 || event.code === 1001) {
        this.setStatus("disconnected");
        return;
      }
      this.scheduleReconnect();
    };

    socket.onerror = (ev: Event | string) => {
      const errorMsg = typeof ev === "string" ? ev : "WebSocket error";
      this.lastError = errorMsg;
      console.warn("WebSocket error:", errorMsg);
      // Не закрывать соединение сразу - дать onclose обработать это
      // socket.close() может вызвать двойное закрытие
    };
  }

  disconnect() {
    this.destroyed = true;
    this.stopPingInterval();
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    if (this.socket) {
      this.socket.close();
      this.socket = null;
    }
    this.setStatus("disconnected");
  }

  private startPingInterval() {
    this.stopPingInterval();
    if (typeof window === "undefined") return;
    
    this.pingInterval = window.setInterval(() => {
      if (this.socket && this.socket.readyState === WebSocket.OPEN) {
        try {
          this.socket.send(JSON.stringify({ action: "ping" }));
          this.lastMessageAt = Date.now();
        } catch (err) {
          console.warn("Failed to send ping:", err);
          this.stopPingInterval();
        }
      } else {
        this.stopPingInterval();
      }
    }, this.PING_INTERVAL);
  }

  private stopPingInterval() {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  subscribe(topic: WsTopic) {
    this.subscriptions.add(topic);
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ action: "subscribe", topic }));
    }
  }

  requestSnapshot() {
    if (this.socket && this.socket.readyState === WebSocket.OPEN) {
      this.socket.send(JSON.stringify({ action: "subscribe", topic: "snapshot", reason: "resync" }));
    }
  }

  onEvent(cb: (event: EventEnvelope) => void) {
    this.eventHandlers.push(cb);
  }

  getStatus(): RealtimeStatus {
    this.pruneEvents();
    return {
      eventsPerSec: this.computeEventsPerSec(),
      lastEventTs: this._lastEventTs,
      status: this.status,
      lastMessageAt: this.lastMessageAt,
      lastHeartbeatAt: this.lastHeartbeatAt,
      reconnectAttempts: this.attempts,
      url: this.options.url ?? getWsUrl(),
      lastError: this.lastError,
    };
  }

  private setStatus(status: WsStatus) {
    this.status = status;
    this.options.onStatusChange?.(status);
  }

  private scheduleReconnect() {
    if (this.destroyed) return;
    
    this.attempts += 1;
    const idx = Math.min(this.attempts - 1, this.backoffSteps.length - 1);
    const delay = this.backoffSteps[idx];
    this.setStatus("reconnecting");
    
    // Ограничить количество попыток переподключения
    if (this.attempts > 20) {
      console.warn(`Max reconnection attempts reached (${this.attempts}), stopping`);
      this.setStatus("disconnected");
      this.lastError = "Max reconnection attempts reached";
      return;
    }
    
    this.reconnectTimer = window.setTimeout(() => {
      if (!this.destroyed) {
        this.connect();
      }
    }, delay);
  }

  private flushSubscriptions() {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return;
    this.subscriptions.forEach((topic) => {
      try {
        this.socket?.send(JSON.stringify({ action: "subscribe", topic }));
      } catch (err) {
        console.warn("Failed to subscribe topic", topic, err);
      }
    });
  }

  private normalizeEvent(message: any): EventEnvelope {
    const topic = message?.topic ?? message?.context?.topic;
    const data = message?.data ?? message;
    const tsRaw = data?.ts ?? message?.ts ?? message?.timestamp ?? Date.now();
    const tsParsed =
      typeof tsRaw === "number"
        ? tsRaw
        : tsRaw
        ? Date.parse(tsRaw)
        : NaN;
    const ts = Number.isFinite(tsParsed) ? tsParsed : Date.now();
    const seqRaw = data?.seq ?? message?.seq;
    const seq = seqRaw === undefined || seqRaw === null ? undefined : Number(seqRaw);
    const payloadCandidate =
      data?.payload ?? data?.data ?? (typeof data === "object" ? data : { value: data });
    const payload =
      payloadCandidate && typeof payloadCandidate === "object" ? payloadCandidate : { value: payloadCandidate };
    const correlationId = data?.correlation_id ?? data?.correlationId ?? message?.correlation_id;

    return {
      seq,
      ts,
      type: (data?.type || topic || "UNKNOWN").toString().toUpperCase(),
      severity: (data?.severity || "INFO").toString().toUpperCase() as EventEnvelope["severity"],
      source: data?.source ?? topic ?? "ws",
      payload,
      correlation_id: correlationId,
      context: data?.context ?? (topic ? { topic } : {}),
      topic,
      raw: message,
    };
  }

  private recordEventTimestamp(ts: number) {
    this._lastEventTs = ts;
    const now = Date.now();
    this.eventTimestamps.push(now);
    this.pruneEvents();
  }

  private handleHeartbeat(env: EventEnvelope) {
    const isHeartbeat =
      env.type?.includes("HEARTBEAT") || env.topic === "system_heartbeat" || env.context?.topic === "system_heartbeat";
    if (isHeartbeat) {
      this.lastHeartbeatAt = env.ts || Date.now();
    }
  }

  private pruneEvents() {
    const cutoff = Date.now() - this.windowMs;
    while (this.eventTimestamps.length && this.eventTimestamps[0] < cutoff) {
      this.eventTimestamps.shift();
    }
  }

  private computeEventsPerSec() {
    this.pruneEvents();
    if (!this.eventTimestamps.length) return 0;
    const spanMs = Math.max(Date.now() - this.eventTimestamps[0], 1);
    return Number((this.eventTimestamps.length / (spanMs / 1000)).toFixed(2));
  }
}

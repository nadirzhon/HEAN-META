"use client";

import { useHeanStore } from "@/store/heanStore";
import type { SystemEvent, AgentThought } from "@/store/heanStore";

const WS_URL = "ws://localhost:8000/ws";
const RECONNECT_BASE_DELAY = 1000;
const RECONNECT_MAX_DELAY = 30000;

let ws: WebSocket | null = null;
let reconnectAttempts = 0;
let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
let heartbeatTimer: ReturnType<typeof setInterval> | null = null;

function getReconnectDelay(): number {
  const delay = Math.min(
    RECONNECT_BASE_DELAY * Math.pow(2, reconnectAttempts),
    RECONNECT_MAX_DELAY
  );
  return delay + Math.random() * 1000; // jitter
}

function generateEventId(): string {
  return `evt_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

function handleMessage(data: Record<string, unknown>) {
  const store = useHeanStore.getState();
  const eventType = (data.type as string) || "UNKNOWN";
  const timestamp = new Date().toISOString();

  // Update heartbeat
  store.setLastHeartbeat(Date.now());

  // Add to system events log
  store.addSystemEvent({
    id: generateEventId(),
    timestamp,
    type: eventType,
    summary: formatEventSummary(eventType, data),
    data,
  });

  // Route to specific store slices based on event type
  switch (eventType) {
    case "HEARTBEAT":
      // Just heartbeat tracking, already handled above
      break;

    case "STATUS":
    case "system_status":
      if (data.equity !== undefined) {
        store.setEngineStatus({
          status: (data.status as string) || "unknown",
          running: (data.running as boolean) ?? true,
          equity: (data.equity as number) || 0,
          daily_pnl: (data.daily_pnl as number) || 0,
          initial_capital: (data.initial_capital as number) || 300,
        });
        store.addEquityPoint({
          timestamp: Date.now(),
          equity: (data.equity as number) || 0,
          pnl: (data.daily_pnl as number) || 0,
        });
      }
      break;

    case "SIGNAL":
      store.addAgentThought({
        id: generateEventId(),
        agent: "brain",
        timestamp,
        content: `Signal: ${data.symbol || "?"} ${data.direction || data.side || "?"} — confidence ${((data.confidence as number) || 0).toFixed(2)}`,
      });
      break;

    case "ORDER_FILLED":
    case "order_decisions":
      store.addAgentThought({
        id: generateEventId(),
        agent: "council",
        timestamp,
        content: `Order filled: ${data.symbol || "?"} ${data.side || "?"} ${data.qty || data.size || "?"}`,
      });
      break;

    case "RISK_ALERT":
    case "RISK_BLOCKED":
      store.addAgentThought({
        id: generateEventId(),
        agent: "risk",
        timestamp,
        content: `Risk: ${data.reason || data.message || eventType}`,
      });
      break;

    case "BRAIN_ANALYSIS":
    case "ai_catalyst":
      if (data.summary) {
        store.setBrainAnalysis({
          timestamp,
          summary: data.summary as string,
          market_sentiment: (data.market_sentiment as string) || "neutral",
          recommendations: (data.recommendations as string[]) || [],
        });
        store.addAgentThought({
          id: generateEventId(),
          agent: "brain",
          timestamp,
          content: data.summary as string,
        });
      }
      break;

    case "POSITION_OPENED":
    case "POSITION_CLOSED":
    case "POSITION_UPDATE":
      // Trigger a positions refresh
      break;

    case "REGIME_UPDATE":
    case "CONTEXT_UPDATE":
      store.addAgentThought({
        id: generateEventId(),
        agent: "meta",
        timestamp,
        content: `Regime: ${data.regime || data.phase || "update"}`,
      });
      break;
  }
}

function formatEventSummary(type: string, data: Record<string, unknown>): string {
  switch (type) {
    case "SIGNAL":
      return `${data.symbol} ${data.direction || data.side} signal (conf: ${((data.confidence as number) || 0).toFixed(2)})`;
    case "ORDER_FILLED":
      return `Filled: ${data.symbol} ${data.side} ${data.qty || data.size}`;
    case "ORDER_PLACED":
      return `Placed: ${data.symbol} ${data.side} ${data.qty || data.size}`;
    case "RISK_ALERT":
      return `Risk alert: ${data.reason || data.message || "unknown"}`;
    case "RISK_BLOCKED":
      return `Blocked: ${data.reason || "risk limit"}`;
    case "HEARTBEAT":
      return "System heartbeat";
    case "POSITION_OPENED":
      return `Position opened: ${data.symbol} ${data.side}`;
    case "POSITION_CLOSED":
      return `Position closed: ${data.symbol} PnL: ${data.pnl || "?"}`;
    case "BRAIN_ANALYSIS":
      return `Brain: ${(data.summary as string)?.slice(0, 80) || "analysis complete"}`;
    case "REGIME_UPDATE":
      return `Regime → ${data.regime || data.phase || "update"}`;
    default:
      return `${type}: ${JSON.stringify(data).slice(0, 100)}`;
  }
}

export function connectWebSocket() {
  if (ws?.readyState === WebSocket.OPEN) return;

  const store = useHeanStore.getState();
  store.setConnectionStatus("reconnecting");

  try {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      reconnectAttempts = 0;
      useHeanStore.getState().setConnectionStatus("connected");
      useHeanStore.getState().setLastHeartbeat(Date.now());

      // Start heartbeat monitor
      if (heartbeatTimer) clearInterval(heartbeatTimer);
      heartbeatTimer = setInterval(() => {
        const last = useHeanStore.getState().lastHeartbeat;
        if (last && Date.now() - last > 30000) {
          useHeanStore.getState().setConnectionStatus("reconnecting");
        }
      }, 10000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        handleMessage(data);
      } catch {
        // Non-JSON message, ignore
      }
    };

    ws.onclose = () => {
      useHeanStore.getState().setConnectionStatus("disconnected");
      if (heartbeatTimer) clearInterval(heartbeatTimer);
      scheduleReconnect();
    };

    ws.onerror = () => {
      ws?.close();
    };
  } catch {
    scheduleReconnect();
  }
}

function scheduleReconnect() {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  reconnectAttempts++;
  const delay = getReconnectDelay();
  reconnectTimer = setTimeout(connectWebSocket, delay);
}

export function disconnectWebSocket() {
  if (reconnectTimer) clearTimeout(reconnectTimer);
  if (heartbeatTimer) clearInterval(heartbeatTimer);
  reconnectTimer = null;
  heartbeatTimer = null;
  reconnectAttempts = 0;
  ws?.close();
  ws = null;
  useHeanStore.getState().setConnectionStatus("disconnected");
}

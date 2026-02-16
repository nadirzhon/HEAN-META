"use client";

import { useState, useEffect, useCallback } from "react";
import type {
  SystemNode,
  NodeConnection,
  Impulse,
  AgentThought,
  NeuroMapState,
  NodeStatus,
  AgentRole,
} from "../../types/neuromap";

// ---------------------------------------------------------------------------
// Default architecture layout (800x600 viewBox)
// ---------------------------------------------------------------------------

const DEFAULT_NODES: SystemNode[] = [
  {
    id: "market",
    label: "Market Data",
    x: 320,
    y: 20,
    width: 160,
    height: 52,
    color: "stream",
    status: "active",
    eventsPerSec: 0,
    description: "Bybit WebSocket ticks, orderbook, funding",
  },
  {
    id: "eventbus",
    label: "EventBus",
    x: 310,
    y: 120,
    width: 180,
    height: 52,
    color: "axon",
    status: "active",
    eventsPerSec: 0,
    description: "Async priority queues + circuit breaker",
  },
  {
    id: "strategies",
    label: "Strategies",
    x: 60,
    y: 240,
    width: 150,
    height: 52,
    color: "axon",
    status: "active",
    eventsPerSec: 0,
    description: "11 trading strategies with filter cascades",
  },
  {
    id: "risk",
    label: "Risk",
    x: 325,
    y: 240,
    width: 150,
    height: 52,
    color: "axon",
    status: "active",
    eventsPerSec: 0,
    description: "RiskGovernor state machine + KillSwitch",
  },
  {
    id: "physics",
    label: "Physics",
    x: 590,
    y: 240,
    width: 150,
    height: 52,
    color: "stream",
    status: "active",
    eventsPerSec: 0,
    description: "Thermodynamics, phase detection, Szilard engine",
  },
  {
    id: "brain",
    label: "Brain",
    x: 620,
    y: 120,
    width: 130,
    height: 52,
    color: "stream",
    status: "idle",
    eventsPerSec: 0,
    description: "Claude AI market analysis",
  },
  {
    id: "portfolio",
    label: "Portfolio",
    x: 60,
    y: 120,
    width: 130,
    height: 52,
    color: "axon",
    status: "active",
    eventsPerSec: 0,
    description: "Accounting, capital allocation, profit capture",
  },
  {
    id: "council",
    label: "Council",
    x: 60,
    y: 360,
    width: 130,
    height: 52,
    color: "axon",
    status: "idle",
    eventsPerSec: 0,
    description: "Multi-agent trade review consensus",
  },
  {
    id: "execution",
    label: "Execution",
    x: 230,
    y: 380,
    width: 340,
    height: 52,
    color: "axon",
    status: "active",
    eventsPerSec: 0,
    description: "Order routing with idempotency",
  },
  {
    id: "exchange",
    label: "Bybit Exchange",
    x: 250,
    y: 490,
    width: 300,
    height: 52,
    color: "stream",
    status: "active",
    eventsPerSec: 0,
    description: "Bybit Testnet REST + WebSocket",
  },
];

const DEFAULT_CONNECTIONS: NodeConnection[] = [
  { id: "c-market-bus", sourceId: "market", targetId: "eventbus", active: true },
  { id: "c-bus-strategies", sourceId: "eventbus", targetId: "strategies", active: true },
  { id: "c-bus-risk", sourceId: "eventbus", targetId: "risk", active: true },
  { id: "c-bus-physics", sourceId: "eventbus", targetId: "physics", active: true },
  { id: "c-bus-brain", sourceId: "eventbus", targetId: "brain", active: false },
  { id: "c-bus-portfolio", sourceId: "eventbus", targetId: "portfolio", active: true },
  { id: "c-strategies-execution", sourceId: "strategies", targetId: "execution", active: true },
  { id: "c-risk-execution", sourceId: "risk", targetId: "execution", active: true },
  { id: "c-execution-exchange", sourceId: "execution", targetId: "exchange", active: true },
  { id: "c-strategies-council", sourceId: "strategies", targetId: "council", active: false },
];

// ---------------------------------------------------------------------------
// Simulated agent thoughts for demo mode
// ---------------------------------------------------------------------------

const BRAIN_THOUGHTS = [
  "BTC showing divergence between spot and perp OI. Possible short squeeze above 68.5k.",
  "ETH/BTC ratio compressing to 0.048 -- historical support level. Monitoring for breakout.",
  "Funding rates across top 10 alts have flipped negative. Contrarian long signal forming.",
  "Macro regime: risk-on. DXY weakening, correlates with crypto upside historically.",
  "Volatility term structure inverted on BTC. Market pricing near-term event risk.",
];

const COUNCIL_THOUGHTS = [
  "APPROVED: BTC long impulse signal. 3/3 agents concur. Confidence: 0.82.",
  "BLOCKED: ETH short signal rejected by risk quorum. Drawdown proximity too high.",
  "REVIEWING: SOL momentum trade. Awaiting physics phase confirmation.",
  "APPROVED: Funding harvest on DOGE perp. Negative funding at -0.03%. Low risk.",
  "BLOCKED: Grid strategy on AVAX paused. Insufficient liquidity depth.",
];

const RISK_THOUGHTS = [
  "State: NORMAL. All thresholds green. Drawdown: 2.1% / 20% limit.",
  "Soft brake engaged on SOL -- 3 consecutive losses. Reducing position size 50%.",
  "KillSwitch check: equity $294.50 vs initial $300. Well within safe zone.",
  "Quarantine lifted on LINK. Recovery period complete. Re-enabling signals.",
  "State: NORMAL. Portfolio heat: 34%. Correlation risk: low across active positions.",
];

function pickRandom<T>(arr: T[]): T {
  return arr[Math.floor(Math.random() * arr.length)];
}

// ---------------------------------------------------------------------------
// Hook
// ---------------------------------------------------------------------------

let nextImpulseId = 0;
let nextThoughtId = 0;

export function useNeuroMapState(): NeuroMapState & {
  fireImpulse: (connectionId: string) => void;
} {
  const [nodes, setNodes] = useState<SystemNode[]>(DEFAULT_NODES);
  const [connections] = useState<NodeConnection[]>(DEFAULT_CONNECTIONS);
  const [impulses, setImpulses] = useState<Impulse[]>([]);
  const [agentThoughts, setAgentThoughts] = useState<AgentThought[]>([]);
  const [totalEventsPerSec, setTotalEventsPerSec] = useState(0);

  // Fire an impulse on a given connection
  const fireImpulse = useCallback((connectionId: string) => {
    const impulse: Impulse = {
      id: `imp-${nextImpulseId++}`,
      connectionId,
      progress: 0,
      createdAt: Date.now(),
    };
    setImpulses((prev) => [...prev, impulse]);

    // Bump event counter on the target node
    const conn = DEFAULT_CONNECTIONS.find((c) => c.id === connectionId);
    if (conn) {
      setNodes((prev) =>
        prev.map((n) =>
          n.id === conn.targetId
            ? { ...n, eventsPerSec: n.eventsPerSec + 1 }
            : n
        )
      );
    }

    // Remove impulse after animation completes
    setTimeout(() => {
      setImpulses((prev) => prev.filter((i) => i.id !== impulse.id));
    }, 900);
  }, []);

  // Simulate data flow activity
  useEffect(() => {
    const activeConnections = DEFAULT_CONNECTIONS.filter((c) => c.active);

    const interval = setInterval(() => {
      // Pick 1-3 random active connections to fire
      const count = 1 + Math.floor(Math.random() * 3);
      for (let i = 0; i < count; i++) {
        const conn = pickRandom(activeConnections);
        fireImpulse(conn.id);
      }
    }, 600);

    return () => clearInterval(interval);
  }, [fireImpulse]);

  // Decay event counters every second
  useEffect(() => {
    const interval = setInterval(() => {
      setNodes((prev) => {
        let total = 0;
        const next = prev.map((n) => {
          const decayed = Math.max(0, n.eventsPerSec - 1);
          total += decayed;
          return { ...n, eventsPerSec: decayed };
        });
        setTotalEventsPerSec(total);
        return next;
      });
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Simulate random status changes (occasional degradation/recovery)
  useEffect(() => {
    const interval = setInterval(() => {
      setNodes((prev) =>
        prev.map((n) => {
          const roll = Math.random();
          let newStatus: NodeStatus = n.status;
          if (roll < 0.02) {
            newStatus = "degraded";
          } else if (roll < 0.005) {
            newStatus = "error";
          } else if (n.status !== "idle" && roll > 0.1) {
            newStatus = "active";
          }
          return { ...n, status: newStatus };
        })
      );
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  // Simulate agent thoughts appearing
  useEffect(() => {
    const scheduleNextThought = () => {
      const delay = 3000 + Math.random() * 5000; // 3-8 seconds
      return setTimeout(() => {
        const agents: AgentRole[] = ["brain", "council", "risk"];
        const agent = pickRandom(agents);
        const messagePool =
          agent === "brain"
            ? BRAIN_THOUGHTS
            : agent === "council"
              ? COUNCIL_THOUGHTS
              : RISK_THOUGHTS;

        const thought: AgentThought = {
          id: `thought-${nextThoughtId++}`,
          agent,
          message: pickRandom(messagePool),
          timestamp: Date.now(),
          isStreaming: true,
        };

        setAgentThoughts((prev) => {
          const next = [thought, ...prev].slice(0, 5);
          return next;
        });

        // Stop "streaming" after a delay
        setTimeout(() => {
          setAgentThoughts((prev) =>
            prev.map((t) =>
              t.id === thought.id ? { ...t, isStreaming: false } : t
            )
          );
        }, 800 + Math.random() * 1200);

        timerId = scheduleNextThought();
      }, delay);
    };

    let timerId = scheduleNextThought();
    return () => clearTimeout(timerId);
  }, []);

  return {
    nodes,
    connections,
    impulses,
    agentThoughts,
    totalEventsPerSec,
    fireImpulse,
  };
}

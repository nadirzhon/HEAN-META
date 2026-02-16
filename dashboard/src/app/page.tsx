"use client";

import { useEffect } from "react";
import TabView from "@/components/TabView";
import { connectWebSocket, disconnectWebSocket } from "@/services/websocket";
import { useHeanStore } from "@/store/heanStore";
import { api } from "@/services/api";

function useInitialLoad() {
  const store = useHeanStore();

  useEffect(() => {
    // Initial data load from API
    async function loadData() {
      try {
        const [engineStatus, positions, orders, strategies, riskState, killSwitch, metrics] =
          await Promise.allSettled([
            api.getEngineStatus(),
            api.getPositions(),
            api.getOrders(),
            api.getStrategies(),
            api.getRiskGovernor(),
            api.getKillSwitch(),
            api.getTradingMetrics(),
          ]);

        if (engineStatus.status === "fulfilled") {
          store.setEngineStatus(engineStatus.value);
          store.addEquityPoint({
            timestamp: Date.now(),
            equity: engineStatus.value.equity,
            pnl: engineStatus.value.daily_pnl,
          });
        }
        if (positions.status === "fulfilled") store.setPositions(positions.value as never[]);
        if (orders.status === "fulfilled") store.setOrders(orders.value as never[]);
        if (strategies.status === "fulfilled") store.setStrategies(strategies.value as never[]);
        if (riskState.status === "fulfilled") store.setRiskState(riskState.value);
        if (killSwitch.status === "fulfilled") store.setKillSwitch(killSwitch.value);
        if (metrics.status === "fulfilled") store.setTradingMetrics(metrics.value.counters);
      } catch {
        // API might not be running yet — that's OK
      }
    }

    loadData();

    // Connect WebSocket for real-time updates
    connectWebSocket();

    // Periodic refresh every 10 seconds
    const interval = setInterval(loadData, 10000);

    return () => {
      clearInterval(interval);
      disconnectWebSocket();
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
}

export default function DashboardPage() {
  useInitialLoad();

  return <TabView />;
}

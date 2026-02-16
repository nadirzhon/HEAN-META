"use client";

import { useMemo } from "react";
import { motion } from "framer-motion";
import { useHeanStore } from "@/store/heanStore";
import { EquityChart } from "../cockpit/EquityChart";
import { AssetDonutChart } from "../cockpit/AssetDonutChart";
import { MetricRow } from "../cockpit/MetricRow";
import { LiveFeed } from "../cockpit/LiveFeed";
import { PositionsSummary } from "../cockpit/PositionsSummary";

/**
 * Cockpit tab — primary overview dashboard.
 * Reads all data from the centralized Zustand store.
 */
export default function CockpitTab() {
  const engineStatus = useHeanStore((s) => s.engineStatus);
  const positions = useHeanStore((s) => s.positions);
  const riskState = useHeanStore((s) => s.riskState);
  const tradingMetrics = useHeanStore((s) => s.tradingMetrics);
  const equityHistory = useHeanStore((s) => s.equityHistory);
  const systemEvents = useHeanStore((s) => s.systemEvents);

  // Calculate asset allocation from positions
  const { assetAllocation, totalValue } = useMemo(() => {
    const alloc = positions.map((pos) => {
      const value = Math.abs(pos.size * (pos.current_price || pos.entry_price));
      return { symbol: pos.symbol, value, percentage: 0 };
    });
    const total = alloc.reduce((sum, a) => sum + a.value, 0);
    alloc.forEach((a) => {
      a.percentage = total > 0 ? (a.value / total) * 100 : 0;
    });
    return { assetAllocation: alloc, totalValue: total };
  }, [positions]);

  // Convert system events to LiveFeed format
  const liveEvents = useMemo(
    () =>
      systemEvents.slice(0, 50).map((e) => ({
        id: e.id,
        type: e.type,
        timestamp: new Date(e.timestamp).getTime(),
        summary: e.summary,
      })),
    [systemEvents]
  );

  const signalsPerMin = tradingMetrics?.last_1m.signals_detected ?? 0;

  // Loading state
  if (!engineStatus) {
    return (
      <div className="flex items-center justify-center h-full">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <div className="w-10 h-10 border-2 border-axon border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-starlight/40 text-sm">Connecting to HEAN backend...</p>
          <p className="text-starlight/25 text-xs">Make sure the system is running at localhost:8000</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="max-w-[1600px] mx-auto space-y-6">
      {/* Metric Row */}
      <MetricRow
        equity={engineStatus.equity}
        dailyPnl={engineStatus.daily_pnl}
        winRate={0}
        riskState={riskState?.risk_state || "UNKNOWN"}
        activePositions={positions.length}
        signalsPerMin={signalsPerMin}
      />

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="lg:col-span-3"
        >
          <EquityChart
            data={equityHistory}
            currentEquity={engineStatus.equity}
            initialCapital={engineStatus.initial_capital}
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="lg:col-span-2"
        >
          <AssetDonutChart data={assetAllocation} totalValue={totalValue} />
        </motion.div>
      </div>

      {/* Bottom Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <PositionsSummary positions={positions} />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <LiveFeed events={liveEvents} />
        </motion.div>
      </div>
    </div>
  );
}

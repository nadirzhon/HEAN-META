"use client";

import { motion } from "framer-motion";
import { useHeanStore } from "@/store/heanStore";
import type { StrategyState, SystemEvent } from "@/store/heanStore";
import GlassCard from "@/components/ui/GlassCard";
import StatusBadge from "@/components/ui/StatusBadge";
import clsx from "clsx";

/* ── Strategy Card ── */

function StrategyCard({ strategy, signals, delay }: {
  strategy: StrategyState;
  signals: SystemEvent[];
  delay: number;
}) {
  const pnl = strategy.pnl ?? 0;
  const winRate = strategy.win_rate ?? 0;
  const isPositive = pnl >= 0;

  return (
    <GlassCard delay={delay} hover>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-sm font-semibold text-supernova">{strategy.type}</h3>
            <p className="text-[10px] text-starlight/40 font-mono">{strategy.strategy_id}</p>
          </div>
          <StatusBadge
            label={strategy.enabled ? (strategy.state || "ACTIVE") : "DISABLED"}
            variant={strategy.enabled ? "success" : "normal"}
          />
        </div>

        {/* Mini PnL Bar */}
        <div className="space-y-1">
          <div className="flex justify-between text-[11px]">
            <span className="text-starlight/50">PnL</span>
            <span className={clsx("font-medium font-mono", isPositive ? "text-positive" : "text-negative")}>
              {isPositive ? "+" : ""}{pnl.toFixed(2)} USDT
            </span>
          </div>
          <div className="h-1.5 bg-white/5 rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(Math.abs(pnl) / 10 * 100, 100)}%` }}
              transition={{ duration: 0.8, delay: delay + 0.2 }}
              className={clsx(
                "h-full rounded-full",
                isPositive ? "bg-positive" : "bg-negative"
              )}
            />
          </div>
        </div>

        {/* Stats Row */}
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <p className="text-[10px] text-starlight/40">Win Rate</p>
            <p className="text-sm font-medium text-supernova">{(winRate * 100).toFixed(0)}%</p>
          </div>
          <div>
            <p className="text-[10px] text-starlight/40">Signals</p>
            <p className="text-sm font-medium text-supernova">{strategy.signals_count ?? 0}</p>
          </div>
          <div>
            <p className="text-[10px] text-starlight/40">Positions</p>
            <p className="text-sm font-medium text-supernova">{strategy.positions_count ?? 0}</p>
          </div>
        </div>

        {/* Recent Signals Feed */}
        {signals.length > 0 && (
          <div className="space-y-1 pt-2 border-t border-glass-border">
            <p className="text-[10px] text-starlight/40 uppercase tracking-wide">Recent Signals</p>
            <div className="space-y-1 max-h-24 overflow-auto">
              {signals.slice(0, 3).map((sig) => (
                <div key={sig.id} className="flex items-center gap-2 text-[11px]">
                  <span className="text-starlight/30 font-mono">
                    {new Date(sig.timestamp).toLocaleTimeString("en-US", { hour12: false })}
                  </span>
                  <span className="text-starlight/70 truncate">{sig.summary}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </GlassCard>
  );
}

/* ── Tactical Tab ── */

export default function TacticalTab() {
  const strategies = useHeanStore((s) => s.strategies);
  const events = useHeanStore((s) => s.systemEvents);

  // Get signal events per strategy
  function getStrategySignals(strategyType: string): SystemEvent[] {
    return events.filter(
      (e) => e.type === "SIGNAL" && e.data?.strategy === strategyType
    );
  }

  if (strategies.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-4"
        >
          <svg className="w-16 h-16 text-starlight/20 mx-auto" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
          </svg>
          <p className="text-starlight/40 text-sm">No strategies loaded</p>
          <p className="text-starlight/25 text-xs">Waiting for HEAN backend connection...</p>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h2 className="text-lg font-semibold text-supernova">Tactical Center</h2>
          <p className="text-xs text-starlight/40">
            {strategies.filter((s) => s.enabled).length} active / {strategies.length} total strategies
          </p>
        </div>
      </motion.div>

      {/* Strategy Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {strategies.map((strategy, i) => (
          <StrategyCard
            key={strategy.strategy_id}
            strategy={strategy}
            signals={getStrategySignals(strategy.type)}
            delay={i * 0.08}
          />
        ))}
      </div>
    </div>
  );
}

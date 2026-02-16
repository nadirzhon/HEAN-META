"use client";

import { motion } from "framer-motion";
import clsx from "clsx";
import { AnimatedNumber } from "../ui/AnimatedNumber";

interface MetricCardProps {
  label: string;
  value: number | string;
  prefix?: string;
  suffix?: string;
  decimals?: number;
  colorize?: boolean;
  badgeColor?: string;
  index?: number;
}

function MetricCard({
  label,
  value,
  prefix = "",
  suffix = "",
  decimals = 0,
  colorize = false,
  badgeColor,
  index = 0,
}: MetricCardProps) {
  const isNumeric = typeof value === "number";

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05, duration: 0.3 }}
      style={{
        background: "rgba(28,28,49,0.5)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: "12px",
        padding: "16px 20px",
        flex: "1",
        minWidth: "140px",
      }}
    >
      <div style={{ color: "#C5C5DD", fontSize: "12px", marginBottom: "8px" }}>
        {label}
      </div>
      {isNumeric ? (
        <div style={{ fontSize: "24px", fontWeight: "bold" }}>
          <AnimatedNumber
            value={value}
            prefix={prefix}
            suffix={suffix}
            decimals={decimals}
            colorPositive={colorize ? "#00FF88" : "#FFFFFF"}
            colorNegative={colorize ? "#FF4466" : "#FFFFFF"}
            colorNeutral="#FFFFFF"
          />
        </div>
      ) : (
        <div
          style={{
            fontSize: "16px",
            fontWeight: "600",
            display: "inline-block",
            padding: "4px 12px",
            borderRadius: "6px",
            background: badgeColor
              ? `${badgeColor}20`
              : "rgba(255,255,255,0.1)",
            color: badgeColor || "#FFFFFF",
          }}
        >
          {value}
        </div>
      )}
    </motion.div>
  );
}

interface MetricRowProps {
  equity: number;
  dailyPnl: number;
  winRate: number;
  riskState: string;
  activePositions: number;
  signalsPerMin: number;
}

/**
 * Horizontal row of key metrics in glass cards.
 * Each metric animates when values change.
 */
export function MetricRow({
  equity,
  dailyPnl,
  winRate,
  riskState,
  activePositions,
  signalsPerMin,
}: MetricRowProps) {
  // Map risk states to colors
  const getRiskStateColor = (state: string): string => {
    const stateUpper = state.toUpperCase();
    if (stateUpper === "NORMAL") return "#00FF88";
    if (stateUpper === "SOFT_BRAKE") return "#FFD600";
    if (stateUpper === "QUARANTINE") return "#FF8800";
    if (stateUpper === "HARD_STOP") return "#FF4466";
    return "#C5C5DD";
  };

  return (
    <div
      style={{
        display: "flex",
        gap: "16px",
        flexWrap: "wrap",
        marginBottom: "24px",
      }}
    >
      <MetricCard
        label="Total Equity"
        value={equity}
        prefix="$"
        decimals={2}
        index={0}
      />
      <MetricCard
        label="Daily PnL"
        value={dailyPnl}
        prefix={dailyPnl >= 0 ? "+$" : "$"}
        decimals={2}
        colorize={true}
        index={1}
      />
      <MetricCard
        label="Win Rate"
        value={winRate}
        suffix="%"
        decimals={1}
        index={2}
      />
      <MetricCard
        label="Risk State"
        value={riskState}
        badgeColor={getRiskStateColor(riskState)}
        index={3}
      />
      <MetricCard
        label="Active Positions"
        value={activePositions}
        index={4}
      />
      <MetricCard
        label="Signals/min"
        value={signalsPerMin}
        decimals={1}
        index={5}
      />
    </div>
  );
}

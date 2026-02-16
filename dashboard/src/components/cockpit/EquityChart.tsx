"use client";

import { motion } from "framer-motion";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { AnimatedNumber } from "../ui/AnimatedNumber";

interface EquityDataPoint {
  timestamp: number;
  equity: number;
  pnl: number;
}

interface EquityChartProps {
  data: EquityDataPoint[];
  currentEquity: number;
  initialCapital: number;
}

/**
 * Equity curve chart with animated area gradient.
 * Shows portfolio value over time with smooth transitions.
 */
export function EquityChart({
  data,
  currentEquity,
  initialCapital,
}: EquityChartProps) {
  const dailyChange = currentEquity - initialCapital;
  const dailyChangePercent =
    initialCapital > 0 ? (dailyChange / initialCapital) * 100 : 0;

  // Custom tooltip component
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload.length) return null;

    const point = payload[0].payload as EquityDataPoint;
    const date = new Date(point.timestamp).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
    const change =
      initialCapital > 0 ? ((point.equity - initialCapital) / initialCapital) * 100 : 0;

    return (
      <div
        style={{
          background: "rgba(28,28,49,0.95)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "8px",
          padding: "12px",
          color: "#C5C5DD",
        }}
      >
        <div style={{ fontSize: "12px", marginBottom: "4px" }}>{date}</div>
        <div style={{ fontSize: "16px", fontWeight: "bold", color: "#FFFFFF" }}>
          ${point.equity.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </div>
        <div
          style={{
            fontSize: "14px",
            color: change >= 0 ? "#00FF88" : "#FF4466",
          }}
        >
          {change >= 0 ? "+" : ""}
          {change.toFixed(2)}%
        </div>
      </div>
    );
  };

  // Empty state
  if (data.length === 0) {
    return (
      <div
        style={{
          background: "rgba(28,28,49,0.5)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "16px",
          padding: "24px",
          height: "300px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: "16px",
        }}
      >
        <motion.div
          animate={{ opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 2, repeat: Infinity }}
          style={{ fontSize: "48px" }}
        >
          📊
        </motion.div>
        <div style={{ color: "#C5C5DD", fontSize: "14px" }}>
          No equity data yet
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        background: "rgba(28,28,49,0.5)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: "16px",
        padding: "24px",
      }}
    >
      {/* Header with current equity */}
      <div style={{ marginBottom: "16px" }}>
        <div style={{ color: "#C5C5DD", fontSize: "12px", marginBottom: "4px" }}>
          Total Equity
        </div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "12px" }}>
          <div style={{ fontSize: "32px", fontWeight: "bold", color: "#FFFFFF" }}>
            <AnimatedNumber
              value={currentEquity}
              prefix="$"
              decimals={2}
              colorPositive="#FFFFFF"
              colorNegative="#FFFFFF"
              colorNeutral="#FFFFFF"
            />
          </div>
          <div
            style={{
              fontSize: "16px",
              color: dailyChange >= 0 ? "#00FF88" : "#FF4466",
            }}
          >
            <AnimatedNumber
              value={dailyChange}
              prefix={dailyChange >= 0 ? "+$" : "-$"}
              decimals={2}
            />
            <span style={{ marginLeft: "8px" }}>
              ({dailyChangePercent >= 0 ? "+" : ""}
              {dailyChangePercent.toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={data}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#A45BFF" stopOpacity={0.6} />
                <stop offset="100%" stopColor="#A45BFF" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(255,255,255,0.05)"
              vertical={false}
            />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(ts) =>
                new Date(ts).toLocaleTimeString("en-US", {
                  hour: "2-digit",
                  minute: "2-digit",
                })
              }
              stroke="#C5C5DD"
              fontSize={12}
              tickLine={false}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
            />
            <YAxis
              stroke="#C5C5DD"
              fontSize={12}
              tickLine={false}
              axisLine={{ stroke: "rgba(255,255,255,0.1)" }}
              tickFormatter={(value) => `$${value.toLocaleString()}`}
            />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="equity"
              stroke="#A45BFF"
              strokeWidth={2}
              fill="url(#equityGradient)"
              animationDuration={1000}
              animationBegin={0}
            />
          </AreaChart>
        </ResponsiveContainer>
      </motion.div>
    </div>
  );
}

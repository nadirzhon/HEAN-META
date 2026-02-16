"use client";

import { motion } from "framer-motion";
import { PieChart, Pie, Cell, ResponsiveContainer, Legend } from "recharts";

interface AssetAllocation {
  symbol: string;
  value: number;
  percentage: number;
}

interface AssetDonutChartProps {
  data: AssetAllocation[];
  totalValue: number;
}

const COLORS = ["#A45BFF", "#00D4FF", "#00FF88", "#FF8800", "#FF4466"];

/**
 * Donut chart showing portfolio allocation by asset.
 * Animated sector growth on data changes.
 */
export function AssetDonutChart({ data, totalValue }: AssetDonutChartProps) {
  // Empty state
  if (data.length === 0) {
    return (
      <div
        style={{
          background: "rgba(28,28,49,0.5)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "16px",
          padding: "24px",
          height: "400px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: "16px",
        }}
      >
        <motion.svg
          width="120"
          height="120"
          viewBox="0 0 120 120"
          animate={{ rotate: 360 }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        >
          <circle
            cx="60"
            cy="60"
            r="50"
            fill="none"
            stroke="rgba(255,255,255,0.1)"
            strokeWidth="20"
          />
        </motion.svg>
        <div style={{ color: "#C5C5DD", fontSize: "14px" }}>
          No positions
        </div>
      </div>
    );
  }

  // Custom legend component
  const renderLegend = (props: any) => {
    return (
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: "8px",
          marginTop: "16px",
        }}
      >
        {data.map((entry, index) => (
          <motion.div
            key={entry.symbol}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              padding: "8px 12px",
              background: "rgba(255,255,255,0.03)",
              borderRadius: "8px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <div
                style={{
                  width: "12px",
                  height: "12px",
                  borderRadius: "2px",
                  background: COLORS[index % COLORS.length],
                }}
              />
              <span style={{ color: "#C5C5DD", fontSize: "14px" }}>
                {entry.symbol}
              </span>
            </div>
            <div style={{ display: "flex", gap: "12px", alignItems: "center" }}>
              <span style={{ color: "#FFFFFF", fontSize: "14px", fontWeight: "500" }}>
                ${entry.value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
              </span>
              <span style={{ color: COLORS[index % COLORS.length], fontSize: "12px" }}>
                {entry.percentage.toFixed(1)}%
              </span>
            </div>
          </motion.div>
        ))}
      </div>
    );
  };

  return (
    <div
      style={{
        background: "rgba(28,28,49,0.5)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: "16px",
        padding: "24px",
      }}
    >
      <div style={{ color: "#C5C5DD", fontSize: "12px", marginBottom: "16px" }}>
        Portfolio Allocation
      </div>

      <ResponsiveContainer width="100%" height={250}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius="60%"
            outerRadius="80%"
            dataKey="value"
            animationDuration={800}
            animationBegin={0}
          >
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={COLORS[index % COLORS.length]}
                stroke="none"
              />
            ))}
          </Pie>
          {/* Center label */}
          <text
            x="50%"
            y="50%"
            textAnchor="middle"
            dominantBaseline="middle"
            style={{ fill: "#FFFFFF", fontSize: "24px", fontWeight: "bold" }}
          >
            ${totalValue.toLocaleString("en-US", { maximumFractionDigits: 0 })}
          </text>
        </PieChart>
      </ResponsiveContainer>

      {renderLegend({})}
    </div>
  );
}

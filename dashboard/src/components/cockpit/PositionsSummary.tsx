"use client";

import { motion } from "framer-motion";
import { AnimatedNumber } from "../ui/AnimatedNumber";

interface Position {
  position_id: string;
  symbol: string;
  side: "LONG" | "SHORT" | string;
  size: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
}

interface PositionsSummaryProps {
  positions: Position[];
}

/**
 * Compact table showing open positions with PnL.
 * Rows animate in on data load.
 */
export function PositionsSummary({ positions }: PositionsSummaryProps) {
  // Empty state
  if (positions.length === 0) {
    return (
      <div
        style={{
          background: "rgba(28,28,49,0.5)",
          border: "1px solid rgba(255,255,255,0.1)",
          borderRadius: "16px",
          padding: "24px",
          height: "100%",
          minHeight: "400px",
          display: "flex",
          flexDirection: "column",
          justifyContent: "center",
          alignItems: "center",
          gap: "16px",
        }}
      >
        <motion.div
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
          style={{ fontSize: "48px" }}
        >
          📋
        </motion.div>
        <div style={{ color: "#C5C5DD", fontSize: "14px" }}>
          No open positions
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
        height: "100%",
        minHeight: "400px",
      }}
    >
      {/* Header */}
      <div
        style={{
          color: "#C5C5DD",
          fontSize: "14px",
          fontWeight: "600",
          marginBottom: "16px",
        }}
      >
        Open Positions ({positions.length})
      </div>

      {/* Table */}
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr
              style={{
                borderBottom: "1px solid rgba(255,255,255,0.1)",
              }}
            >
              <th
                style={{
                  textAlign: "left",
                  padding: "8px 12px",
                  color: "#C5C5DD",
                  fontSize: "11px",
                  fontWeight: "600",
                  textTransform: "uppercase",
                }}
              >
                Symbol
              </th>
              <th
                style={{
                  textAlign: "left",
                  padding: "8px 12px",
                  color: "#C5C5DD",
                  fontSize: "11px",
                  fontWeight: "600",
                  textTransform: "uppercase",
                }}
              >
                Side
              </th>
              <th
                style={{
                  textAlign: "right",
                  padding: "8px 12px",
                  color: "#C5C5DD",
                  fontSize: "11px",
                  fontWeight: "600",
                  textTransform: "uppercase",
                }}
              >
                Size
              </th>
              <th
                style={{
                  textAlign: "right",
                  padding: "8px 12px",
                  color: "#C5C5DD",
                  fontSize: "11px",
                  fontWeight: "600",
                  textTransform: "uppercase",
                }}
              >
                Entry
              </th>
              <th
                style={{
                  textAlign: "right",
                  padding: "8px 12px",
                  color: "#C5C5DD",
                  fontSize: "11px",
                  fontWeight: "600",
                  textTransform: "uppercase",
                }}
              >
                PnL
              </th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position, index) => (
              <motion.tr
                key={position.position_id}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05, duration: 0.3 }}
                style={{
                  borderBottom: "1px solid rgba(255,255,255,0.05)",
                }}
              >
                {/* Symbol */}
                <td
                  style={{
                    padding: "12px",
                    color: "#FFFFFF",
                    fontSize: "14px",
                    fontWeight: "600",
                  }}
                >
                  {position.symbol}
                </td>

                {/* Side */}
                <td style={{ padding: "12px" }}>
                  <div
                    style={{
                      display: "inline-block",
                      padding: "4px 10px",
                      borderRadius: "4px",
                      fontSize: "11px",
                      fontWeight: "600",
                      background:
                        position.side === "LONG"
                          ? "rgba(0, 255, 136, 0.2)"
                          : "rgba(255, 68, 102, 0.2)",
                      color: position.side === "LONG" ? "#00FF88" : "#FF4466",
                    }}
                  >
                    {position.side}
                  </div>
                </td>

                {/* Size */}
                <td
                  style={{
                    padding: "12px",
                    textAlign: "right",
                    color: "#C5C5DD",
                    fontSize: "13px",
                    fontFamily: "monospace",
                  }}
                >
                  {position.size.toLocaleString("en-US", {
                    minimumFractionDigits: 4,
                    maximumFractionDigits: 4,
                  })}
                </td>

                {/* Entry Price */}
                <td
                  style={{
                    padding: "12px",
                    textAlign: "right",
                    color: "#C5C5DD",
                    fontSize: "13px",
                    fontFamily: "monospace",
                  }}
                >
                  $
                  {position.entry_price.toLocaleString("en-US", {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </td>

                {/* PnL */}
                <td
                  style={{
                    padding: "12px",
                    textAlign: "right",
                    fontSize: "14px",
                    fontWeight: "600",
                    fontFamily: "monospace",
                  }}
                >
                  <AnimatedNumber
                    value={position.unrealized_pnl}
                    prefix={position.unrealized_pnl >= 0 ? "+$" : "$"}
                    decimals={2}
                  />
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

"use client";

import { useRef } from "react";
import { motion, useInView } from "framer-motion";
import Heading from "@/components/ui/Heading";
import { useExplainability } from "@/hooks/useExplainability";

/* ------------------------------------------------------------------ */
/*  Node & connection definitions for the HEAN architecture diagram   */
/* ------------------------------------------------------------------ */

interface DiagramNode {
  id: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  color: "axon" | "stream" | "both";
}

interface Connection {
  from: string;
  to: string;
}

const NODES: DiagramNode[] = [
  // Row 1 — source
  { id: "market", label: "Market Data", x: 60, y: 40, w: 140, h: 48, color: "stream" },

  // Row 2 — hub
  { id: "bus", label: "EventBus", x: 310, y: 40, w: 140, h: 48, color: "both" },

  // Row 3 — processing fan-out
  { id: "strategies", label: "Strategies", x: 160, y: 160, w: 130, h: 48, color: "axon" },
  { id: "risk", label: "Risk Engine", x: 320, y: 160, w: 130, h: 48, color: "axon" },
  { id: "physics", label: "Physics", x: 480, y: 160, w: 130, h: 48, color: "stream" },

  // Row 4 — execution
  { id: "execution", label: "Execution", x: 310, y: 280, w: 140, h: 48, color: "axon" },

  // Row 5 — exchange
  { id: "bybit", label: "Bybit Exchange", x: 310, y: 390, w: 140, h: 48, color: "stream" },
];

const CONNECTIONS: Connection[] = [
  { from: "market", to: "bus" },
  { from: "bus", to: "strategies" },
  { from: "bus", to: "risk" },
  { from: "bus", to: "physics" },
  { from: "strategies", to: "execution" },
  { from: "risk", to: "execution" },
  { from: "execution", to: "bybit" },
];

/* ------------------------------------------------------------------ */
/*  Helpers                                                           */
/* ------------------------------------------------------------------ */

function nodeCenter(node: DiagramNode): { cx: number; cy: number } {
  return { cx: node.x + node.w / 2, cy: node.y + node.h / 2 };
}

function nodeById(id: string): DiagramNode {
  return NODES.find((n) => n.id === id)!;
}

/** Determine which edge of the target node the line should connect to. */
function connectionPath(conn: Connection): { x1: number; y1: number; x2: number; y2: number } {
  const from = nodeCenter(nodeById(conn.from));
  const to = nodeCenter(nodeById(conn.to));
  const toNode = nodeById(conn.to);
  const fromNode = nodeById(conn.from);

  let x1 = from.cx;
  let y1 = from.cy;
  let x2 = to.cx;
  let y2 = to.cy;

  // Exit from bottom of source if target is below, otherwise from right edge
  if (to.cy > from.cy + 10) {
    y1 = fromNode.y + fromNode.h;
    y2 = toNode.y;
  } else {
    x1 = fromNode.x + fromNode.w;
    x2 = toNode.x;
  }

  return { x1, y1, x2, y2 };
}

const BORDER_COLORS: Record<DiagramNode["color"], string> = {
  axon: "#A45BFF",
  stream: "#00D4FF",
  both: "#A45BFF",
};

/* ------------------------------------------------------------------ */
/*  Sub-components                                                    */
/* ------------------------------------------------------------------ */

function AnimatedNode({
  node,
  index,
  isInView,
}: {
  node: DiagramNode;
  index: number;
  isInView: boolean;
}) {
  const borderColor = BORDER_COLORS[node.color];
  const glowColor =
    node.color === "stream"
      ? "rgba(0, 212, 255, 0.25)"
      : "rgba(164, 91, 255, 0.25)";

  return (
    <motion.g
      initial={{ opacity: 0, y: 12 }}
      animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 12 }}
      transition={{ duration: 0.5, delay: 0.15 + index * 0.1, ease: "easeOut" }}
    >
      {/* Glow filter */}
      <defs>
        <filter id={`glow-${node.id}`} x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feFlood floodColor={glowColor} result="color" />
          <feComposite in="color" in2="blur" operator="in" result="glow" />
          <feMerge>
            <feMergeNode in="glow" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Node rectangle */}
      <rect
        x={node.x}
        y={node.y}
        width={node.w}
        height={node.h}
        rx={10}
        ry={10}
        fill="rgba(28, 28, 49, 0.6)"
        stroke={borderColor}
        strokeWidth={1.5}
        filter={`url(#glow-${node.id})`}
      />

      {/* Label */}
      <text
        x={node.x + node.w / 2}
        y={node.y + node.h / 2 + 1}
        textAnchor="middle"
        dominantBaseline="central"
        fill="#FFFFFF"
        fontSize={14}
        fontWeight={600}
        fontFamily="system-ui, sans-serif"
      >
        {node.label}
      </text>
    </motion.g>
  );
}

function AnimatedConnection({
  conn,
  index,
  isInView,
}: {
  conn: Connection;
  index: number;
  isInView: boolean;
}) {
  const { x1, y1, x2, y2 } = connectionPath(conn);
  const gradientId = `line-grad-${conn.from}-${conn.to}`;

  // Compute a simple path with an optional bend for non-straight lines
  const isStraight = Math.abs(x1 - x2) < 2 || Math.abs(y1 - y2) < 2;
  const pathD = isStraight
    ? `M${x1},${y1} L${x2},${y2}`
    : `M${x1},${y1} L${x1},${(y1 + y2) / 2} L${x2},${(y1 + y2) / 2} L${x2},${y2}`;

  // Rough path length for stroke-dasharray
  const estimatedLength = isStraight
    ? Math.hypot(x2 - x1, y2 - y1)
    : Math.abs(y2 - y1) / 2 + Math.abs(x2 - x1) + Math.abs(y2 - y1) / 2;

  return (
    <g>
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#A45BFF" stopOpacity={0.8} />
          <stop offset="100%" stopColor="#00D4FF" stopOpacity={0.8} />
        </linearGradient>
      </defs>

      {/* Glow line underneath */}
      <motion.path
        d={pathD}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={3}
        strokeLinecap="round"
        strokeLinejoin="round"
        opacity={0.25}
        initial={{ pathLength: 0, opacity: 0 }}
        animate={
          isInView
            ? { pathLength: 1, opacity: 0.25 }
            : { pathLength: 0, opacity: 0 }
        }
        transition={{ duration: 0.7, delay: 0.5 + index * 0.12, ease: "easeInOut" }}
        style={{
          strokeDasharray: estimatedLength,
          filter: "blur(4px)",
        }}
      />

      {/* Sharp line on top */}
      <motion.path
        d={pathD}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={1.5}
        strokeLinecap="round"
        strokeLinejoin="round"
        initial={{ pathLength: 0, opacity: 0 }}
        animate={
          isInView ? { pathLength: 1, opacity: 1 } : { pathLength: 0, opacity: 0 }
        }
        transition={{ duration: 0.7, delay: 0.5 + index * 0.12, ease: "easeInOut" }}
      />

      {/* Arrowhead dot at the end */}
      <motion.circle
        cx={x2}
        cy={y2}
        r={3}
        fill="#00D4FF"
        initial={{ opacity: 0, scale: 0 }}
        animate={isInView ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0 }}
        transition={{ duration: 0.3, delay: 1.0 + index * 0.12 }}
      />
    </g>
  );
}

/* ------------------------------------------------------------------ */
/*  Main section                                                      */
/* ------------------------------------------------------------------ */

export default function Technology() {
  const sectionRef = useRef<HTMLElement>(null);
  const isInView = useInView(sectionRef, { once: true, margin: "-80px" });
  const { text } = useExplainability();

  return (
    <section
      ref={sectionRef}
      id="technology"
      className="relative py-32 md:py-40 overflow-hidden"
    >
      {/* Subtle background radial glow */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 50% 40%, rgba(164,91,255,0.06) 0%, transparent 60%)",
        }}
      />

      <div className="relative mx-auto max-w-5xl px-6 text-center">
        {/* Section heading */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          animate={isInView ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 }}
          transition={{ duration: 0.6 }}
          className="mb-16"
        >
          <Heading as="h2" gradient>
            {text("How It Works", "System Architecture")}
          </Heading>
        </motion.div>

        {/* SVG Diagram */}
        <div className="mx-auto w-full max-w-[800px] overflow-x-auto">
          <svg
            viewBox="0 0 760 470"
            className="w-full h-auto min-w-[600px]"
            xmlns="http://www.w3.org/2000/svg"
            preserveAspectRatio="xMidYMid meet"
          >
            {/* Connections drawn first (behind nodes) */}
            {CONNECTIONS.map((conn, i) => (
              <AnimatedConnection
                key={`${conn.from}-${conn.to}`}
                conn={conn}
                index={i}
                isInView={isInView}
              />
            ))}

            {/* Nodes */}
            {NODES.map((node, i) => (
              <AnimatedNode
                key={node.id}
                node={node}
                index={i}
                isInView={isInView}
              />
            ))}

            {/* Flow labels between rows */}
            <motion.text
              x={380}
              y={128}
              textAnchor="middle"
              fill="#C5C5DD"
              fontSize={11}
              fontFamily="system-ui, sans-serif"
              fontStyle="italic"
              initial={{ opacity: 0 }}
              animate={isInView ? { opacity: 0.6 } : { opacity: 0 }}
              transition={{ delay: 1.4, duration: 0.5 }}
            >
              {text("spreading out", "fan-out processing")}
            </motion.text>

            <motion.text
              x={380}
              y={248}
              textAnchor="middle"
              fill="#C5C5DD"
              fontSize={11}
              fontFamily="system-ui, sans-serif"
              fontStyle="italic"
              initial={{ opacity: 0 }}
              animate={isInView ? { opacity: 0.6 } : { opacity: 0 }}
              transition={{ delay: 1.6, duration: 0.5 }}
            >
              {text("coming together", "signal convergence")}
            </motion.text>

            <motion.text
              x={380}
              y={358}
              textAnchor="middle"
              fill="#C5C5DD"
              fontSize={11}
              fontFamily="system-ui, sans-serif"
              fontStyle="italic"
              initial={{ opacity: 0 }}
              animate={isInView ? { opacity: 0.6 } : { opacity: 0 }}
              transition={{ delay: 1.8, duration: 0.5 }}
            >
              {text("sending orders", "order routing")}
            </motion.text>
          </svg>
        </div>

        {/* Legend */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={isInView ? { opacity: 1 } : { opacity: 0 }}
          transition={{ delay: 2.0, duration: 0.6 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-6 text-xs text-starlight/60"
        >
          <span className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-axon" />
            {text("Brain power", "Decision layer")}
          </span>
          <span className="flex items-center gap-2">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-stream" />
            {text("Information", "Data layer")}
          </span>
          <span className="flex items-center gap-2">
            <span
              className="inline-block h-0.5 w-6 rounded-full"
              style={{
                background: "linear-gradient(90deg, #A45BFF, #00D4FF)",
              }}
            />
            {text("Message flow", "Event flow")}
          </span>
        </motion.div>
      </div>
    </section>
  );
}

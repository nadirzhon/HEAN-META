"use client";

import React, { useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { SystemNode, Impulse } from "../../types/neuromap";
import { NEURO_COLORS } from "../../types/neuromap";

interface AnimatedConnectionProps {
  id: string;
  sourceNode: SystemNode;
  targetNode: SystemNode;
  impulses: Impulse[];
  active: boolean;
}

/**
 * Computes a smooth SVG path between the center-bottom of the source node
 * and the center-top of the target node, using a cubic bezier curve.
 * Special-cases horizontal connections (same y-band) to route via sides.
 */
function computePath(source: SystemNode, target: SystemNode): string {
  const sx = source.x + source.width / 2;
  const sy = source.y + source.height;
  const tx = target.x + target.width / 2;
  const ty = target.y;

  // If nodes are roughly at the same Y level, connect from sides
  const sameLevel = Math.abs(source.y - target.y) < 30;
  if (sameLevel) {
    const leftToRight = source.x < target.x;
    const startX = leftToRight ? source.x + source.width : source.x;
    const startY = source.y + source.height / 2;
    const endX = leftToRight ? target.x : target.x + target.width;
    const endY = target.y + target.height / 2;
    const midX = (startX + endX) / 2;
    return `M ${startX} ${startY} C ${midX} ${startY}, ${midX} ${endY}, ${endX} ${endY}`;
  }

  // Default: top-to-bottom bezier
  const dy = ty - sy;
  const cpOffset = Math.max(30, Math.abs(dy) * 0.4);
  return `M ${sx} ${sy} C ${sx} ${sy + cpOffset}, ${tx} ${ty - cpOffset}, ${tx} ${ty}`;
}

/**
 * Computes a point along an SVG path at a given progress (0..1).
 * Uses the browser's SVGPathElement.getPointAtLength API when available,
 * otherwise falls back to a linear interpolation of the endpoints.
 */
function getPointOnPath(
  pathD: string,
  progress: number
): { x: number; y: number } | null {
  if (typeof document === "undefined") return null;

  try {
    const svgNS = "http://www.w3.org/2000/svg";
    const pathEl = document.createElementNS(svgNS, "path");
    pathEl.setAttribute("d", pathD);
    const totalLength = pathEl.getTotalLength();
    const pt = pathEl.getPointAtLength(progress * totalLength);
    return { x: pt.x, y: pt.y };
  } catch {
    return null;
  }
}

/**
 * AnimatedConnection renders a curved SVG path between two nodes with:
 * - Gradient stroke (axon purple --> stream cyan)
 * - Dashed "flowing" animation on the stroke
 * - Impulse dots that travel along the path when events fire
 */
export const AnimatedConnection: React.FC<AnimatedConnectionProps> = ({
  id,
  sourceNode,
  targetNode,
  impulses,
  active,
}) => {
  const gradientId = `conn-grad-${id}`;
  const glowFilterId = `conn-glow-${id}`;
  const pathD = useMemo(
    () => computePath(sourceNode, targetNode),
    [sourceNode, targetNode]
  );

  return (
    <g>
      <defs>
        {/* Gradient for the connection line */}
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={NEURO_COLORS.axon} stopOpacity="0.8" />
          <stop
            offset="100%"
            stopColor={NEURO_COLORS.stream}
            stopOpacity="0.8"
          />
        </linearGradient>

        {/* Glow filter for impulse dots */}
        <filter
          id={glowFilterId}
          x="-100%"
          y="-100%"
          width="300%"
          height="300%"
        >
          <feGaussianBlur in="SourceGraphic" stdDeviation="3" />
        </filter>

        {/* Radial gradient for impulse dots */}
        <radialGradient id={`imp-grad-${id}`}>
          <stop offset="0%" stopColor={NEURO_COLORS.supernova} stopOpacity="1" />
          <stop offset="40%" stopColor={NEURO_COLORS.stream} stopOpacity="0.9" />
          <stop offset="100%" stopColor={NEURO_COLORS.axon} stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Base connection line */}
      <motion.path
        d={pathD}
        fill="none"
        stroke={`url(#${gradientId})`}
        strokeWidth={active ? 1.8 : 1}
        strokeOpacity={active ? 0.5 : 0.15}
        strokeLinecap="round"
      />

      {/* Flowing dash overlay (only when active) */}
      {active && (
        <motion.path
          d={pathD}
          fill="none"
          stroke={`url(#${gradientId})`}
          strokeWidth={1.2}
          strokeOpacity={0.35}
          strokeLinecap="round"
          strokeDasharray="6 10"
          animate={{
            strokeDashoffset: [0, -32],
          }}
          transition={{
            duration: 1.5,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      )}

      {/* Impulse dots traveling along the path */}
      <AnimatePresence>
        {impulses.map((impulse) => (
          <ImpulseDot
            key={impulse.id}
            pathD={pathD}
            gradientId={`imp-grad-${id}`}
            glowFilterId={glowFilterId}
          />
        ))}
      </AnimatePresence>
    </g>
  );
};

/**
 * ImpulseDot is a small glowing circle that animates along a path
 * from start to end over 0.8 seconds.
 */
const ImpulseDot: React.FC<{
  pathD: string;
  gradientId: string;
  glowFilterId: string;
}> = ({ pathD, gradientId, glowFilterId }) => {
  // Pre-compute a set of positions along the path for the animation
  const positions = useMemo(() => {
    const steps = 20;
    const points: { x: number; y: number }[] = [];
    for (let i = 0; i <= steps; i++) {
      const pt = getPointOnPath(pathD, i / steps);
      if (pt) points.push(pt);
    }
    return points;
  }, [pathD]);

  if (positions.length < 2) return null;

  const xValues = positions.map((p) => p.x);
  const yValues = positions.map((p) => p.y);

  return (
    <motion.g
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.1 }}
    >
      {/* Glow behind the dot */}
      <motion.circle
        r={8}
        fill={`url(#${gradientId})`}
        filter={`url(#${glowFilterId})`}
        opacity={0.6}
        animate={{
          cx: xValues,
          cy: yValues,
        }}
        transition={{
          duration: 0.8,
          ease: "easeInOut",
        }}
      />

      {/* Core dot */}
      <motion.circle
        r={3.5}
        fill={NEURO_COLORS.supernova}
        opacity={0.95}
        animate={{
          cx: xValues,
          cy: yValues,
        }}
        transition={{
          duration: 0.8,
          ease: "easeInOut",
        }}
      />
    </motion.g>
  );
};

export default AnimatedConnection;

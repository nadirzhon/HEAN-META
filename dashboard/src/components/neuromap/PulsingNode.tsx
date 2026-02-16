"use client";

import React, { useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { NodeColor, NodeStatus } from "../../types/neuromap";
import { NEURO_COLORS, resolveColor } from "../../types/neuromap";

interface PulsingNodeProps {
  id: string;
  label: string;
  x: number;
  y: number;
  width: number;
  height: number;
  color: NodeColor;
  status: NodeStatus;
  eventCount: number;
  onClick?: (nodeId: string) => void;
}

const STATUS_COLORS: Record<NodeStatus, string> = {
  active: NEURO_COLORS.statusActive,
  degraded: NEURO_COLORS.statusDegraded,
  error: NEURO_COLORS.statusError,
  idle: NEURO_COLORS.statusIdle,
};

/**
 * PulsingNode renders a single system component as a rounded rectangle
 * in the SVG graph. It has:
 * - A subtle breathing glow animation when idle
 * - A bright pulse when events are received (eventCount > 0)
 * - A status indicator dot
 * - An event counter badge
 */
export const PulsingNode: React.FC<PulsingNodeProps> = ({
  id,
  label,
  x,
  y,
  width,
  height,
  color,
  status,
  eventCount,
  onClick,
}) => {
  const hexColor = resolveColor(color);
  const statusColor = STATUS_COLORS[status];
  const filterId = `glow-${id}`;
  const gradientId = `node-grad-${id}`;

  const handleClick = useCallback(() => {
    onClick?.(id);
  }, [id, onClick]);

  const isActive = eventCount > 0;

  return (
    <g
      style={{ cursor: "pointer" }}
      onClick={handleClick}
      role="button"
      aria-label={`${label} node, status: ${status}`}
    >
      {/* Glow filter */}
      <defs>
        <filter id={filterId} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur
            in="SourceGraphic"
            stdDeviation={isActive ? "6" : "3"}
          />
        </filter>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor={hexColor} stopOpacity="0.25" />
          <stop offset="100%" stopColor={hexColor} stopOpacity="0.08" />
        </linearGradient>
      </defs>

      {/* Glow layer (behind the rect) */}
      <motion.rect
        x={x}
        y={y}
        rx={12}
        ry={12}
        width={width}
        height={height}
        fill="none"
        stroke={hexColor}
        strokeWidth={2}
        filter={`url(#${filterId})`}
        animate={
          isActive
            ? {
                opacity: [0.3, 0.9, 0.3],
                scale: [1, 1.08, 1],
              }
            : {
                opacity: [0.4, 0.7, 0.4],
                scale: [1, 1.02, 1],
              }
        }
        transition={
          isActive
            ? { duration: 0.5, ease: "easeOut" }
            : { duration: 3, repeat: Infinity, ease: "easeInOut" }
        }
        style={{ transformOrigin: `${x + width / 2}px ${y + height / 2}px` }}
      />

      {/* Main rectangle */}
      <motion.rect
        x={x}
        y={y}
        rx={12}
        ry={12}
        width={width}
        height={height}
        fill={`url(#${gradientId})`}
        stroke={hexColor}
        strokeWidth={1.5}
        strokeOpacity={0.6}
        animate={
          isActive
            ? {
                strokeOpacity: [0.6, 1, 0.6],
              }
            : {
                strokeOpacity: [0.3, 0.6, 0.3],
              }
        }
        transition={
          isActive
            ? { duration: 0.5, ease: "easeOut" }
            : { duration: 3, repeat: Infinity, ease: "easeInOut" }
        }
      />

      {/* Glass background fill */}
      <rect
        x={x + 1}
        y={y + 1}
        rx={11}
        ry={11}
        width={width - 2}
        height={height - 2}
        fill={NEURO_COLORS.glassBg}
      />

      {/* Label */}
      <text
        x={x + width / 2}
        y={y + height / 2 + 1}
        textAnchor="middle"
        dominantBaseline="central"
        fill={NEURO_COLORS.supernova}
        fontSize={13}
        fontFamily="'Inter', 'SF Pro Display', system-ui, sans-serif"
        fontWeight={600}
        letterSpacing="0.02em"
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        {label}
      </text>

      {/* Status indicator dot */}
      <motion.circle
        cx={x + 14}
        cy={y + 14}
        r={4}
        fill={statusColor}
        animate={
          status === "active"
            ? { opacity: [0.7, 1, 0.7] }
            : status === "error"
              ? { opacity: [0.4, 1, 0.4] }
              : {}
        }
        transition={
          status === "active"
            ? { duration: 2, repeat: Infinity, ease: "easeInOut" }
            : status === "error"
              ? { duration: 0.6, repeat: Infinity, ease: "easeInOut" }
              : {}
        }
      />

      {/* Event counter badge */}
      <AnimatePresence>
        {eventCount > 0 && (
          <motion.g
            initial={{ opacity: 0, scale: 0.5 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.5 }}
            transition={{ duration: 0.2 }}
          >
            <rect
              x={x + width - 32}
              y={y - 8}
              rx={8}
              ry={8}
              width={36}
              height={18}
              fill={hexColor}
              fillOpacity={0.85}
            />
            <text
              x={x + width - 14}
              y={y + 2}
              textAnchor="middle"
              dominantBaseline="central"
              fill={NEURO_COLORS.supernova}
              fontSize={10}
              fontFamily="'JetBrains Mono', 'SF Mono', monospace"
              fontWeight={700}
              style={{ pointerEvents: "none", userSelect: "none" }}
            >
              {eventCount}/s
            </text>
          </motion.g>
        )}
      </AnimatePresence>
    </g>
  );
};

export default PulsingNode;

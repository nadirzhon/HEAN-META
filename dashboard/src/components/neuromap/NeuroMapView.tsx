"use client";

import React, { useCallback, useMemo } from "react";
import { motion } from "framer-motion";
import type { SystemNode, NodeConnection, Impulse } from "../../types/neuromap";
import { NEURO_COLORS } from "../../types/neuromap";
import { PulsingNode } from "./PulsingNode";
import { AnimatedConnection } from "./AnimatedConnection";

interface NeuroMapViewProps {
  nodes: SystemNode[];
  connections: NodeConnection[];
  impulses: Impulse[];
  onNodeClick?: (nodeId: string) => void;
}

/**
 * NeuroMapView is the main SVG-based interactive node graph showing
 * the HEAN system architecture with real-time animated data flow.
 *
 * - 800x600 viewBox, scales responsively via preserveAspectRatio
 * - Hierarchical layout matching HEAN's signal chain
 * - Animated connections with traveling impulse dots
 * - Pulsing nodes with status indicators
 */
export const NeuroMapView: React.FC<NeuroMapViewProps> = ({
  nodes,
  connections,
  impulses,
  onNodeClick,
}) => {
  // Build a lookup map for nodes by id
  const nodeMap = useMemo(() => {
    const map = new Map<string, SystemNode>();
    for (const node of nodes) {
      map.set(node.id, node);
    }
    return map;
  }, [nodes]);

  // Group impulses by connection id
  const impulsesByConnection = useMemo(() => {
    const map = new Map<string, Impulse[]>();
    for (const imp of impulses) {
      const list = map.get(imp.connectionId) || [];
      list.push(imp);
      map.set(imp.connectionId, list);
    }
    return map;
  }, [impulses]);

  const handleNodeClick = useCallback(
    (nodeId: string) => {
      onNodeClick?.(nodeId);
    },
    [onNodeClick]
  );

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        position: "relative",
      }}
    >
      <svg
        viewBox="0 0 800 570"
        preserveAspectRatio="xMidYMid meet"
        style={{
          width: "100%",
          height: "100%",
          overflow: "visible",
        }}
      >
        {/* Global defs */}
        <defs>
          {/* Subtle grid pattern for the background */}
          <pattern
            id="neuro-grid"
            width="40"
            height="40"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke={NEURO_COLORS.glassBorder}
              strokeWidth="0.5"
              opacity="0.3"
            />
          </pattern>

          {/* Central glow for the EventBus area */}
          <radialGradient id="eventbus-aura" cx="50%" cy="25%" r="40%">
            <stop offset="0%" stopColor={NEURO_COLORS.axon} stopOpacity="0.06" />
            <stop
              offset="50%"
              stopColor={NEURO_COLORS.stream}
              stopOpacity="0.03"
            />
            <stop offset="100%" stopColor="transparent" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Background grid */}
        <rect width="800" height="570" fill="url(#neuro-grid)" opacity="0.5" />

        {/* Central aura */}
        <motion.ellipse
          cx="400"
          cy="200"
          rx="300"
          ry="200"
          fill="url(#eventbus-aura)"
          animate={{
            rx: [290, 310, 290],
            ry: [190, 210, 190],
          }}
          transition={{
            duration: 6,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />

        {/* Render connections first (behind nodes) */}
        {connections.map((conn) => {
          const sourceNode = nodeMap.get(conn.sourceId);
          const targetNode = nodeMap.get(conn.targetId);
          if (!sourceNode || !targetNode) return null;

          return (
            <AnimatedConnection
              key={conn.id}
              id={conn.id}
              sourceNode={sourceNode}
              targetNode={targetNode}
              impulses={impulsesByConnection.get(conn.id) || []}
              active={conn.active}
            />
          );
        })}

        {/* Render nodes */}
        {nodes.map((node) => (
          <PulsingNode
            key={node.id}
            id={node.id}
            label={node.label}
            x={node.x}
            y={node.y}
            width={node.width}
            height={node.height}
            color={node.color}
            status={node.status}
            eventCount={node.eventsPerSec}
            onClick={handleNodeClick}
          />
        ))}
      </svg>
    </div>
  );
};

export default NeuroMapView;

"use client";

import React, { useCallback, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { NEURO_COLORS, resolveColor } from "../../types/neuromap";
import type { NodeColor } from "../../types/neuromap";
import { NeuroMapView } from "../neuromap/NeuroMapView";
import { AgentStatePanel } from "../neuromap/AgentStatePanel";
import { useNeuroMapState } from "../neuromap/useNeuroMapState";

// ---------------------------------------------------------------------------
// Legend item
// ---------------------------------------------------------------------------

const LegendItem: React.FC<{
  color: NodeColor;
  label: string;
}> = ({ color, label }) => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      gap: 6,
    }}
  >
    <div
      style={{
        width: 10,
        height: 10,
        borderRadius: 3,
        backgroundColor: resolveColor(color),
        opacity: 0.8,
      }}
    />
    <span
      style={{
        color: NEURO_COLORS.starlight,
        fontSize: 11,
        fontFamily: "'Inter', system-ui, sans-serif",
        opacity: 0.7,
      }}
    >
      {label}
    </span>
  </div>
);

// ---------------------------------------------------------------------------
// Status legend item (for the status dots)
// ---------------------------------------------------------------------------

const StatusLegendItem: React.FC<{
  color: string;
  label: string;
}> = ({ color, label }) => (
  <div
    style={{
      display: "flex",
      alignItems: "center",
      gap: 5,
    }}
  >
    <div
      style={{
        width: 7,
        height: 7,
        borderRadius: "50%",
        backgroundColor: color,
      }}
    />
    <span
      style={{
        color: NEURO_COLORS.starlight,
        fontSize: 10,
        fontFamily: "'Inter', system-ui, sans-serif",
        opacity: 0.5,
      }}
    >
      {label}
    </span>
  </div>
);

// ---------------------------------------------------------------------------
// Node detail tooltip
// ---------------------------------------------------------------------------

const NodeDetailTooltip: React.FC<{
  nodeId: string | null;
  nodes: Array<{
    id: string;
    label: string;
    description?: string;
    status: string;
    eventsPerSec: number;
  }>;
  onClose: () => void;
}> = ({ nodeId, nodes, onClose }) => {
  const node = nodes.find((n) => n.id === nodeId);

  return (
    <AnimatePresence>
      {node && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 10 }}
          transition={{ duration: 0.2 }}
          style={{
            position: "absolute",
            bottom: 16,
            left: 16,
            right: 16,
            background: NEURO_COLORS.glassBg,
            border: `1px solid ${NEURO_COLORS.glassBorder}`,
            borderRadius: 10,
            padding: "12px 16px",
            backdropFilter: "blur(12px)",
            WebkitBackdropFilter: "blur(12px)",
            zIndex: 10,
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
            }}
          >
            <span
              style={{
                color: NEURO_COLORS.supernova,
                fontSize: 14,
                fontWeight: 700,
                fontFamily: "'Inter', system-ui, sans-serif",
              }}
            >
              {node.label}
            </span>
            <button
              onClick={onClose}
              style={{
                background: "none",
                border: "none",
                color: NEURO_COLORS.starlight,
                cursor: "pointer",
                fontSize: 16,
                opacity: 0.5,
                padding: "0 4px",
              }}
              aria-label="Close detail panel"
            >
              x
            </button>
          </div>
          {node.description && (
            <p
              style={{
                color: NEURO_COLORS.starlight,
                fontSize: 12,
                marginTop: 6,
                marginBottom: 0,
                opacity: 0.7,
                lineHeight: 1.4,
                fontFamily: "'Inter', system-ui, sans-serif",
              }}
            >
              {node.description}
            </p>
          )}
          <div
            style={{
              display: "flex",
              gap: 16,
              marginTop: 8,
              fontSize: 11,
              fontFamily: "'JetBrains Mono', monospace",
              color: NEURO_COLORS.starlight,
              opacity: 0.6,
            }}
          >
            <span>Status: {node.status}</span>
            <span>Events/s: {node.eventsPerSec}</span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

// ---------------------------------------------------------------------------
// Main tab component
// ---------------------------------------------------------------------------

/**
 * NeuroMapTab is the top-level integration component that brings together:
 * - The SVG node graph (65% width on desktop)
 * - The AgentStatePanel (35% width on desktop)
 * - Header with live event counter
 * - Legend showing node color meanings
 * - Responsive: stacks vertically on mobile
 */
export const NeuroMapTab: React.FC = () => {
  const state = useNeuroMapState();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const handleNodeClick = useCallback((nodeId: string) => {
    setSelectedNodeId((prev) => (prev === nodeId ? null : nodeId));
  }, []);

  const handleCloseDetail = useCallback(() => {
    setSelectedNodeId(null);
  }, []);

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        gap: 0,
        fontFamily: "'Inter', system-ui, sans-serif",
        color: NEURO_COLORS.starlight,
      }}
    >
      {/* Header bar */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "12px 20px",
          flexShrink: 0,
          borderBottom: `1px solid ${NEURO_COLORS.glassBorder}`,
        }}
      >
        {/* Left: title */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <svg
            width={20}
            height={20}
            viewBox="0 0 24 24"
            fill="none"
            stroke={NEURO_COLORS.axon}
            strokeWidth={2}
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83" />
          </svg>
          <h2
            style={{
              margin: 0,
              color: NEURO_COLORS.supernova,
              fontSize: 16,
              fontWeight: 800,
              letterSpacing: "0.06em",
              textTransform: "uppercase",
            }}
          >
            Neuro-Map
          </h2>
        </div>

        {/* Center: live event counter */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <motion.div
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              backgroundColor: NEURO_COLORS.statusActive,
            }}
            animate={{
              opacity: [0.5, 1, 0.5],
              scale: [0.9, 1.1, 0.9],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut",
            }}
          />
          <span
            style={{
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12,
              color: NEURO_COLORS.starlight,
              opacity: 0.8,
            }}
          >
            {state.totalEventsPerSec} events/s
          </span>
        </div>

        {/* Right: legend */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 14,
          }}
        >
          <LegendItem color="axon" label="Decision" />
          <LegendItem color="stream" label="Data" />
          <div
            style={{
              width: 1,
              height: 16,
              backgroundColor: NEURO_COLORS.glassBorder,
            }}
          />
          <StatusLegendItem color={NEURO_COLORS.statusActive} label="Active" />
          <StatusLegendItem color={NEURO_COLORS.statusDegraded} label="Degraded" />
          <StatusLegendItem color={NEURO_COLORS.statusError} label="Error" />
        </div>
      </div>

      {/* Main content area */}
      <div
        style={{
          flex: 1,
          display: "flex",
          gap: 0,
          minHeight: 0,
          overflow: "hidden",
        }}
        className="neuromap-layout"
      >
        {/* Node graph area */}
        <div
          style={{
            flex: "0 0 65%",
            position: "relative",
            padding: "8px 12px",
            minWidth: 0,
          }}
          className="neuromap-graph"
        >
          <NeuroMapView
            nodes={state.nodes}
            connections={state.connections}
            impulses={state.impulses}
            onNodeClick={handleNodeClick}
          />

          {/* Node detail tooltip overlay */}
          <NodeDetailTooltip
            nodeId={selectedNodeId}
            nodes={state.nodes}
            onClose={handleCloseDetail}
          />
        </div>

        {/* Divider */}
        <div
          style={{
            width: 1,
            backgroundColor: NEURO_COLORS.glassBorder,
            flexShrink: 0,
          }}
          className="neuromap-divider"
        />

        {/* Agent state panel */}
        <div
          style={{
            flex: "0 0 calc(35% - 1px)",
            padding: "8px 12px",
            minWidth: 0,
            minHeight: 0,
            overflow: "hidden",
          }}
          className="neuromap-agents"
        >
          <AgentStatePanel thoughts={state.agentThoughts} />
        </div>
      </div>

      {/* Responsive styles injected via a style tag */}
      <style>{`
        @media (max-width: 900px) {
          .neuromap-layout {
            flex-direction: column !important;
          }
          .neuromap-graph {
            flex: 0 0 auto !important;
            max-height: 55vh;
          }
          .neuromap-divider {
            width: auto !important;
            height: 1px !important;
          }
          .neuromap-agents {
            flex: 1 1 auto !important;
            min-height: 200px;
          }
        }
      `}</style>
    </div>
  );
};

export default NeuroMapTab;

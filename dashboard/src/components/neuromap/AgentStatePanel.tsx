"use client";

import React from "react";
import { motion, AnimatePresence } from "framer-motion";
import type { AgentThought, AgentRole } from "../../types/neuromap";
import { NEURO_COLORS } from "../../types/neuromap";

interface AgentStatePanelProps {
  thoughts: AgentThought[];
}

// ---------------------------------------------------------------------------
// Agent metadata
// ---------------------------------------------------------------------------

interface AgentMeta {
  name: string;
  color: string;
  icon: string; // SVG path data for a small icon
}

const AGENT_META: Record<AgentRole, AgentMeta> = {
  brain: {
    name: "Brain",
    color: NEURO_COLORS.stream,
    // Simple brain icon (circle with waves)
    icon: "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z",
  },
  council: {
    name: "Council",
    color: NEURO_COLORS.axon,
    // Group icon (three people)
    icon: "M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z",
  },
  risk: {
    name: "Risk Governor",
    color: NEURO_COLORS.statusDegraded,
    // Shield icon
    icon: "M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm0 10.99h7c-.53 4.12-3.28 7.79-7 8.94V12H5V6.3l7-3.11v8.8z",
  },
};

// ---------------------------------------------------------------------------
// Typing indicator
// ---------------------------------------------------------------------------

const TypingIndicator: React.FC<{ color: string }> = ({ color }) => (
  <span
    style={{
      display: "inline-flex",
      gap: 3,
      alignItems: "center",
      marginLeft: 4,
    }}
  >
    {[0, 1, 2].map((i) => (
      <motion.span
        key={i}
        style={{
          display: "inline-block",
          width: 5,
          height: 5,
          borderRadius: "50%",
          backgroundColor: color,
        }}
        animate={{
          opacity: [0.3, 1, 0.3],
          y: [0, -3, 0],
        }}
        transition={{
          duration: 0.8,
          repeat: Infinity,
          delay: i * 0.15,
          ease: "easeInOut",
        }}
      />
    ))}
  </span>
);

// ---------------------------------------------------------------------------
// Single thought card
// ---------------------------------------------------------------------------

const ThoughtCard: React.FC<{ thought: AgentThought }> = ({ thought }) => {
  const meta = AGENT_META[thought.agent];
  const timeStr = new Date(thought.timestamp).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -10, scale: 0.95 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      style={{
        background: NEURO_COLORS.glassBg,
        border: `1px solid ${NEURO_COLORS.glassBorder}`,
        borderLeft: `3px solid ${meta.color}`,
        borderRadius: 10,
        padding: "10px 14px",
        marginBottom: 8,
      }}
    >
      {/* Header: avatar + name + timestamp */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 6,
        }}
      >
        {/* Agent icon */}
        <svg
          width={18}
          height={18}
          viewBox="0 0 24 24"
          fill={meta.color}
          style={{ flexShrink: 0 }}
        >
          <path d={meta.icon} />
        </svg>

        <span
          style={{
            color: meta.color,
            fontSize: 12,
            fontWeight: 700,
            fontFamily: "'Inter', system-ui, sans-serif",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          {meta.name}
        </span>

        <span
          style={{
            marginLeft: "auto",
            color: NEURO_COLORS.starlight,
            fontSize: 10,
            fontFamily: "'JetBrains Mono', monospace",
            opacity: 0.5,
          }}
        >
          {timeStr}
        </span>
      </div>

      {/* Message body */}
      <div
        style={{
          color: NEURO_COLORS.starlight,
          fontSize: 12.5,
          lineHeight: 1.5,
          fontFamily: "'Inter', system-ui, sans-serif",
        }}
      >
        {thought.message}
        {thought.isStreaming && <TypingIndicator color={meta.color} />}
      </div>
    </motion.div>
  );
};

// ---------------------------------------------------------------------------
// Panel
// ---------------------------------------------------------------------------

/**
 * AgentStatePanel shows a live feed of "thoughts" from HEAN's AI agents:
 * Brain, Council, and Risk Governor. New thoughts slide in from below
 * and old ones fade out. A typing indicator appears while the agent
 * is "streaming" its thought.
 */
export const AgentStatePanel: React.FC<AgentStatePanelProps> = ({
  thoughts,
}) => {
  return (
    <div
      style={{
        background: NEURO_COLORS.glassBg,
        border: `1px solid ${NEURO_COLORS.glassBorder}`,
        borderRadius: 14,
        padding: "16px 14px",
        backdropFilter: "blur(12px)",
        WebkitBackdropFilter: "blur(12px)",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 14,
          paddingBottom: 10,
          borderBottom: `1px solid ${NEURO_COLORS.glassBorder}`,
        }}
      >
        <svg
          width={16}
          height={16}
          viewBox="0 0 24 24"
          fill={NEURO_COLORS.axon}
        >
          <path d="M21 6h-2v9H6v2c0 .55.45 1 1 1h11l4 4V7c0-.55-.45-1-1-1zm-4 6V3c0-.55-.45-1-1-1H3c-.55 0-1 .45-1 1v14l4-4h10c.55 0 1-.45 1-1z" />
        </svg>
        <span
          style={{
            color: NEURO_COLORS.supernova,
            fontSize: 13,
            fontWeight: 700,
            fontFamily: "'Inter', system-ui, sans-serif",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
          }}
        >
          Agent Activity
        </span>
      </div>

      {/* Thoughts feed */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          overflowX: "hidden",
          minHeight: 0,
          /* Custom scrollbar */
          scrollbarWidth: "thin",
          scrollbarColor: `${NEURO_COLORS.axon}44 transparent`,
        }}
      >
        <AnimatePresence mode="popLayout">
          {thoughts.map((thought) => (
            <ThoughtCard key={thought.id} thought={thought} />
          ))}
        </AnimatePresence>

        {thoughts.length === 0 && (
          <div
            style={{
              color: NEURO_COLORS.starlight,
              fontSize: 12,
              opacity: 0.4,
              textAlign: "center",
              padding: "40px 0",
              fontFamily: "'Inter', system-ui, sans-serif",
            }}
          >
            Waiting for agent activity...
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentStatePanel;

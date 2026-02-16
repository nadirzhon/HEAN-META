"use client";

import { motion, AnimatePresence } from "framer-motion";
import { useEffect, useRef } from "react";

interface LiveEvent {
  id: string;
  type: string;
  timestamp: number;
  summary: string;
}

interface LiveFeedProps {
  events: LiveEvent[];
  maxEvents?: number;
}

/**
 * Live scrolling feed of real-time system events.
 * New events slide in from top with fade animation.
 */
export function LiveFeed({ events, maxEvents = 50 }: LiveFeedProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to top when new events arrive
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = 0;
    }
  }, [events]);

  // Map event types to badge colors
  const getEventColor = (type: string): string => {
    const typeUpper = type.toUpperCase();
    if (typeUpper.includes("SIGNAL")) return "#A45BFF";
    if (typeUpper.includes("ORDER_FILLED") || typeUpper.includes("FILLED"))
      return "#00D4FF";
    if (typeUpper.includes("RISK") || typeUpper.includes("ALERT"))
      return "#FF8800";
    if (typeUpper.includes("ERROR")) return "#FF4466";
    if (typeUpper.includes("POSITION")) return "#00FF88";
    return "#C5C5DD";
  };

  // Format timestamp
  const formatTime = (timestamp: number): string => {
    return new Date(timestamp).toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  };

  // Limit displayed events
  const displayedEvents = events.slice(0, maxEvents);

  return (
    <div
      style={{
        background: "rgba(28,28,49,0.5)",
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: "16px",
        padding: "24px",
        display: "flex",
        flexDirection: "column",
        height: "100%",
        minHeight: "400px",
      }}
    >
      {/* Header */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "8px",
          marginBottom: "16px",
        }}
      >
        <div style={{ color: "#C5C5DD", fontSize: "14px", fontWeight: "600" }}>
          Live Feed
        </div>
        <motion.div
          animate={{
            opacity: [1, 0.3, 1],
            scale: [1, 1.2, 1],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
          style={{
            width: "8px",
            height: "8px",
            borderRadius: "50%",
            background: "#00FF88",
          }}
        />
      </div>

      {/* Event list */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: "8px",
        }}
      >
        <AnimatePresence initial={false}>
          {displayedEvents.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                height: "100%",
                gap: "12px",
                color: "#C5C5DD",
              }}
            >
              <div style={{ fontSize: "32px", opacity: 0.5 }}>📡</div>
              <div style={{ fontSize: "14px" }}>Waiting for events...</div>
            </motion.div>
          ) : (
            displayedEvents.map((event, index) => (
              <motion.div
                key={event.id}
                initial={{ opacity: 0, y: -10, height: 0 }}
                animate={{ opacity: 1, y: 0, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "12px",
                  padding: "10px 12px",
                  background: "rgba(255,255,255,0.02)",
                  borderRadius: "8px",
                  borderLeft: `3px solid ${getEventColor(event.type)}`,
                }}
              >
                {/* Timestamp */}
                <div
                  style={{
                    color: "rgba(197, 197, 221, 0.6)",
                    fontSize: "11px",
                    fontFamily: "monospace",
                    minWidth: "70px",
                  }}
                >
                  {formatTime(event.timestamp)}
                </div>

                {/* Event type badge */}
                <div
                  style={{
                    padding: "2px 8px",
                    borderRadius: "4px",
                    fontSize: "10px",
                    fontWeight: "600",
                    textTransform: "uppercase",
                    background: `${getEventColor(event.type)}20`,
                    color: getEventColor(event.type),
                    minWidth: "80px",
                    textAlign: "center",
                  }}
                >
                  {event.type.replace(/_/g, " ")}
                </div>

                {/* Summary */}
                <div
                  style={{
                    flex: 1,
                    color: "#C5C5DD",
                    fontSize: "13px",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {event.summary}
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}

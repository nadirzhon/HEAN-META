"use client";

import { useState, useMemo, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useHeanStore } from "@/store/heanStore";
import type { SystemEvent } from "@/store/heanStore";
import clsx from "clsx";

/* ── Event Type Badge ── */

const EVENT_COLORS: Record<string, { bg: string; text: string }> = {
  SIGNAL: { bg: "bg-axon/15", text: "text-axon" },
  ORDER_PLACED: { bg: "bg-stream/15", text: "text-stream" },
  ORDER_FILLED: { bg: "bg-positive/15", text: "text-positive" },
  ORDER_CANCELLED: { bg: "bg-starlight/10", text: "text-starlight/60" },
  ORDER_REJECTED: { bg: "bg-negative/15", text: "text-negative" },
  RISK_ALERT: { bg: "bg-warning/15", text: "text-warning" },
  RISK_BLOCKED: { bg: "bg-negative/15", text: "text-negative" },
  ERROR: { bg: "bg-negative/15", text: "text-negative" },
  HEARTBEAT: { bg: "bg-starlight/5", text: "text-starlight/30" },
  POSITION_OPENED: { bg: "bg-positive/15", text: "text-positive" },
  POSITION_CLOSED: { bg: "bg-stream/15", text: "text-stream" },
  BRAIN_ANALYSIS: { bg: "bg-axon/15", text: "text-axon" },
  REGIME_UPDATE: { bg: "bg-stream/15", text: "text-stream" },
};

function EventTypeBadge({ type }: { type: string }) {
  const colors = EVENT_COLORS[type] || { bg: "bg-starlight/10", text: "text-starlight/50" };
  return (
    <span className={clsx("text-[10px] px-2 py-0.5 rounded-full font-medium font-mono", colors.bg, colors.text)}>
      {type}
    </span>
  );
}

/* ── JSON Viewer ── */

function JSONViewer({ data }: { data: Record<string, unknown> }) {
  return (
    <motion.div
      initial={{ opacity: 0, height: 0 }}
      animate={{ opacity: 1, height: "auto" }}
      exit={{ opacity: 0, height: 0 }}
      transition={{ duration: 0.2 }}
      className="overflow-hidden"
    >
      <pre className="mt-2 p-3 bg-space rounded-lg text-[11px] font-mono text-starlight/60 overflow-auto max-h-60 border border-glass-border">
        {JSON.stringify(data, null, 2)}
      </pre>
    </motion.div>
  );
}

/* ── Event Row ── */

function EventRow({ event }: { event: SystemEvent }) {
  const [expanded, setExpanded] = useState(false);
  const time = new Date(event.timestamp).toLocaleTimeString("en-US", { hour12: false });

  return (
    <div
      className={clsx(
        "px-4 py-2.5 border-b border-glass-border/50 cursor-pointer transition-colors",
        expanded ? "bg-white/[0.02]" : "hover:bg-white/[0.02]"
      )}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-center gap-3">
        {/* Timestamp */}
        <span className="text-[11px] font-mono text-starlight/30 shrink-0 w-[68px]">
          {time}
        </span>

        {/* Type Badge */}
        <div className="shrink-0">
          <EventTypeBadge type={event.type} />
        </div>

        {/* Summary */}
        <span className="text-xs text-starlight/70 truncate flex-1">
          {event.summary}
        </span>

        {/* Expand icon */}
        {event.data && (
          <svg
            className={clsx("w-3.5 h-3.5 text-starlight/30 shrink-0 transition-transform", expanded && "rotate-180")}
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={2}
            stroke="currentColor"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="m19 9-7 7-7-7" />
          </svg>
        )}
      </div>

      {/* Expanded JSON */}
      <AnimatePresence>
        {expanded && event.data && <JSONViewer data={event.data} />}
      </AnimatePresence>
    </div>
  );
}

/* ── Filter Bar ── */

const EVENT_TYPE_FILTERS = [
  "ALL",
  "SIGNAL",
  "ORDER_FILLED",
  "ORDER_PLACED",
  "RISK_ALERT",
  "RISK_BLOCKED",
  "ERROR",
  "POSITION_OPENED",
  "POSITION_CLOSED",
  "BRAIN_ANALYSIS",
];

/* ── Black Box Tab ── */

export default function BlackBoxTab() {
  const events = useHeanStore((s) => s.systemEvents);
  const [filter, setFilter] = useState("ALL");
  const [search, setSearch] = useState("");
  const listRef = useRef<HTMLDivElement>(null);

  const filteredEvents = useMemo(() => {
    let result = events;
    if (filter !== "ALL") {
      result = result.filter((e) => e.type === filter);
    }
    if (search) {
      const q = search.toLowerCase();
      result = result.filter(
        (e) =>
          e.summary.toLowerCase().includes(q) ||
          e.type.toLowerCase().includes(q)
      );
    }
    return result;
  }, [events, filter, search]);

  const scrollToTop = useCallback(() => {
    listRef.current?.scrollTo({ top: 0, behavior: "smooth" });
  }, []);

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between shrink-0"
      >
        <div>
          <h2 className="text-lg font-semibold text-supernova">Black Box</h2>
          <p className="text-xs text-starlight/40">
            {filteredEvents.length} events {filter !== "ALL" && `(${filter})`}
          </p>
        </div>
        <button
          onClick={scrollToTop}
          className="text-[11px] text-starlight/40 hover:text-starlight/70 transition-colors"
        >
          Scroll to top
        </button>
      </motion.div>

      {/* Search + Filter Bar */}
      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="flex flex-col sm:flex-row gap-3 shrink-0"
      >
        {/* Search */}
        <div className="relative flex-1">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-starlight/30" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" d="m21 21-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z" />
          </svg>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search events..."
            className="w-full pl-10 pr-4 py-2 bg-space-lighter border border-glass-border rounded-lg text-sm text-starlight placeholder-starlight/30 outline-none focus:border-axon/40 transition-colors"
          />
        </div>

        {/* Filter pills */}
        <div className="flex flex-wrap gap-1.5">
          {EVENT_TYPE_FILTERS.map((type) => (
            <button
              key={type}
              onClick={() => setFilter(type)}
              className={clsx(
                "text-[10px] px-2.5 py-1 rounded-full font-medium transition-all",
                filter === type
                  ? "bg-axon/20 text-axon border border-axon/30"
                  : "bg-white/5 text-starlight/40 border border-transparent hover:text-starlight/60 hover:bg-white/8"
              )}
            >
              {type}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Event List */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
        ref={listRef}
        className="flex-1 glass overflow-auto"
        style={{ padding: 0 }}
      >
        {filteredEvents.length === 0 ? (
          <div className="flex items-center justify-center h-48">
            <div className="text-center space-y-2">
              <svg className="w-10 h-10 text-starlight/15 mx-auto" fill="none" viewBox="0 0 24 24" strokeWidth={1} stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375" />
              </svg>
              <p className="text-starlight/30 text-xs">No events recorded</p>
            </div>
          </div>
        ) : (
          filteredEvents.map((event) => (
            <EventRow key={event.id} event={event} />
          ))
        )}
      </motion.div>
    </div>
  );
}

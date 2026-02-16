"use client";

import { useState, useEffect, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import clsx from "clsx";
import { useHeanStore } from "@/store/heanStore";

/* ── Tab definitions ── */

interface TabDef {
  id: string;
  label: string;
  icon: React.ReactNode;
  hotkey: string;
}

const TABS: TabDef[] = [
  {
    id: "cockpit",
    label: "Cockpit",
    hotkey: "1",
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 016 3.75h2.25A2.25 2.25 0 0110.5 6v2.25a2.25 2.25 0 01-2.25 2.25H6a2.25 2.25 0 01-2.25-2.25V6zM3.75 15.75A2.25 2.25 0 016 13.5h2.25a2.25 2.25 0 012.25 2.25V18a2.25 2.25 0 01-2.25 2.25H6A2.25 2.25 0 013.75 18v-2.25zM13.5 6a2.25 2.25 0 012.25-2.25H18A2.25 2.25 0 0120.25 6v2.25A2.25 2.25 0 0118 10.5h-2.25a2.25 2.25 0 01-2.25-2.25V6zM13.5 15.75a2.25 2.25 0 012.25-2.25H18a2.25 2.25 0 012.25 2.25V18A2.25 2.25 0 0118 20.25h-2.25A2.25 2.25 0 0113.5 18v-2.25z" />
      </svg>
    ),
  },
  {
    id: "neuromap",
    label: "Neuro-Map",
    hotkey: "2",
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 3.75H6A2.25 2.25 0 003.75 6v1.5M16.5 3.75H18A2.25 2.25 0 0120.25 6v1.5m0 9V18A2.25 2.25 0 0118 20.25h-1.5m-9 0H6A2.25 2.25 0 013.75 18v-1.5M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
      </svg>
    ),
  },
  {
    id: "tactical",
    label: "Tactical Center",
    hotkey: "3",
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3v11.25A2.25 2.25 0 006 16.5h2.25M3.75 3h-1.5m1.5 0h16.5m0 0h1.5m-1.5 0v11.25A2.25 2.25 0 0118 16.5h-2.25m-7.5 0h7.5m-7.5 0l-1 3m8.5-3l1 3m0 0l.5 1.5m-.5-1.5h-9.5m0 0l-.5 1.5M9 11.25v1.5M12 9v3.75m3-6v6" />
      </svg>
    ),
  },
  {
    id: "blackbox",
    label: "Black Box",
    hotkey: "4",
    icon: (
      <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M20.25 6.375c0 2.278-3.694 4.125-8.25 4.125S3.75 8.653 3.75 6.375m16.5 0c0-2.278-3.694-4.125-8.25-4.125S3.75 4.097 3.75 6.375m16.5 0v11.25c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125V6.375m16.5 0v3.75m-16.5-3.75v3.75m16.5 0v3.75C20.25 16.153 16.556 18 12 18s-8.25-1.847-8.25-4.125v-3.75m16.5 0c0 2.278-3.694 4.125-8.25 4.125s-8.25-1.847-8.25-4.125" />
      </svg>
    ),
  },
];

/* ── Tab Content Components (lazy) ── */

import dynamic from "next/dynamic";

const CockpitTab = dynamic(() => import("@/components/tabs/CockpitTab"), {
  loading: () => <TabSkeleton label="Cockpit" />,
});
const NeuroMapTab = dynamic(() => import("@/components/tabs/NeuroMapTab"), {
  loading: () => <TabSkeleton label="Neuro-Map" />,
});
const TacticalTab = dynamic(() => import("@/components/tabs/TacticalTab"), {
  loading: () => <TabSkeleton label="Tactical Center" />,
});
const BlackBoxTab = dynamic(() => import("@/components/tabs/BlackBoxTab"), {
  loading: () => <TabSkeleton label="Black Box" />,
});

function TabSkeleton({ label }: { label: string }) {
  return (
    <div className="flex items-center justify-center h-[calc(100vh-80px)]">
      <div className="text-center space-y-4">
        <div className="w-8 h-8 border-2 border-axon border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-starlight/60 text-sm">Loading {label}...</p>
      </div>
    </div>
  );
}

const TAB_COMPONENTS: Record<string, React.ComponentType> = {
  cockpit: CockpitTab,
  neuromap: NeuroMapTab,
  tactical: TacticalTab,
  blackbox: BlackBoxTab,
};

/* ── Connection Status Indicator ── */

function ConnectionIndicator() {
  const connectionStatus = useHeanStore((s) => s.connectionStatus);

  const statusConfig = {
    connected: { className: "live-dot", label: "Live" },
    disconnected: { className: "live-dot live-dot-danger", label: "Offline" },
    reconnecting: { className: "live-dot live-dot-warning", label: "Reconnecting" },
  };

  const config = statusConfig[connectionStatus];

  return (
    <div className="flex items-center gap-2 text-xs">
      <div className={config.className} />
      <span className="text-starlight/60">{config.label}</span>
    </div>
  );
}

/* ── Main TabView ── */

export default function TabView() {
  const [activeTab, setActiveTab] = useState("cockpit");

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    const tab = TABS.find((t) => t.hotkey === e.key);
    if (tab && !e.metaKey && !e.ctrlKey && !e.altKey) {
      const active = document.activeElement;
      if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
      setActiveTab(tab.id);
    }
  }, []);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  const ActiveComponent = TAB_COMPONENTS[activeTab];

  return (
    <div className="flex flex-col h-screen">
      {/* Header Bar */}
      <header className="flex items-center justify-between px-6 h-14 border-b border-glass-border bg-space-light/80 backdrop-blur-md shrink-0">
        {/* Logo */}
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-supernova tracking-tight">
            <span className="gradient-text">HEAN</span>
            <span className="text-starlight/40 font-normal ml-2 text-sm">Heimdall</span>
          </h1>
        </div>

        {/* Tabs */}
        <nav className="flex items-center gap-1">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={clsx(
                "relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
                activeTab === tab.id
                  ? "text-supernova"
                  : "text-starlight/50 hover:text-starlight/80 hover:bg-white/5"
              )}
            >
              {tab.icon}
              <span className="hidden sm:inline">{tab.label}</span>
              <span className="text-[10px] text-starlight/30 hidden lg:inline ml-1">{tab.hotkey}</span>

              {/* Active indicator */}
              {activeTab === tab.id && (
                <motion.div
                  layoutId="tab-indicator"
                  className="absolute inset-0 rounded-lg bg-white/5 border border-glass-border"
                  transition={{ type: "spring", bounce: 0.15, duration: 0.5 }}
                />
              )}
            </button>
          ))}
        </nav>

        {/* Connection Status */}
        <ConnectionIndicator />
      </header>

      {/* Tab Content */}
      <main className="flex-1 overflow-hidden">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeTab}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2, ease: "easeOut" }}
            className="h-full overflow-auto p-4 md:p-6"
          >
            <ActiveComponent />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

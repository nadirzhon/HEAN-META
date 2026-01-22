import { useEffect, useMemo, useRef, useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { Badge } from "@/app/components/ui/badge";
import { Button } from "@/app/components/ui/button";
import { Input } from "@/app/components/ui/input";
import { ScrollArea } from "@/app/components/ui/scroll-area";
import { Separator } from "@/app/components/ui/separator";
import { Switch } from "@/app/components/ui/switch";
import { EventFeedItem } from "@/app/types/trading";

interface EventFeedProps {
  events: EventFeedItem[];
}

const categoryList = ["system", "orders", "positions", "risk", "strategies", "other"] as const;

const severityTone = (severity?: string) => {
  const s = (severity ?? "INFO").toUpperCase();
  if (s === "ERROR") return "bg-[var(--trading-red)]/20 text-[var(--trading-red)] border-[var(--trading-red)]/40";
  if (s === "WARN" || s === "WARNING") return "bg-[var(--trading-amber)]/20 text-[var(--trading-amber)] border-[var(--trading-amber)]/40";
  return "bg-[var(--trading-cyan)]/15 text-[var(--trading-cyan)] border-[var(--trading-cyan)]/40";
};

export function EventFeed({ events }: EventFeedProps) {
  const [search, setSearch] = useState("");
  const [activeTab, setActiveTab] = useState<"TRADING" | "SYSTEM">("TRADING");
  const [filters, setFilters] = useState<Record<string, boolean>>({
    system: true,
    orders: true,
    positions: true,
    risk: true,
    strategies: true,
    other: true,
  });
  const [paused, setPaused] = useState(false);
  const listRef = useRef<HTMLDivElement | null>(null);
  const prevCount = useRef(events.length);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    return events.filter((evt) => {
      // Filter by tab
      const isSystem = evt.category === "system" || evt.type?.includes("HEARTBEAT") || evt.topic === "system_heartbeat";
      if (activeTab === "SYSTEM" && !isSystem) return false;
      if (activeTab === "TRADING" && isSystem) return false;
      
      if (!filters[evt.category]) return false;
      if (!q) return true;
      return (
        evt.message.toLowerCase().includes(q) ||
        (evt.payload && JSON.stringify(evt.payload).toLowerCase().includes(q)) ||
        (evt.type ?? "").toLowerCase().includes(q)
      );
    });
  }, [events, filters, search, activeTab]);

  useEffect(() => {
    if (paused) return;
    if (events.length > prevCount.current && listRef.current) {
      listRef.current.scrollTop = 0;
    }
    prevCount.current = events.length;
  }, [events, paused]);

  const toggleFilter = (key: string) => {
    setFilters((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  return (
    <div className="rounded-lg border border-border bg-card/40 backdrop-blur-sm flex flex-col h-full">
      <div className="px-4 py-3 flex items-center justify-between gap-2 border-b border-border/50">
        <div>
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Live Event Feed</div>
          <div className="text-[10px] text-muted-foreground/70">ring buffer · last 200 events</div>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 text-[11px]">
            <Switch checked={!paused} onCheckedChange={(checked) => setPaused(!checked)} />
            <span className="text-muted-foreground">{paused ? "Autoscroll off" : "Autoscroll on"}</span>
          </div>
        </div>
      </div>
      
      {/* Tabs */}
      <div className="px-4 py-2 border-b border-border/50 flex gap-2">
        <Button
          size="sm"
          variant={activeTab === "TRADING" ? "default" : "ghost"}
          className="text-[11px] font-mono uppercase"
          onClick={() => setActiveTab("TRADING")}
        >
          TRADING
        </Button>
        <Button
          size="sm"
          variant={activeTab === "SYSTEM" ? "default" : "ghost"}
          className="text-[11px] font-mono uppercase"
          onClick={() => setActiveTab("SYSTEM")}
        >
          SYSTEM
        </Button>
      </div>

      <div className="px-4 py-3 space-y-2">
        <Input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search events..."
          className="h-8 text-sm"
        />
        <div className="flex flex-wrap gap-2">
          {categoryList.map((cat) => (
            <Button
              key={cat}
              size="sm"
              variant={filters[cat] ? "default" : "outline"}
              className="text-[11px] font-mono uppercase"
              onClick={() => toggleFilter(cat)}
            >
              {cat}
            </Button>
          ))}
        </div>
      </div>

      <Separator />

      <ScrollArea className="px-4 flex-1">
        <div ref={listRef} className="space-y-3 pb-4">
          {filtered.length === 0 && (
            <div className="text-center text-muted-foreground text-sm py-6">No events match filters</div>
          )}
          {filtered.map((evt) => (
            <div key={evt.id} className="border border-border/40 rounded-md p-3 bg-card/50 shadow-sm">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Badge variant="outline" className={`font-mono ${severityTone(evt.severity)}`}>
                    {evt.severity}
                  </Badge>
                  <span className="text-[11px] uppercase tracking-wider text-muted-foreground">{evt.category}</span>
                  <span className="text-xs font-mono">{evt.type}</span>
                </div>
                <span className="text-[11px] text-muted-foreground font-mono">
                  {formatDistanceToNow(evt.ts, { addSuffix: true })}
                </span>
              </div>
              <div className="text-sm mt-1 font-mono text-foreground">{evt.message}</div>
              {evt.payload && (
                <div className="mt-2 grid grid-cols-2 gap-2 text-[11px] text-muted-foreground/90">
                  {Object.entries(evt.payload)
                    .slice(0, 4)
                    .map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-2">
                        <span className="uppercase tracking-wider">{k}</span>
                        <span className="font-mono text-right truncate max-w-[120px]">
                          {typeof v === "object" ? JSON.stringify(v) : String(v)}
                        </span>
                      </div>
                    ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}

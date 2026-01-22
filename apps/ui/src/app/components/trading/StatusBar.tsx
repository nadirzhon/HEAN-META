import { useMemo, useState, useEffect } from "react";
import { Activity, Clock3, Network, Timer, Users, Zap } from "lucide-react";
import { Badge } from "@/app/components/ui/badge";
import { PulseState, HealthStatus, TelemetryState, WsMeta } from "@/app/hooks/useTradingData";

interface StatusBarProps {
  pulse: PulseState;
  telemetry: TelemetryState;
  ws: WsMeta;
  health: HealthStatus;
  backendAvailable: boolean;
  lastSync?: Date | null;
}

const tone = (value: string) => {
  switch (value) {
    case "RUNNING":
    case "ACTIVE":
      return "text-[var(--trading-green)]";
    case "PAUSED":
    case "DEFENSIVE":
      return "text-[var(--trading-amber)]";
    case "EMERGENCY":
    case "STOPPED":
      return "text-[var(--trading-red)]";
    default:
      return "text-[var(--trading-cyan)]";
  }
};

const wsTone = (status: PulseState["wsStatus"]) => {
  if (status === "connected") return "text-[var(--trading-green)]";
  if (status === "reconnecting") return "text-[var(--trading-amber)]";
  return "text-[var(--trading-red)]";
};

const formatUptime = (seconds?: number) => {
  if (!seconds && seconds !== 0) return "—";
  const s = Math.max(0, Math.floor(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return `${h}h ${m}m`;
};

export function StatusBar({ pulse, telemetry, ws, health, backendAvailable, lastSync }: StatusBarProps) {
  const [now, setNow] = useState(Date.now());
  
  // Update timestamp every second for live ageLabel
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  
  const { ageLabel, ageTone, dataStatus, pipeBroken } = useMemo(() => {
    // Priority: heartbeat > event > message
    const ts = pulse.lastHeartbeatTs ?? pulse.lastEventTs ?? telemetry.last_event_ts ?? ws.lastMessageAt;
    const ageMs = ts ? Math.max(0, now - ts) : Number.POSITIVE_INFINITY;
    const ageSec = ageMs / 1000;
    const status = ageSec > 15 ? "OFFLINE" : ageSec > 5 ? "DATA STALE" : "OK";
    const tone =
      status === "OFFLINE"
        ? "text-[var(--trading-red)]"
        : status === "DATA STALE"
        ? "text-[var(--trading-amber)]"
        : "text-[var(--trading-green)]";
    const pipe = pulse.engineState === "RUNNING" && (pulse.eventsPerSec || telemetry.events_per_sec || 0) === 0;
    return {
      ageLabel: ts ? `${Math.round(ageSec)}s ago` : "—",
      ageTone: tone,
      dataStatus: status,
      pipeBroken: pipe,
    };
  }, [pulse.engineState, pulse.eventsPerSec, pulse.lastEventTs, pulse.lastHeartbeatTs, telemetry.events_per_sec, telemetry.last_event_ts, ws.lastMessageAt, now]);

  return (
    <div className="fixed top-0 inset-x-0 z-40 border-b border-border bg-card/80 backdrop-blur-md">
      <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center gap-4 overflow-x-auto">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs uppercase tracking-wider text-muted-foreground">Engine</span>
          <Badge variant="outline" className={`${tone(pulse.engineState)} border-current font-mono`}>
            {pulse.engineState}
          </Badge>
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-center gap-2">
          <Network className={`h-4 w-4 ${wsTone(ws.status)}`} />
          <span className="text-xs uppercase tracking-wider text-muted-foreground">WS</span>
          <Badge variant="outline" className={`${wsTone(ws.status)} border-current font-mono`}>
            {ws.status}
          </Badge>
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-center gap-2">
          <Timer className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs uppercase tracking-wider text-muted-foreground">Last event</span>
        <span className={`font-mono text-sm ${ageTone}`}>{ageLabel}</span>
        {dataStatus !== "OK" && (
          <Badge
            variant="outline"
            className={`ml-1 font-mono ${
              dataStatus === "OFFLINE"
                ? "text-[var(--trading-red)] border-[var(--trading-red)]/50"
                : "text-[var(--trading-amber)] border-[var(--trading-amber)]/50"
            }`}
          >
            {dataStatus}
          </Badge>
        )}
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-center gap-2">
          <Zap className="h-4 w-4 text-[var(--trading-cyan)]" />
          <span className="text-xs uppercase tracking-wider text-muted-foreground">Events/sec</span>
          <span className="font-mono text-sm text-[var(--trading-cyan)]">
            {(pulse.eventsPerSec || telemetry.events_per_sec || 0).toFixed(2)}
          </span>
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-center gap-2">
          <Users className="h-4 w-4 text-[var(--trading-cyan)]" />
          <span className="text-xs uppercase tracking-wider text-muted-foreground">WS Clients</span>
          <span className="font-mono text-sm">{telemetry.ws_clients ?? "—"}</span>
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-center gap-2">
          <Clock3 className="h-4 w-4 text-muted-foreground" />
          <span className="text-xs uppercase tracking-wider text-muted-foreground">Uptime</span>
          <span className="font-mono text-sm">{formatUptime(telemetry.uptime)}</span>
        </div>

      {pipeBroken && (
        <>
          <div className="h-4 w-px bg-border" />
          <Badge
            variant="destructive"
            className="bg-[var(--trading-red)]/10 text-[var(--trading-red)] border-[var(--trading-red)]/60 font-mono"
          >
            PIPE BROKEN
          </Badge>
        </>
      )}

        <div className="h-4 w-px bg-border" />

        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wider text-muted-foreground">Mode</span>
          <Badge variant="outline" className="text-[var(--trading-purple)] border-[var(--trading-purple)]/50 font-mono">
            {pulse.mode.toUpperCase()}
          </Badge>
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-center gap-2">
          <span className="text-xs uppercase tracking-wider text-muted-foreground">Health</span>
          <Badge
            variant="outline"
            className={`font-mono ${
              health === "OK"
                ? "text-[var(--trading-green)] border-[var(--trading-green)]/50"
                : health === "WARNING"
                ? "text-[var(--trading-amber)] border-[var(--trading-amber)]/50"
                : "text-[var(--trading-red)] border-[var(--trading-red)]/50"
            }`}
          >
            {health}
          </Badge>
        </div>

        {!pulse.mockMode && backendAvailable && ws.status === "connected" && (
          <>
            <div className="h-4 w-px bg-border" />
            <Badge className="bg-[var(--trading-green)]/20 text-[var(--trading-green)] border-[var(--trading-green)]/50 font-bold">
              REAL DATA
            </Badge>
          </>
        )}

        {pulse.mockMode && (
          <>
            <div className="h-4 w-px bg-border" />
            <Badge variant="destructive" className="bg-[var(--trading-amber)]/20 text-[var(--trading-amber)] border-[var(--trading-amber)]/50 font-bold">
              MOCK DATA
            </Badge>
          </>
        )}

        {backendAvailable && ws.status !== "connected" && (
          <>
            <div className="h-4 w-px bg-border" />
            <Badge variant="destructive" className="bg-[var(--trading-red)]/20 text-[var(--trading-red)] border-[var(--trading-red)]/50 font-bold">
              NO REALTIME: WS DOWN
            </Badge>
          </>
        )}

        {!backendAvailable && (
          <>
            <div className="h-4 w-px bg-border" />
            <Badge variant="destructive" className="bg-[var(--trading-red)]/20 text-[var(--trading-red)] border-[var(--trading-red)]/50 font-bold">
              Backend unreachable
            </Badge>
          </>
        )}

        <div className="ml-auto flex items-center gap-3 text-[11px] text-muted-foreground font-mono">
          <span>REST: {pulse.restHealth}</span>
          <span>
            Sync {lastSync ? lastSync.toLocaleTimeString("en-US", { hour12: false }) : "—"}
          </span>
        </div>
      </div>
    </div>
  );
}

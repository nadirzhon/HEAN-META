import { AlertTriangle, Info } from "lucide-react";
import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { WsMeta } from "@/app/hooks/useTradingData";

interface DebugPanelProps {
  apiBase: string;
  wsUrl: string;
  backendAvailable: boolean;
  lastBackendError: string | null;
  ws: WsMeta;
}

const formatTs = (value?: number) => {
  if (!value) return "—";
  return new Date(value).toLocaleTimeString("en-US", { hour12: false });
};

export function DebugPanel({ apiBase, wsUrl, backendAvailable, lastBackendError, ws }: DebugPanelProps) {
  return (
    <Card className="p-4 border-border/60 bg-card/60 backdrop-blur-sm space-y-3 text-sm">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Info className="h-4 w-4 text-muted-foreground" />
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Debug</div>
        </div>
        <Badge variant="outline" className="font-mono text-[11px]">
          {backendAvailable ? "REST OK" : "REST DOWN"}
        </Badge>
      </div>
      <div className="grid grid-cols-1 gap-2 text-[12px] text-muted-foreground">
        <div className="flex justify-between">
          <span>API base</span>
          <span className="font-mono text-foreground truncate max-w-[220px]">{apiBase}</span>
        </div>
        <div className="flex justify-between">
          <span>WS url</span>
          <span className="font-mono text-foreground truncate max-w-[220px]">{wsUrl}</span>
        </div>
        <div className="flex justify-between">
          <span>WS status</span>
          <span className="font-mono text-foreground">
            {ws.status} · last msg {formatTs(ws.lastMessageAt)} · retries {ws.reconnectAttempts}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Last heartbeat</span>
          <span className="font-mono text-foreground">{formatTs(ws.lastHeartbeatAt)}</span>
        </div>
        {ws.lastError && (
          <div className="flex justify-between text-[var(--trading-amber)]">
            <span>WS error</span>
            <span className="font-mono truncate max-w-[220px] text-right">{ws.lastError}</span>
          </div>
        )}
        {lastBackendError && (
          <div className="flex items-start gap-2 text-[var(--trading-amber)]">
            <AlertTriangle className="h-4 w-4" />
            <span className="font-mono text-xs">{lastBackendError}</span>
          </div>
        )}
      </div>
    </Card>
  );
}

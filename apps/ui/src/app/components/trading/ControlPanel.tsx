import { useState } from "react";
import { AlertTriangle, PauseCircle, PlayCircle, Power, RotateCcw } from "lucide-react";
import { Button } from "@/app/components/ui/button";
import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/app/components/ui/tooltip";
import { ControlAction, ControlResponse } from "@/app/api/client";
import { ControlStatus } from "@/app/hooks/useTradingData";
import { toast } from "sonner";

interface ControlPanelProps {
  onAction: (action: ControlAction) => Promise<ControlResponse>;
  support: Partial<Record<ControlAction, boolean>>;
  controlStatus: ControlStatus;
  backendAvailable: boolean;
}

const actionConfig: Array<{
  key: ControlAction;
  label: string;
  icon: any;
  tone: string;
}> = [
  { key: "pause", label: "Pause", icon: PauseCircle, tone: "border-[var(--trading-amber)]/50 text-[var(--trading-amber)]" },
  { key: "resume", label: "Resume", icon: PlayCircle, tone: "border-[var(--trading-cyan)]/50 text-[var(--trading-cyan)]" },
  { key: "kill", label: "Kill", icon: Power, tone: "border-[var(--trading-red)]/60 text-[var(--trading-red)]" },
  { key: "restart", label: "Restart", icon: RotateCcw, tone: "border-[var(--trading-purple)]/50 text-[var(--trading-purple)]" },
];

export function ControlPanel({ onAction, support, controlStatus, backendAvailable }: ControlPanelProps) {
  const [loadingKey, setLoadingKey] = useState<ControlAction | null>(null);

  const handle = async (action: ControlAction) => {
    if (support[action] === false || !backendAvailable) return;
    if (action === "kill" && !window.confirm("Send KILL to engine? This is irreversible.")) return;
    
    setLoadingKey(action);
    let timeoutId: NodeJS.Timeout | null = null;
    
    try {
      // Add timeout to prevent hanging
      const timeoutPromise = new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
          reject(new Error(`Request timeout after 10s`));
        }, 10000);
      });
      
      const actionPromise = onAction(action);
      const res = await Promise.race([actionPromise, timeoutPromise]) as ControlResponse;
      
      if (timeoutId) clearTimeout(timeoutId);
      
      // Standardized response handling
      const message = res?.message ?? res?.status ?? "OK";
      const engineState = (res as any)?.engine_state ?? res?.status;
      
      toast.success(message, { 
        description: `CONTROL_RESULT · ${action}${engineState ? ` · ${engineState}` : ""}` 
      });
    } catch (err: any) {
      if (timeoutId) clearTimeout(timeoutId);
      
      // Handle timeout
      if (err?.message?.includes("timeout")) {
        toast.error("Request timeout", { 
          description: `CONTROL_RESULT · ${action} · Backend did not respond in 10s` 
        });
        return;
      }
      
      // Handle specific error cases
      if (err instanceof Error) {
        // Check if it's a network error
        const isNetworkError = err.message.includes("fetch") || 
                              err.message.includes("network") || 
                              err.message.includes("Failed to fetch") ||
                              err.message.includes("AbortError");
        
        if (isNetworkError) {
          toast.error("Network error - backend unavailable", { 
            description: `CONTROL_RESULT · ${action} · Check connection` 
          });
          return;
        }
        
        // Check if it's a not supported error (404/501)
        if (typeof err.isNotSupported === "function" && err.isNotSupported()) {
          toast.error("Action not supported by backend", { 
            description: `CONTROL_RESULT · ${action} · Endpoint returned 404/501` 
          });
          return;
        }
        
        // Check for conflict (409)
        if (err.status === 409) {
          toast.error("State conflict - action cannot be performed", { 
            description: `CONTROL_RESULT · ${action} · Engine state conflict` 
          });
          return;
        }
        
        // Check for server error (500)
        if (err.status === 500) {
          const detail = (err as any)?.body?.detail || err.message;
          toast.error("Server error", { 
            description: `CONTROL_RESULT · ${action} · ${detail}` 
          });
          return;
        }
      }
      
      // Generic error handling - never crash, always show toast
      const msg = err?.message ?? err?.toString() ?? "Control action failed";
      toast.error(msg, { description: `CONTROL_RESULT · ${action}` });
    } finally {
      setLoadingKey(null);
    }
  };

  const statusTone = (() => {
    if (controlStatus.state === "ok") return "text-[var(--trading-green)] border-[var(--trading-green)]/50";
    if (controlStatus.state === "error") return "text-[var(--trading-red)] border-[var(--trading-red)]/50";
    if (controlStatus.state === "waiting" || controlStatus.state === "pending") return "text-[var(--trading-amber)] border-[var(--trading-amber)]/50";
    return "text-muted-foreground border-border/60";
  })();

  const statusLabel =
    controlStatus.state === "idle"
      ? "Idle"
      : controlStatus.state === "waiting"
      ? `Awaiting CONTROL_RESULT for ${controlStatus.action ?? "action"}`
      : `${controlStatus.action ?? ""} ${controlStatus.state}`.trim();

  return (
    <Card className="p-4 border-border/60 bg-card/50 backdrop-blur-sm space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Engine Control</div>
          <div className="text-[10px] text-muted-foreground/70">Calls backend endpoints directly</div>
        </div>
        <Badge variant="outline" className={`text-[11px] ${statusTone}`}>
          {statusLabel}
        </Badge>
      </div>
      <TooltipProvider delayDuration={150}>
        <div className="grid grid-cols-2 gap-2">
          {actionConfig.map(({ key, label, icon: Icon, tone }) => {
            const unsupported = support[key] === false;
            const disabledReason = unsupported
              ? "Not supported by backend"
              : !backendAvailable
              ? "Backend unavailable"
              : undefined;
            const button = (
              <Button
                key={key}
                variant="outline"
                size="sm"
                disabled={Boolean(disabledReason) || loadingKey === key}
                className={`justify-start ${tone}`}
                onClick={() => handle(key)}
              >
                <Icon className="h-4 w-4" />
                <span>{unsupported ? "Not supported" : label}</span>
              </Button>
            );
            return disabledReason ? (
              <Tooltip key={key}>
                <TooltipTrigger asChild>{button}</TooltipTrigger>
                <TooltipContent>{disabledReason}</TooltipContent>
              </Tooltip>
            ) : (
              button
            );
          })}
        </div>
      </TooltipProvider>
      {controlStatus.message && (
        <div className="text-[12px] text-muted-foreground font-mono">
          {controlStatus.state === "waiting" ? "Waiting: " : ""}
          {controlStatus.message}
        </div>
      )}
      {!backendAvailable && (
        <div className="flex items-center gap-2 text-[12px] text-muted-foreground">
          <AlertTriangle className="h-4 w-4 text-[var(--trading-amber)]" />
          Backend not reachable — controls paused until REST recovers.
        </div>
      )}
      {support.kill === false && (
        <div className="flex items-center gap-2 text-[12px] text-muted-foreground">
          <AlertTriangle className="h-4 w-4 text-[var(--trading-amber)]" />
          Kill endpoint returned 404/501 — button locked.
        </div>
      )}
    </Card>
  );
}

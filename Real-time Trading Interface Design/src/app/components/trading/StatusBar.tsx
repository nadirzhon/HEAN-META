import { Activity, AlertTriangle, Power, ShieldCheck } from "lucide-react";
import { Badge } from "@/app/components/ui/badge";
import { Button } from "@/app/components/ui/button";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/app/components/ui/tooltip";

type SystemMode = "CALM" | "ACTIVE" | "DEFENSIVE" | "EMERGENCY";
type HealthStatus = "OK" | "WARNING" | "CRITICAL";

interface StatusBarProps {
  systemMode: SystemMode;
  confidence: number;
  health: HealthStatus;
  onKillSwitch?: () => void;
}

export function StatusBar({ systemMode, confidence, health, onKillSwitch }: StatusBarProps) {
  const getModeColor = (mode: SystemMode) => {
    switch (mode) {
      case "CALM":
        return "text-[var(--trading-green)]";
      case "ACTIVE":
        return "text-[var(--trading-cyan)]";
      case "DEFENSIVE":
        return "text-[var(--trading-amber)]";
      case "EMERGENCY":
        return "text-[var(--trading-red)]";
    }
  };

  const getHealthColor = (status: HealthStatus) => {
    switch (status) {
      case "OK":
        return "text-[var(--trading-green)]";
      case "WARNING":
        return "text-[var(--trading-amber)]";
      case "CRITICAL":
        return "text-[var(--trading-red)]";
    }
  };

  const getConfidenceColor = (conf: number) => {
    if (conf >= 80) return "text-[var(--trading-green)]";
    if (conf >= 50) return "text-[var(--trading-cyan)]";
    if (conf >= 30) return "text-[var(--trading-amber)]";
    return "text-[var(--trading-red)]";
  };

  return (
    <div className="border-b border-border bg-card/50 backdrop-blur-sm">
      <div className="flex items-center justify-between px-6 py-3">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-muted-foreground" />
            <span className="text-xs text-muted-foreground uppercase tracking-wider">System Mode</span>
            <Badge variant="outline" className={`${getModeColor(systemMode)} border-current font-mono`}>
              {systemMode}
            </Badge>
          </div>

          <div className="h-4 w-px bg-border" />

          <div className="flex items-center gap-2">
            <span className="text-xs text-muted-foreground uppercase tracking-wider">Confidence</span>
            <span className={`font-mono tabular-nums ${getConfidenceColor(confidence)}`}>
              {confidence}%
            </span>
          </div>

          <div className="h-4 w-px bg-border" />

          <div className="flex items-center gap-2">
            {health === "OK" && <ShieldCheck className="h-4 w-4 text-[var(--trading-green)]" />}
            {health === "WARNING" && <AlertTriangle className="h-4 w-4 text-[var(--trading-amber)]" />}
            {health === "CRITICAL" && <AlertTriangle className="h-4 w-4 text-[var(--trading-red)]" />}
            <span className="text-xs text-muted-foreground uppercase tracking-wider">Health</span>
            <Badge variant="outline" className={`${getHealthColor(health)} border-current font-mono`}>
              {health}
            </Badge>
          </div>
        </div>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="outline"
                size="sm"
                className="gap-2 border-[var(--trading-red)]/30 text-[var(--trading-red)] hover:bg-[var(--trading-red)]/10 hover:text-[var(--trading-red)]"
                onClick={onKillSwitch}
              >
                <Power className="h-4 w-4" />
                KILL SWITCH
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Emergency shutdown - closes all positions</p>
            </TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
    </div>
  );
}

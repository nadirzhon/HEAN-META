import { TrendingUp, BarChart3, Circle } from "lucide-react";
import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { Switch } from "@/app/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/app/components/ui/tooltip";

export interface Strategy {
  id: string;
  name: string;
  enabled: boolean;
  symbols: string[];
  aggression: "LOW" | "MEDIUM" | "HIGH";
  signalsPerMinute: number;
  lastDecision?: {
    outcome: string;
    timestamp: Date;
  };
  isWired: boolean;
}

interface StrategyCardProps {
  strategy: Strategy;
  onToggle?: (id: string, enabled: boolean) => void;
}

export function StrategyCard({ strategy, onToggle }: StrategyCardProps) {
  const getAggressionColor = (level: string) => {
    switch (level) {
      case "LOW":
        return "border-[var(--trading-green)]/50 text-[var(--trading-green)]";
      case "MEDIUM":
        return "border-[var(--trading-amber)]/50 text-[var(--trading-amber)]";
      case "HIGH":
        return "border-[var(--trading-red)]/50 text-[var(--trading-red)]";
      default:
        return "border-muted-foreground/50 text-muted-foreground";
    }
  };

  return (
    <Card className="p-5 bg-card/30 backdrop-blur-sm border-border/50">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg ${strategy.enabled ? "bg-[var(--trading-cyan)]/10" : "bg-muted/30"}`}>
            <TrendingUp className={`h-5 w-5 ${strategy.enabled ? "text-[var(--trading-cyan)]" : "text-muted-foreground"}`} />
          </div>
          <div>
            <h4 className="font-mono">{strategy.name}</h4>
            <div className="flex items-center gap-2 mt-1">
              <Circle 
                className={`h-2 w-2 fill-current ${
                  strategy.enabled 
                    ? "text-[var(--trading-green)]" 
                    : "text-[var(--trading-gray)]"
                }`} 
              />
              <span className="text-xs text-muted-foreground uppercase tracking-wider">
                {strategy.enabled ? "Active" : "Disabled"}
              </span>
            </div>
          </div>
        </div>

        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <div>
                <Switch
                  checked={strategy.enabled}
                  onCheckedChange={(checked) => onToggle?.(strategy.id, checked)}
                  disabled={!strategy.isWired}
                />
              </div>
            </TooltipTrigger>
            {!strategy.isWired && (
              <TooltipContent>
                <p>Strategy not wired - toggle disabled</p>
              </TooltipContent>
            )}
          </Tooltip>
        </TooltipProvider>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Symbols</span>
          <div className="flex gap-1">
            {strategy.symbols.slice(0, 3).map((symbol) => (
              <Badge key={symbol} variant="outline" className="font-mono text-xs">
                {symbol}
              </Badge>
            ))}
            {strategy.symbols.length > 3 && (
              <Badge variant="outline" className="font-mono text-xs">
                +{strategy.symbols.length - 3}
              </Badge>
            )}
          </div>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Aggression</span>
          <Badge variant="outline" className={`font-mono ${getAggressionColor(strategy.aggression)}`}>
            {strategy.aggression}
          </Badge>
        </div>

        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">Signals/min</span>
          <div className="flex items-center gap-2">
            <BarChart3 className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="font-mono tabular-nums">{strategy.signalsPerMinute}</span>
          </div>
        </div>

        {strategy.lastDecision && (
          <div className="pt-3 border-t border-border/50">
            <div className="text-xs text-muted-foreground mb-1">Last Decision</div>
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="text-xs">
                {strategy.lastDecision.outcome}
              </Badge>
              <span className="text-xs text-muted-foreground font-mono">
                {strategy.lastDecision.timestamp.toLocaleTimeString("en-US", {
                  hour12: false,
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
          </div>
        )}

        {!strategy.isWired && (
          <div className="pt-3 border-t border-border/50">
            <Badge variant="outline" className="border-[var(--trading-gray)]/50 text-[var(--trading-gray)] text-xs">
              Not wired
            </Badge>
          </div>
        )}
      </div>
    </Card>
  );
}

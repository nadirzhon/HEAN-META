import { Activity, AlertTriangle, Clock, Database } from "lucide-react";
import { Card } from "@/app/components/ui/card";
import { Progress } from "@/app/components/ui/progress";
import { Badge } from "@/app/components/ui/badge";

export interface RiskMetric {
  label: string;
  value: number;
  max: number;
  unit: string;
  status: "ok" | "warning" | "critical";
  isWired: boolean;
}

interface RiskMonitorProps {
  metrics: RiskMetric[];
}

export function RiskMonitor({ metrics }: RiskMonitorProps) {
  const getStatusColor = (status: string) => {
    switch (status) {
      case "ok":
        return "text-[var(--trading-green)]";
      case "warning":
        return "text-[var(--trading-amber)]";
      case "critical":
        return "text-[var(--trading-red)]";
      default:
        return "text-muted-foreground";
    }
  };

  const getProgressColor = (status: string) => {
    switch (status) {
      case "ok":
        return "[&>div]:bg-[var(--trading-green)]";
      case "warning":
        return "[&>div]:bg-[var(--trading-amber)]";
      case "critical":
        return "[&>div]:bg-[var(--trading-red)]";
      default:
        return "";
    }
  };

  const getIcon = (label: string) => {
    if (label.toLowerCase().includes("latency")) return Clock;
    if (label.toLowerCase().includes("eventbus") || label.toLowerCase().includes("pressure")) return Activity;
    if (label.toLowerCase().includes("memory") || label.toLowerCase().includes("blocks")) return Database;
    return AlertTriangle;
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {metrics.map((metric, index) => {
        const Icon = getIcon(metric.label);
        const percentage = (metric.value / metric.max) * 100;

        return (
          <Card key={index} className="p-5 bg-card/30 backdrop-blur-sm border-border/50">
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <Icon className="h-4 w-4 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">{metric.label}</span>
              </div>
              {!metric.isWired && (
                <Badge variant="outline" className="border-[var(--trading-gray)]/50 text-[var(--trading-gray)] text-xs">
                  Not wired
                </Badge>
              )}
            </div>

            {metric.isWired ? (
              <>
                <div className="flex items-end justify-between mb-2">
                  <span className={`text-2xl font-mono tabular-nums ${getStatusColor(metric.status)}`}>
                    {metric.value}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    / {metric.max} {metric.unit}
                  </span>
                </div>
                <Progress 
                  value={percentage} 
                  className={`h-1.5 ${getProgressColor(metric.status)}`}
                />
              </>
            ) : (
              <div className="text-center py-4 text-sm text-muted-foreground">
                Data feed not connected
              </div>
            )}
          </Card>
        );
      })}
    </div>
  );
}

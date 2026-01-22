import { LucideIcon } from "lucide-react";
import { Card } from "@/app/components/ui/card";

interface MetricsCardProps {
  label: string;
  value: string | number;
  delta?: string;
  deltaType?: "positive" | "negative" | "neutral";
  icon?: LucideIcon;
  valueColor?: string;
}

export function MetricsCard({ label, value, delta, deltaType = "neutral", icon: Icon, valueColor }: MetricsCardProps) {
  const getDeltaColor = () => {
    switch (deltaType) {
      case "positive":
        return "text-[var(--trading-green)]";
      case "negative":
        return "text-[var(--trading-red)]";
      default:
        return "text-muted-foreground";
    }
  };

  return (
    <Card className="p-4 bg-card/50 backdrop-blur-sm border-border/50">
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">{label}</div>
          <div className={`text-2xl font-mono tabular-nums ${valueColor || "text-foreground"}`}>
            {value}
          </div>
          {delta && (
            <div className={`text-sm font-mono mt-1 ${getDeltaColor()}`}>
              {delta}
            </div>
          )}
        </div>
        {Icon && (
          <Icon className="h-5 w-5 text-muted-foreground/50" />
        )}
      </div>
    </Card>
  );
}

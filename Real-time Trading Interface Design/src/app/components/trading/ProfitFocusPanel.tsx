import { TrendingUp, TrendingDown, Shield, AlertTriangle } from "lucide-react";
import { Card } from "@/app/components/ui/card";

interface ProfitMetrics {
  expected: number;
  realized: number;
  missed: number;
  protectedLoss: number;
}

interface ProfitFocusPanelProps {
  metrics: ProfitMetrics;
  timeframe: string;
}

export function ProfitFocusPanel({ metrics, timeframe }: ProfitFocusPanelProps) {
  return (
    <Card className="p-6 bg-card/30 backdrop-blur-sm border-border/50">
      <div className="mb-6">
        <h3 className="text-sm uppercase tracking-wider text-muted-foreground mb-1">
          Profit Intelligence
        </h3>
        <p className="text-xs text-muted-foreground">
          Understanding profit beyond realized gains — {timeframe}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Expected Profit */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-[var(--trading-cyan)]" />
            <span className="text-xs text-muted-foreground uppercase tracking-wider">
              Expected
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-mono tabular-nums text-[var(--trading-cyan)]">
              ${metrics.expected.toFixed(2)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground/70">
            Model-based profit forecast
          </p>
        </div>

        {/* Realized Profit */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <TrendingUp className="h-3.5 w-3.5 text-[var(--trading-green)]" />
            <span className="text-xs text-muted-foreground uppercase tracking-wider">
              Realized
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-mono tabular-nums text-[var(--trading-green)]">
              ${metrics.realized.toFixed(2)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground/70">
            Actual closed profit
          </p>
        </div>

        {/* Missed Profit */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <TrendingDown className="h-3.5 w-3.5 text-[var(--trading-amber)]" />
            <span className="text-xs text-muted-foreground uppercase tracking-wider">
              Missed
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-mono tabular-nums text-[var(--trading-amber)]">
              ${metrics.missed.toFixed(2)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground/70">
            Opportunities skipped by filters
          </p>
        </div>

        {/* Protected Loss */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Shield className="h-3.5 w-3.5 text-[var(--trading-green)]" />
            <span className="text-xs text-muted-foreground uppercase tracking-wider">
              Protected
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl font-mono tabular-nums text-[var(--trading-green)]">
              ${metrics.protectedLoss.toFixed(2)}
            </span>
          </div>
          <p className="text-xs text-muted-foreground/70">
            Loss avoided by risk system
          </p>
        </div>
      </div>

      {/* Insight */}
      <div className="mt-6 pt-6 border-t border-border/50">
        <div className="flex items-start gap-3 p-3 bg-[var(--trading-green)]/5 border border-[var(--trading-green)]/20 rounded-lg">
          <AlertTriangle className="h-4 w-4 text-[var(--trading-green)] flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-foreground/90">
              Protected loss is a hidden win. 
              The system prevented ${metrics.protectedLoss.toFixed(2)} in potential drawdown.
            </p>
            <p className="text-xs text-muted-foreground mt-1">
              Profit psychology: Success = Realized + Protected
            </p>
          </div>
        </div>
      </div>
    </Card>
  );
}

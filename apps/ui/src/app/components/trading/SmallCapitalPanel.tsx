/**
 * Small Capital Mode Panel - Shows cost/edge metrics and block reasons
 *
 * Displays:
 * - Average cost vs edge (in bps)
 * - Edge/cost ratio
 * - Top block reasons
 * - Maker fill rate
 * - Decision counts (CREATE/SKIP/BLOCK)
 */

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import { Badge } from "../ui/badge";
import { AlertCircle, TrendingUp, TrendingDown, DollarSign, Shield } from "lucide-react";

interface SmallCapitalPanelProps {
  data: {
    enabled: boolean;
    avg_cost_bps: number | null;
    avg_edge_bps: number | null;
    edge_cost_ratio: number | null;
    top_block_reasons: Array<{ reason: string; count: number }>;
    maker_fill_rate: number | null;
    decision_counts: {
      create: number;
      skip: number;
      block: number;
      total: number;
    };
    min_notional_usd: number | null;
    maker_only_default: boolean | null;
  };
}

export function SmallCapitalPanel({ data }: SmallCapitalPanelProps) {
  if (!data.enabled) {
    return null;
  }

  const edgeCostRatio = data.edge_cost_ratio || 0;
  const isHealthy = edgeCostRatio >= 4.0; // Should be at least 4x cost
  const statusColor = isHealthy ? "text-green-600" : "text-yellow-600";
  const statusIcon = isHealthy ? TrendingUp : AlertCircle;

  return (
    <Card className="w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-base font-semibold flex items-center gap-2">
              <Shield className="h-4 w-4" />
              Small Capital Mode
            </CardTitle>
            <CardDescription className="text-xs mt-1">
              Cost-aware execution for small deposits
            </CardDescription>
          </div>
          <Badge variant={isHealthy ? "default" : "secondary"}>
            {data.maker_only_default ? "Maker-Only" : "Mixed"}
          </Badge>
        </div>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Edge vs Cost Metrics */}
        <div className="grid grid-cols-3 gap-3 text-sm">
          <div className="space-y-1">
            <div className="text-xs text-muted-foreground">Avg Edge</div>
            <div className="font-mono font-semibold text-green-600">
              {data.avg_edge_bps !== null ? `${data.avg_edge_bps.toFixed(1)} bps` : "—"}
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-xs text-muted-foreground">Avg Cost</div>
            <div className="font-mono font-semibold text-red-600">
              {data.avg_cost_bps !== null ? `${data.avg_cost_bps.toFixed(1)} bps` : "—"}
            </div>
          </div>

          <div className="space-y-1">
            <div className="text-xs text-muted-foreground">Edge/Cost</div>
            <div className={`font-mono font-semibold ${statusColor} flex items-center gap-1`}>
              {edgeCostRatio > 0 ? `${edgeCostRatio.toFixed(2)}x` : "—"}
              {edgeCostRatio > 0 && React.createElement(statusIcon, { className: "h-3 w-3" })}
            </div>
          </div>
        </div>

        {/* Decision Stats */}
        <div className="space-y-2">
          <div className="text-xs text-muted-foreground font-medium">Decision Stats</div>
          <div className="flex gap-2 text-xs">
            <Badge variant="outline" className="flex items-center gap-1">
              <DollarSign className="h-3 w-3 text-green-600" />
              Create: {data.decision_counts.create}
            </Badge>
            <Badge variant="secondary" className="flex items-center gap-1">
              Skip: {data.decision_counts.skip}
            </Badge>
            <Badge variant="destructive" className="flex items-center gap-1">
              <AlertCircle className="h-3 w-3" />
              Block: {data.decision_counts.block}
            </Badge>
          </div>
        </div>

        {/* Maker Fill Rate */}
        {data.maker_fill_rate !== null && (
          <div className="space-y-1">
            <div className="text-xs text-muted-foreground font-medium">Maker Fill Rate</div>
            <div className="flex items-center gap-2">
              <div className="flex-1 bg-muted rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full"
                  style={{ width: `${(data.maker_fill_rate * 100).toFixed(0)}%` }}
                />
              </div>
              <div className="text-xs font-mono font-semibold min-w-[3rem] text-right">
                {(data.maker_fill_rate * 100).toFixed(0)}%
              </div>
            </div>
          </div>
        )}

        {/* Top Block Reasons */}
        {data.top_block_reasons && data.top_block_reasons.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs text-muted-foreground font-medium">Top Block Reasons</div>
            <div className="space-y-1 max-h-24 overflow-y-auto">
              {data.top_block_reasons.slice(0, 5).map((item, idx) => (
                <div
                  key={idx}
                  className="flex items-center justify-between text-xs p-1.5 rounded bg-muted/50"
                >
                  <span className="text-xs truncate font-mono text-muted-foreground">
                    {item.reason}
                  </span>
                  <Badge variant="secondary" className="ml-2 text-xs h-5">
                    {item.count}
                  </Badge>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Config Summary */}
        <div className="text-xs text-muted-foreground pt-2 border-t">
          Min Notional: ${data.min_notional_usd?.toFixed(0) || "10"}
          {" • "}
          Requires edge ≥ {data.edge_cost_ratio || 4}x cost
        </div>
      </CardContent>
    </Card>
  );
}

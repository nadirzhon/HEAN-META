import { useEffect, useState } from "react";
import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { Button } from "@/app/components/ui/button";
import { AlertCircle, CheckCircle2, XCircle, Clock, TrendingDown, RotateCcw } from "lucide-react";
import { fetchWhyNotTrading, WhyNotTradingResponse } from "@/app/api/client";
import { fetchKillswitchStatus, resetKillswitch, KillswitchStatusResponse } from "@/app/api/killswitch_client";
import { toast } from "sonner";

export function WhyNotTradingPanel() {
  const [data, setData] = useState<WhyNotTradingResponse | null>(null);
  const [killswitchData, setKillswitchData] = useState<KillswitchStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [resetting, setResetting] = useState(false);

  useEffect(() => {
    const load = async () => {
      try {
        const [whyNotTrading, killswitch] = await Promise.all([
          fetchWhyNotTrading(),
          fetchKillswitchStatus().catch(() => null),
        ]);
        setData(whyNotTrading);
        setKillswitchData(killswitch);
      } catch (err) {
        console.error("Failed to load why_not_trading:", err);
      } finally {
        setLoading(false);
      }
    };
    load();
    const id = window.setInterval(load, 5000); // Refresh every 5s
    return () => clearInterval(id);
  }, []);

  const handleResetKillswitch = async () => {
    if (!window.confirm("Clear killswitch? This will allow new orders. Continue?")) {
      return;
    }
    setResetting(true);
    try {
      const result = await resetKillswitch(true);
      toast.success(result.message || "Killswitch cleared", { description: "CONTROL_RESULT · killswitch_reset" });
      // Reload data
      const [whyNotTrading, killswitch] = await Promise.all([
        fetchWhyNotTrading(),
        fetchKillswitchStatus().catch(() => null),
      ]);
      setData(whyNotTrading);
      setKillswitchData(killswitch);
    } catch (err: any) {
      toast.error(err?.message || "Failed to reset killswitch", { description: "CONTROL_RESULT · killswitch_reset" });
    } finally {
      setResetting(false);
    }
  };

  if (loading) {
    return (
      <Card className="border border-border/70 bg-card/40 backdrop-blur-sm">
        <div className="p-4 text-sm text-muted-foreground">Loading...</div>
      </Card>
    );
  }

  if (!data) {
    return (
      <Card className="border border-border/70 bg-card/40 backdrop-blur-sm">
        <div className="p-4 text-sm text-muted-foreground">Unable to load trading status</div>
      </Card>
    );
  }

  const { engine_running, top_reasons, risk_status, strategy_state, engine_state } = data;

  return (
    <Card className="border border-border/70 bg-card/40 backdrop-blur-sm">
      <div className="p-4 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wider">WHY NOT TRADING?</h3>
          <Badge
            variant={engine_running ? "outline" : "destructive"}
            className={
              engine_running
                ? "text-[var(--trading-green)] border-[var(--trading-green)]/50"
                : "text-[var(--trading-red)] border-[var(--trading-red)]/50"
            }
          >
            {engine_state}
          </Badge>
        </div>

        {engine_running && top_reasons.length === 0 && (
          <div className="flex items-center gap-2 text-sm text-[var(--trading-green)]">
            <CheckCircle2 className="h-4 w-4" />
            <span>Engine is running and ready to trade</span>
          </div>
        )}

        {top_reasons.length > 0 && (
          <div className="space-y-2">
            <div className="text-xs uppercase tracking-wider text-muted-foreground">Top Reasons</div>
            {top_reasons.map((reason, idx) => {
              const Icon =
                reason.severity === "error"
                  ? XCircle
                  : reason.severity === "warning"
                  ? AlertCircle
                  : Clock;
              const color =
                reason.severity === "error"
                  ? "text-[var(--trading-red)]"
                  : reason.severity === "warning"
                  ? "text-[var(--trading-amber)]"
                  : "text-[var(--trading-cyan)]";
              return (
                <div key={idx} className="flex items-start gap-2 p-2 rounded border border-border/50 bg-background/50">
                  <Icon className={`h-4 w-4 mt-0.5 ${color}`} />
                  <div className="flex-1">
                    <div className="text-sm font-mono">{reason.code}</div>
                    <div className="text-xs text-muted-foreground">{reason.message}</div>
                    {reason.count && (
                      <div className="text-xs text-muted-foreground mt-1">Count: {reason.count}</div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {risk_status.killswitch_triggered && killswitchData && (
          <div className="space-y-3 p-3 rounded border border-[var(--trading-red)]/50 bg-[var(--trading-red)]/10">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <XCircle className="h-4 w-4 text-[var(--trading-red)]" />
                <span className="text-sm text-[var(--trading-red)] font-bold">KILLSWITCH TRIGGERED</span>
              </div>
              <Button
                size="sm"
                variant="outline"
                className="h-7 text-xs border-[var(--trading-red)]/50 text-[var(--trading-red)] hover:bg-[var(--trading-red)]/20"
                onClick={handleResetKillswitch}
                disabled={resetting}
              >
                <RotateCcw className="h-3 w-3 mr-1" />
                Clear
              </Button>
            </div>
            
            {killswitchData.triggered_at && (
              <div className="text-xs text-muted-foreground">
                Triggered: {new Date(killswitchData.triggered_at).toLocaleString()}
              </div>
            )}
            
            {killswitchData.reasons.length > 0 && (
              <div className="space-y-1">
                <div className="text-xs font-semibold text-[var(--trading-red)]">Reasons:</div>
                {killswitchData.reasons.map((reason, idx) => (
                  <div key={idx} className="text-xs text-muted-foreground font-mono pl-2 border-l-2 border-[var(--trading-red)]/30">
                    {reason}
                  </div>
                ))}
              </div>
            )}
            
            {killswitchData.current_metrics && (
              <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-[var(--trading-red)]/30">
                <div>
                  <div className="text-muted-foreground">Current Drawdown</div>
                  <div className="font-mono text-[var(--trading-red)]">
                    {killswitchData.current_metrics.current_drawdown_pct.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Limit</div>
                  <div className="font-mono">
                    {killswitchData.thresholds.drawdown_pct.toFixed(2)}%
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Equity</div>
                  <div className="font-mono">
                    ${killswitchData.current_metrics.current_equity.toFixed(2)}
                  </div>
                </div>
                <div>
                  <div className="text-muted-foreground">Max Drawdown</div>
                  <div className="font-mono text-[var(--trading-red)]">
                    {killswitchData.current_metrics.max_drawdown_pct.toFixed(2)}%
                  </div>
                </div>
              </div>
            )}
            
            <div className="text-xs text-[var(--trading-red)] font-semibold pt-1 border-t border-[var(--trading-red)]/30">
              NEW_ORDERS_BLOCKED
            </div>
          </div>
        )}
        
        {risk_status.killswitch_triggered && !killswitchData && (
          <div className="flex items-center gap-2 p-2 rounded border border-[var(--trading-red)]/50 bg-[var(--trading-red)]/10">
            <XCircle className="h-4 w-4 text-[var(--trading-red)]" />
            <span className="text-sm text-[var(--trading-red)] font-bold">KILLSWITCH TRIGGERED</span>
          </div>
        )}

        {risk_status.stop_trading && (
          <div className="flex items-center gap-2 p-2 rounded border border-[var(--trading-red)]/50 bg-[var(--trading-red)]/10">
            <TrendingDown className="h-4 w-4 text-[var(--trading-red)]" />
            <span className="text-sm text-[var(--trading-red)]">Trading stopped by risk manager</span>
          </div>
        )}

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div>
            <div className="text-muted-foreground">Positions</div>
            <div className="font-mono">
              {risk_status.current_positions} / {risk_status.max_positions}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">Orders</div>
            <div className="font-mono">
              {risk_status.current_orders} / {risk_status.max_orders}
            </div>
          </div>
          <div>
            <div className="text-muted-foreground">Strategies</div>
            <div className="font-mono">
              {strategy_state.enabled} / {strategy_state.total} enabled
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

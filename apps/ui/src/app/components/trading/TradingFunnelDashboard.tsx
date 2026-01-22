import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { AlertCircle, TrendingUp, TrendingDown, Activity, DollarSign, BarChart3 } from "lucide-react";
import { TradingFunnelMetrics } from "@/app/hooks/useTradingData";
import { PositionsTable, Position } from "./PositionsTable";
import { OrdersTable, OrderRow } from "./OrdersTable";

interface TradingFunnelDashboardProps {
  funnelMetrics: TradingFunnelMetrics | null;
  positions: Position[];
  orders: OrderRow[];
  recentFills?: OrderRow[];
  backendAvailable: boolean;
}

export function TradingFunnelDashboard({
  funnelMetrics,
  positions,
  orders,
  recentFills = [],
  backendAvailable,
}: TradingFunnelDashboardProps) {
  if (!funnelMetrics) {
    return (
      <Card className="p-6 border-[var(--trading-amber)]/60 bg-[var(--trading-amber)]/10">
        <div className="flex items-center gap-3 text-sm">
          <AlertCircle className="h-5 w-5 text-[var(--trading-amber)]" />
          <div>
            <div className="font-semibold text-[var(--trading-amber)]">TRADING TELEMETRY NOT WIRED</div>
            <div className="text-xs text-muted-foreground">
              Backend trading_metrics topic not available. Check backend logs.
            </div>
          </div>
        </div>
      </Card>
    );
  }

  const hasNoSignals = funnelMetrics.signals_total_session === 0;
  const signalsPerMin = funnelMetrics.signals_total_1m;
  const noSignalsFor60s = !funnelMetrics.last_signal_ts || 
    (Date.now() - new Date(funnelMetrics.last_signal_ts).getTime()) > 60000;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-xl font-mono tracking-tight mb-4">Trading Funnel</h2>
        
        {/* KPI Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 mb-4">
          {/* Signals */}
          <Card className="p-3">
            <div className="text-xs text-muted-foreground mb-1">Signals/min</div>
            <div className="text-lg font-mono">{signalsPerMin}</div>
            <div className="text-xs text-muted-foreground mt-1">Session: {funnelMetrics.signals_total_session}</div>
          </Card>
          
          {/* Decisions */}
          <Card className="p-3">
            <div className="text-xs text-muted-foreground mb-1">Decisions</div>
            <div className="text-sm font-mono">
              <div className="text-green-400">✓ {funnelMetrics.decisions_create}</div>
              <div className="text-yellow-400">⊘ {funnelMetrics.decisions_skip}</div>
              <div className="text-red-400">✗ {funnelMetrics.decisions_block}</div>
            </div>
          </Card>
          
          {/* Orders */}
          <Card className="p-3">
            <div className="text-xs text-muted-foreground mb-1">Orders</div>
            <div className="text-sm font-mono">
              <div>Open: {funnelMetrics.orders_open}</div>
              <div className="text-green-400">Filled: {funnelMetrics.orders_filled}</div>
              <div className="text-red-400">Canceled: {funnelMetrics.orders_canceled}</div>
              <div className="text-red-400">Rejected: {funnelMetrics.orders_rejected}</div>
            </div>
          </Card>
          
          {/* Positions */}
          <Card className="p-3">
            <div className="text-xs text-muted-foreground mb-1">Positions</div>
            <div className="text-sm font-mono">
              <div>Open: {funnelMetrics.positions_open}</div>
              <div className="text-muted-foreground">Closed: {funnelMetrics.positions_closed}</div>
            </div>
          </Card>
          
          {/* PnL */}
          <Card className="p-3">
            <div className="text-xs text-muted-foreground mb-1">PnL</div>
            <div className="text-sm font-mono">
              <div className={funnelMetrics.pnl_unrealized >= 0 ? "text-green-400" : "text-red-400"}>
                Unreal: ${funnelMetrics.pnl_unrealized.toFixed(2)}
              </div>
              <div className={funnelMetrics.pnl_realized >= 0 ? "text-green-400" : "text-red-400"}>
                Real: ${funnelMetrics.pnl_realized.toFixed(2)}
              </div>
            </div>
          </Card>
          
          {/* Equity/Margin */}
          <Card className="p-3">
            <div className="text-xs text-muted-foreground mb-1">Equity/Margin</div>
            <div className="text-sm font-mono">
              <div>Equity: ${funnelMetrics.equity.toFixed(2)}</div>
              <div className="text-muted-foreground">Used: ${funnelMetrics.used_margin.toFixed(2)}</div>
              <div className="text-muted-foreground">Free: ${funnelMetrics.free_margin.toFixed(2)}</div>
            </div>
          </Card>
        </div>
        
        {/* WHY ORDERS ARE LOW Panel */}
        <Card className="p-4 mb-4 border-[var(--trading-amber)]/60">
          <div className="flex items-center gap-2 mb-3">
            <BarChart3 className="h-4 w-4 text-[var(--trading-amber)]" />
            <h3 className="font-semibold">WHY ORDERS ARE LOW</h3>
          </div>
          
          {hasNoSignals && noSignalsFor60s && (
            <div className="mb-3 p-2 bg-[var(--trading-amber)]/10 rounded text-sm">
              <div className="font-semibold text-[var(--trading-amber)]">No signals detected</div>
              <div className="text-xs text-muted-foreground">
                {funnelMetrics.last_signal_ts 
                  ? `Last signal: ${new Date(funnelMetrics.last_signal_ts).toLocaleTimeString()}`
                  : "No signals in session"}
              </div>
            </div>
          )}
          
          {funnelMetrics.top_reasons.length > 0 ? (
            <div className="space-y-2">
              {funnelMetrics.top_reasons.slice(0, 5).map((reason, idx) => (
                <div key={idx} className="flex items-center justify-between text-sm">
                  <Badge variant="outline" className="font-mono text-xs">
                    {reason.code}
                  </Badge>
                  <div className="text-muted-foreground">
                    {reason.count} ({reason.pct}%)
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">No recent skip/block reasons</div>
          )}
          
          <div className="mt-3 pt-3 border-t border-border/50 text-xs text-muted-foreground grid grid-cols-3 gap-2">
            <div>
              Last signal: {funnelMetrics.last_signal_ts 
                ? new Date(funnelMetrics.last_signal_ts).toLocaleTimeString()
                : "—"}
            </div>
            <div>
              Last order: {funnelMetrics.last_order_ts
                ? new Date(funnelMetrics.last_order_ts).toLocaleTimeString()
                : "—"}
            </div>
            <div>
              Last fill: {funnelMetrics.last_fill_ts
                ? new Date(funnelMetrics.last_fill_ts).toLocaleTimeString()
                : "—"}
            </div>
          </div>
        </Card>
        
        {/* Tables */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card className="p-4">
            <h3 className="font-semibold mb-3">Open Positions</h3>
            {positions.length > 0 ? (
              <PositionsTable positions={positions} />
            ) : (
              <div className="text-sm text-muted-foreground">No open positions</div>
            )}
          </Card>
          
          <Card className="p-4">
            <h3 className="font-semibold mb-3">Open Orders</h3>
            {orders.length > 0 ? (
              <OrdersTable orders={orders} />
            ) : (
              <div className="text-sm text-muted-foreground">No open orders</div>
            )}
          </Card>
        </div>
        
        {recentFills.length > 0 && (
          <Card className="p-4">
            <h3 className="font-semibold mb-3">Recent Fills</h3>
            <OrdersTable orders={recentFills} />
          </Card>
        )}
      </div>
    </div>
  );
}

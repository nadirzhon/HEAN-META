import { AlertTriangle, WifiOff } from "lucide-react";
import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { StatusBar } from "@/app/components/trading/StatusBar";
import { EventFeed } from "@/app/components/trading/EventFeed";
import { ControlPanel } from "@/app/components/trading/ControlPanel";
import { PortfolioCard } from "@/app/components/trading/PortfolioCard";
import { DebugPanel } from "@/app/components/trading/DebugPanel";
import { WhyNotTradingPanel } from "@/app/components/trading/WhyNotTradingPanel";
import { TradingFunnelDashboard } from "@/app/components/trading/TradingFunnelDashboard";
import { useTradingData } from "@/app/hooks/useTradingData";
import { Toaster } from "sonner";
import { ErrorBoundary } from "@/app/components/ErrorBoundary";

function App() {
  const {
    account,
    portfolio,
    metrics,
    telemetry,
    ws,
    pulse,
    health,
    backendAvailable,
    lastBackendError,
    eventFeed,
    controlSupport,
    controlStatus,
    runControl,
    usingLiveData,
    error,
    lastSync,
    apiBase,
    wsUrl,
    funnelMetrics,
    positions,
    orders,
  } = useTradingData();

  const wsUnhealthy = ws.status !== "connected";

  return (
    <ErrorBoundary>
      <div className="dark min-h-screen bg-background">
      <StatusBar
        pulse={pulse}
        telemetry={telemetry}
        ws={ws}
        backendAvailable={backendAvailable}
        health={health}
        lastSync={lastSync}
      />

      <div className="max-w-screen-2xl mx-auto px-4 sm:px-6 pt-[78px] pb-6 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-mono tracking-tight">HEAN Cockpit</h1>
            <p className="text-sm text-muted-foreground">Operator view · live telemetry + controls</p>
            {pulse.restHealth === "degraded" && (
              <div className="mt-1 text-xs text-[var(--trading-amber)]">REST polling partially degraded — some endpoints unavailable.</div>
            )}
            {pulse.restHealth === "error" && (
              <div className="mt-1 text-xs text-[var(--trading-red)]">REST polling failed — showing cached/mock data.</div>
            )}
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            <Badge
              variant="outline"
              className={
                usingLiveData
                  ? "text-[var(--trading-green)] border-[var(--trading-green)]/40"
                  : "text-[var(--trading-amber)] border-[var(--trading-amber)]/40"
              }
            >
              {usingLiveData ? "Live" : "Mock"}
            </Badge>
            <Badge variant="outline" className="font-mono">
              Equity ${metrics.equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}
            </Badge>
            <Badge 
              variant="outline" 
              className={`font-mono ${
                metrics.dailyPnl >= 0 
                  ? "text-[var(--trading-green)] border-[var(--trading-green)]/40" 
                  : "text-[var(--trading-red)] border-[var(--trading-red)]/40"
              }`}
            >
              PnL ${metrics.dailyPnl >= 0 ? "+" : ""}${metrics.dailyPnl.toFixed(2)}
            </Badge>
            <Badge variant="outline" className="font-mono">
              Orders: {orders.length}
            </Badge>
            {metrics.initialCapital && (
              <Badge variant="outline" className="font-mono text-muted-foreground">
                Start: ${metrics.initialCapital.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </Badge>
            )}
          </div>
        </div>

        {wsUnhealthy && (
          <Card className="border-[var(--trading-amber)]/60 bg-[var(--trading-amber)]/10 px-4 py-3 flex items-center gap-3 text-sm">
            <WifiOff className="h-4 w-4 text-[var(--trading-amber)]" />
            <div>
              <div className="font-semibold text-[var(--trading-amber)]">WebSocket not connected</div>
              <div className="text-xs text-muted-foreground">
                Autoreconnect in progress · attempts {ws.reconnectAttempts} · last message{" "}
                {ws.lastMessageAt ? new Date(ws.lastMessageAt).toLocaleTimeString("en-US", { hour12: false }) : "—"}
              </div>
            </div>
          </Card>
        )}

        {!backendAvailable && (
          <Card className="border-[var(--trading-amber)]/60 bg-[var(--trading-amber)]/10 px-4 py-3 flex items-center gap-3 text-sm">
            <AlertTriangle className="h-4 w-4 text-[var(--trading-amber)]" />
            <div>
              <div className="font-semibold text-[var(--trading-amber)]">Backend unavailable</div>
              <div className="text-xs text-muted-foreground">
                Polling /telemetry/summary & /portfolio/summary failed. Showing cached values.
                {lastBackendError ? ` (${lastBackendError})` : ""}
              </div>
            </div>
          </Card>
        )}

        {error && (
          <Card className="border-[var(--trading-amber)]/50 bg-[var(--trading-amber)]/10 px-4 py-3 flex items-center gap-3 text-sm">
            <AlertTriangle className="h-4 w-4 text-[var(--trading-amber)]" />
            <span>API sync warning: {error}</span>
          </Card>
        )}

        <TradingFunnelDashboard
          funnelMetrics={funnelMetrics}
          positions={positions}
          orders={orders}
          backendAvailable={backendAvailable}
        />

        <div className="grid grid-cols-12 gap-4">
          <div className="col-span-12 lg:col-span-4 space-y-3">
            <PortfolioCard portfolio={portfolio} account={account} backendAvailable={backendAvailable} />
            <WhyNotTradingPanel />
            <ControlPanel
              onAction={runControl}
              support={controlSupport}
              controlStatus={controlStatus}
              backendAvailable={backendAvailable}
            />
            <DebugPanel
              apiBase={apiBase}
              wsUrl={wsUrl}
              backendAvailable={backendAvailable}
              lastBackendError={lastBackendError}
              ws={ws}
            />
          </div>

          <div className="col-span-12 lg:col-span-8 space-y-3">
            <EventFeed events={eventFeed} />
          </div>
        </div>
      </div>
      <Toaster richColors closeButton position="top-center" />
    </div>
    </ErrorBoundary>
  );
}

export default App;

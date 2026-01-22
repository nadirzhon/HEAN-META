import { useState } from "react";
import { Activity, Brain, Shield, Target, HelpCircle, Sparkles, Eye } from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/app/components/ui/tabs";
import { Card } from "@/app/components/ui/card";
import { StatusBar } from "@/app/components/trading/StatusBar";
import { MetricsCard } from "@/app/components/trading/MetricsCard";
import { SystemMessage } from "@/app/components/trading/SystemMessage";
import { PositionsTable, Position } from "@/app/components/trading/PositionsTable";
import { DecisionsTimeline, Decision } from "@/app/components/trading/DecisionsTimeline";
import { DecisionFlow, FlowStep } from "@/app/components/trading/DecisionFlow";
import { RiskMonitor, RiskMetric } from "@/app/components/trading/RiskMonitor";
import { StrategyCard, Strategy } from "@/app/components/trading/StrategyCard";
import { WhyResultPanel, TradeResult, ResultFactor } from "@/app/components/trading/WhyResultPanel";
import { WhatIfSimulator } from "@/app/components/trading/WhatIfSimulator";
import { MarketRealityView } from "@/app/components/trading/MarketRealityView";

// Mock data
const mockPositions: Position[] = [
  {
    id: "1",
    symbol: "BTC/USD",
    side: "LONG",
    entryPrice: 43250.50,
    lastPrice: 43890.25,
    unrealizedPnL: 639.75,
    takeProfit: 44500.00,
    stopLoss: 42800.00,
    ttl: 3420,
    status: "ACTIVE",
  },
  {
    id: "2",
    symbol: "ETH/USD",
    side: "SHORT",
    entryPrice: 2340.80,
    lastPrice: 2310.15,
    unrealizedPnL: 30.65,
    takeProfit: 2250.00,
    stopLoss: 2380.00,
    ttl: 1890,
    status: "ACTIVE",
  },
  {
    id: "3",
    symbol: "SOL/USD",
    side: "LONG",
    entryPrice: 98.45,
    lastPrice: 96.80,
    unrealizedPnL: -1.65,
    takeProfit: 102.00,
    stopLoss: 95.00,
    ttl: 2640,
    status: "ACTIVE",
  },
];

const mockDecisions: Decision[] = [
  {
    id: "1",
    type: "ORDER_DECISION",
    symbol: "BTC/USD",
    reasonCode: "MOMENTUM_BREAKOUT",
    outcome: "ENTRY",
    timestamp: new Date(Date.now() - 120000),
  },
  {
    id: "2",
    type: "EXIT_DECISION",
    symbol: "AVAX/USD",
    reasonCode: "TP_REACHED",
    outcome: "TP_HIT",
    timestamp: new Date(Date.now() - 300000),
  },
  {
    id: "3",
    type: "ORDER_DECISION",
    symbol: "ETH/USD",
    reasonCode: "RSI_OVERSOLD",
    outcome: "ENTRY",
    timestamp: new Date(Date.now() - 480000),
  },
  {
    id: "4",
    type: "EXIT_DECISION",
    symbol: "MATIC/USD",
    reasonCode: "TTL_EXPIRED",
    outcome: "TIMEOUT_TTL",
    timestamp: new Date(Date.now() - 720000),
  },
  {
    id: "5",
    type: "ORDER_DECISION",
    symbol: "LINK/USD",
    reasonCode: "VOLATILITY_TOO_HIGH",
    outcome: "SUPPRESSED",
    timestamp: new Date(Date.now() - 900000),
  },
];

const mockDecisionFlows: Array<{ symbol: string; steps: FlowStep[]; timestamp: Date }> = [
  {
    symbol: "BTC/USD",
    timestamp: new Date(Date.now() - 120000),
    steps: [
      { stage: "TICK", status: "passed", reasonCode: "Price data received" },
      { stage: "SIGNAL", status: "passed", reasonCode: "MOMENTUM_BREAKOUT detected" },
      { stage: "FILTER", status: "passed", reasonCode: "Trend alignment confirmed" },
      { stage: "RISK", status: "passed", reasonCode: "Position size approved" },
      { stage: "ORDER", status: "passed", reasonCode: "Order placed successfully" },
      { stage: "EXIT", status: "passed", reasonCode: "Exit monitors active" },
    ],
  },
  {
    symbol: "LINK/USD",
    timestamp: new Date(Date.now() - 900000),
    steps: [
      { stage: "TICK", status: "passed", reasonCode: "Price data received" },
      { stage: "SIGNAL", status: "passed", reasonCode: "RSI_OVERSOLD detected" },
      { stage: "FILTER", status: "blocked", reasonCode: "Volatility too high", details: "ATR > 2.5%" },
      { stage: "RISK", status: "suppressed" },
      { stage: "ORDER", status: "suppressed" },
      { stage: "EXIT", status: "suppressed" },
    ],
  },
  {
    symbol: "ETH/USD",
    timestamp: new Date(Date.now() - 480000),
    steps: [
      { stage: "TICK", status: "passed", reasonCode: "Price data received" },
      { stage: "SIGNAL", status: "passed", reasonCode: "Mean reversion signal" },
      { stage: "FILTER", status: "passed", reasonCode: "Volume confirmed" },
      { stage: "RISK", status: "suppressed", reasonCode: "Max positions reached", details: "3/3 slots used" },
      { stage: "ORDER", status: "suppressed" },
      { stage: "EXIT", status: "suppressed" },
    ],
  },
];

const mockRiskMetrics: RiskMetric[] = [
  {
    label: "EventBus Pressure (P0)",
    value: 142,
    max: 1000,
    unit: "msg/s",
    status: "ok",
    isWired: true,
  },
  {
    label: "EventBus Pressure (P1)",
    value: 387,
    max: 500,
    unit: "msg/s",
    status: "warning",
    isWired: true,
  },
  {
    label: "Exit Latency",
    value: 23,
    max: 100,
    unit: "ms",
    status: "ok",
    isWired: true,
  },
  {
    label: "Dropped Ticks Rate",
    value: 0.3,
    max: 5,
    unit: "%",
    status: "ok",
    isWired: true,
  },
  {
    label: "Decision Memory Blocks",
    value: 1247,
    max: 10000,
    unit: "blocks",
    status: "ok",
    isWired: true,
  },
  {
    label: "WebSocket Lag",
    value: 0,
    max: 500,
    unit: "ms",
    status: "ok",
    isWired: false,
  },
];

const mockStrategies: Strategy[] = [
  {
    id: "1",
    name: "Momentum Breakout",
    enabled: true,
    symbols: ["BTC/USD", "ETH/USD", "SOL/USD"],
    aggression: "HIGH",
    signalsPerMinute: 12,
    lastDecision: {
      outcome: "ENTRY",
      timestamp: new Date(Date.now() - 120000),
    },
    isWired: true,
  },
  {
    id: "2",
    name: "Mean Reversion",
    enabled: true,
    symbols: ["BTC/USD", "ETH/USD", "AVAX/USD", "MATIC/USD"],
    aggression: "MEDIUM",
    signalsPerMinute: 8,
    lastDecision: {
      outcome: "SUPPRESSED",
      timestamp: new Date(Date.now() - 480000),
    },
    isWired: true,
  },
  {
    id: "3",
    name: "Arbitrage Scanner",
    enabled: false,
    symbols: ["BTC/USD", "ETH/USD"],
    aggression: "LOW",
    signalsPerMinute: 3,
    isWired: false,
  },
  {
    id: "4",
    name: "Volatility Capture",
    enabled: true,
    symbols: ["SOL/USD", "AVAX/USD", "LINK/USD", "MATIC/USD", "DOT/USD"],
    aggression: "HIGH",
    signalsPerMinute: 15,
    lastDecision: {
      outcome: "HOLD",
      timestamp: new Date(Date.now() - 60000),
    },
    isWired: true,
  },
];

// Mock trade results for Why Result view
const mockTradeResults: TradeResult[] = [
  {
    symbol: "AVAX/USD",
    side: "LONG",
    result: 12.45,
    entryPrice: 38.20,
    exitPrice: 39.65,
    duration: "23m 14s",
    timestamp: new Date(Date.now() - 300000),
    factors: [
      {
        stage: "Entry Signal",
        passed: true,
        explanation: "Entry aligned with impulse regime — strong momentum detected",
        impact: "positive",
      },
      {
        stage: "Position Sizing",
        passed: true,
        explanation: "Risk allowed full position size — portfolio exposure within limits",
        impact: "positive",
      },
      {
        stage: "Volatility Filter",
        passed: false,
        explanation: "Volatility filter partially suppressed exposure — ATR above threshold, position reduced by 40%",
        impact: "negative",
      },
      {
        stage: "Exit Execution",
        passed: true,
        explanation: "Exit triggered by TP_HIT — target price reached, clean execution",
        impact: "positive",
      },
    ],
  },
  {
    symbol: "SOL/USD",
    side: "SHORT",
    result: -3.20,
    entryPrice: 96.80,
    exitPrice: 97.12,
    duration: "14m 08s",
    timestamp: new Date(Date.now() - 1200000),
    factors: [
      {
        stage: "Entry Signal",
        passed: true,
        explanation: "Mean reversion signal triggered — RSI oversold condition met",
        impact: "positive",
      },
      {
        stage: "Trend Alignment",
        passed: false,
        explanation: "Trend filter showed conflicting signals — entered counter-trend position",
        impact: "negative",
      },
      {
        stage: "Position Sizing",
        passed: true,
        explanation: "Position size reduced to 50% due to trend uncertainty",
        impact: "neutral",
      },
      {
        stage: "Exit Execution",
        passed: false,
        explanation: "Stop loss hit — price moved against position before target reached",
        impact: "negative",
      },
    ],
  },
  {
    symbol: "BTC/USD",
    side: "LONG",
    result: 84.30,
    entryPrice: 43150.00,
    exitPrice: 43890.25,
    duration: "1h 34m",
    timestamp: new Date(Date.now() - 120000),
    factors: [
      {
        stage: "Entry Signal",
        passed: true,
        explanation: "Momentum breakout signal confirmed — volume surge detected",
        impact: "positive",
      },
      {
        stage: "Trend Alignment",
        passed: true,
        explanation: "Multiple timeframe trend alignment — all indicators bullish",
        impact: "positive",
      },
      {
        stage: "Risk Management",
        passed: true,
        explanation: "Optimal risk/reward ratio (1:3.2) — stop loss properly positioned",
        impact: "positive",
      },
      {
        stage: "Exit Execution",
        passed: true,
        explanation: "Partial TP hit, trailing stop secured remainder — maximized profit",
        impact: "positive",
      },
    ],
  },
];

function App() {
  const [activeTab, setActiveTab] = useState("overview");

  const handleKillSwitch = () => {
    console.log("KILL SWITCH ACTIVATED");
    // In production: emergency shutdown logic
  };

  const handleStrategyToggle = (id: string, enabled: boolean) => {
    console.log(`Strategy ${id} toggled to ${enabled}`);
    // In production: API call to enable/disable strategy
  };

  // Calculate summary metrics
  const totalPnL = mockPositions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);
  const openPositionsCount = mockPositions.length;
  const openOrdersCount = mockPositions.filter(p => p.status === "PENDING").length;

  return (
    <div className="dark min-h-screen bg-background">
      <StatusBar
        systemMode="ACTIVE"
        confidence={78}
        health="OK"
        onKillSwitch={handleKillSwitch}
      />

      <div className="container mx-auto px-6 py-6">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-3xl font-mono tracking-tight">HEAN</h1>
            <p className="text-sm text-muted-foreground mt-1">
              Algorithmic Trading System
            </p>
          </div>
          <div className="text-right">
            <div className="text-xs text-muted-foreground uppercase tracking-wider">System Time</div>
            <div className="font-mono tabular-nums text-sm">
              {new Date().toLocaleString("en-US", { 
                month: "short", 
                day: "numeric", 
                year: "numeric",
                hour: "2-digit", 
                minute: "2-digit",
                second: "2-digit",
                hour12: false 
              })}
            </div>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="bg-card/50 backdrop-blur-sm border border-border">
            <TabsTrigger value="overview" className="gap-2 data-[state=active]:bg-muted">
              <Activity className="h-4 w-4" />
              Overview
            </TabsTrigger>
            <TabsTrigger value="decisions" className="gap-2 data-[state=active]:bg-muted">
              <Brain className="h-4 w-4" />
              Decisions
            </TabsTrigger>
            <TabsTrigger value="why-result" className="gap-2 data-[state=active]:bg-muted">
              <HelpCircle className="h-4 w-4" />
              Why Result
            </TabsTrigger>
            <TabsTrigger value="risk" className="gap-2 data-[state=active]:bg-muted">
              <Shield className="h-4 w-4" />
              Risk & Health
            </TabsTrigger>
            <TabsTrigger value="strategies" className="gap-2 data-[state=active]:bg-muted">
              <Target className="h-4 w-4" />
              Strategies
            </TabsTrigger>
            <TabsTrigger value="what-if" className="gap-2 data-[state=active]:bg-muted">
              <Sparkles className="h-4 w-4" />
              What-If Mode
            </TabsTrigger>
            <TabsTrigger value="market-reality" className="gap-2 data-[state=active]:bg-muted">
              <Eye className="h-4 w-4" />
              Market Reality
            </TabsTrigger>
          </TabsList>

          {/* OVERVIEW TAB */}
          <TabsContent value="overview" className="space-y-6">
            {/* Core Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
              <MetricsCard
                label="Equity"
                value="$124,567"
                delta="+$2,345"
                deltaType="positive"
                valueColor="text-[var(--trading-cyan)]"
              />
              <MetricsCard
                label="Daily PnL"
                value={`$${totalPnL.toFixed(2)}`}
                delta="+1.89%"
                deltaType="positive"
                valueColor={totalPnL >= 0 ? "text-[var(--trading-green)]" : "text-[var(--trading-red)]"}
              />
              <MetricsCard
                label="Open Positions"
                value={openPositionsCount}
              />
              <MetricsCard
                label="Open Orders"
                value={openOrdersCount}
              />
              <MetricsCard
                label="Market Regime"
                value="TREND"
                valueColor="text-[var(--trading-amber)]"
              />
            </div>

            {/* System Message */}
            <SystemMessage 
              message="Market volatility is elevated, impulse is weakening, risk limits are active, exits are protected." 
              marketRegime="TREND"
              volatilityLevel="HIGH"
              riskState="LIMITED"
            />

            {/* Positions Table */}
            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                Active Positions
              </h3>
              <PositionsTable positions={mockPositions} />
            </div>

            {/* Recent Decisions */}
            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                Recent Decisions
              </h3>
              <Card className="p-6 bg-card/30 backdrop-blur-sm border-border/50">
                <DecisionsTimeline decisions={mockDecisions} />
              </Card>
            </div>
          </TabsContent>

          {/* DECISIONS TAB */}
          <TabsContent value="decisions" className="space-y-6">
            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                Decision Flow Visualization
              </h3>
              <p className="text-sm text-muted-foreground mb-6">
                Each decision flows through multiple stages: TICK → SIGNAL → FILTER → RISK → ORDER → EXIT
              </p>
              <div className="space-y-4">
                {mockDecisionFlows.map((flow, index) => (
                  <DecisionFlow
                    key={index}
                    symbol={flow.symbol}
                    steps={flow.steps}
                    timestamp={flow.timestamp}
                  />
                ))}
              </div>
            </div>

            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                Decision History
              </h3>
              <Card className="p-6 bg-card/30 backdrop-blur-sm border-border/50">
                <DecisionsTimeline decisions={mockDecisions} />
              </Card>
            </div>
          </TabsContent>

          {/* WHY RESULT TAB */}
          <TabsContent value="why-result" className="space-y-6">
            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                Trade Result Analysis
              </h3>
              <p className="text-sm text-muted-foreground mb-6">
                Analyze the factors that influenced the outcome of recent trades.
              </p>
              <div className="space-y-4">
                {mockTradeResults.map((result, index) => (
                  <WhyResultPanel
                    key={index}
                    result={result}
                  />
                ))}
              </div>
            </div>
          </TabsContent>

          {/* RISK & HEALTH TAB */}
          <TabsContent value="risk" className="space-y-6">
            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                System Health Metrics
              </h3>
              <RiskMonitor metrics={mockRiskMetrics} />
            </div>

            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                Performance Indicators
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <MetricsCard
                  label="Win Rate"
                  value="67.3%"
                  valueColor="text-[var(--trading-green)]"
                />
                <MetricsCard
                  label="Avg Trade Duration"
                  value="47m"
                  valueColor="text-[var(--trading-cyan)]"
                />
                <MetricsCard
                  label="Sharpe Ratio"
                  value="2.14"
                  valueColor="text-[var(--trading-cyan)]"
                />
              </div>
            </div>
          </TabsContent>

          {/* STRATEGIES TAB */}
          <TabsContent value="strategies" className="space-y-6">
            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                Active Strategies
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {mockStrategies.map((strategy) => (
                  <StrategyCard
                    key={strategy.id}
                    strategy={strategy}
                    onToggle={handleStrategyToggle}
                  />
                ))}
              </div>
            </div>

            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                Strategy Performance
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Card className="p-5 bg-card/30 backdrop-blur-sm border-border/50">
                  <div className="text-sm text-muted-foreground mb-2">Total Signals Today</div>
                  <div className="text-3xl font-mono tabular-nums text-[var(--trading-cyan)]">1,247</div>
                </Card>
                <Card className="p-5 bg-card/30 backdrop-blur-sm border-border/50">
                  <div className="text-sm text-muted-foreground mb-2">Signal → Entry Rate</div>
                  <div className="text-3xl font-mono tabular-nums text-[var(--trading-green)]">12.4%</div>
                </Card>
              </div>
            </div>
          </TabsContent>

          {/* WHAT-IF MODE TAB */}
          <TabsContent value="what-if" className="space-y-6">
            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                What-If Mode
              </h3>
              <p className="text-sm text-muted-foreground mb-6">
                Simulate trades with different parameters to see potential outcomes.
              </p>
              <WhatIfSimulator />
            </div>
          </TabsContent>

          {/* MARKET REALITY TAB */}
          <TabsContent value="market-reality" className="space-y-6">
            <div>
              <h3 className="mb-4 text-xs text-muted-foreground uppercase tracking-wider">
                Market Reality View
              </h3>
              <p className="text-sm text-muted-foreground mb-6">
                Visualize the current market conditions and trends.
              </p>
              <MarketRealityView />
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

export default App;
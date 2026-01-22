import { useState } from "react";
import { TrendingUp, Eye, Layers } from "lucide-react";
import { Button } from "@/app/components/ui/button";
import { Switch } from "@/app/components/ui/switch";
import { Label } from "@/app/components/ui/label";
import { AdvancedChart } from "@/app/components/trading/AdvancedChart";
import { ProfitFocusPanel } from "@/app/components/trading/ProfitFocusPanel";
import { StrategyRiskTriangle } from "@/app/components/trading/StrategyRiskTriangle";
import { TimeControl } from "@/app/components/trading/TimeControl";
import { SystemNervousSystem } from "@/app/components/trading/SystemNervousSystem";
import { AutonomousExplanation } from "@/app/components/trading/AutonomousExplanation";
import { 
  generateCandleData, 
  generateDecisionMarkers, 
  generateProfitZones,
  generateOrderBlocks 
} from "@/app/utils/chartData";

export function MarketRealityView() {
  const [showProfitIntelligence, setShowProfitIntelligence] = useState(true);
  const [showDecisions, setShowDecisions] = useState(true);
  const [showGhostOverlay, setShowGhostOverlay] = useState(false);

  // Generate chart data
  const candleData = generateCandleData(180, 43000);
  const decisionMarkers = showDecisions ? generateDecisionMarkers(candleData) : [];
  const profitZones = generateProfitZones(candleData);
  const orderBlocks = generateOrderBlocks(candleData);

  const minTime = candleData[0]?.timestamp || Date.now();
  const maxTime = candleData[candleData.length - 1]?.timestamp || Date.now();
  const currentTime = maxTime;

  return (
    <div className="relative min-h-screen">
      {/* System Nervous System Background */}
      <SystemNervousSystem 
        eventBusLoad={42}
        tickRate={87}
        decisionRate={12}
      />

      <div className="relative z-10 space-y-6">
        {/* Header Controls */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-mono tracking-tight mb-1">Market Reality</h2>
            <p className="text-sm text-muted-foreground">
              Visual Trading Intelligence System
            </p>
          </div>

          <div className="flex items-center gap-6">
            <div className="flex items-center gap-2">
              <Switch
                id="profit-intelligence"
                checked={showProfitIntelligence}
                onCheckedChange={setShowProfitIntelligence}
              />
              <Label htmlFor="profit-intelligence" className="text-sm cursor-pointer">
                Profit Intelligence
              </Label>
            </div>

            <div className="flex items-center gap-2">
              <Switch
                id="decisions"
                checked={showDecisions}
                onCheckedChange={setShowDecisions}
              />
              <Label htmlFor="decisions" className="text-sm cursor-pointer">
                Decision Markers
              </Label>
            </div>

            <div className="flex items-center gap-2">
              <Switch
                id="ghost-overlay"
                checked={showGhostOverlay}
                onCheckedChange={setShowGhostOverlay}
              />
              <Label htmlFor="ghost-overlay" className="text-sm cursor-pointer">
                What-If Ghost
              </Label>
            </div>
          </div>
        </div>

        {/* Main Chart Area */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left Sidebar */}
          <div className="col-span-3 space-y-6">
            <ProfitFocusPanel
              metrics={{
                expected: 847.50,
                realized: 668.45,
                missed: 234.80,
                protectedLoss: 1247.30,
              }}
              timeframe="Today"
            />

            <TimeControl
              minTime={minTime}
              maxTime={maxTime}
              currentTime={currentTime}
              onTimeChange={(time) => console.log("Time changed:", time)}
            />

            <StrategyRiskTriangle
              position={{
                market: 45,
                strategy: 30,
                risk: 25,
              }}
            />
          </div>

          {/* Main Chart */}
          <div className="col-span-9">
            <div className="relative">
              {showGhostOverlay && (
                <div className="absolute inset-0 z-20 pointer-events-none">
                  <div className="absolute top-10 right-10 bg-card/90 backdrop-blur-sm border border-[var(--trading-cyan)]/30 rounded-lg p-4 max-w-xs">
                    <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">
                      Ghost Simulation
                    </div>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">If aggression +20%:</span>
                        <span className="text-[var(--trading-cyan)]">+$340 profit</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Risk increase:</span>
                        <span className="text-[var(--trading-amber)]">+15%</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Bus overload risk:</span>
                        <span className="text-[var(--trading-red)]">Medium</span>
                      </div>
                    </div>
                    <div className="mt-3 pt-3 border-t border-border/50">
                      <p className="text-xs text-muted-foreground italic">
                        Looking into parallel futures...
                      </p>
                    </div>
                  </div>
                </div>
              )}

              <AdvancedChart
                data={candleData}
                orderBlocks={orderBlocks}
                decisionMarkers={decisionMarkers}
                profitZones={showProfitIntelligence ? profitZones : []}
                showProfitIntelligence={showProfitIntelligence}
                height={700}
              />
            </div>

            {/* Chart Insights */}
            <div className="mt-4 grid grid-cols-3 gap-4">
              <div className="p-4 bg-card/30 backdrop-blur-sm border border-border/50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="h-4 w-4 text-[var(--trading-cyan)]" />
                  <span className="text-xs text-muted-foreground uppercase tracking-wider">
                    Regime Strength
                  </span>
                </div>
                <div className="text-2xl font-mono tabular-nums text-[var(--trading-cyan)]">
                  78%
                </div>
              </div>

              <div className="p-4 bg-card/30 backdrop-blur-sm border border-border/50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Eye className="h-4 w-4 text-[var(--trading-amber)]" />
                  <span className="text-xs text-muted-foreground uppercase tracking-wider">
                    Profit Zones Visible
                  </span>
                </div>
                <div className="text-2xl font-mono tabular-nums text-[var(--trading-amber)]">
                  {profitZones.length}
                </div>
              </div>

              <div className="p-4 bg-card/30 backdrop-blur-sm border border-border/50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <Layers className="h-4 w-4 text-[var(--trading-green)]" />
                  <span className="text-xs text-muted-foreground uppercase tracking-wider">
                    Decision Points
                  </span>
                </div>
                <div className="text-2xl font-mono tabular-nums text-[var(--trading-green)]">
                  {decisionMarkers.length}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Autonomous Explanation Engine */}
      <AutonomousExplanation
        systemMode="ACTIVE"
        marketRegime="TREND"
        positionCount={3}
        riskLevel={35}
        volatility={62}
      />
    </div>
  );
}

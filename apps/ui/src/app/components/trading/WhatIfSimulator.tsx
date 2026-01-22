import { useState } from "react";
import { Sliders, TrendingUp, AlertTriangle, Activity, Database } from "lucide-react";
import { Card } from "@/app/components/ui/card";
import { Slider } from "@/app/components/ui/slider";
import { Badge } from "@/app/components/ui/badge";
import { Alert, AlertDescription } from "@/app/components/ui/alert";

interface SimulationParams {
  aggression: number; // 0-100
  maxPositions: number; // 1-10
  symbolsCount: number; // 1-20
}

interface SimulationForecast {
  expectedTrades: number;
  expectedDrawdown: { min: number; max: number };
  eventBusStress: number; // 0-100
  exitRiskProbability: number; // 0-100
}

export function WhatIfSimulator() {
  const [params, setParams] = useState<SimulationParams>({
    aggression: 50,
    maxPositions: 3,
    symbolsCount: 5,
  });

  // Simulation logic (mock calculations)
  const calculateForecast = (p: SimulationParams): SimulationForecast => {
    const aggressionFactor = p.aggression / 100;
    const positionFactor = p.maxPositions / 10;
    const symbolFactor = p.symbolsCount / 20;

    return {
      expectedTrades: Math.round(
        (aggressionFactor * 100 + positionFactor * 50 + symbolFactor * 30) * 1.5
      ),
      expectedDrawdown: {
        min: Math.round(aggressionFactor * 500 + positionFactor * 200),
        max: Math.round(aggressionFactor * 1200 + positionFactor * 500),
      },
      eventBusStress: Math.round(
        (aggressionFactor * 40 + positionFactor * 20 + symbolFactor * 30)
      ),
      exitRiskProbability: Math.round(
        (aggressionFactor * 25 + positionFactor * 15)
      ),
    };
  };

  const forecast = calculateForecast(params);

  const getStressLevel = (stress: number): { color: string; label: string } => {
    if (stress < 30) return { color: "text-[var(--trading-green)]", label: "LOW" };
    if (stress < 60) return { color: "text-[var(--trading-amber)]", label: "MODERATE" };
    return { color: "text-[var(--trading-red)]", label: "HIGH" };
  };

  const getRiskLevel = (risk: number): { color: string; label: string } => {
    if (risk < 20) return { color: "text-[var(--trading-green)]", label: "LOW" };
    if (risk < 40) return { color: "text-[var(--trading-amber)]", label: "MODERATE" };
    return { color: "text-[var(--trading-red)]", label: "HIGH" };
  };

  const stressLevel = getStressLevel(forecast.eventBusStress);
  const riskLevel = getRiskLevel(forecast.exitRiskProbability);

  return (
    <div className="space-y-6">
      {/* Safety Notice */}
      <Alert className="border-[var(--trading-cyan)]/30 bg-[var(--trading-cyan)]/5">
        <Sliders className="h-4 w-4 text-[var(--trading-cyan)]" />
        <AlertDescription className="text-sm text-foreground/90">
          <span className="font-medium">Simulation Mode</span> — No real trading occurs. 
          Adjust parameters to forecast system behavior.
        </AlertDescription>
      </Alert>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Control Panel */}
        <Card className="p-6 bg-card/30 backdrop-blur-sm border-border/50">
          <div className="flex items-center gap-2 mb-6">
            <Sliders className="h-5 w-5 text-muted-foreground" />
            <h3 className="text-sm uppercase tracking-wider text-muted-foreground">
              Parameters
            </h3>
          </div>

          <div className="space-y-8">
            {/* Aggression Slider */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm text-foreground">Aggression Level</label>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-mono tabular-nums text-[var(--trading-cyan)]">
                    {params.aggression}
                  </span>
                  <span className="text-sm text-muted-foreground">%</span>
                </div>
              </div>
              <Slider
                value={[params.aggression]}
                onValueChange={([value]) => setParams({ ...params, aggression: value })}
                min={0}
                max={100}
                step={5}
                className="[&_[role=slider]]:bg-[var(--trading-cyan)] [&_[role=slider]]:border-[var(--trading-cyan)]"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-2">
                <span>Conservative</span>
                <span>Aggressive</span>
              </div>
            </div>

            {/* Max Positions Slider */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm text-foreground">Max Positions</label>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-mono tabular-nums text-[var(--trading-cyan)]">
                    {params.maxPositions}
                  </span>
                </div>
              </div>
              <Slider
                value={[params.maxPositions]}
                onValueChange={([value]) => setParams({ ...params, maxPositions: value })}
                min={1}
                max={10}
                step={1}
                className="[&_[role=slider]]:bg-[var(--trading-cyan)] [&_[role=slider]]:border-[var(--trading-cyan)]"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-2">
                <span>1</span>
                <span>10</span>
              </div>
            </div>

            {/* Symbols Count Slider */}
            <div>
              <div className="flex items-center justify-between mb-3">
                <label className="text-sm text-foreground">Active Symbols</label>
                <div className="flex items-center gap-2">
                  <span className="text-lg font-mono tabular-nums text-[var(--trading-cyan)]">
                    {params.symbolsCount}
                  </span>
                </div>
              </div>
              <Slider
                value={[params.symbolsCount]}
                onValueChange={([value]) => setParams({ ...params, symbolsCount: value })}
                min={1}
                max={20}
                step={1}
                className="[&_[role=slider]]:bg-[var(--trading-cyan)] [&_[role=slider]]:border-[var(--trading-cyan)]"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-2">
                <span>1</span>
                <span>20</span>
              </div>
            </div>
          </div>
        </Card>

        {/* Forecast Panel */}
        <Card className="p-6 bg-card/30 backdrop-blur-sm border-border/50">
          <div className="flex items-center gap-2 mb-6">
            <TrendingUp className="h-5 w-5 text-muted-foreground" />
            <h3 className="text-sm uppercase tracking-wider text-muted-foreground">
              Forecast
            </h3>
          </div>

          <div className="space-y-6">
            {/* Expected Trades */}
            <div>
              <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">
                Expected Trades per Day
              </div>
              <div className="text-3xl font-mono tabular-nums text-[var(--trading-cyan)]">
                {forecast.expectedTrades}
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Based on historical signal frequency
              </div>
            </div>

            {/* Expected Drawdown */}
            <div className="pt-4 border-t border-border/50">
              <div className="text-xs text-muted-foreground uppercase tracking-wider mb-2">
                Expected Drawdown Range
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-mono tabular-nums text-[var(--trading-amber)]">
                  ${forecast.expectedDrawdown.min}
                </span>
                <span className="text-muted-foreground">—</span>
                <span className="text-2xl font-mono tabular-nums text-[var(--trading-amber)]">
                  ${forecast.expectedDrawdown.max}
                </span>
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Worst-case intraday loss estimate
              </div>
            </div>

            {/* EventBus Stress */}
            <div className="pt-4 border-t border-border/50">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="h-4 w-4 text-muted-foreground" />
                <div className="text-xs text-muted-foreground uppercase tracking-wider">
                  EventBus Stress Forecast
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-2xl font-mono tabular-nums">
                  {forecast.eventBusStress}%
                </div>
                <Badge 
                  variant="outline" 
                  className={`${stressLevel.color} border-current`}
                >
                  {stressLevel.label}
                </Badge>
              </div>
            </div>

            {/* Exit Risk */}
            <div className="pt-4 border-t border-border/50">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                <div className="text-xs text-muted-foreground uppercase tracking-wider">
                  Exit Risk Probability
                </div>
              </div>
              <div className="flex items-center gap-3">
                <div className="text-2xl font-mono tabular-nums">
                  {forecast.exitRiskProbability}%
                </div>
                <Badge 
                  variant="outline" 
                  className={`${riskLevel.color} border-current`}
                >
                  {riskLevel.label}
                </Badge>
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                Chance of delayed exit execution
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Recommendations */}
      {(forecast.eventBusStress > 70 || forecast.exitRiskProbability > 40) && (
        <Alert className="border-[var(--trading-amber)]/30 bg-[var(--trading-amber)]/5">
          <AlertTriangle className="h-4 w-4 text-[var(--trading-amber)]" />
          <AlertDescription className="text-sm text-foreground/90">
            <span className="font-medium">High Stress Detected</span> — 
            {forecast.eventBusStress > 70 && " EventBus may struggle with message throughput."}
            {forecast.exitRiskProbability > 40 && " Exit latency risk increases significantly."}
            {" Consider reducing aggression or position count."}
          </AlertDescription>
        </Alert>
      )}

      {(params.aggression < 30 && params.maxPositions <= 2) && (
        <Alert className="border-[var(--trading-green)]/30 bg-[var(--trading-green)]/5">
          <Database className="h-4 w-4 text-[var(--trading-green)]" />
          <AlertDescription className="text-sm text-foreground/90">
            <span className="font-medium">Conservative Configuration</span> — 
            System will operate with minimal stress and controlled risk exposure.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}

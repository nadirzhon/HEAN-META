import { ComposedChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine, Area } from "recharts";
import { Card } from "@/app/components/ui/card";
import { CandleData, OrderBlock, DecisionMarker, ProfitZone } from "@/app/utils/chartData";
import { ArrowUp, ArrowDown } from "lucide-react";

interface AdvancedChartProps {
  data: CandleData[];
  orderBlocks?: OrderBlock[];
  decisionMarkers?: DecisionMarker[];
  profitZones?: ProfitZone[];
  showProfitIntelligence?: boolean;
  height?: number;
}

export function AdvancedChart({ 
  data, 
  orderBlocks = [], 
  decisionMarkers = [],
  profitZones = [],
  showProfitIntelligence = false,
  height = 600 
}: AdvancedChartProps) {
  
  if (data.length === 0) return null;

  // Transform candlestick data for recharts
  const chartData = data.map((candle) => ({
    timestamp: candle.timestamp,
    time: new Date(candle.timestamp).toLocaleTimeString("en-US", { 
      hour: "2-digit", 
      minute: "2-digit",
      hour12: false 
    }),
    open: candle.open,
    close: candle.close,
    high: candle.high,
    low: candle.low,
    volume: candle.volume,
    // For candlestick body
    bodyTop: Math.max(candle.open, candle.close),
    bodyBottom: Math.min(candle.open, candle.close),
    isGreen: candle.close >= candle.open,
  }));

  const minPrice = Math.min(...data.map(d => d.low)) - 100;
  const maxPrice = Math.max(...data.map(d => d.high)) + 100;

  // Custom candlestick renderer
  const CustomCandlestick = (props: any) => {
    const { x, y, width, height, payload } = props;
    if (!payload) return null;

    const wickX = x + width / 2;
    const color = payload.isGreen ? "var(--trading-green)" : "var(--trading-red)";
    
    return (
      <g>
        {/* Wick */}
        <line
          x1={wickX}
          y1={y}
          x2={wickX}
          y2={y + height}
          stroke={color}
          strokeWidth={1}
        />
        {/* Body */}
        <rect
          x={x}
          y={y + height * 0.3}
          width={width}
          height={height * 0.4}
          fill={color}
          opacity={0.8}
        />
      </g>
    );
  };

  const getProfitZoneColor = (type: string) => {
    switch (type) {
      case "optimal":
        return "rgba(16, 185, 129, 0.1)"; // green
      case "weak":
        return "rgba(245, 158, 11, 0.08)"; // amber
      case "risk-dominant":
        return "rgba(239, 68, 68, 0.1)"; // red
      default:
        return "transparent";
    }
  };

  return (
    <Card className="p-0 bg-card/30 backdrop-blur-sm border-border/50 overflow-hidden">
      {/* Chart Header */}
      <div className="px-6 py-4 border-b border-border/50 flex items-center justify-between">
        <div>
          <h3 className="text-sm font-mono text-muted-foreground uppercase tracking-wider">
            BTC/USD
          </h3>
          <div className="flex items-center gap-4 mt-1">
            <span className="text-2xl font-mono tabular-nums text-foreground">
              ${data[data.length - 1]?.close.toFixed(2)}
            </span>
            <span className={`text-sm font-mono ${
              data[data.length - 1]?.close >= data[data.length - 1]?.open
                ? "text-[var(--trading-green)]"
                : "text-[var(--trading-red)]"
            }`}>
              {data[data.length - 1]?.close >= data[data.length - 1]?.open ? "+" : ""}
              {(data[data.length - 1]?.close - data[data.length - 1]?.open).toFixed(2)}
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span className="uppercase tracking-wider">1m</span>
          <div className="h-4 w-px bg-border" />
          <span>TREND Regime</span>
        </div>
      </div>

      {/* Main Chart */}
      <div className="relative" style={{ height }}>
        {/* Profit Intelligence Zones */}
        {showProfitIntelligence && profitZones.length > 0 && (
          <div className="absolute inset-0 pointer-events-none z-10">
            {profitZones.map((zone, idx) => {
              const startIdx = chartData.findIndex(d => d.timestamp >= zone.startTime);
              const endIdx = chartData.findIndex(d => d.timestamp >= zone.endTime);
              
              if (startIdx === -1 || endIdx === -1) return null;
              
              const xStart = (startIdx / chartData.length) * 100;
              const xEnd = (endIdx / chartData.length) * 100;
              
              return (
                <div
                  key={idx}
                  className="absolute top-0 bottom-20"
                  style={{
                    left: `${xStart}%`,
                    width: `${xEnd - xStart}%`,
                    backgroundColor: getProfitZoneColor(zone.type),
                    borderLeft: `1px dashed ${
                      zone.type === "optimal" 
                        ? "var(--trading-green)" 
                        : zone.type === "weak"
                        ? "var(--trading-amber)"
                        : "var(--trading-red)"
                    }`,
                    borderRight: `1px dashed ${
                      zone.type === "optimal" 
                        ? "var(--trading-green)" 
                        : zone.type === "weak"
                        ? "var(--trading-amber)"
                        : "var(--trading-red)"
                    }`,
                  }}
                >
                  <div className="absolute top-4 left-2 text-xs text-muted-foreground bg-card/80 backdrop-blur-sm px-2 py-1 rounded">
                    {zone.label}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Decision Markers */}
        {decisionMarkers.length > 0 && (
          <div className="absolute inset-0 pointer-events-none z-20">
            {decisionMarkers.map((marker, idx) => {
              const dataIdx = chartData.findIndex(d => d.timestamp >= marker.timestamp);
              if (dataIdx === -1) return null;
              
              const xPos = (dataIdx / chartData.length) * 100;
              const yPos = ((maxPrice - marker.price) / (maxPrice - minPrice)) * 100;
              
              const isEntry = marker.type === "entry";
              const isLong = marker.side === "LONG";
              
              return (
                <div
                  key={idx}
                  className="absolute group cursor-pointer"
                  style={{
                    left: `${xPos}%`,
                    top: `${yPos}%`,
                    transform: "translate(-50%, -50%)",
                  }}
                >
                  {/* Arrow */}
                  <div className={`
                    flex items-center justify-center w-8 h-8 rounded-full
                    ${isEntry
                      ? isLong 
                        ? "bg-[var(--trading-cyan)]/20 border-2 border-[var(--trading-cyan)]"
                        : "bg-[var(--trading-purple)]/20 border-2 border-[var(--trading-purple)]"
                      : "bg-[var(--trading-amber)]/20 border-2 border-[var(--trading-amber)]"
                    }
                  `}>
                    {isEntry ? (
                      isLong ? <ArrowUp className="h-4 w-4 text-[var(--trading-cyan)]" /> : <ArrowDown className="h-4 w-4 text-[var(--trading-purple)]" />
                    ) : (
                      <div className="h-2 w-2 bg-[var(--trading-amber)] rounded-full" />
                    )}
                  </div>
                  
                  {/* Tooltip on hover */}
                  <div className="absolute left-10 top-0 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none z-30">
                    <div className="bg-card border border-border rounded-lg p-3 shadow-lg min-w-[200px]">
                      <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
                        {marker.type} • {marker.side}
                      </div>
                      <div className="text-sm text-foreground mb-2">{marker.reason}</div>
                      <div className="text-xs text-muted-foreground">
                        Confidence: {marker.confidence.toFixed(0)}%
                      </div>
                      {marker.suppressedAlternatives && marker.suppressedAlternatives.length > 0 && (
                        <div className="mt-2 pt-2 border-t border-border/50">
                          <div className="text-xs text-muted-foreground">Suppressed:</div>
                          {marker.suppressedAlternatives.map((alt, i) => (
                            <div key={i} className="text-xs text-muted-foreground/70">• {alt}</div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}

        <ResponsiveContainer width="100%" height="80%">
          <ComposedChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 0 }}>
            <defs>
              <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="var(--trading-cyan)" stopOpacity={0.3} />
                <stop offset="100%" stopColor="var(--trading-cyan)" stopOpacity={0.05} />
              </linearGradient>
            </defs>
            
            <XAxis 
              dataKey="time" 
              stroke="var(--color-muted-foreground)" 
              tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }}
              axisLine={{ stroke: "var(--color-border)" }}
            />
            <YAxis 
              yAxisId="price"
              domain={[minPrice, maxPrice]}
              stroke="var(--color-muted-foreground)"
              tick={{ fill: "var(--color-muted-foreground)", fontSize: 11 }}
              axisLine={{ stroke: "var(--color-border)" }}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
            />
            
            {/* Order Blocks */}
            {orderBlocks.map((block, idx) => (
              <ReferenceLine
                key={idx}
                yAxisId="price"
                y={block.price}
                stroke={
                  block.type === "resistance" 
                    ? "var(--trading-red)" 
                    : block.type === "support"
                    ? "var(--trading-green)"
                    : "var(--trading-cyan)"
                }
                strokeDasharray="3 3"
                strokeWidth={1.5}
                label={{ 
                  value: block.label, 
                  position: "right",
                  fill: "var(--color-muted-foreground)",
                  fontSize: 11
                }}
              />
            ))}
            
            {/* Candlesticks using Bar */}
            <Bar
              yAxisId="price"
              dataKey={(d: any) => [d.low, d.high]}
              shape={<CustomCandlestick />}
            />
            
            <Tooltip
              contentStyle={{
                backgroundColor: "var(--color-card)",
                border: "1px solid var(--color-border)",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              labelStyle={{ color: "var(--color-muted-foreground)" }}
              formatter={(value: any, name: string) => {
                if (name === "volume") return [value.toLocaleString(), "Volume"];
                return [`$${value.toFixed(2)}`, name];
              }}
            />
          </ComposedChart>
        </ResponsiveContainer>

        {/* Volume Chart */}
        <ResponsiveContainer width="100%" height="20%">
          <ComposedChart data={chartData} margin={{ top: 0, right: 30, left: 20, bottom: 20 }}>
            <XAxis 
              dataKey="time" 
              stroke="var(--color-muted-foreground)" 
              tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }}
              axisLine={{ stroke: "var(--color-border)" }}
            />
            <YAxis 
              stroke="var(--color-muted-foreground)"
              tick={{ fill: "var(--color-muted-foreground)", fontSize: 10 }}
              axisLine={{ stroke: "var(--color-border)" }}
              tickFormatter={(value) => `${(value / 1000000).toFixed(1)}M`}
            />
            <Area
              type="monotone"
              dataKey="volume"
              stroke="var(--trading-cyan)"
              fill="url(#volumeGradient)"
              strokeWidth={1}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </Card>
  );
}

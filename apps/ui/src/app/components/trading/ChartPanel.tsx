import { useEffect, useRef } from "react";
import { ColorType, LineStyle, UTCTimestamp, createChart } from "lightweight-charts";
import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { CandleData } from "@/app/utils/chartData";
import { ChartMarker } from "@/app/types/trading";

interface ChartPanelProps {
  symbol: string;
  candles: CandleData[];
  priceLine: Array<{ time: number; value: number }>;
  markers: ChartMarker[];
  hasMarketData: boolean;
  positionLabel?: string;
  height?: number;
}

const toUtc = (time: number): UTCTimestamp => Math.floor(time / 1000) as UTCTimestamp;
const COLORS = {
  green: "#10b981",
  red: "#ef4444",
  cyan: "#06b6d4",
  amber: "#f59e0b",
  purple: "#a855f7",
};

export function ChartPanel({
  symbol,
  candles,
  priceLine,
  markers,
  hasMarketData,
  positionLabel,
  height = 420,
}: ChartPanelProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const candleSeriesRef = useRef<ReturnType<typeof createChart>["addCandlestickSeries"]>();
  const lineSeriesRef = useRef<ReturnType<typeof createChart>["addLineSeries"]>();
  const chartRef = useRef<ReturnType<typeof createChart>>();

  useEffect(() => {
    if (!containerRef.current || chartRef.current) return;
    const chart = createChart(containerRef.current, {
      height,
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#cbd5e1",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(255,255,255,0.04)" },
        horzLines: { color: "rgba(255,255,255,0.06)" },
      },
      rightPriceScale: {
        borderColor: "rgba(255,255,255,0.08)",
      },
      timeScale: {
        borderColor: "rgba(255,255,255,0.08)",
        timeVisible: true,
        secondsVisible: true,
      },
      crosshair: {
        mode: 1,
      },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: COLORS.green,
      downColor: COLORS.red,
      borderVisible: false,
      wickUpColor: COLORS.green,
      wickDownColor: COLORS.red,
    });

    const lineSeries = chart.addLineSeries({
      color: COLORS.cyan,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
    });

    chartRef.current = chart;
    candleSeriesRef.current = candleSeries;
    lineSeriesRef.current = lineSeries;

    const onResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", onResize);
    onResize();

    return () => {
      window.removeEventListener("resize", onResize);
      chart.remove();
    };
  }, [height]);

  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    const lineSeries = lineSeriesRef.current;
    if (!chartRef.current || !candleSeries || !lineSeries) return;

    if (candles.length) {
      candleSeries.setData(
        candles.map((c) => ({
          time: toUtc(c.timestamp),
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
        }))
      );
      chartRef.current.timeScale().fitContent();
    } else {
      candleSeries.setData([]);
    }

    if (priceLine.length) {
      lineSeries.setData(priceLine.map((p) => ({ time: toUtc(p.time), value: p.value })));
    }

    if (markers.length && candles.length) {
      candleSeries.setMarkers(
        markers.map((m) => ({
          time: toUtc(m.time),
          position: m.position,
          color: m.color,
          shape: m.shape,
          text: m.text,
        }))
      );
    } else if (markers.length) {
      lineSeries.setMarkers(
        markers.map((m) => ({
          time: toUtc(m.time),
          position: m.position,
          color: m.color,
          shape: m.shape,
          text: m.text,
        }))
      );
    }
  }, [candles, priceLine, markers]);

  return (
    <Card className="border border-border/70 bg-card/40 backdrop-blur-sm relative overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-border/50">
        <div>
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Market View</div>
          <div className="font-mono text-lg">{symbol}</div>
        </div>
        {positionLabel && (
          <Badge variant="outline" className="text-[var(--trading-cyan)] border-[var(--trading-cyan)]/40">
            {positionLabel}
          </Badge>
        )}
      </div>
      <div ref={containerRef} className="w-full" style={{ height }} />
      {!hasMarketData && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/60 backdrop-blur-sm border border-dashed border-border/60">
          <div className="text-center space-y-2">
            <div className="text-sm font-mono">waiting for live market data…</div>
            <div className="text-xs text-muted-foreground">Engine will stream candles and ticks here as soon as they are available.</div>
          </div>
        </div>
      )}
    </Card>
  );
}

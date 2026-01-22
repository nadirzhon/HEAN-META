import { CheckCircle2, XCircle, TrendingUp, TrendingDown } from "lucide-react";
import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";

export interface ResultFactor {
  stage: string;
  passed: boolean;
  explanation: string;
  impact?: "positive" | "negative" | "neutral";
}

export interface TradeResult {
  symbol: string;
  side: "LONG" | "SHORT";
  result: number; // PnL in USDT
  entryPrice: number;
  exitPrice: number;
  duration: string;
  factors: ResultFactor[];
  timestamp: Date;
}

interface WhyResultPanelProps {
  result: TradeResult;
}

export function WhyResultPanel({ result }: WhyResultPanelProps) {
  const isProfit = result.result >= 0;

  return (
    <Card className="p-6 bg-card/30 backdrop-blur-sm border-border/50">
      {/* Header */}
      <div className="flex items-start justify-between mb-6 pb-6 border-b border-border/50">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <span className="text-2xl font-mono">{result.symbol}</span>
            <Badge 
              variant="outline" 
              className={
                result.side === "LONG" 
                  ? "border-[var(--trading-cyan)]/50 text-[var(--trading-cyan)]"
                  : "border-[var(--trading-purple)]/50 text-[var(--trading-purple)]"
              }
            >
              {result.side}
            </Badge>
          </div>
          <div className="text-sm text-muted-foreground font-mono">
            {result.entryPrice.toFixed(2)} → {result.exitPrice.toFixed(2)} • {result.duration}
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {result.timestamp.toLocaleString("en-US", {
              month: "short",
              day: "numeric",
              hour: "2-digit",
              minute: "2-digit",
              hour12: false,
            })}
          </div>
        </div>

        <div className="text-right">
          <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Result</div>
          <div className={`text-3xl font-mono tabular-nums flex items-center gap-2 ${
            isProfit ? "text-[var(--trading-green)]" : "text-[var(--trading-red)]"
          }`}>
            {isProfit ? <TrendingUp className="h-6 w-6" /> : <TrendingDown className="h-6 w-6" />}
            {isProfit ? "+" : ""}{result.result.toFixed(2)} USDT
          </div>
        </div>
      </div>

      {/* Explanation */}
      <div>
        <h4 className="text-xs text-muted-foreground uppercase tracking-wider mb-4">
          Why this result?
        </h4>
        <div className="space-y-3">
          {result.factors.map((factor, index) => (
            <div key={index} className="flex items-start gap-3">
              <div className="flex-shrink-0 mt-0.5">
                {factor.passed ? (
                  <CheckCircle2 className="h-5 w-5 text-[var(--trading-green)]" />
                ) : (
                  <XCircle className="h-5 w-5 text-[var(--trading-red)]" />
                )}
              </div>
              <div className="flex-1">
                <div className="text-sm text-foreground/90">{factor.explanation}</div>
                {factor.impact && (
                  <div className={`text-xs mt-1 ${
                    factor.impact === "positive" 
                      ? "text-[var(--trading-green)]" 
                      : factor.impact === "negative"
                      ? "text-[var(--trading-red)]"
                      : "text-[var(--trading-amber)]"
                  }`}>
                    {factor.impact === "positive" && "Enhanced profitability"}
                    {factor.impact === "negative" && "Limited profitability"}
                    {factor.impact === "neutral" && "Position size adjusted"}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Summary */}
      <div className="mt-6 pt-6 border-t border-border/50">
        <div className="text-xs text-muted-foreground">
          <span className="uppercase tracking-wider">Summary: </span>
          <span className="text-foreground/80">
            {result.factors.filter(f => f.passed).length} of {result.factors.length} decision factors aligned optimally.
            {isProfit 
              ? " Trade executed according to risk parameters."
              : " Some factors limited final outcome."
            }
          </span>
        </div>
      </div>
    </Card>
  );
}

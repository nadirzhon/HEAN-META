import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { Progress } from "@/app/components/ui/progress";
import { Wallet } from "lucide-react";

export interface WalletState {
  wallet_balance: number;
  available_balance: number;
  equity: number;
  used_margin: number;
  reserved_margin: number;
  unrealized_pnl: number;
  realized_pnl: number;
  timestamp?: string;
}

interface WalletSummaryProps {
  state: WalletState;
  engineRunning: boolean;
}

const fmt = (v: number) =>
  `$${v.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;

export function WalletSummary({ state, engineRunning }: WalletSummaryProps) {
  const { wallet_balance, available_balance, equity, used_margin, reserved_margin, unrealized_pnl, realized_pnl, timestamp } =
    state;
  const usedPct =
    wallet_balance > 0 ? Math.min(100, ((used_margin + reserved_margin) / wallet_balance) * 100) : 0;

  return (
    <Card className="p-4 bg-card/40 backdrop-blur-sm border-border/60">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Wallet className="h-5 w-5 text-[var(--trading-cyan)]" />
          <span className="text-sm uppercase tracking-wider text-muted-foreground">
            Wallet
          </span>
        </div>
        <Badge variant="outline" className={engineRunning ? "text-[var(--trading-green)] border-[var(--trading-green)]/40" : ""}>
          {engineRunning ? "Running" : "Stopped"}
        </Badge>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
        <div>
          <div className="text-muted-foreground">Equity</div>
          <div className="font-mono text-lg">{fmt(equity)}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Wallet Balance</div>
          <div className="font-mono text-lg">{fmt(wallet_balance)}</div>
        </div>
        <div>
          <div className="text-muted-foreground">Available</div>
          <div className="font-mono text-lg text-[var(--trading-green)]">
            {fmt(available_balance)}
          </div>
        </div>
        <div>
          <div className="text-muted-foreground">Used Margin</div>
          <div className="font-mono text-lg text-[var(--trading-amber)]">
            {fmt(used_margin + reserved_margin)}
          </div>
        </div>
        <div>
          <div className="text-muted-foreground">Unrealized PnL</div>
          <div
            className={`font-mono text-lg ${
              unrealized_pnl >= 0 ? "text-[var(--trading-green)]" : "text-[var(--trading-red)]"
            }`}
          >
            {fmt(unrealized_pnl)}
          </div>
        </div>
        <div>
          <div className="text-muted-foreground">Realized PnL</div>
          <div
            className={`font-mono text-lg ${
              realized_pnl >= 0 ? "text-[var(--trading-green)]" : "text-[var(--trading-red)]"
            }`}
          >
            {fmt(realized_pnl)}
          </div>
        </div>
      </div>

      <div className="mt-4">
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <span>Margin usage</span>
          <span>{usedPct.toFixed(1)}%</span>
        </div>
        <Progress value={usedPct} className="h-1.5 [&>div]:bg-[var(--trading-amber)]" />
      </div>

      {timestamp && (
        <div className="mt-3 text-xs text-muted-foreground">
          Updated: {new Date(timestamp).toLocaleTimeString("en-US", { hour12: false })}
        </div>
      )}
    </Card>
  );
}

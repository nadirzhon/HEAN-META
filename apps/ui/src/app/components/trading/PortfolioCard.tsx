import { AlertTriangle, ShieldHalf, Wallet, Wallet2 } from "lucide-react";
import { Card } from "@/app/components/ui/card";
import { Badge } from "@/app/components/ui/badge";
import { PortfolioSummary } from "@/app/api/client";
import { WalletState } from "@/app/components/trading/WalletSummary";

interface PortfolioCardProps {
  portfolio: PortfolioSummary | null;
  account: WalletState;
  backendAvailable: boolean;
}

const formatMoney = (value?: number | null) => {
  if (value === undefined || value === null || Number.isNaN(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
};

export function PortfolioCard({ portfolio, account, backendAvailable }: PortfolioCardProps) {
  const equity = portfolio?.equity ?? account.equity;
  const balance = portfolio?.balance ?? portfolio?.wallet_balance ?? account.wallet_balance;
  const freeMargin = portfolio?.free_margin ?? account.available_balance;
  const usedMargin = portfolio?.used_margin ?? account.used_margin;
  const unrealized = portfolio?.unrealized_pnl ?? account.unrealized_pnl;
  const realized = portfolio?.realized_pnl ?? account.realized_pnl;
  const available = backendAvailable && portfolio !== null;

  return (
    <Card className="p-4 border-border/60 bg-card/60 backdrop-blur-sm space-y-4">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Wallet className="h-5 w-5 text-[var(--trading-cyan)]" />
          <div>
            <div className="text-xs uppercase tracking-wider text-muted-foreground">Portfolio</div>
            <div className="text-sm text-muted-foreground/80">Equity · Margin · PnL</div>
          </div>
        </div>
        {!available && (
          <Badge variant="destructive" className="bg-[var(--trading-amber)]/20 text-[var(--trading-amber)] border-[var(--trading-amber)]/40">
            данные недоступны
          </Badge>
        )}
      </div>

      <div className="rounded-md border border-border/50 bg-background/40 p-3 flex items-center justify-between">
        <div>
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Equity</div>
          <div className="text-2xl font-mono text-[var(--trading-green)]">${formatMoney(equity)}</div>
        </div>
        <div className="text-right">
          <div className="text-[11px] uppercase tracking-wider text-muted-foreground">Balance</div>
          <div className="text-lg font-mono text-foreground">${formatMoney(balance)}</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-md border border-border/40 bg-card/40 p-3">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
            <span>Free Margin</span>
            <ShieldHalf className="h-4 w-4 text-[var(--trading-cyan)]" />
          </div>
          <div className="font-mono text-lg">${formatMoney(freeMargin)}</div>
        </div>
        <div className="rounded-md border border-border/40 bg-card/40 p-3">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
            <span>Used Margin</span>
            <Wallet2 className="h-4 w-4 text-[var(--trading-amber)]" />
          </div>
          <div className="font-mono text-lg">${formatMoney(usedMargin)}</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="rounded-md border border-border/40 bg-card/40 p-3">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
            <span>Unrealized PnL</span>
            <Badge
              variant="outline"
              className={`font-mono ${
                (unrealized ?? 0) >= 0
                  ? "text-[var(--trading-green)] border-[var(--trading-green)]/50"
                  : "text-[var(--trading-red)] border-[var(--trading-red)]/50"
              }`}
            >
              {(unrealized ?? 0) >= 0 ? "UP" : "DOWN"}
            </Badge>
          </div>
          <div
            className={`font-mono text-lg ${
              (unrealized ?? 0) >= 0 ? "text-[var(--trading-green)]" : "text-[var(--trading-red)]"
            }`}
          >
            ${formatMoney(unrealized)}
          </div>
        </div>
        <div className="rounded-md border border-border/40 bg-card/40 p-3">
          <div className="flex items-center justify-between text-[11px] uppercase tracking-wider text-muted-foreground">
            <span>Realized PnL</span>
            <Badge
              variant="outline"
              className={`font-mono ${
                (realized ?? 0) >= 0
                  ? "text-[var(--trading-green)] border-[var(--trading-green)]/50"
                  : "text-[var(--trading-red)] border-[var(--trading-red)]/50"
              }`}
            >
              {(realized ?? 0) >= 0 ? "UP" : "DOWN"}
            </Badge>
          </div>
          <div
            className={`font-mono text-lg ${
              (realized ?? 0) >= 0 ? "text-[var(--trading-green)]" : "text-[var(--trading-red)]"
            }`}
          >
            ${formatMoney(realized)}
          </div>
        </div>
      </div>

      {!available && (
        <div className="flex items-center gap-2 text-[12px] text-muted-foreground">
          <AlertTriangle className="h-4 w-4 text-[var(--trading-amber)]" />
          <span>Загрузить портфель не удалось — показываем последнее известное значение.</span>
        </div>
      )}
    </Card>
  );
}

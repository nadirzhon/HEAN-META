import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/app/components/ui/table";
import { Badge } from "@/app/components/ui/badge";

export interface Position {
  id: string;
  symbol: string;
  side: "LONG" | "SHORT";
  entryPrice: number;
  lastPrice: number;
  unrealizedPnL: number;
  takeProfit?: number;
  stopLoss?: number;
  ttl?: number; // time to live in seconds
  status: "ACTIVE" | "CLOSING" | "PENDING";
}

interface PositionsTableProps {
  positions: Position[];
}

export function PositionsTable({ positions }: PositionsTableProps) {
  const formatPrice = (price: number) => price.toFixed(2);
  const formatPnL = (pnl: number) => {
    const sign = pnl >= 0 ? "+" : "";
    return `${sign}$${pnl.toFixed(2)}`;
  };

  const getPnLColor = (pnl: number) => {
    return pnl >= 0 ? "text-[var(--trading-green)]" : "text-[var(--trading-red)]";
  };

  const getSideColor = (side: string) => {
    return side === "LONG" ? "text-[var(--trading-cyan)]" : "text-[var(--trading-purple)]";
  };

  const formatTTL = (seconds?: number) => {
    if (!seconds) return "—";
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}m ${secs}s`;
  };

  return (
    <div className="rounded-lg border border-border bg-card/30 backdrop-blur-sm">
      <Table>
        <TableHeader>
          <TableRow className="border-border/50 hover:bg-transparent">
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Symbol</TableHead>
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Side</TableHead>
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Entry</TableHead>
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Last</TableHead>
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">PnL</TableHead>
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">TP</TableHead>
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">SL</TableHead>
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">TTL</TableHead>
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {positions.length === 0 ? (
            <TableRow>
              <TableCell colSpan={9} className="text-center text-muted-foreground py-8">
                No open positions
              </TableCell>
            </TableRow>
          ) : (
            positions.map((position) => (
              <TableRow key={position.id} className="border-border/50 hover:bg-muted/30">
                <TableCell className="font-mono">{position.symbol}</TableCell>
                <TableCell>
                  <span className={`font-mono ${getSideColor(position.side)}`}>{position.side}</span>
                </TableCell>
                <TableCell className="text-right font-mono tabular-nums">${formatPrice(position.entryPrice)}</TableCell>
                <TableCell className="text-right font-mono tabular-nums">${formatPrice(position.lastPrice)}</TableCell>
                <TableCell className={`text-right font-mono tabular-nums ${getPnLColor(position.unrealizedPnL)}`}>
                  {formatPnL(position.unrealizedPnL)}
                </TableCell>
                <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                  {position.takeProfit ? `$${formatPrice(position.takeProfit)}` : "—"}
                </TableCell>
                <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                  {position.stopLoss ? `$${formatPrice(position.stopLoss)}` : "—"}
                </TableCell>
                <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                  {formatTTL(position.ttl)}
                </TableCell>
                <TableCell>
                  <Badge 
                    variant="outline" 
                    className={
                      position.status === "ACTIVE" 
                        ? "border-[var(--trading-green)]/50 text-[var(--trading-green)]" 
                        : position.status === "CLOSING"
                        ? "border-[var(--trading-amber)]/50 text-[var(--trading-amber)]"
                        : "border-[var(--trading-gray)]/50 text-[var(--trading-gray)]"
                    }
                  >
                    {position.status}
                  </Badge>
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
}

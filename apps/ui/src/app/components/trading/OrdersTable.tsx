import { useState } from "react";
import { formatDistanceToNow } from "date-fns";
import { Badge } from "@/app/components/ui/badge";
import { Button } from "@/app/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/app/components/ui/table";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/app/components/ui/dialog";
import { OrderRow } from "@/app/types/trading";
import { Maximize2, Minimize2 } from "lucide-react";

interface OrdersTableProps {
  orders: OrderRow[];
  compact?: boolean;
}

const fmtPrice = (v?: number) => (typeof v === "number" ? `$${v.toFixed(2)}` : "—");
const fmtSize = (v?: number) => (typeof v === "number" ? v.toFixed(3) : "—");

const statusTone = (status?: string) => {
  const s = (status ?? "").toUpperCase();
  if (s === "FILLED") return "border-[var(--trading-green)]/50 text-[var(--trading-green)]";
  if (s === "PARTIAL") return "border-[var(--trading-amber)]/50 text-[var(--trading-amber)]";
  if (s === "CANCELED" || s === "REJECTED") return "border-[var(--trading-red)]/50 text-[var(--trading-red)]";
  return "border-[var(--trading-cyan)]/50 text-[var(--trading-cyan)]";
};

const sideTone = (side: string) =>
  side === "SELL" ? "text-[var(--trading-purple)]" : "text-[var(--trading-cyan)]";

export function OrdersTable({ orders, compact = true }: OrdersTableProps) {
  const [expanded, setExpanded] = useState(false);

  const tableContent = (
    <Table>
      <TableHeader className="sticky top-0 bg-card/95 backdrop-blur-sm z-10">
        <TableRow className="border-border/50 hover:bg-transparent">
          <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Symbol</TableHead>
          <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Side</TableHead>
          <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Size</TableHead>
          {!compact && (
            <>
              <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Filled</TableHead>
              <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Price</TableHead>
              <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">TP</TableHead>
              <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">SL</TableHead>
              <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Strategy</TableHead>
            </>
          )}
          <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Status</TableHead>
          {!compact && (
            <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Age</TableHead>
          )}
        </TableRow>
      </TableHeader>
      <TableBody>
        {orders.length === 0 ? (
          <TableRow>
            <TableCell colSpan={compact ? 4 : 10} className="text-center text-muted-foreground py-8">
              No open orders
            </TableCell>
          </TableRow>
        ) : (
          orders.map((order) => (
            <TableRow key={order.id} className="border-border/50 hover:bg-muted/30">
              <TableCell className="font-mono text-xs">{order.symbol}</TableCell>
              <TableCell className={`font-mono text-xs ${sideTone(order.side)}`}>{order.side}</TableCell>
              <TableCell className="text-right font-mono tabular-nums text-xs">{fmtSize(order.size)}</TableCell>
              {!compact && (
                <>
                  <TableCell className="text-right font-mono tabular-nums text-xs text-muted-foreground">
                    {fmtSize(order.filled)}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-xs">{fmtPrice(order.price)}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-xs text-muted-foreground">
                    {fmtPrice(order.takeProfit)}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums text-xs text-muted-foreground">
                    {fmtPrice(order.stopLoss)}
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">{order.strategyId ?? "—"}</TableCell>
                </>
              )}
              <TableCell>
                <Badge variant="outline" className={`text-[10px] ${statusTone(order.status)}`}>
                  {order.status}
                </Badge>
              </TableCell>
              {!compact && (
                <TableCell className="text-right text-xs text-muted-foreground font-mono">
                  {order.createdAt
                    ? formatDistanceToNow(order.createdAt, { addSuffix: true })
                    : "—"}
                </TableCell>
              )}
            </TableRow>
          ))
        )}
      </TableBody>
    </Table>
  );

  if (compact && !expanded) {
    return (
      <div className="rounded-lg border border-border bg-card/30 backdrop-blur-sm">
        <div className="relative" style={{ maxHeight: "300px", overflowY: "auto" }}>
          {tableContent}
        </div>
        {orders.length > 0 && (
          <div className="p-2 border-t border-border/50 flex justify-end">
            <Dialog open={expanded} onOpenChange={setExpanded}>
              <DialogTrigger asChild>
                <Button variant="ghost" size="sm" className="h-7 text-xs">
                  <Maximize2 className="h-3 w-3 mr-1" />
                  Expand
                </Button>
              </DialogTrigger>
              <DialogContent className="max-w-[95vw] max-h-[90vh] overflow-auto">
                <DialogHeader>
                  <DialogTitle>Open Orders ({orders.length})</DialogTitle>
                </DialogHeader>
                <div className="rounded-lg border border-border bg-card/30 backdrop-blur-sm">
                  <Table>
                    <TableHeader>
                      <TableRow className="border-border/50 hover:bg-transparent">
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Symbol</TableHead>
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Side</TableHead>
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Size</TableHead>
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Filled</TableHead>
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Price</TableHead>
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">TP</TableHead>
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">SL</TableHead>
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Strategy</TableHead>
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider">Status</TableHead>
                        <TableHead className="text-xs text-muted-foreground uppercase tracking-wider text-right">Age</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {orders.map((order) => (
                        <TableRow key={order.id} className="border-border/50 hover:bg-muted/30">
                          <TableCell className="font-mono">{order.symbol}</TableCell>
                          <TableCell className={`font-mono ${sideTone(order.side)}`}>{order.side}</TableCell>
                          <TableCell className="text-right font-mono tabular-nums">{fmtSize(order.size)}</TableCell>
                          <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                            {fmtSize(order.filled)}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums">{fmtPrice(order.price)}</TableCell>
                          <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                            {fmtPrice(order.takeProfit)}
                          </TableCell>
                          <TableCell className="text-right font-mono tabular-nums text-muted-foreground">
                            {fmtPrice(order.stopLoss)}
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">{order.strategyId ?? "—"}</TableCell>
                          <TableCell>
                            <Badge variant="outline" className={statusTone(order.status)}>
                              {order.status}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-right text-xs text-muted-foreground font-mono">
                            {order.createdAt
                              ? formatDistanceToNow(order.createdAt, { addSuffix: true })
                              : "—"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </DialogContent>
            </Dialog>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card/30 backdrop-blur-sm">
      {tableContent}
    </div>
  );
}

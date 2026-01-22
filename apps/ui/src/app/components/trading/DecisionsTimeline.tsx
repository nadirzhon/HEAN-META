import { Clock } from "lucide-react";
import { Badge } from "@/app/components/ui/badge";

export interface Decision {
  id: string;
  type: "ORDER_DECISION" | "EXIT_DECISION";
  symbol: string;
  reasonCode: string;
  outcome: "HOLD" | "CLOSE" | "TP_HIT" | "SL_HIT" | "TIMEOUT_TTL" | "ENTRY" | "SUPPRESSED";
  timestamp: Date;
}

interface DecisionsTimelineProps {
  decisions: Decision[];
}

export function DecisionsTimeline({ decisions }: DecisionsTimelineProps) {
  const getOutcomeColor = (outcome: string) => {
    switch (outcome) {
      case "ENTRY":
        return "border-[var(--trading-cyan)]/50 text-[var(--trading-cyan)]";
      case "TP_HIT":
        return "border-[var(--trading-green)]/50 text-[var(--trading-green)]";
      case "CLOSE":
      case "SL_HIT":
        return "border-[var(--trading-red)]/50 text-[var(--trading-red)]";
      case "TIMEOUT_TTL":
        return "border-[var(--trading-amber)]/50 text-[var(--trading-amber)]";
      case "SUPPRESSED":
        return "border-[var(--trading-gray)]/50 text-[var(--trading-gray)]";
      default:
        return "border-muted-foreground/50 text-muted-foreground";
    }
  };

  const formatTimestamp = (date: Date) => {
    return date.toLocaleTimeString("en-US", { 
      hour12: false, 
      hour: "2-digit", 
      minute: "2-digit", 
      second: "2-digit" 
    });
  };

  return (
    <div className="space-y-3">
      {decisions.length === 0 ? (
        <div className="text-center text-muted-foreground py-8">
          No recent decisions
        </div>
      ) : (
        decisions.map((decision, index) => (
          <div key={decision.id} className="relative flex gap-4">
            {/* Timeline line */}
            {index < decisions.length - 1 && (
              <div className="absolute left-[11px] top-8 bottom-0 w-px bg-border" />
            )}
            
            {/* Timeline dot */}
            <div className="relative flex-shrink-0 mt-1">
              <div className="h-6 w-6 rounded-full border-2 border-[var(--trading-cyan)] bg-card flex items-center justify-center">
                <div className="h-2 w-2 rounded-full bg-[var(--trading-cyan)]" />
              </div>
            </div>

            {/* Content */}
            <div className="flex-1 pb-4 min-w-0">
              <div className="flex items-start justify-between gap-4 mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-mono text-foreground">{decision.symbol}</span>
                  <Badge variant="outline" className={getOutcomeColor(decision.outcome)}>
                    {decision.outcome}
                  </Badge>
                </div>
                <div className="flex items-center gap-1 text-xs text-muted-foreground font-mono flex-shrink-0">
                  <Clock className="h-3 w-3" />
                  {formatTimestamp(decision.timestamp)}
                </div>
              </div>
              <div className="text-xs text-muted-foreground">
                <span className="uppercase tracking-wider">{decision.type}</span>
                <span className="mx-2">•</span>
                <span>{decision.reasonCode}</span>
              </div>
            </div>
          </div>
        ))
      )}
    </div>
  );
}

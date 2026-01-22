import { CheckCircle2, XCircle, AlertCircle, ArrowRight } from "lucide-react";
import { Card } from "@/app/components/ui/card";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/app/components/ui/tooltip";

export interface FlowStep {
  stage: "TICK" | "SIGNAL" | "FILTER" | "RISK" | "ORDER" | "EXIT";
  status: "passed" | "suppressed" | "blocked";
  reasonCode?: string;
  details?: string;
}

interface DecisionFlowProps {
  symbol: string;
  steps: FlowStep[];
  timestamp: Date;
}

export function DecisionFlow({ symbol, steps, timestamp }: DecisionFlowProps) {
  const getStepIcon = (status: string) => {
    switch (status) {
      case "passed":
        return <CheckCircle2 className="h-5 w-5" />;
      case "suppressed":
        return <AlertCircle className="h-5 w-5" />;
      case "blocked":
        return <XCircle className="h-5 w-5" />;
    }
  };

  const getStepColor = (status: string) => {
    switch (status) {
      case "passed":
        return "border-[var(--trading-green)] text-[var(--trading-green)] bg-[var(--trading-green)]/10";
      case "suppressed":
        return "border-[var(--trading-amber)] text-[var(--trading-amber)] bg-[var(--trading-amber)]/10";
      case "blocked":
        return "border-[var(--trading-red)] text-[var(--trading-red)] bg-[var(--trading-red)]/10";
    }
  };

  return (
    <Card className="p-6 bg-card/30 backdrop-blur-sm border-border/50">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h4 className="font-mono text-lg">{symbol}</h4>
          <p className="text-xs text-muted-foreground font-mono">
            {timestamp.toLocaleString("en-US", { 
              month: "short", 
              day: "numeric", 
              hour: "2-digit", 
              minute: "2-digit", 
              second: "2-digit",
              hour12: false 
            })}
          </p>
        </div>
      </div>

      <TooltipProvider>
        <div className="flex items-center gap-2">
          {steps.map((step, index) => (
            <div key={step.stage} className="flex items-center">
              <Tooltip>
                <TooltipTrigger asChild>
                  <div 
                    className={`flex flex-col items-center gap-2 px-4 py-3 rounded-lg border transition-all cursor-default ${getStepColor(step.status)}`}
                  >
                    {getStepIcon(step.status)}
                    <span className="text-xs font-mono uppercase tracking-wider">{step.stage}</span>
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <div className="space-y-1">
                    <p className="font-mono text-xs">{step.stage}</p>
                    {step.reasonCode && (
                      <p className="text-xs text-muted-foreground">{step.reasonCode}</p>
                    )}
                    {step.details && (
                      <p className="text-xs text-muted-foreground">{step.details}</p>
                    )}
                  </div>
                </TooltipContent>
              </Tooltip>

              {index < steps.length - 1 && (
                <ArrowRight className="h-4 w-4 mx-2 text-muted-foreground" />
              )}
            </div>
          ))}
        </div>
      </TooltipProvider>
    </Card>
  );
}

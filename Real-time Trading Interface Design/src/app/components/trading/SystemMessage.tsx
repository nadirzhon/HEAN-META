import { Info } from "lucide-react";

interface SystemMessageProps {
  message: string;
  marketRegime?: "TREND" | "RANGE" | "CHAOS";
  volatilityLevel?: "LOW" | "MEDIUM" | "HIGH";
  riskState?: "LIMITED" | "ACTIVE" | "ELEVATED";
}

export function SystemMessage({ 
  message, 
  marketRegime = "TREND", 
  volatilityLevel = "MEDIUM", 
  riskState = "LIMITED" 
}: SystemMessageProps) {
  // Generate contextual message based on system state
  const getContextualMessage = () => {
    if (message) return message;
    
    const regimeText = {
      TREND: "Market shows clear directional movement",
      RANGE: "Market is consolidating in range",
      CHAOS: "Market volatility is extreme"
    }[marketRegime];

    const volatilityText = {
      LOW: "volatility is subdued",
      MEDIUM: "volatility is moderate", 
      HIGH: "volatility is elevated"
    }[volatilityLevel];

    const riskText = {
      LIMITED: "risk limits are active, exits are protected",
      ACTIVE: "risk exposure is normal",
      ELEVATED: "risk exposure is elevated, monitoring closely"
    }[riskState];

    return `${regimeText}, ${volatilityText}, ${riskText}.`;
  };

  const getRegimeColor = () => {
    switch (marketRegime) {
      case "TREND":
        return "border-[var(--trading-cyan)]/20 bg-[var(--trading-cyan)]/10";
      case "RANGE":
        return "border-[var(--trading-amber)]/20 bg-[var(--trading-amber)]/10";
      case "CHAOS":
        return "border-[var(--trading-red)]/20 bg-[var(--trading-red)]/10";
    }
  };

  return (
    <div className={`flex items-start gap-3 px-4 py-3 border rounded-lg ${getRegimeColor()}`}>
      <Info className="h-5 w-5 text-[var(--trading-cyan)] flex-shrink-0 mt-0.5" />
      <div className="flex-1">
        <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
          System Brain
        </div>
        <p className="text-sm text-foreground/90 leading-relaxed">
          {getContextualMessage()}
        </p>
      </div>
    </div>
  );
}
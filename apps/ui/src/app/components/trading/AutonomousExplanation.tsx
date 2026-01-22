import { useEffect, useState } from "react";
import { Brain } from "lucide-react";

interface AutonomousExplanationProps {
  systemMode: "CALM" | "ACTIVE" | "DEFENSIVE" | "EMERGENCY";
  marketRegime: "TREND" | "RANGE" | "CHAOS";
  positionCount: number;
  riskLevel: number; // 0-100
  volatility: number; // 0-100
}

export function AutonomousExplanation({ 
  systemMode, 
  marketRegime, 
  positionCount,
  riskLevel,
  volatility 
}: AutonomousExplanationProps) {
  const [explanation, setExplanation] = useState("");

  useEffect(() => {
    const generateExplanation = () => {
      let sentence = "";

      // Start with market assessment
      if (marketRegime === "TREND") {
        sentence += volatility > 60 
          ? "The system is trading selectively in volatile trend conditions"
          : "The system is following directional momentum";
      } else if (marketRegime === "RANGE") {
        sentence += "The system is trading mean-reversion setups in range conditions";
      } else {
        sentence += "The system has reduced aggression due to chaotic market conditions";
      }

      // Add position context
      if (positionCount === 0) {
        sentence += ", currently observing without exposure";
      } else if (positionCount >= 3) {
        sentence += ", managing multiple positions simultaneously";
      } else {
        sentence += `, actively managing ${positionCount} position${positionCount > 1 ? 's' : ''}`;
      }

      // Add risk assessment
      if (riskLevel > 70) {
        sentence += ", prioritizing capital protection over opportunity";
      } else if (riskLevel > 40) {
        sentence += ", balancing risk and reward";
      } else {
        sentence += ", with controlled risk exposure";
      }

      // Add defensive context
      if (systemMode === "DEFENSIVE") {
        sentence += ". Exits are tightened, new entries suppressed.";
      } else if (systemMode === "EMERGENCY") {
        sentence += ". Emergency protocols active, liquidation priority.";
      } else if (volatility > 70) {
        sentence += ". Position sizing reduced due to elevated volatility.";
      } else {
        sentence += ". All risk parameters within normal ranges.";
      }

      setExplanation(sentence);
    };

    generateExplanation();
  }, [systemMode, marketRegime, positionCount, riskLevel, volatility]);

  return (
    <div className="fixed bottom-6 left-6 right-6 z-50 pointer-events-none">
      <div className="max-w-4xl mx-auto">
        <div className="flex items-start gap-3 px-6 py-4 bg-card/95 backdrop-blur-md border border-border/50 rounded-lg shadow-2xl pointer-events-auto">
          <div className="flex-shrink-0 mt-1">
            <Brain className="h-5 w-5 text-[var(--trading-cyan)]" />
          </div>
          <div className="flex-1">
            <div className="text-xs text-muted-foreground uppercase tracking-wider mb-1">
              Autonomous Explanation Engine
            </div>
            <p className="text-sm text-foreground/95 leading-relaxed">
              {explanation}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

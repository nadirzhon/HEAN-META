import { Card } from "@/app/components/ui/card";

interface StrategyRiskPosition {
  market: number; // 0-100
  strategy: number; // 0-100
  risk: number; // 0-100
}

interface StrategyRiskTriangleProps {
  position: StrategyRiskPosition;
}

export function StrategyRiskTriangle({ position }: StrategyRiskTriangleProps) {
  // Calculate barycentric coordinates for triangle
  const size = 280;
  const padding = 40;
  
  // Triangle vertices (equilateral)
  const topVertex = { x: size / 2, y: padding };
  const leftVertex = { x: padding, y: size - padding };
  const rightVertex = { x: size - padding, y: size - padding };
  
  // Convert percentages to barycentric coordinates
  const market = position.market / 100;
  const strategy = position.strategy / 100;
  const risk = position.risk / 100;
  
  // Normalize
  const total = market + strategy + risk;
  const normMarket = market / total;
  const normStrategy = strategy / total;
  const normRisk = risk / total;
  
  // Calculate point position in triangle
  const pointX = 
    normMarket * topVertex.x +
    normStrategy * leftVertex.x +
    normRisk * rightVertex.x;
  const pointY =
    normMarket * topVertex.y +
    normStrategy * leftVertex.y +
    normRisk * rightVertex.y;
  
  // Determine warning state
  const isRiskDominant = normRisk > 0.5;
  const isBalanced = normMarket > 0.25 && normStrategy > 0.25 && normRisk > 0.25;
  
  return (
    <Card className="p-6 bg-card/30 backdrop-blur-sm border-border/50">
      <div className="mb-6">
        <h3 className="text-sm uppercase tracking-wider text-muted-foreground mb-1">
          Strategy × Market × Risk
        </h3>
        <p className="text-xs text-muted-foreground">
          System state visualization — no numbers, pure intuition
        </p>
      </div>

      <svg width={size} height={size} className="mx-auto">
        <defs>
          <linearGradient id="triangleGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="var(--trading-cyan)" stopOpacity={0.05} />
            <stop offset="50%" stopColor="var(--trading-amber)" stopOpacity={0.05} />
            <stop offset="100%" stopColor="var(--trading-red)" stopOpacity={0.05} />
          </linearGradient>
          
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge>
              <feMergeNode in="coloredBlur"/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        
        {/* Triangle background */}
        <polygon
          points={`${topVertex.x},${topVertex.y} ${leftVertex.x},${leftVertex.y} ${rightVertex.x},${rightVertex.y}`}
          fill="url(#triangleGradient)"
          stroke="var(--color-border)"
          strokeWidth="2"
        />
        
        {/* Grid lines */}
        <line
          x1={topVertex.x}
          y1={topVertex.y}
          x2={(leftVertex.x + rightVertex.x) / 2}
          y2={leftVertex.y}
          stroke="var(--color-border)"
          strokeWidth="1"
          strokeDasharray="4 4"
          opacity="0.3"
        />
        <line
          x1={leftVertex.x}
          y1={leftVertex.y}
          x2={(topVertex.x + rightVertex.x) / 2}
          y2={(topVertex.y + rightVertex.y) / 2}
          stroke="var(--color-border)"
          strokeWidth="1"
          strokeDasharray="4 4"
          opacity="0.3"
        />
        <line
          x1={rightVertex.x}
          y1={rightVertex.y}
          x2={(topVertex.x + leftVertex.x) / 2}
          y2={(topVertex.y + leftVertex.y) / 2}
          stroke="var(--color-border)"
          strokeWidth="1"
          strokeDasharray="4 4"
          opacity="0.3"
        />
        
        {/* Vertex labels */}
        <text
          x={topVertex.x}
          y={topVertex.y - 10}
          textAnchor="middle"
          fill="var(--trading-cyan)"
          fontSize="12"
          fontFamily="monospace"
          fontWeight="500"
        >
          MARKET
        </text>
        <text
          x={leftVertex.x - 10}
          y={leftVertex.y + 5}
          textAnchor="end"
          fill="var(--trading-amber)"
          fontSize="12"
          fontFamily="monospace"
          fontWeight="500"
        >
          STRATEGY
        </text>
        <text
          x={rightVertex.x + 10}
          y={rightVertex.y + 5}
          textAnchor="start"
          fill="var(--trading-red)"
          fontSize="12"
          fontFamily="monospace"
          fontWeight="500"
        >
          RISK
        </text>
        
        {/* Current position */}
        <circle
          cx={pointX}
          cy={pointY}
          r="8"
          fill={isRiskDominant ? "var(--trading-red)" : isBalanced ? "var(--trading-green)" : "var(--trading-cyan)"}
          filter="url(#glow)"
          opacity="0.9"
        >
          <animate
            attributeName="r"
            values="8;10;8"
            dur="2s"
            repeatCount="indefinite"
          />
        </circle>
        <circle
          cx={pointX}
          cy={pointY}
          r="16"
          fill="none"
          stroke={isRiskDominant ? "var(--trading-red)" : isBalanced ? "var(--trading-green)" : "var(--trading-cyan)"}
          strokeWidth="2"
          opacity="0.3"
        >
          <animate
            attributeName="r"
            values="16;20;16"
            dur="2s"
            repeatCount="indefinite"
          />
        </circle>
      </svg>

      {/* Status indicator */}
      <div className="mt-6 pt-6 border-t border-border/50">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground uppercase tracking-wider">
            System State
          </span>
          <div className="flex items-center gap-2">
            <div className={`h-2 w-2 rounded-full ${
              isRiskDominant 
                ? "bg-[var(--trading-red)]" 
                : isBalanced 
                ? "bg-[var(--trading-green)]" 
                : "bg-[var(--trading-cyan)]"
            }`} />
            <span className="text-sm font-mono">
              {isRiskDominant 
                ? "Risk Dominant" 
                : isBalanced 
                ? "Balanced" 
                : "Market Following"}
            </span>
          </div>
        </div>
        {isRiskDominant && (
          <p className="text-xs text-[var(--trading-red)] mt-2">
            Warning: System approaching risk constraint boundary
          </p>
        )}
      </div>
    </Card>
  );
}

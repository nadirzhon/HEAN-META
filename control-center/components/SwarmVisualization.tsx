'use client';

import { useEffect, useState } from 'react';

interface SwarmData {
  symbol: string;
  consensus_percentage: number;
  buy_votes: number;
  sell_votes: number;
  total_agents: number;
  average_confidence: number;
  execution_signal_strength: number;
  consensus_reached: boolean;
  price_level_ofi?: number[];  // OFI at each price level for heat map
}

interface SwarmVisualizationProps {
  swarmData?: SwarmData | null;
  symbol?: string;
}

export default function SwarmVisualization({ swarmData, symbol }: SwarmVisualizationProps) {
  const [heatMapData, setHeatMapData] = useState<number[]>([]);

  useEffect(() => {
    if (swarmData?.price_level_ofi) {
      setHeatMapData(swarmData.price_level_ofi);
    }
  }, [swarmData]);

  if (!swarmData) {
    return (
      <div className="hean-card">
        <h2 className="text-xl font-bold mb-4 hean-text-glow">Swarm Intelligence</h2>
        <div className="text-hean-secondary text-center py-8">
          Waiting for swarm data...
        </div>
      </div>
    );
  }

  const consensusPercentage = swarmData.consensus_percentage || 0;
  const confidence = swarmData.average_confidence || 0;
  const signalStrength = swarmData.execution_signal_strength || 0;
  const buyPercentage = swarmData.total_agents > 0 
    ? (swarmData.buy_votes / swarmData.total_agents) * 100 
    : 0;
  const sellPercentage = swarmData.total_agents > 0 
    ? (swarmData.sell_votes / swarmData.total_agents) * 100 
    : 0;

  // Determine consensus color
  const getConsensusColor = () => {
    if (consensusPercentage >= 80) {
      return swarmData.buy_votes > swarmData.sell_votes ? 'text-green-400' : 'text-red-400';
    }
    return 'text-hean-secondary';
  };

  // Generate heat map colors based on OFI values
  const getHeatMapColor = (ofiValue: number) => {
    // OFI ranges from -1 (strong sell) to +1 (strong buy)
    const normalized = (ofiValue + 1) / 2; // Map to [0, 1]
    const intensity = Math.abs(ofiValue);
    
    if (ofiValue > 0.3) {
      // Strong buying pressure - green gradient
      return `rgba(34, 197, 94, ${intensity})`; // green-500
    } else if (ofiValue < -0.3) {
      // Strong selling pressure - red gradient
      return `rgba(239, 68, 68, ${intensity})`; // red-500
    } else {
      // Neutral - gray
      return `rgba(156, 163, 175, ${0.3})`; // gray-400
    }
  };

  return (
    <div className="hean-card">
      <h2 className="text-xl font-bold mb-4 hean-text-glow">
        Swarm Intelligence
        {symbol && <span className="text-sm font-normal text-hean-secondary ml-2">({symbol})</span>}
      </h2>

      {/* Swarm Confidence Meter */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-hean-secondary text-sm">Swarm Confidence</span>
          <span className={`text-lg font-bold ${getConsensusColor()}`}>
            {(consensusPercentage).toFixed(1)}%
          </span>
        </div>
        
        <div className="w-full bg-hean-bg-secondary rounded-full h-4 mb-2 overflow-hidden">
          <div
            className={`h-full transition-all duration-300 ${
              swarmData.buy_votes > swarmData.sell_votes 
                ? 'bg-gradient-to-r from-green-500 to-green-400' 
                : 'bg-gradient-to-r from-red-500 to-red-400'
            }`}
            style={{ width: `${consensusPercentage}%` }}
          />
        </div>

        <div className="flex justify-between text-xs text-hean-secondary">
          <span>Consensus Threshold: 80%</span>
          <span className={swarmData.consensus_reached ? 'text-green-400 font-bold' : ''}>
            {swarmData.consensus_reached ? '✓ CONSENSUS REACHED' : 'Waiting for consensus...'}
          </span>
        </div>
      </div>

      {/* Agent Vote Breakdown */}
      <div className="mb-6">
        <h3 className="text-sm font-semibold text-hean-secondary mb-3">Agent Votes</h3>
        <div className="space-y-2">
          <div className="flex items-center">
            <div className="w-20 text-sm text-green-400">BUY</div>
            <div className="flex-1 bg-hean-bg-secondary rounded-full h-6 mr-2 overflow-hidden">
              <div
                className="h-full bg-green-500 transition-all duration-300"
                style={{ width: `${buyPercentage}%` }}
              />
            </div>
            <div className="w-16 text-right text-sm text-hean-secondary">
              {swarmData.buy_votes}/{swarmData.total_agents}
            </div>
          </div>
          
          <div className="flex items-center">
            <div className="w-20 text-sm text-red-400">SELL</div>
            <div className="flex-1 bg-hean-bg-secondary rounded-full h-6 mr-2 overflow-hidden">
              <div
                className="h-full bg-red-500 transition-all duration-300"
                style={{ width: `${sellPercentage}%` }}
              />
            </div>
            <div className="w-16 text-right text-sm text-hean-secondary">
              {swarmData.sell_votes}/{swarmData.total_agents}
            </div>
          </div>
        </div>
      </div>

      {/* Signal Strength Indicator */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-hean-secondary text-sm">Execution Signal Strength</span>
          <span className="text-sm font-bold text-hean-primary">
            {(signalStrength * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-hean-bg-secondary rounded-full h-3">
          <div
            className="h-full bg-gradient-to-r from-hean-primary to-hean-accent transition-all duration-300"
            style={{ width: `${signalStrength * 100}%` }}
          />
        </div>
      </div>

      {/* Average Confidence */}
      <div className="mb-6">
        <div className="flex justify-between items-center mb-2">
          <span className="text-hean-secondary text-sm">Average Agent Confidence</span>
          <span className="text-sm font-bold text-hean-primary">
            {(confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="w-full bg-hean-bg-secondary rounded-full h-3">
          <div
            className="h-full bg-hean-accent transition-all duration-300"
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>

      {/* OFI Heat Map Visualization */}
      {heatMapData.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-hean-secondary mb-3">
            Order-Flow Imbalance Heat Map (3D Orderbook)
          </h3>
          <div className="grid grid-cols-10 gap-1">
            {heatMapData.slice(0, 50).map((ofiValue, index) => (
              <div
                key={index}
                className="aspect-square rounded transition-all duration-200 hover:scale-110 cursor-pointer"
                style={{
                  backgroundColor: getHeatMapColor(ofiValue),
                }}
                title={`Level ${index}: OFI = ${ofiValue.toFixed(3)}`}
              />
            ))}
          </div>
          <div className="flex justify-between text-xs text-hean-secondary mt-2">
            <span>Strong Sell</span>
            <span>Neutral</span>
            <span>Strong Buy</span>
          </div>
          <div className="flex justify-between mt-1">
            <div className="w-8 h-3 bg-red-500 rounded"></div>
            <div className="w-8 h-3 bg-gray-400 rounded"></div>
            <div className="w-8 h-3 bg-green-500 rounded"></div>
          </div>
        </div>
      )}

      {/* Agent Distribution (if available) */}
      <div className="mt-6 pt-4 border-t border-hean-bg-secondary">
        <div className="text-xs text-hean-secondary">
          <div className="flex justify-between mb-1">
            <span>Total Agents:</span>
            <span className="text-hean-primary font-bold">{swarmData.total_agents}</span>
          </div>
          <div className="text-xs text-hean-secondary mt-2">
            Specialized Agents: Delta Analyzers, OFI Analyzers, VPIN Analyzers, Micro-Momentum
          </div>
        </div>
      </div>
    </div>
  );
}

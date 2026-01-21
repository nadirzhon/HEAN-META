'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/lib/websocket';

interface ArbCycle {
  id: string;
  pair_a: string;
  pair_b: string;
  pair_c: string;
  asset_a: string;
  asset_b: string;
  asset_c: string;
  profit_bps: number;
  profit_ratio: number;
  max_size: number;
  detection_time: number;
  execution_time_us?: number;
  status: 'detected' | 'executing' | 'completed' | 'failed';
  revenue?: number;
}

export default function LiveArbChains() {
  const [cycles, setCycles] = useState<ArbCycle[]>([]);
  const [stats, setStats] = useState({
    totalCycles: 0,
    completedCycles: 0,
    totalRevenue: 0,
    avgExecutionTime: 0,
    successRate: 0,
  });
  const { lastMessage } = useWebSocket();

  useEffect(() => {
    // Subscribe to triangular arbitrage events
    if (lastMessage?.topic === 'triangular_arb' || lastMessage?.topic === 'arb_cycles') {
      const data = lastMessage.data;
      
      if (data.cycle) {
        // New cycle detected
        const newCycle: ArbCycle = {
          id: data.cycle.id || `${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          pair_a: data.cycle.pair_a || data.cycle.pairA || '',
          pair_b: data.cycle.pair_b || data.cycle.pairB || '',
          pair_c: data.cycle.pair_c || data.cycle.pairC || '',
          asset_a: data.cycle.asset_a || data.cycle.assetA || '',
          asset_b: data.cycle.asset_b || data.cycle.assetB || '',
          asset_c: data.cycle.asset_c || data.cycle.assetC || '',
          profit_bps: data.cycle.profit_bps || data.cycle.profitBps || 0,
          profit_ratio: data.cycle.profit_ratio || data.cycle.profitRatio || 0,
          max_size: data.cycle.max_size || data.cycle.maxSize || 0,
          detection_time: data.cycle.detection_time || Date.now(),
          execution_time_us: data.cycle.execution_time_us || data.cycle.executionTimeUs,
          status: data.cycle.status || 'detected',
          revenue: data.cycle.revenue || 0,
        };

        setCycles((prev) => {
          const updated = [newCycle, ...prev].slice(0, 50); // Keep last 50 cycles
          return updated;
        });
      }

      if (data.stats) {
        // Update statistics
        setStats({
          totalCycles: data.stats.total_cycles || data.stats.totalCycles || 0,
          completedCycles: data.stats.completed_cycles || data.stats.completedCycles || 0,
          totalRevenue: data.stats.total_revenue || data.stats.totalRevenue || 0,
          avgExecutionTime: data.stats.avg_execution_time_us || data.stats.avgExecutionTimeUs || 0,
          successRate: data.stats.success_rate || data.stats.successRate || 0,
        });
      }
    }
  }, [lastMessage]);

  // Update cycle status based on execution events
  useEffect(() => {
    if (lastMessage?.topic === 'order_filled' || lastMessage?.topic === 'order_cancelled') {
      const order = lastMessage.data?.order;
      if (order?.metadata?.atomic_group) {
        setCycles((prev) =>
          prev.map((cycle) => {
            if (cycle.id === order.metadata.atomic_group) {
              return {
                ...cycle,
                status: lastMessage.topic === 'order_filled' ? 'completed' : 'failed',
                revenue: order.metadata.revenue || cycle.revenue,
              };
            }
            return cycle;
          })
        );
      }
    }
  }, [lastMessage]);

  const formatTime = (us: number) => {
    if (us < 1000) return `${us.toFixed(2)}μs`;
    if (us < 1000000) return `${(us / 1000).toFixed(2)}ms`;
    return `${(us / 1000000).toFixed(2)}s`;
  };

  const formatProfit = (bps: number) => {
    return `${bps >= 0 ? '+' : ''}${bps.toFixed(2)}bps`;
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'detected':
        return 'text-yellow-400';
      case 'executing':
        return 'text-blue-400';
      case 'completed':
        return 'text-green-400';
      case 'failed':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="bg-hean-dark border border-hean-border rounded-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold hean-text-glow">Live Arb Chains</h2>
        <div className="flex gap-4 text-sm">
          <div className="text-hean-secondary">
            <span className="font-semibold">Total:</span> {stats.totalCycles}
          </div>
          <div className="text-green-400">
            <span className="font-semibold">Completed:</span> {stats.completedCycles}
          </div>
          <div className="text-yellow-400">
            <span className="font-semibold">Success:</span> {stats.successRate.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Statistics Bar */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-hean-darker rounded p-3">
          <div className="text-hean-secondary text-xs mb-1">Total Revenue</div>
          <div className="text-green-400 text-lg font-bold">
            ${stats.totalRevenue.toFixed(2)}
          </div>
        </div>
        <div className="bg-hean-darker rounded p-3">
          <div className="text-hean-secondary text-xs mb-1">Avg Execution</div>
          <div className="text-blue-400 text-lg font-bold">
            {formatTime(stats.avgExecutionTime)}
          </div>
        </div>
        <div className="bg-hean-darker rounded p-3">
          <div className="text-hean-secondary text-xs mb-1">Active Cycles</div>
          <div className="text-yellow-400 text-lg font-bold">
            {cycles.filter((c) => c.status === 'executing' || c.status === 'detected').length}
          </div>
        </div>
        <div className="bg-hean-darker rounded p-3">
          <div className="text-hean-secondary text-xs mb-1">Success Rate</div>
          <div className="text-green-400 text-lg font-bold">
            {stats.successRate.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Cycles Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-hean-border text-hean-secondary">
              <th className="text-left py-2 px-3">Cycle</th>
              <th className="text-left py-2 px-3">Profit</th>
              <th className="text-left py-2 px-3">Size</th>
              <th className="text-left py-2 px-3">Execution</th>
              <th className="text-left py-2 px-3">Revenue</th>
              <th className="text-left py-2 px-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {cycles.slice(0, 20).map((cycle) => (
              <tr
                key={cycle.id}
                className="border-b border-hean-border hover:bg-hean-darker transition-colors"
              >
                <td className="py-2 px-3">
                  <div className="font-mono text-xs">
                    <div className="flex items-center gap-1">
                      <span className="text-blue-400">{cycle.asset_a}</span>
                      <span className="text-hean-secondary">→</span>
                      <span className="text-purple-400">{cycle.asset_b}</span>
                      <span className="text-hean-secondary">→</span>
                      <span className="text-green-400">{cycle.asset_c}</span>
                      <span className="text-hean-secondary">→</span>
                      <span className="text-blue-400">{cycle.asset_a}</span>
                    </div>
                    <div className="text-hean-secondary text-xs mt-1">
                      {cycle.pair_a} • {cycle.pair_b} • {cycle.pair_c}
                    </div>
                  </div>
                </td>
                <td className="py-2 px-3">
                  <span
                    className={`font-bold ${
                      cycle.profit_bps >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}
                  >
                    {formatProfit(cycle.profit_bps)}
                  </span>
                  <div className="text-hean-secondary text-xs">
                    {(cycle.profit_ratio * 100).toFixed(4)}%
                  </div>
                </td>
                <td className="py-2 px-3 font-mono text-xs">
                  {cycle.max_size.toFixed(6)}
                </td>
                <td className="py-2 px-3 font-mono text-xs">
                  {cycle.execution_time_us
                    ? formatTime(cycle.execution_time_us)
                    : cycle.status === 'detected'
                    ? '< 500μs'
                    : '-'}
                </td>
                <td className="py-2 px-3">
                  {cycle.revenue ? (
                    <span className="text-green-400 font-bold">
                      ${cycle.revenue.toFixed(4)}
                    </span>
                  ) : (
                    <span className="text-hean-secondary">-</span>
                  )}
                </td>
                <td className="py-2 px-3">
                  <span className={`font-semibold ${getStatusColor(cycle.status)}`}>
                    {cycle.status.toUpperCase()}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {cycles.length === 0 && (
        <div className="text-center py-12 text-hean-secondary">
          <div className="text-4xl mb-4">⏳</div>
          <div>Waiting for triangular arbitrage opportunities...</div>
          <div className="text-xs mt-2">
            Monitoring 50+ trading pairs simultaneously
          </div>
        </div>
      )}
    </div>
  );
}

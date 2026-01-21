'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/lib/websocket';

interface NodeStatus {
  region: string;
  role: string;
  is_healthy: boolean;
  is_alive: boolean;
  cpu_usage: number;
  memory_usage: number;
  network_latency_ms: number;
  exchange_latencies: Record<string, number>;
  active_connections: number;
}

interface NetworkStats {
  local_region: string;
  local_role: string;
  master_node: string | null;
  nodes: Record<string, NodeStatus>;
  execution_count: Record<string, number>;
  failover_count: number;
  active_positions: number;
  active_orders: number;
}

interface NetworkMapProps {
  stats?: NetworkStats;
}

export default function NetworkMap({ stats }: NetworkMapProps) {
  const { lastMessage } = useWebSocket();
  const [networkData, setNetworkData] = useState<NetworkStats | null>(stats || null);
  const [viewMode, setViewMode] = useState<'3d' | '2d'>('3d');
  const [selectedNode, setSelectedNode] = useState<string | null>(null);

  useEffect(() => {
    if (lastMessage?.topic === 'network_status' || lastMessage?.topic === 'system_status') {
      try {
        const data = typeof lastMessage.data === 'string' 
          ? JSON.parse(lastMessage.data) 
          : lastMessage.data;
        
        if (data.network_stats) {
          setNetworkData(data.network_stats);
        }
      } catch (e) {
        console.error('Failed to parse network status:', e);
      }
    }
  }, [lastMessage]);

  const regions = ['TOKYO', 'SINGAPORE', 'FRANKFURT'];
  const exchanges = ['bybit', 'binance', 'okx'];

  // Calculate node positions for 3D visualization
  const getNodePosition = (region: string, index: number) => {
    const positions: Record<string, { x: number; y: number; z: number }> = {
      TOKYO: { x: -150, y: 50, z: -100 },
      SINGAPORE: { x: 0, y: 0, z: 100 },
      FRANKFURT: { x: 150, y: -50, z: -100 },
    };
    return positions[region] || { x: 0, y: 0, z: 0 };
  };

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'MASTER':
        return 'bg-green-500';
      case 'HEDGE':
        return 'bg-blue-500';
      case 'STANDBY':
        return 'bg-gray-500';
      default:
        return 'bg-gray-700';
    }
  };

  const getHealthColor = (isHealthy: boolean, isAlive: boolean) => {
    if (!isAlive) return 'bg-red-600';
    if (!isHealthy) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  if (!networkData) {
    return (
      <div className="bg-hean-dark/50 rounded-lg border border-hean-border p-8">
        <div className="text-center text-hean-secondary">
          <div className="animate-spin inline-block w-8 h-8 border-4 border-hean-primary border-t-transparent rounded-full mb-4"></div>
          <p>Loading network map...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-hean-dark/50 rounded-lg border border-hean-border p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-hean-primary mb-2">Global Network Map</h2>
          <p className="text-sm text-hean-secondary">Phase 19: Distributed Execution Mesh</p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setViewMode(viewMode === '3d' ? '2d' : '3d')}
            className={`px-4 py-2 rounded-lg border ${
              viewMode === '3d'
                ? 'bg-hean-primary text-hean-dark border-hean-primary'
                : 'bg-hean-dark text-hean-primary border-hean-border hover:bg-hean-dark/80'
            } transition-colors`}
          >
            {viewMode === '3d' ? '3D View' : '2D View'}
          </button>
        </div>
      </div>

      {/* Network Overview Stats */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="bg-hean-dark/80 rounded-lg p-4 border border-hean-border">
          <div className="text-sm text-hean-secondary mb-1">Master Node</div>
          <div className="text-xl font-bold text-hean-primary">
            {networkData.master_node || 'N/A'}
          </div>
        </div>
        <div className="bg-hean-dark/80 rounded-lg p-4 border border-hean-border">
          <div className="text-sm text-hean-secondary mb-1">Failover Count</div>
          <div className="text-xl font-bold text-red-400">{networkData.failover_count}</div>
        </div>
        <div className="bg-hean-dark/80 rounded-lg p-4 border border-hean-border">
          <div className="text-sm text-hean-secondary mb-1">Active Positions</div>
          <div className="text-xl font-bold text-hean-primary">{networkData.active_positions}</div>
        </div>
        <div className="bg-hean-dark/80 rounded-lg p-4 border border-hean-border">
          <div className="text-sm text-hean-secondary mb-1">Active Orders</div>
          <div className="text-xl font-bold text-hean-primary">{networkData.active_orders}</div>
        </div>
      </div>

      {/* 3D Network Visualization */}
      {viewMode === '3d' ? (
        <div className="relative w-full h-[600px] bg-hean-dark/30 rounded-lg border border-hean-border overflow-hidden mb-6">
          <svg
            viewBox="-200 -150 400 300"
            className="w-full h-full"
            style={{ perspective: '1000px' }}
          >
            {/* Grid background */}
            <defs>
              <pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse">
                <path d="M 20 0 L 0 0 0 20" fill="none" stroke="rgba(255,255,255,0.05)" strokeWidth="1" />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />

            {/* Exchange nodes (bottom) */}
            {exchanges.map((exchange, idx) => {
              const angle = (idx * 2 * Math.PI) / exchanges.length;
              const x = Math.cos(angle) * 80;
              const y = Math.sin(angle) * 80;
              return (
                <g key={exchange}>
                  <circle
                    cx={x}
                    cy={y}
                    r="8"
                    fill="#8b5cf6"
                    className="cursor-pointer hover:fill-purple-400 transition-colors"
                  />
                  <text
                    x={x}
                    y={y - 15}
                    textAnchor="middle"
                    className="fill-hean-secondary text-xs font-mono"
                  >
                    {exchange.toUpperCase()}
                  </text>
                </g>
              );
            })}

            {/* Regional nodes */}
            {regions.map((region, idx) => {
              const node = networkData.nodes[region];
              if (!node) return null;

              const pos = getNodePosition(region, idx);
              const isMaster = networkData.master_node === region;
              const isLocal = networkData.local_region === region;

              return (
                <g key={region}>
                  {/* Connection lines to exchanges */}
                  {exchanges.map((exchange, eIdx) => {
                    const exAngle = (eIdx * 2 * Math.PI) / exchanges.length;
                    const exX = Math.cos(exAngle) * 80;
                    const exY = Math.sin(exAngle) * 80;
                    const latency = node.exchange_latencies[exchange] || 0;
                    const opacity = latency > 0 ? Math.max(0.1, 1 - latency / 500) : 0.1;

                    return (
                      <line
                        key={`${region}-${exchange}`}
                        x1={pos.x}
                        y1={pos.y}
                        x2={exX}
                        y2={exY}
                        stroke={isMaster ? '#10b981' : '#3b82f6'}
                        strokeWidth={isMaster ? 2 : 1}
                        opacity={opacity}
                        strokeDasharray={isMaster ? '0' : '3,3'}
                      />
                    );
                  })}

                  {/* Connection lines between nodes */}
                  {regions.map((otherRegion) => {
                    if (otherRegion <= region) return null;
                    const otherNode = networkData.nodes[otherRegion];
                    if (!otherNode) return null;

                    const otherPos = getNodePosition(otherRegion, regions.indexOf(otherRegion));
                    return (
                      <line
                        key={`${region}-${otherRegion}`}
                        x1={pos.x}
                        y1={pos.y}
                        x2={otherPos.x}
                        y2={otherPos.y}
                        stroke="#6b7280"
                        strokeWidth="1"
                        opacity="0.3"
                        strokeDasharray="5,5"
                      />
                    );
                  })}

                  {/* Node circle */}
                  <g
                    className="cursor-pointer transition-transform hover:scale-110"
                    onClick={() => setSelectedNode(selectedNode === region ? null : region)}
                  >
                    <circle
                      cx={pos.x}
                      cy={pos.y}
                      r={isMaster ? 16 : 12}
                      fill={getRoleColor(node.role)}
                      stroke={isLocal ? '#fbbf24' : '#374151'}
                      strokeWidth={isMaster ? 3 : isLocal ? 2 : 1}
                      className="drop-shadow-lg"
                    />
                    {/* Health indicator */}
                    <circle
                      cx={pos.x + (isMaster ? 10 : 8)}
                      cy={pos.y - (isMaster ? 10 : 8)}
                      r="4"
                      fill={getHealthColor(node.is_healthy, node.is_alive)}
                      stroke="#000"
                      strokeWidth="1"
                    />
                    {/* Node label */}
                    <text
                      x={pos.x}
                      y={pos.y + 25}
                      textAnchor="middle"
                      className="fill-hean-primary text-sm font-bold"
                    >
                      {region}
                    </text>
                    <text
                      x={pos.x}
                      y={pos.y + 38}
                      textAnchor="middle"
                      className="fill-hean-secondary text-xs"
                    >
                      {node.role}
                    </text>
                  </g>
                </g>
              );
            })}

            {/* Latency labels */}
            {selectedNode && networkData.nodes[selectedNode] && (
              <g>
                {exchanges.map((exchange, eIdx) => {
                  const node = networkData.nodes[selectedNode];
                  const latency = node.exchange_latencies[exchange] || 0;
                  if (latency === 0) return null;

                  const pos = getNodePosition(selectedNode, regions.indexOf(selectedNode));
                  const exAngle = (eIdx * 2 * Math.PI) / exchanges.length;
                  const exX = Math.cos(exAngle) * 80;
                  const exY = Math.sin(exAngle) * 80;
                  const midX = (pos.x + exX) / 2;
                  const midY = (pos.y + exY) / 2;

                  return (
                    <g key={`label-${exchange}`}>
                      <rect
                        x={midX - 25}
                        y={midY - 8}
                        width="50"
                        height="16"
                        fill="rgba(0,0,0,0.8)"
                        rx="4"
                      />
                      <text
                        x={midX}
                        y={midY + 4}
                        textAnchor="middle"
                        className="fill-hean-primary text-xs font-mono"
                      >
                        {latency.toFixed(1)}ms
                      </text>
                    </g>
                  );
                })}
              </g>
            )}
          </svg>
        </div>
      ) : (
        /* 2D Table View */
        <div className="overflow-x-auto mb-6">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-hean-border">
                <th className="text-left p-3 text-hean-secondary font-semibold">Region</th>
                <th className="text-left p-3 text-hean-secondary font-semibold">Role</th>
                <th className="text-left p-3 text-hean-secondary font-semibold">Status</th>
                <th className="text-left p-3 text-hean-secondary font-semibold">CPU</th>
                <th className="text-left p-3 text-hean-secondary font-semibold">Memory</th>
                <th className="text-left p-3 text-hean-secondary font-semibold">Network Latency</th>
                <th className="text-left p-3 text-hean-secondary font-semibold">Exchange Latencies</th>
                <th className="text-left p-3 text-hean-secondary font-semibold">Connections</th>
              </tr>
            </thead>
            <tbody>
              {regions.map((region) => {
                const node = networkData.nodes[region];
                if (!node) return null;

                return (
                  <tr
                    key={region}
                    className={`border-b border-hean-border/50 hover:bg-hean-dark/50 cursor-pointer ${
                      selectedNode === region ? 'bg-hean-dark/70' : ''
                    } ${networkData.master_node === region ? 'ring-2 ring-green-500' : ''}`}
                    onClick={() => setSelectedNode(selectedNode === region ? null : region)}
                  >
                    <td className="p-3 text-hean-primary font-mono">{region}</td>
                    <td className="p-3">
                      <span
                        className={`px-2 py-1 rounded text-xs font-semibold ${
                          node.role === 'MASTER'
                            ? 'bg-green-500/20 text-green-400'
                            : node.role === 'HEDGE'
                            ? 'bg-blue-500/20 text-blue-400'
                            : 'bg-gray-500/20 text-gray-400'
                        }`}
                      >
                        {node.role}
                      </span>
                    </td>
                    <td className="p-3">
                      <div className="flex items-center gap-2">
                        <div
                          className={`w-2 h-2 rounded-full ${
                            getHealthColor(node.is_healthy, node.is_alive)
                          }`}
                        />
                        <span className="text-hean-secondary">
                          {node.is_alive ? (node.is_healthy ? 'Healthy' : 'Unhealthy') : 'Offline'}
                        </span>
                      </div>
                    </td>
                    <td className="p-3 text-hean-secondary">
                      {(node.cpu_usage * 100).toFixed(1)}%
                    </td>
                    <td className="p-3 text-hean-secondary">
                      {(node.memory_usage * 100).toFixed(1)}%
                    </td>
                    <td className="p-3 text-hean-secondary font-mono">
                      {node.network_latency_ms === Infinity
                        ? '∞'
                        : `${node.network_latency_ms.toFixed(2)}ms`}
                    </td>
                    <td className="p-3">
                      <div className="flex flex-col gap-1">
                        {Object.entries(node.exchange_latencies).map(([ex, lat]) => (
                          <span key={ex} className="text-xs font-mono text-hean-secondary">
                            {ex}: {lat.toFixed(1)}ms
                          </span>
                        ))}
                      </div>
                    </td>
                    <td className="p-3 text-hean-secondary">{node.active_connections}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Node Details Panel */}
      {selectedNode && networkData.nodes[selectedNode] && (
        <div className="mt-6 bg-hean-dark/80 rounded-lg border border-hean-border p-4">
          <h3 className="text-lg font-bold text-hean-primary mb-4">
            Node Details: {selectedNode}
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-sm text-hean-secondary mb-2">Role</div>
              <div className="text-hean-primary font-semibold">
                {networkData.nodes[selectedNode].role}
              </div>
            </div>
            <div>
              <div className="text-sm text-hean-secondary mb-2">Health Status</div>
              <div className="text-hean-primary font-semibold">
                {networkData.nodes[selectedNode].is_alive
                  ? networkData.nodes[selectedNode].is_healthy
                    ? 'Healthy'
                    : 'Unhealthy'
                  : 'Offline'}
              </div>
            </div>
            <div>
              <div className="text-sm text-hean-secondary mb-2">System Resources</div>
              <div className="text-hean-secondary">
                CPU: {(networkData.nodes[selectedNode].cpu_usage * 100).toFixed(1)}% | Memory:{' '}
                {(networkData.nodes[selectedNode].memory_usage * 100).toFixed(1)}%
              </div>
            </div>
            <div>
              <div className="text-sm text-hean-secondary mb-2">Active Connections</div>
              <div className="text-hean-primary font-semibold">
                {networkData.nodes[selectedNode].active_connections}
              </div>
            </div>
            <div className="col-span-2">
              <div className="text-sm text-hean-secondary mb-2">Exchange Latencies</div>
              <div className="flex gap-4 flex-wrap">
                {Object.entries(networkData.nodes[selectedNode].exchange_latencies).map(
                  ([exchange, latency]) => (
                    <div key={exchange} className="text-hean-secondary font-mono">
                      {exchange.toUpperCase()}: {latency.toFixed(2)}ms
                    </div>
                  )
                )}
              </div>
            </div>
            {networkData.execution_count[selectedNode] !== undefined && (
              <div>
                <div className="text-sm text-hean-secondary mb-2">Executions</div>
                <div className="text-hean-primary font-semibold">
                  {networkData.execution_count[selectedNode]}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

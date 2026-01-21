'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/lib/websocket';
import { useDashboardData } from '@/lib/hooks';
import PanicButton from '@/components/PanicButton';
import MarketPulse from '@/components/MarketPulse';
import TradingMetrics from '@/components/TradingMetrics';
import OrderFeed from '@/components/OrderFeed';
import SystemStatus from '@/components/SystemStatus';
import ReconnectingOverlay from '@/components/ReconnectingOverlay';
import NetworkMap from '@/components/NetworkMap';
import LiveArbChains from '@/components/LiveArbChains';
import SwarmVisualization from '@/components/SwarmVisualization';
import IcebergOrders from '@/components/IcebergOrders';

export default function SingularityDashboard() {
  const [isConnected, setIsConnected] = useState(false);
  const { subscribe, unsubscribe, lastMessage } = useWebSocket();
  const { data: dashboardData, error, isLoading } = useDashboardData();

  useEffect(() => {
    // Subscribe to key topics
    subscribe('ticker_btcusdt');
    subscribe('ticker_ethusdt');
    subscribe('signals');
    subscribe('orders');
    subscribe('ai_reasoning');
    subscribe('system_status');
    subscribe('triangular_arb');
    subscribe('arb_cycles');

    return () => {
      unsubscribe('ticker_btcusdt');
      unsubscribe('ticker_ethusdt');
      unsubscribe('signals');
      unsubscribe('orders');
      unsubscribe('ai_reasoning');
      unsubscribe('system_status');
      unsubscribe('triangular_arb');
      unsubscribe('arb_cycles');
    };
  }, [subscribe, unsubscribe]);

  useEffect(() => {
    if (lastMessage) {
      setIsConnected(true);
    }
  }, [lastMessage]);

  return (
    <div className="min-h-screen bg-hean-dark text-hean-primary p-8">
      {/* Header */}
      <header className="mb-8 flex justify-between items-center">
        <div>
          <h1 className="text-4xl font-bold hean-text-glow mb-2">
            HEAN SINGULARITY
          </h1>
          <p className="text-hean-secondary text-sm">
            Cyber-Command Center | Production Trading System
          </p>
        </div>
        <PanicButton />
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-12 gap-6">
        {/* Left Column - Market Pulse */}
        <div className="col-span-12 lg:col-span-8">
          <MarketPulse lastMessage={lastMessage} />
        </div>

        {/* Right Column - Status & Controls */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <SystemStatus 
            isConnected={isConnected}
            dashboardData={dashboardData}
          />
          <TradingMetrics dashboardData={dashboardData} />
          <SwarmVisualization 
            swarmData={dashboardData?.swarm}
            symbol={dashboardData?.current_symbol}
          />
        </div>

        {/* Bottom - Live Arb Chains */}
        <div className="col-span-12">
          <LiveArbChains />
        </div>

        {/* Bottom - Order Feed */}
        <div className="col-span-12">
          <OrderFeed lastMessage={lastMessage} />
        </div>

        {/* Ultra-Low Latency: Iceberg Orders Visualization */}
        <div className="col-span-12 lg:col-span-6">
          <IcebergOrders icebergOrders={dashboardData?.iceberg_orders} />
        </div>

        {/* Phase 19: Network Map */}
        <div className="col-span-12 lg:col-span-6">
          <NetworkMap stats={dashboardData?.network_stats} />
        </div>
      </div>

      {/* Reconnecting Overlay */}
      {!isConnected && <ReconnectingOverlay />}
    </div>
  );
}
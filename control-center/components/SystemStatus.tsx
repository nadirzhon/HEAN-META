'use client';

import { useHealthStatus } from '@/lib/hooks';

interface SystemStatusProps {
  isConnected: boolean;
  dashboardData?: any;
}

export default function SystemStatus({ isConnected, dashboardData }: SystemStatusProps) {
  const { data: healthData } = useHealthStatus();

  const getStatusColor = (status: string | boolean) => {
    if (status === 'healthy' || status === 'running' || status === 'connected' || status === true) {
      return 'text-hean-primary';
    }
    return 'text-hean-danger';
  };

  const getStatusIcon = (status: string | boolean) => {
    if (status === 'healthy' || status === 'running' || status === 'connected' || status === true) {
      return '●';
    }
    return '○';
  };

  return (
    <div className="hean-card">
      <h2 className="text-xl font-bold mb-4 hean-text-glow">System Status</h2>
      
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <span className="text-hean-secondary">WebSocket</span>
          <span className={`font-bold ${getStatusColor(isConnected)}`}>
            {getStatusIcon(isConnected)} {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>

        {healthData?.components && (
          <>
            <div className="flex justify-between items-center">
              <span className="text-hean-secondary">API</span>
              <span className={`font-bold ${getStatusColor(healthData.components.api)}`}>
                {getStatusIcon(healthData.components.api)} {healthData.components.api}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-hean-secondary">Event Bus</span>
              <span className={`font-bold ${getStatusColor(healthData.components.event_bus)}`}>
                {getStatusIcon(healthData.components.event_bus)} {healthData.components.event_bus}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-hean-secondary">Redis</span>
              <span className={`font-bold ${getStatusColor(healthData.components.redis === 'connected')}`}>
                {getStatusIcon(healthData.components.redis === 'connected')} {healthData.components.redis}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-hean-secondary">Engine</span>
              <span className={`font-bold ${getStatusColor(healthData.components.engine === 'running')}`}>
                {getStatusIcon(healthData.components.engine === 'running')} {healthData.components.engine}
              </span>
            </div>
          </>
        )}

        <div className="mt-4 pt-4 border-t border-hean-primary border-opacity-20">
          <div className="text-xs text-hean-secondary">
            Last update: {healthData?.timestamp ? new Date(healthData.timestamp).toLocaleTimeString() : '—'}
          </div>
        </div>
      </div>
    </div>
  );
}
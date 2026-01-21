'use client';

interface TradingMetricsProps {
  dashboardData?: any;
}

export default function TradingMetrics({ dashboardData }: TradingMetricsProps) {
  const metrics = dashboardData?.metrics || {};

  const formatCurrency = (value: number | undefined) => {
    if (value === undefined) return '—';
    return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  const formatPercent = (value: number | undefined) => {
    if (value === undefined) return '—';
    return `${value > 0 ? '+' : ''}${value.toFixed(2)}%`;
  };

  return (
    <div className="hean-card">
      <h2 className="text-xl font-bold mb-4 hean-text-glow">Trading Metrics</h2>
      
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <span className="text-hean-secondary">Equity</span>
          <span className="text-xl font-bold text-hean-primary">
            {formatCurrency(metrics.equity)}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-hean-secondary">P&L (24h)</span>
          <span className={`text-xl font-bold ${
            (metrics.daily_pnl || 0) >= 0 ? 'text-hean-primary' : 'text-hean-danger'
          }`}>
            {formatCurrency(metrics.daily_pnl)}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-hean-secondary">Return %</span>
          <span className={`text-xl font-bold ${
            (metrics.return_pct || 0) >= 0 ? 'text-hean-primary' : 'text-hean-danger'
          }`}>
            {formatPercent(metrics.return_pct)}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-hean-secondary">Open Positions</span>
          <span className="text-xl font-bold text-hean-secondary">
            {metrics.open_positions || 0}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-hean-secondary">Trades (24h)</span>
          <span className="text-xl font-bold text-hean-secondary">
            {metrics.daily_trades || 0}
          </span>
        </div>

        <div className="flex justify-between items-center">
          <span className="text-hean-secondary">Win Rate</span>
          <span className="text-xl font-bold text-hean-primary">
            {metrics.win_rate ? `${(metrics.win_rate * 100).toFixed(1)}%` : '—'}
          </span>
        </div>

        {/* Ultra-Low Latency Metrics */}
        <div className="mt-6 pt-4 border-t border-hean-border">
          <h3 className="text-sm font-semibold text-hean-secondary mb-3">Ultra-Low Latency Precision</h3>
          
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-sm text-hean-secondary">Microsecond Jitter</span>
              <span className={`text-lg font-bold ${
                (metrics.microsecond_jitter || 0) < 50 ? 'text-green-400' : 
                (metrics.microsecond_jitter || 0) < 100 ? 'text-yellow-400' : 'text-hean-danger'
              }`}>
                {metrics.microsecond_jitter !== undefined ? `${metrics.microsecond_jitter.toFixed(2)} μs` : '—'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-hean-secondary">Order-Fill Probability</span>
              <span className={`text-lg font-bold ${
                (metrics.order_fill_probability || 0) > 0.8 ? 'text-green-400' : 
                (metrics.order_fill_probability || 0) > 0.5 ? 'text-yellow-400' : 'text-hean-danger'
              }`}>
                {metrics.order_fill_probability !== undefined ? `${(metrics.order_fill_probability * 100).toFixed(1)}%` : '—'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-hean-secondary">Avg Latency (ns)</span>
              <span className={`text-lg font-bold ${
                (metrics.avg_latency_ns || 0) < 50000 ? 'text-green-400' : 
                (metrics.avg_latency_ns || 0) < 100000 ? 'text-yellow-400' : 'text-hean-danger'
              }`}>
                {metrics.avg_latency_ns !== undefined ? `${(metrics.avg_latency_ns / 1000).toFixed(2)} μs` : '—'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-hean-secondary">VPIN (Informed Trading)</span>
              <span className={`text-lg font-bold ${
                (metrics.vpin || 0) > 0.7 ? 'text-hean-danger' : 
                (metrics.vpin || 0) > 0.5 ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {metrics.vpin !== undefined ? metrics.vpin.toFixed(3) : '—'}
              </span>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm text-hean-secondary">Spoofing Detections</span>
              <span className={`text-lg font-bold ${
                (metrics.spoofing_count || 0) > 0 ? 'text-hean-danger' : 'text-green-400'
              }`}>
                {metrics.spoofing_count !== undefined ? metrics.spoofing_count : 0}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
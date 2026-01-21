'use client';

interface IcebergOrder {
  symbol: string;
  price: number;
  suspectedSize: number;
  side: 'bid' | 'ask';
  detectionTime: number;
}

interface IcebergOrdersProps {
  icebergOrders?: IcebergOrder[];
}

export default function IcebergOrders({ icebergOrders = [] }: IcebergOrdersProps) {
  const formatPrice = (price: number) => {
    return price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  const formatSize = (size: number) => {
    if (size >= 1000000) {
      return `${(size / 1000000).toFixed(2)}M`;
    } else if (size >= 1000) {
      return `${(size / 1000).toFixed(2)}K`;
    }
    return size.toFixed(2);
  };

  const getTimeAgo = (timestamp: number) => {
    const secondsAgo = Math.floor((Date.now() - timestamp) / 1000);
    if (secondsAgo < 60) {
      return `${secondsAgo}s ago`;
    } else if (secondsAgo < 3600) {
      return `${Math.floor(secondsAgo / 60)}m ago`;
    }
    return `${Math.floor(secondsAgo / 3600)}h ago`;
  };

  return (
    <div className="hean-card">
      <h2 className="text-xl font-bold mb-4 hean-text-glow">Iceberg Orders Detected</h2>
      
      {icebergOrders.length === 0 ? (
        <div className="text-center py-8 text-hean-secondary">
          <p>No iceberg orders detected</p>
          <p className="text-sm mt-2">Large hidden orders will appear here in real-time</p>
        </div>
      ) : (
        <div className="space-y-3">
          {icebergOrders.slice(0, 10).map((order, index) => (
            <div
              key={index}
              className="p-3 bg-hean-surface border border-hean-border rounded-lg hover:border-hean-primary transition-colors"
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-hean-primary">{order.symbol}</span>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                    order.side === 'bid' 
                      ? 'bg-green-500/20 text-green-400' 
                      : 'bg-red-500/20 text-red-400'
                  }`}>
                    {order.side.toUpperCase()}
                  </span>
                </div>
                <span className="text-xs text-hean-secondary">{getTimeAgo(order.detectionTime)}</span>
              </div>
              
              <div className="grid grid-cols-2 gap-3 mt-2">
                <div>
                  <span className="text-xs text-hean-secondary block mb-1">Price</span>
                  <span className="text-sm font-mono text-hean-primary">
                    {formatPrice(order.price)}
                  </span>
                </div>
                <div>
                  <span className="text-xs text-hean-secondary block mb-1">Suspected Size</span>
                  <span className="text-sm font-mono text-hean-primary">
                    {formatSize(order.suspectedSize)}
                  </span>
                </div>
              </div>
              
              {/* Visual indicator bar */}
              <div className="mt-3 h-2 bg-hean-surface-dark rounded-full overflow-hidden">
                <div
                  className={`h-full transition-all ${
                    order.side === 'bid' ? 'bg-green-500/50' : 'bg-red-500/50'
                  }`}
                  style={{
                    width: `${Math.min(100, (order.suspectedSize / 100000) * 100)}%`
                  }}
                />
              </div>
            </div>
          ))}
          
          {icebergOrders.length > 10 && (
            <div className="text-center pt-2 text-sm text-hean-secondary">
              +{icebergOrders.length - 10} more detected
            </div>
          )}
        </div>
      )}
    </div>
  );
}

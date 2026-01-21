'use client';

import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface MarketData {
  timestamp: number;
  price: number;
  volume?: number;
}

interface MarketPulseProps {
  lastMessage: any;
}

export default function MarketPulse({ lastMessage }: MarketPulseProps) {
  const [btcData, setBtcData] = useState<MarketData[]>([]);
  const [ethData, setEthData] = useState<MarketData[]>([]);
  const [currentPrice, setCurrentPrice] = useState<{ btc?: number; eth?: number }>({});

  useEffect(() => {
    if (lastMessage?.topic?.startsWith('ticker_')) {
      const symbol = lastMessage.topic.replace('ticker_', '').toUpperCase();
      const price = lastMessage.data?.price;
      const volume = lastMessage.data?.volume;

      if (price) {
        const timestamp = Date.now();
        const newData: MarketData = { timestamp, price, volume };

        if (symbol.includes('BTC')) {
          setBtcData((prev) => [...prev.slice(-100), newData]);
          setCurrentPrice((prev) => ({ ...prev, btc: price }));
        } else if (symbol.includes('ETH')) {
          setEthData((prev) => [...prev.slice(-100), newData]);
          setCurrentPrice((prev) => ({ ...prev, eth: price }));
        }
      }
    }
  }, [lastMessage]);

  const formatPrice = (price: number | undefined) => {
    if (!price) return '—';
    return `$${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
  };

  return (
    <div className="hean-card">
      <h2 className="text-2xl font-bold mb-6 hean-text-glow">Market Pulse</h2>
      
      <div className="grid grid-cols-2 gap-6 mb-6">
        {/* BTC */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-hean-secondary">BTC/USDT</span>
            <span className="text-2xl font-bold hean-text-glow live-pulse">
              {formatPrice(currentPrice.btc)}
            </span>
          </div>
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={btcData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 255, 136, 0.1)" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(val) => new Date(val).toLocaleTimeString()}
                stroke="rgba(0, 255, 136, 0.5)"
              />
              <YAxis stroke="rgba(0, 255, 136, 0.5)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0a0e27', 
                  border: '1px solid rgba(0, 255, 136, 0.3)',
                  color: '#00ff88'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="#00ff88" 
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* ETH */}
        <div className="space-y-2">
          <div className="flex justify-between items-center">
            <span className="text-hean-secondary">ETH/USDT</span>
            <span className="text-2xl font-bold hean-text-glow-secondary live-pulse">
              {formatPrice(currentPrice.eth)}
            </span>
          </div>
          <ResponsiveContainer width="100%" height={150}>
            <LineChart data={ethData}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(0, 212, 255, 0.1)" />
              <XAxis 
                dataKey="timestamp" 
                tickFormatter={(val) => new Date(val).toLocaleTimeString()}
                stroke="rgba(0, 212, 255, 0.5)"
              />
              <YAxis stroke="rgba(0, 212, 255, 0.5)" />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#0a0e27', 
                  border: '1px solid rgba(0, 212, 255, 0.3)',
                  color: '#00d4ff'
                }}
              />
              <Line 
                type="monotone" 
                dataKey="price" 
                stroke="#00d4ff" 
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="text-xs text-hean-secondary">
        Real-time market data streaming from C++ core
      </div>
    </div>
  );
}
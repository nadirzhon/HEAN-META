'use client';

import { useEffect, useState } from 'react';

interface Order {
  id: string;
  type: string;
  symbol?: string;
  side?: string;
  quantity?: number;
  price?: number;
  timestamp: string;
}

interface OrderFeedProps {
  lastMessage: any;
}

export default function OrderFeed({ lastMessage }: OrderFeedProps) {
  const [orders, setOrders] = useState<Order[]>([]);
  const [signals, setSignals] = useState<any[]>([]);
  const [aiReasoning, setAiReasoning] = useState<any[]>([]);

  useEffect(() => {
    if (lastMessage?.topic === 'orders' && lastMessage?.data) {
      const order: Order = {
        id: lastMessage.data.order_id || Date.now().toString(),
        type: lastMessage.data.type || 'order',
        symbol: lastMessage.data.symbol,
        side: lastMessage.data.side,
        quantity: lastMessage.data.quantity,
        price: lastMessage.data.price,
        timestamp: lastMessage.timestamp,
      };
      setOrders((prev) => [order, ...prev].slice(0, 50)); // Keep last 50
    }

    if (lastMessage?.topic === 'signals' && lastMessage?.data) {
      const signal = {
        ...lastMessage.data,
        timestamp: lastMessage.timestamp,
      };
      setSignals((prev) => [signal, ...prev].slice(0, 50));
    }

    if (lastMessage?.topic === 'ai_reasoning' && lastMessage?.data) {
      const reasoning = {
        ...lastMessage.data,
        timestamp: lastMessage.timestamp,
      };
      setAiReasoning((prev) => [reasoning, ...prev].slice(0, 20));
    }
  }, [lastMessage]);

  const formatTime = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch {
      return '—';
    }
  };

  return (
    <div className="hean-card">
      <h2 className="text-2xl font-bold mb-6 hean-text-glow">Live Activity Feed</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Orders */}
        <div>
          <h3 className="text-lg font-semibold mb-3 text-hean-secondary">Orders</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {orders.length === 0 ? (
              <p className="text-sm text-hean-secondary">No orders yet</p>
            ) : (
              orders.map((order) => (
                <div
                  key={order.id}
                  className="hean-glass rounded-lg p-3 text-sm border border-hean-primary border-opacity-20"
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-bold">{order.symbol || '—'}</span>
                    <span className="text-xs text-hean-secondary">{formatTime(order.timestamp)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className={`${order.side === 'buy' ? 'text-hean-primary' : 'text-hean-danger'}`}>
                      {order.side?.toUpperCase()}
                    </span>
                    <span>{order.quantity} @ ${order.price?.toFixed(2)}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Signals */}
        <div>
          <h3 className="text-lg font-semibold mb-3 text-hean-secondary">Signals</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {signals.length === 0 ? (
              <p className="text-sm text-hean-secondary">No signals yet</p>
            ) : (
              signals.map((signal, idx) => (
                <div
                  key={idx}
                  className="hean-glass rounded-lg p-3 text-sm border border-hean-secondary border-opacity-20"
                >
                  <div className="flex justify-between items-start mb-1">
                    <span className="font-bold">{signal.symbol || '—'}</span>
                    <span className="text-xs text-hean-secondary">{formatTime(signal.timestamp)}</span>
                  </div>
                  <div className="text-xs text-hean-secondary">
                    {signal.strategy_id || 'Unknown strategy'}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* AI Reasoning */}
        <div>
          <h3 className="text-lg font-semibold mb-3 text-hean-secondary">AI Reasoning</h3>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {aiReasoning.length === 0 ? (
              <p className="text-sm text-hean-secondary">No AI reasoning yet</p>
            ) : (
              aiReasoning.map((reasoning, idx) => (
                <div
                  key={idx}
                  className="hean-glass rounded-lg p-3 text-sm border border-hean-accent border-opacity-20"
                >
                  <div className="text-xs text-hean-secondary mb-1">
                    {formatTime(reasoning.timestamp)}
                  </div>
                  <div className="text-xs text-hean-primary line-clamp-3">
                    {JSON.stringify(reasoning.data || reasoning, null, 2).substring(0, 150)}...
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
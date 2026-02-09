#!/usr/bin/env python3
"""
Strategy Orchestrator - High-level trading logic in Python
Connects all components: Rust services, C++ indicators, Go API
"""

import asyncio
import numpy as np
from typing import Dict, List
from dataclasses import dataclass
from datetime import datetime
import zmq.asyncio
import orjson
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Signal:
    symbol: str
    side: str  # BUY/SELL
    strength: float  # 0.0 - 1.0
    reason: str
    timestamp: float

@dataclass
class Order:
    order_id: int
    symbol_id: int
    side: int  # 0=BUY, 1=SELL
    quantity: float
    price: float
    timestamp_ns: int

class StrategyOrchestrator:
    """
    Orchestrates trading strategies and sends orders to Rust order router.
    Python is perfect for this layer:
    - Latency not critical (10-50ms is fine for decisions)
    - Complex business logic
    - ML inference
    - Portfolio optimization
    """

    def __init__(self):
        # ZeroMQ connection to Rust order router
        self.context = zmq.asyncio.Context()
        self.order_socket = self.context.socket(zmq.PUSH)
        self.order_socket.connect("tcp://127.0.0.1:5555")

        # State
        self.positions: Dict[str, float] = {}
        self.signals_history: List[Signal] = []
        self.order_counter = 0

        # Symbol mapping (must match Rust)
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT",
            "DOGEUSDT", "MATICUSDT", "DOTUSDT", "AVAXUSDT", "ATOMUSDT"
        ]

        logger.info("🎯 Strategy Orchestrator initialized")

    def symbol_to_id(self, symbol: str) -> int:
        try:
            return self.symbols.index(symbol)
        except ValueError:
            return 0

    async def run(self):
        """Main event loop"""
        logger.info("🚀 Starting Strategy Orchestrator...")

        while True:
            try:
                # 1. Get market data (mock for now)
                market_data = await self.get_market_data()

                # 2. Generate signals
                signals = await self.generate_signals(market_data)

                # 3. Send orders for signals
                for signal in signals:
                    await self.send_order(signal)

                # Sleep for 1 second (strategy cycle)
                await asyncio.sleep(1.0)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(5.0)

    async def get_market_data(self) -> Dict:
        """Get current market data (mock)"""
        # In production: fetch from Market Data Processor
        return {
            'BTCUSDT': {
                'price': 45000.0,
                'volume': 1000.0,
                'prices': np.random.randn(100) * 1000 + 45000
            }
        }

    async def generate_signals(self, market_data: Dict) -> List[Signal]:
        """
        Generate trading signals from strategies.
        This is where Python shines - easy to experiment with strategies.
        """
        signals = []

        for symbol, data in market_data.items():
            # Strategy 1: Simple momentum
            current_price = data['price']

            # Generate random signal for demo
            if np.random.rand() > 0.95:  # 5% chance
                side = 'BUY' if np.random.rand() > 0.5 else 'SELL'
                signals.append(Signal(
                    symbol=symbol,
                    side=side,
                    strength=0.7,
                    reason='Momentum signal',
                    timestamp=datetime.now().timestamp()
                ))

                logger.info(f"📊 Signal: {side} {symbol} @ {current_price}")

        return signals

    async def send_order(self, signal: Signal):
        """Send order to Rust order router via ZeroMQ"""
        try:
            self.order_counter += 1

            order = Order(
                order_id=self.order_counter,
                symbol_id=self.symbol_to_id(signal.symbol),
                side=0 if signal.side == 'BUY' else 1,
                quantity=0.1,  # Fixed size for demo
                price=45000.0,  # Mock price
                timestamp_ns=int(datetime.now().timestamp() * 1e9)
            )

            # Serialize order (bincode-compatible format)
            import struct
            msg = struct.pack('<QQHQDD',
                order.order_id,
                order.timestamp_ns,
                order.symbol_id,
                order.side,
                order.quantity,
                order.price
            )

            await self.order_socket.send(msg)
            logger.info(f"✅ Order sent: {signal.side} {signal.symbol}")

        except Exception as e:
            logger.error(f"Failed to send order: {e}")

async def main():
    orchestrator = StrategyOrchestrator()
    await orchestrator.run()

if __name__ == '__main__':
    asyncio.run(main())

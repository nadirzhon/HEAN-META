"""
bybit_tensorized.py

Tensorized Orderflow System for monitoring ALL 200+ Bybit Perpetual pairs
simultaneously as a single multi-dimensional matrix.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np
import websockets
from websockets.client import WebSocketClientProtocol

from ..core.python.causal_brain import CausalBrain, MarketState

logger = logging.getLogger(__name__)


class ProxyNode:
    """Represents a proxy node for self-healing routing"""
    def __init__(self, name: str, endpoint: str, region: str):
        self.name = name
        self.endpoint = endpoint
        self.region = region
        self.latency_ms = float('inf')
        self.last_check = 0.0
        self.failure_count = 0
        self.success_rate = 1.0
        self.is_active = True

    def update_latency(self, latency_ms: float):
        """Update latency measurement"""
        self.latency_ms = latency_ms
        self.last_check = time.time()
        if latency_ms < 5000:  # Reasonable threshold
            self.failure_count = 0
            self.success_rate = min(1.0, self.success_rate + 0.01)
        else:
            self.failure_count += 1
            self.success_rate = max(0.0, self.success_rate - 0.1)

    def get_score(self) -> float:
        """Get proxy score for routing selection"""
        if not self.is_active:
            return -1.0
        return self.success_rate / (self.latency_ms + 1.0)


@dataclass
class OrderflowTensor:
    """Multi-dimensional tensor representation of exchange orderflow"""
    # Shape: [num_symbols, time_window, features]
    # Features: [price, volume, bid_volume, ask_volume, spread, imbalance, ...]
    data: np.ndarray
    symbols: list[str]
    timestamps: np.ndarray
    feature_names: list[str] = field(default_factory=lambda: [
        'price', 'volume', 'bid_volume', 'ask_volume',
        'spread', 'imbalance', 'volatility', 'momentum'
    ])

    def get_symbol_slice(self, symbol: str) -> np.ndarray:
        """Get orderflow tensor for a specific symbol"""
        if symbol not in self.symbols:
            return None
        idx = self.symbols.index(symbol)
        return self.data[idx, :, :]

    def get_feature_matrix(self, feature: str) -> np.ndarray:
        """Get feature matrix across all symbols"""
        if feature not in self.feature_names:
            return None
        fidx = self.feature_names.index(feature)
        return self.data[:, :, fidx]

    def get_correlation_matrix(self) -> np.ndarray:
        """Compute correlation matrix across all symbols"""
        # Use price returns for correlation
        price_idx = self.feature_names.index('price')
        prices = self.data[:, :, price_idx]

        # Compute returns
        returns = np.diff(prices, axis=1)

        # Compute correlation
        returns_std = np.std(returns, axis=1, keepdims=True)
        returns_normalized = returns / (returns_std + 1e-8)

        correlation = np.corrcoef(returns_normalized)
        return correlation


class BybitTensorizedMonitor:
    """
    Monitor all Bybit perpetual pairs simultaneously using tensorized orderflow.
    """

    def __init__(self, api_key: str | None = None,
                 api_secret: str | None = None,
                 causal_brain: CausalBrain | None = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.causal_brain = causal_brain

        # Proxy nodes for self-healing routing
        self.proxy_nodes: list[ProxyNode] = []
        self.current_proxy: ProxyNode | None = None
        self.proxy_rotation_interval = 30.0  # seconds

        # Symbol management
        self.all_symbols: list[str] = []
        self.active_websockets: dict[str, WebSocketClientProtocol] = {}

        # Tensor storage
        self.tensor_window_size = 1000  # Last 1000 updates per symbol
        self.orderflow_tensor: OrderflowTensor | None = None

        # Market data storage
        self.latest_prices: dict[str, float] = {}
        self.latest_spreads: dict[str, float] = {}
        self.latest_volumes: dict[str, float] = {}
        self.latest_volatilities: dict[str, float] = {}
        self.orderbook_snapshots: dict[str, dict] = {}

        # Connection state
        self.running = False
        self.monitor_task: asyncio.Task | None = None
        self.proxy_check_task: asyncio.Task | None = None

        # Statistics
        self.updates_per_second = 0.0
        self.total_updates = 0
        self.last_update_time = time.time()

    def add_proxy_node(self, name: str, endpoint: str, region: str):
        """Add a proxy node for routing"""
        proxy = ProxyNode(name, endpoint, region)
        self.proxy_nodes.append(proxy)
        if not self.current_proxy:
            self.current_proxy = proxy

    def _select_fastest_proxy(self) -> ProxyNode | None:
        """Select the fastest available proxy node"""
        if not self.proxy_nodes:
            return None

        # Filter active proxies
        active_proxies = [p for p in self.proxy_nodes if p.is_active]
        if not active_proxies:
            # Reset all proxies if none are active
            for p in self.proxy_nodes:
                p.is_active = True
            active_proxies = self.proxy_nodes

        # Select by score
        best_proxy = max(active_proxies, key=lambda p: p.get_score())
        return best_proxy

    async def _check_proxy_latency(self, proxy: ProxyNode) -> float:
        """Check latency to a proxy node"""
        try:
            start = time.time()
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{proxy.endpoint}/v5/public/time",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 200:
                        latency = (time.time() - start) * 1000
                        proxy.update_latency(latency)
                        return latency
        except Exception as e:
            logger.warning(f"Proxy {proxy.name} latency check failed: {e}")
            proxy.update_latency(float('inf'))
            proxy.is_active = False

        return float('inf')

    async def _proxy_health_monitor(self):
        """Continuously monitor proxy health and switch if needed"""
        while self.running:
            try:
                # Check all proxies
                tasks = [self._check_proxy_latency(p) for p in self.proxy_nodes]
                await asyncio.gather(*tasks, return_exceptions=True)

                # Select fastest
                new_proxy = self._select_fastest_proxy()
                if new_proxy and new_proxy != self.current_proxy:
                    logger.info(f"Switching to fastest proxy: {new_proxy.name} "
                              f"(latency: {new_proxy.latency_ms:.2f}ms)")
                    self.current_proxy = new_proxy

                await asyncio.sleep(self.proxy_rotation_interval)
            except Exception as e:
                logger.error(f"Proxy health monitor error: {e}")
                await asyncio.sleep(5)

    async def _fetch_all_perpetual_symbols(self) -> list[str]:
        """Fetch all perpetual trading pairs from Bybit"""
        endpoint = "https://api.bybit.com/v5/market/instruments-info"
        params = {"category": "linear"}

        try:
            url = endpoint
            if self.current_proxy and self.current_proxy.endpoint.startswith("http"):
                url = f"{self.current_proxy.endpoint}{endpoint}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("retCode") == 0:
                            symbols = [
                                item["symbol"]
                                for item in data.get("result", {}).get("list", [])
                                if item.get("status") == "Trading"
                            ]
                            return symbols
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")

        return []

    async def _initialize_tensor(self, symbols: list[str]):
        """Initialize the orderflow tensor"""
        num_symbols = len(symbols)
        num_features = 8  # price, volume, bid_volume, ask_volume, spread, imbalance, volatility, momentum

        self.orderflow_tensor = OrderflowTensor(
            data=np.zeros((num_symbols, self.tensor_window_size, num_features)),
            symbols=symbols,
            timestamps=np.zeros(self.tensor_window_size),
            feature_names=[
                'price', 'volume', 'bid_volume', 'ask_volume',
                'spread', 'imbalance', 'volatility', 'momentum'
            ]
        )

    def _update_tensor(self, symbol: str, update: dict):
        """Update the orderflow tensor with new data"""
        if not self.orderflow_tensor or symbol not in self.orderflow_tensor.symbols:
            return

        symbol_idx = self.orderflow_tensor.symbols.index(symbol)

        # Rotate tensor (FIFO)
        self.orderflow_tensor.data[symbol_idx, :-1, :] = self.orderflow_tensor.data[symbol_idx, 1:, :]
        self.orderflow_tensor.timestamps[:-1] = self.orderflow_tensor.timestamps[1:]

        # Insert new data at the end
        timestamp = time.time()
        price = float(update.get("price", 0.0))
        volume = float(update.get("volume", 0.0))
        bid_volume = float(update.get("bid_volume", 0.0))
        ask_volume = float(update.get("ask_volume", 0.0))
        spread = float(update.get("spread", 0.0))
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)

        # Compute volatility (rolling)
        if price > 0:
            price_history = self.orderflow_tensor.data[symbol_idx, -100:, 0]
            returns = np.diff(price_history[price_history > 0])
            volatility = np.std(returns) if len(returns) > 1 else 0.0

            # Momentum (price change rate)
            if len(price_history) > 10:
                momentum = (price - np.mean(price_history[-10:])) / np.mean(price_history[-10:])
            else:
                momentum = 0.0
        else:
            volatility = 0.0
            momentum = 0.0

        self.orderflow_tensor.data[symbol_idx, -1, :] = [
            price, volume, bid_volume, ask_volume,
            spread, imbalance, volatility, momentum
        ]
        self.orderflow_tensor.timestamps[-1] = timestamp

        # Update latest values
        self.latest_prices[symbol] = price
        self.latest_spreads[symbol] = spread
        self.latest_volumes[symbol] = volume
        self.latest_volatilities[symbol] = volatility

    async def _handle_orderbook_update(self, symbol: str, data: dict):
        """Handle orderbook update from WebSocket"""
        try:
            if "data" in data:
                orderbook = data["data"]
                bids = orderbook.get("b", [])
                asks = orderbook.get("a", [])

                if bids and asks:
                    best_bid = float(bids[0][0])
                    best_ask = float(asks[0][0])
                    mid_price = (best_bid + best_ask) / 2.0
                    spread = best_ask - best_bid

                    bid_volume = sum(float(b[1]) for b in bids[:5])
                    ask_volume = sum(float(a[1]) for a in asks[:5])
                    total_volume = bid_volume + ask_volume

                    update = {
                        "price": mid_price,
                        "volume": total_volume,
                        "bid_volume": bid_volume,
                        "ask_volume": ask_volume,
                        "spread": spread
                    }

                    self._update_tensor(symbol, update)
                    self.orderbook_snapshots[symbol] = {
                        "bids": bids[:10],
                        "asks": asks[:10],
                        "timestamp": time.time()
                    }
        except Exception as e:
            logger.error(f"Error handling orderbook update for {symbol}: {e}")

    async def _handle_trade_update(self, symbol: str, data: dict):
        """Handle trade update from WebSocket"""
        try:
            if "data" in data:
                trades = data["data"]
                if trades:
                    latest_trade = trades[0]
                    price = float(latest_trade.get("p", 0.0))
                    volume = float(latest_trade.get("v", 0.0))
                    latest_trade.get("S", "")

                    # Get orderbook snapshot for context
                    orderbook = self.orderbook_snapshots.get(symbol, {})
                    bids = orderbook.get("bids", [])
                    asks = orderbook.get("asks", [])

                    bid_volume = sum(float(b[1]) for b in bids[:5]) if bids else 0.0
                    ask_volume = sum(float(a[1]) for a in asks[:5]) if asks else 0.0
                    spread = float(asks[0][0]) - float(bids[0][0]) if bids and asks else 0.0

                    update = {
                        "price": price,
                        "volume": volume,
                        "bid_volume": bid_volume,
                        "ask_volume": ask_volume,
                        "spread": spread
                    }

                    self._update_tensor(symbol, update)
                    self.latest_prices[symbol] = price
        except Exception as e:
            logger.error(f"Error handling trade update for {symbol}: {e}")

    async def _websocket_listener(self, symbol: str, stream_type: str):
        """Listen to WebSocket stream for a symbol"""
        ws_url = "wss://stream.bybit.com/v5/public/linear"

        # Use proxy if available
        if self.current_proxy and hasattr(self.current_proxy, 'ws_endpoint'):
            ws_url = self.current_proxy.ws_endpoint

        topic = f"orderbook.50.{symbol}" if stream_type == "orderbook" else f"publicTrade.{symbol}"

        retry_count = 0
        max_retries = 5

        while self.running and retry_count < max_retries:
            try:
                async with websockets.connect(ws_url) as ws:
                    # Subscribe to topic
                    subscribe_msg = {
                        "op": "subscribe",
                        "args": [topic]
                    }
                    await ws.send(json.dumps(subscribe_msg))

                    self.active_websockets[symbol] = ws

                    async for message in ws:
                        if not self.running:
                            break

                        data = json.loads(message)

                        if stream_type == "orderbook":
                            await self._handle_orderbook_update(symbol, data)
                        else:
                            await self._handle_trade_update(symbol, data)

                        self.total_updates += 1

                        # Update throughput stats
                        now = time.time()
                        elapsed = now - self.last_update_time
                        if elapsed >= 1.0:
                            self.updates_per_second = self.total_updates / elapsed
                            self.total_updates = 0
                            self.last_update_time = now

                            # Send market state to causal brain
                            if self.causal_brain:
                                await self._send_market_state_to_brain()

                self.active_websockets.pop(symbol, None)
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                retry_count += 1
                await asyncio.sleep(min(2 ** retry_count, 30))

                # Switch proxy if connection fails
                if retry_count >= 3:
                    new_proxy = self._select_fastest_proxy()
                    if new_proxy and new_proxy != self.current_proxy:
                        self.current_proxy = new_proxy

    async def _send_market_state_to_brain(self):
        """Send current market state to CausalBrain"""
        if not self.causal_brain or not self.orderflow_tensor:
            return

        # Compute correlations from tensor
        correlation_matrix = self.orderflow_tensor.get_correlation_matrix()
        correlations = {}

        for i, sym1 in enumerate(self.orderflow_tensor.symbols):
            for j, sym2 in enumerate(self.orderflow_tensor.symbols):
                if i < j:
                    correlations[(sym1, sym2)] = float(correlation_matrix[i, j])

        # Create market state
        state = MarketState(
            timestamp=time.time(),
            symbols=list(self.latest_prices.keys()),
            prices=self.latest_prices.copy(),
            volumes=self.latest_volumes.copy(),
            spreads=self.latest_spreads.copy(),
            volatilities=self.latest_volatilities.copy(),
            correlations=correlations
        )

        self.causal_brain.update_market_state(state)

    async def start(self):
        """Start monitoring all perpetual pairs"""
        if self.running:
            return

        self.running = True

        # Fetch all symbols
        logger.info("Fetching all Bybit perpetual symbols...")
        self.all_symbols = await self._fetch_all_perpetual_symbols()
        logger.info(f"Found {len(self.all_symbols)} perpetual pairs")

        # Initialize tensor
        await self._initialize_tensor(self.all_symbols)

        # Start proxy health monitor
        if self.proxy_nodes:
            self.proxy_check_task = asyncio.create_task(self._proxy_health_monitor())

        # Start WebSocket listeners for each symbol
        # Batch connections to avoid overwhelming
        batch_size = 20
        for i in range(0, len(self.all_symbols), batch_size):
            batch = self.all_symbols[i:i+batch_size]
            for symbol in batch:
                # Listen to both orderbook and trades
                asyncio.create_task(self._websocket_listener(symbol, "orderbook"))
                asyncio.create_task(self._websocket_listener(symbol, "trades"))

            # Small delay between batches
            await asyncio.sleep(0.5)

    async def stop(self):
        """Stop monitoring"""
        self.running = False

        # Close all WebSocket connections
        for ws in self.active_websockets.values():
            await ws.close()
        self.active_websockets.clear()

        # Cancel tasks
        if self.proxy_check_task:
            self.proxy_check_task.cancel()

    def get_tensor(self) -> OrderflowTensor | None:
        """Get current orderflow tensor"""
        return self.orderflow_tensor

    def get_stats(self) -> dict:
        """Get monitoring statistics"""
        return {
            "num_symbols": len(self.all_symbols),
            "active_connections": len(self.active_websockets),
            "updates_per_second": self.updates_per_second,
            "current_proxy": self.current_proxy.name if self.current_proxy else "direct",
            "proxy_latency_ms": self.current_proxy.latency_ms if self.current_proxy else 0.0
        }

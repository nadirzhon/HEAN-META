"""Python wrapper for C++ Triangular Arbitrage Scanner."""

import time
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TriangularCycle:
    """Triangular arbitrage cycle detection result."""
    
    pair_a: str      # First pair (A->B)
    pair_b: str      # Second pair (B->C)
    pair_c: str      # Third pair (C->A)
    asset_a: str     # Starting asset
    asset_b: str     # Intermediate asset
    asset_c: str     # Final asset
    profit_ratio: float  # (Price_AB * Price_BC * Price_CA) - 1
    profit_bps: float    # Profit in basis points
    max_size: float      # Maximum tradeable size (limited by liquidity)
    detection_time_ns: int  # Detection timestamp


class TriangularScanner:
    """High-Frequency Triangular Arbitrage Scanner.
    
    Monitors 50+ trading pairs simultaneously and detects triangular arbitrage
    opportunities with ultra-low latency (< 500 microseconds).
    """
    
    def __init__(
        self,
        bus: EventBus,
        fee_buffer: float = 0.001,  # 0.1% fee buffer
        min_profit_bps: float = 5.0  # Minimum profit in basis points
    ) -> None:
        """Initialize the triangular scanner.
        
        Args:
            bus: Event bus for publishing signals
            fee_buffer: Fee buffer as ratio (e.g., 0.001 = 0.1%)
            min_profit_bps: Minimum profit in basis points (default 5 bps = 0.05%)
        """
        self._bus = bus
        self._fee_buffer = fee_buffer
        self._min_profit_bps = min_profit_bps
        self._running = False
        
        # Try to import C++ scanner
        self._cpp_scanner: Any = None
        try:
            import graph_engine_py  # type: ignore
            if hasattr(graph_engine_py, 'TriangularScanner'):
                self._cpp_scanner = graph_engine_py.TriangularScanner(fee_buffer, min_profit_bps)
                logger.info("C++ TriangularScanner initialized successfully")
            else:
                logger.warning("C++ TriangularScanner not available in graph_engine_py")
        except ImportError:
            logger.warning("graph_engine_py not available. Using Python fallback implementation.")
        
        # Price cache for pairs (bid/ask with timestamps)
        self._price_cache: dict[str, dict[str, Any]] = {}
        
        # Active cycles tracking
        self._active_cycles: dict[tuple[str, str, str], TriangularCycle] = {}
        
    async def start(self) -> None:
        """Start the triangular scanner."""
        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._bus.subscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook)
        logger.info("Triangular Arbitrage Scanner started")
    
    async def stop(self) -> None:
        """Stop the triangular scanner."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._bus.unsubscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook)
        logger.info("Triangular Arbitrage Scanner stopped")
    
    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update prices."""
        tick: Tick = event.data["tick"]
        
        # Update price cache (use bid/ask if available, otherwise use price)
        if tick.symbol not in self._price_cache:
            self._price_cache[tick.symbol] = {}
        
        self._price_cache[tick.symbol]["price"] = tick.price
        self._price_cache[tick.symbol]["bid"] = tick.bid or tick.price
        self._price_cache[tick.symbol]["ask"] = tick.ask or tick.price
        self._price_cache[tick.symbol]["timestamp"] = tick.timestamp
        
        # Update C++ scanner if available
        if self._cpp_scanner:
            try:
                bid = tick.bid or tick.price
                ask = tick.ask or tick.price
                bid_size = 1.0  # Default size if not available
                ask_size = 1.0
                timestamp_ns = int(tick.timestamp.timestamp() * 1e9)
                
                self._cpp_scanner.update_pair(
                    tick.symbol, bid, ask, bid_size, ask_size, timestamp_ns
                )
            except Exception as e:
                logger.warning(f"Failed to update C++ scanner: {e}")
        
        # Periodic scan (every tick - ultra-high frequency)
        await self._scan_cycles()
    
    async def _handle_orderbook(self, event: Event) -> None:
        """Handle orderbook updates for better bid/ask prices."""
        orderbook_data = event.data.get("orderbook", {})
        symbol = orderbook_data.get("symbol")
        
        if not symbol:
            return
        
        bids = orderbook_data.get("bids", [])
        asks = orderbook_data.get("asks", [])
        
        if not bids or not asks:
            return
        
        # Update price cache with best bid/ask
        if symbol not in self._price_cache:
            self._price_cache[symbol] = {}
        
        self._price_cache[symbol]["bid"] = float(bids[0][0])
        self._price_cache[symbol]["ask"] = float(asks[0][0])
        self._price_cache[symbol]["bid_size"] = float(bids[0][1])
        self._price_cache[symbol]["ask_size"] = float(asks[0][1])
        self._price_cache[symbol]["timestamp"] = datetime.utcnow()
        
        # Update C++ scanner
        if self._cpp_scanner:
            try:
                bid = float(bids[0][0])
                ask = float(asks[0][0])
                bid_size = float(bids[0][1])
                ask_size = float(asks[0][1])
                timestamp_ns = int(time.time() * 1e9)
                
                self._cpp_scanner.update_pair(
                    symbol, bid, ask, bid_size, ask_size, timestamp_ns
                )
            except Exception as e:
                logger.warning(f"Failed to update C++ scanner from orderbook: {e}")
        
        # Scan immediately after orderbook update
        await self._scan_cycles()
    
    async def _scan_cycles(self) -> None:
        """Scan for triangular arbitrage opportunities."""
        if not self._running:
            return
        
        start_time = time.perf_counter()
        
        # Try C++ scanner first (ultra-low latency)
        if self._cpp_scanner:
            try:
                cycles = self._cpp_scanner.scan_cycles()
                if cycles:
                    for cycle in cycles[:5]:  # Top 5 cycles
                        await self._execute_cycle(cycle)
            except Exception as e:
                logger.warning(f"C++ scanner error: {e}, falling back to Python")
                cycles = await self._scan_cycles_python()
                if cycles:
                    for cycle in cycles[:5]:
                        await self._execute_cycle(cycle)
        else:
            # Python fallback (slower but functional)
            cycles = await self._scan_cycles_python()
            if cycles:
                for cycle in cycles[:5]:
                    await self._execute_cycle(cycle)
        
        latency_us = (time.perf_counter() - start_time) * 1e6
        
        # Log if latency exceeds target
        if latency_us > 500:
            logger.warning(f"Triangular scan latency: {latency_us:.2f}us (target: <500us)")
        else:
            logger.debug(f"Triangular scan latency: {latency_us:.2f}us")
    
    async def _scan_cycles_python(self) -> list[TriangularCycle]:
        """Python fallback implementation (simpler, slower)."""
        # This is a simplified Python implementation
        # The C++ version is much faster for 50+ pairs
        
        cycles: list[TriangularCycle] = []
        
        # Extract base/quote from symbols
        def parse_symbol(symbol: str) -> tuple[str, str]:
            quotes = ["USDT", "USDC", "BTC", "ETH", "BNB", "BUSD"]
            for q in quotes:
                if symbol.endswith(q):
                    return symbol[:-len(q)], q
            # Fallback
            if len(symbol) > 4:
                return symbol[:-4], symbol[-4:]
            return symbol, ""
        
        # Build asset graph
        asset_graph: dict[str, dict[str, tuple[str, float, float]]] = {}  # asset -> {pair: (quote, bid, ask)}
        
        for symbol, price_data in self._price_cache.items():
            if "bid" not in price_data or "ask" not in price_data:
                continue
            
            base, quote = parse_symbol(symbol)
            if not base or not quote:
                continue
            
            bid = price_data["bid"]
            ask = price_data["ask"]
            
            # Forward: base -> quote (sell base at bid)
            if base not in asset_graph:
                asset_graph[base] = {}
            asset_graph[base][symbol] = (quote, bid, ask)
            
            # Reverse: quote -> base (buy base at ask)
            if quote not in asset_graph:
                asset_graph[quote] = {}
            asset_graph[quote][symbol] = (base, 1.0 / ask, 1.0 / bid)
        
        # Find triangular cycles (A->B->C->A)
        for start_asset in asset_graph:
            # Try all paths of length 3
            for pair1, (asset2, price1, _) in asset_graph.get(start_asset, {}).items():
                for pair2, (asset3, price2, _) in asset_graph.get(asset2, {}).items():
                    if pair2 == pair1:  # Skip same pair
                        continue
                    for pair3, (asset4, price3, _) in asset_graph.get(asset3, {}).items():
                        if pair3 in [pair1, pair2]:  # Skip duplicate pairs
                            continue
                        if asset4 == start_asset:  # Found cycle!
                            profit_ratio = (price1 * price2 * price3) - 1.0
                            profit_bps = profit_ratio * 10000.0
                            
                            required_profit_bps = self._min_profit_bps + 10  # Fee buffer
                            
                            if profit_bps >= required_profit_bps:
                                cycle = TriangularCycle(
                                    pair_a=pair1,
                                    pair_b=pair2,
                                    pair_c=pair3,
                                    asset_a=start_asset,
                                    asset_b=asset2,
                                    asset_c=asset3,
                                    profit_ratio=profit_ratio,
                                    profit_bps=profit_bps,
                                    max_size=1.0,  # Simplified
                                    detection_time_ns=int(time.time() * 1e9)
                                )
                                cycles.append(cycle)
        
        # Sort by profit (highest first)
        cycles.sort(key=lambda c: c.profit_bps, reverse=True)
        
        return cycles[:10]  # Top 10
    
    async def _execute_cycle(self, cycle: TriangularCycle) -> None:
        """Execute a detected triangular arbitrage cycle.
        
        Creates atomic multi-leg order requests.
        """
        cycle_key = (cycle.pair_a, cycle.pair_b, cycle.pair_c)
        
        # Skip if cycle already active
        if cycle_key in self._active_cycles:
            return
        
        # Calculate position sizes (equal notional)
        # For triangular arbitrage, we want to maintain consistent notional
        notional = min(100.0, cycle.max_size)  # Conservative sizing
        
        # Determine trade directions from cycle
        # This is simplified - real implementation needs proper path analysis
        logger.info(
            f"Triangular arbitrage opportunity detected: "
            f"{cycle.pair_a} -> {cycle.pair_b} -> {cycle.pair_c} -> {cycle.asset_a}, "
            f"profit={cycle.profit_bps:.2f}bps, max_size={cycle.max_size:.4f}"
        )
        
        # Emit signal for atomic multi-leg execution
        cycle_id = f"{cycle.asset_a}-{cycle.asset_b}-{cycle.asset_c}-{cycle.detection_time_ns}"
        # The executor will handle atomic execution with rollback
        signal = Signal(
            strategy_id="triangular_arbitrage",
            symbol=cycle.pair_a,  # Primary symbol
            side="buy",  # Will be determined by executor based on cycle
            size=notional,
            entry_price=0.0,  # Will be filled by executor
            metadata={
                "triangular_cycle": True,
                "cycle_id": cycle_id,
                "pair_a": cycle.pair_a,
                "pair_b": cycle.pair_b,
                "pair_c": cycle.pair_c,
                "asset_a": cycle.asset_a,
                "asset_b": cycle.asset_b,
                "asset_c": cycle.asset_c,
                "profit_bps": cycle.profit_bps,
                "profit_ratio": cycle.profit_ratio,
                "max_size": cycle.max_size,
                "detection_time_ns": cycle.detection_time_ns,
                "atomic_execution": True,  # Flag for atomic multi-leg orders
                "triangular_stats": self.get_stats(),
            }
        )
        
        await self._bus.publish(
            Event(
                event_type=EventType.SIGNAL,
                data={"signal": signal}
            )
        )
        
        # Track active cycle
        self._active_cycles[cycle_key] = cycle
    
    def get_stats(self) -> dict[str, Any]:
        """Get scanner statistics."""
        if self._cpp_scanner:
            try:
                stats = self._cpp_scanner.get_stats()
                return {
                    "total_scans": stats.get("total_scans", 0),
                    "cycles_found": stats.get("cycles_found", 0),
                    "avg_latency_us": stats.get("avg_latency_us", 0.0),
                    "active_pairs": self._cpp_scanner.get_active_pair_count(),
                }
            except Exception as e:
                logger.warning(f"Failed to get C++ scanner stats: {e}")
        
        return {
            "total_scans": 0,
            "cycles_found": len(self._active_cycles),
            "avg_latency_us": 0.0,
            "active_pairs": len(self._price_cache),
        }

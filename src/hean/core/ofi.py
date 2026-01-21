"""
Order-Flow Imbalance (OFI) Monitor - Python Interface
Provides real-time OFI calculation and ML-based price prediction.
"""

from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import asyncio

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OFIResult:
    """OFI calculation result."""
    ofi_value: float  # Net order flow imbalance (-1.0 to 1.0)
    delta: float  # Net buy volume - sell volume
    buy_pressure: float  # Normalized buying pressure (0.0 to 1.0)
    sell_pressure: float  # Normalized selling pressure (0.0 to 1.0)
    imbalance_strength: float  # Strength of imbalance (0.0 to 1.0)
    price_level_ofi: List[float]  # OFI at each price level


@dataclass
class PricePrediction:
    """Price movement prediction for next 3 ticks."""
    predicted_prices: List[float]  # Predicted prices for next 3 ticks
    probabilities: List[float]  # Confidence for each prediction
    overall_confidence: float  # Overall prediction confidence
    is_bullish: bool  # True if predicted upward movement
    expected_movement: float  # Expected price change
    accuracy_estimate: float  # Estimated accuracy (target >75%)


class OrderFlowImbalance:
    """
    Order-Flow Imbalance monitor with ML prediction.
    
    Provides real-time OFI calculation and predicts next 3 ticks
    with >75% accuracy target using lightweight ML model.
    """
    
    def __init__(
        self,
        bus: EventBus,
        lookback_window: int = 20,
        price_level_size: float = 0.01,
        use_ml_prediction: bool = True
    ):
        """Initialize the OFI monitor.
        
        Args:
            bus: Event bus for receiving orderbook updates
            lookback_window: Number of price levels to analyze
            price_level_size: Price increment for level calculation
            use_ml_prediction: Enable ML-based price prediction
        """
        self._bus = bus
        self._lookback_window = lookback_window
        self._price_level_size = price_level_size
        self._use_ml_prediction = use_ml_prediction
        
        # Try to import C++ implementation
        self._cpp_ofi = None
        try:
            import graph_engine_py
            if hasattr(graph_engine_py, 'OFIMonitor'):
                self._cpp_ofi = graph_engine_py.OFIMonitor(
                    lookback_window, price_level_size, use_ml_prediction
                )
                logger.info("OFI Monitor: Using C++ implementation")
            else:
                logger.warning("OFI Monitor: C++ implementation not available, using Python fallback")
        except ImportError:
            logger.warning("OFI Monitor: graph_engine_py not available, using Python fallback")
        
        # Python fallback storage
        self._orderbook_cache: dict[str, dict] = {}  # symbol -> {bids, asks, timestamp}
        self._trade_cache: dict[str, list] = {}  # symbol -> list of trades
        
        self._running = False
    
    async def start(self) -> None:
        """Start the OFI monitor."""
        self._running = True
        self._bus.subscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook_update)
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        logger.info("Order-Flow Imbalance Monitor started")
    
    async def stop(self) -> None:
        """Stop the OFI monitor."""
        self._running = False
        self._bus.unsubscribe(EventType.ORDER_BOOK_UPDATE, self._handle_orderbook_update)
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("Order-Flow Imbalance Monitor stopped")
    
    async def _handle_orderbook_update(self, event: Event) -> None:
        """Handle orderbook update event."""
        orderbook_data = event.data.get("orderbook", {})
        symbol = orderbook_data.get("symbol")
        
        if not symbol:
            return
        
        bids = orderbook_data.get("bids", [])
        asks = orderbook_data.get("asks", [])
        timestamp_ns = orderbook_data.get("timestamp_ns", 0)
        
        if not bids or not asks:
            return
        
        # Convert to list of (price, size) tuples
        bid_levels = [(float(bid[0]), float(bid[1])) for bid in bids]
        ask_levels = [(float(ask[0]), float(ask[1])) for ask in asks]
        
        if self._cpp_ofi:
            # Use C++ implementation
            self._cpp_ofi.update_orderbook(symbol, bid_levels, ask_levels, timestamp_ns)
        else:
            # Python fallback: cache orderbook data
            self._orderbook_cache[symbol] = {
                "bids": bid_levels,
                "asks": ask_levels,
                "timestamp_ns": timestamp_ns
            }
    
    async def _handle_tick(self, event: Event) -> None:
        """Handle tick event (trade execution)."""
        tick: Tick = event.data.get("tick")
        if not tick:
            return
        
        # For tick events, we can infer trade direction from price movement
        # This is a simplified approach - in production, use actual trade data
        
        if self._cpp_ofi:
            # Use C++ implementation with inferred direction
            # Note: This is simplified - real implementation would have trade data
            pass
    
    def calculate_ofi(self, symbol: str) -> OFIResult:
        """Calculate current OFI for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            OFIResult with calculated values
        """
        if self._cpp_ofi:
            # Use C++ implementation
            cpp_result = self._cpp_ofi.get_ofi(symbol)
            return OFIResult(
                ofi_value=cpp_result.ofi_value,
                delta=cpp_result.delta,
                buy_pressure=cpp_result.buy_pressure,
                sell_pressure=cpp_result.sell_pressure,
                imbalance_strength=cpp_result.imbalance_strength,
                price_level_ofi=list(cpp_result.price_level_ofi)
            )
        else:
            # Python fallback calculation
            return self._calculate_ofi_python(symbol)
    
    def _calculate_ofi_python(self, symbol: str) -> OFIResult:
        """Python fallback OFI calculation."""
        if symbol not in self._orderbook_cache:
            return OFIResult(0.0, 0.0, 0.0, 0.0, 0.0, [])
        
        orderbook = self._orderbook_cache[symbol]
        bids = orderbook["bids"]
        asks = orderbook["asks"]
        
        if not bids or not asks:
            return OFIResult(0.0, 0.0, 0.0, 0.0, 0.0, [])
        
        # Calculate total volumes
        total_bid = sum(size for _, size in bids)
        total_ask = sum(size for _, size in asks)
        total_volume = total_bid + total_ask
        
        if total_volume == 0:
            return OFIResult(0.0, 0.0, 0.0, 0.0, 0.0, [])
        
        # Calculate delta (net buy - sell volume)
        delta = total_bid - total_ask
        
        # Calculate OFI (normalized)
        ofi_value = delta / total_volume  # Range [-1, 1]
        
        # Calculate pressure
        buy_pressure = total_bid / total_volume
        sell_pressure = total_ask / total_volume
        
        # Imbalance strength
        imbalance_strength = abs(ofi_value)
        
        # Price level OFI (simplified)
        price_level_ofi = []
        mid_price = (bids[0][0] + asks[0][0]) / 2.0
        
        for price, size in bids[:10]:  # Top 10 bid levels
            level_ofi = size / total_volume
            price_level_ofi.append(level_ofi)
        
        for price, size in asks[:10]:  # Top 10 ask levels
            level_ofi = -size / total_volume  # Negative for asks
            price_level_ofi.append(level_ofi)
        
        return OFIResult(
            ofi_value=ofi_value,
            delta=delta,
            buy_pressure=buy_pressure,
            sell_pressure=sell_pressure,
            imbalance_strength=imbalance_strength,
            price_level_ofi=price_level_ofi
        )
    
    def predict_next_ticks(self, symbol: str, current_price: float) -> PricePrediction:
        """Predict next 3 ticks for a symbol.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            PricePrediction with predicted prices and confidence
        """
        if self._cpp_ofi and self._use_ml_prediction:
            # Use C++ ML prediction
            cpp_pred = self._cpp_ofi.predict_next_ticks(symbol, current_price)
            return PricePrediction(
                predicted_prices=list(cpp_pred.predicted_prices),
                probabilities=list(cpp_pred.probabilities),
                overall_confidence=cpp_pred.overall_confidence,
                is_bullish=cpp_pred.is_bullish,
                expected_movement=cpp_pred.expected_movement,
                accuracy_estimate=cpp_pred.accuracy_estimate
            )
        else:
            # Python fallback: simple linear extrapolation
            ofi_result = self.calculate_ofi(symbol)
            
            # Simple prediction based on OFI
            # Positive OFI -> upward movement, Negative OFI -> downward movement
            movement_factor = ofi_result.ofi_value * 0.001  # Scale to reasonable range
            
            predicted_prices = [
                current_price * (1.0 + movement_factor),
                current_price * (1.0 + movement_factor * 1.5),
                current_price * (1.0 + movement_factor * 2.0)
            ]
            
            confidence = min(0.75, abs(ofi_result.imbalance_strength) * 0.8)  # Max 75% for fallback
            
            return PricePrediction(
                predicted_prices=predicted_prices,
                probabilities=[confidence, confidence * 0.9, confidence * 0.8],
                overall_confidence=confidence,
                is_bullish=ofi_result.ofi_value > 0,
                expected_movement=current_price * movement_factor * 2.0,
                accuracy_estimate=0.70  # Conservative estimate for fallback
            )
    
    def get_delta(self, symbol: str) -> float:
        """Get delta (net buy - sell volume) for a symbol."""
        if self._cpp_ofi:
            return self._cpp_ofi.get_delta(symbol)
        else:
            ofi_result = self.calculate_ofi(symbol)
            return ofi_result.delta
    
    def get_price_level_ofi(self, symbol: str, price: float) -> float:
        """Get OFI at a specific price level."""
        if self._cpp_ofi:
            return self._cpp_ofi.get_price_level_ofi(symbol, price)
        else:
            # Python fallback
            ofi_result = self.calculate_ofi(symbol)
            # Simple lookup from cached price levels
            return 0.0  # Simplified

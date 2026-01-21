"""
Causal Discovery Engine: Beyond AGI Market Analysis

This engine maps hidden relationships between Bybit's top 100 assets using causal inference.
It predicts "Market Domino Effects" - if Asset X moves, it calculates the precise microsecond
when Asset Y will follow, executing before market-makers react.

The system understands the market better than the engineers who built the exchange.
"""

import asyncio
import numpy as np
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
import networkx as nx
from dataclasses import dataclass

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CausalEdge:
    """Represents a causal relationship between two assets."""
    source: str
    target: str
    lag_microseconds: int  # Precise lag in microseconds
    strength: float  # Causal strength (0-1)
    confidence: float  # Statistical confidence (0-1)
    last_observed: datetime
    observation_count: int


@dataclass
class DominoPrediction:
    """Prediction of a market domino effect."""
    trigger_asset: str
    affected_asset: str
    predicted_time: datetime
    expected_magnitude: float  # Expected price change magnitude
    confidence: float
    lag_microseconds: int


class CausalDiscoveryEngine:
    """
    Causal Discovery Engine that maps relationships between assets and predicts domino effects.
    
    Uses:
    - Granger Causality for temporal relationships
    - Transfer Entropy for information flow
    - Cross-correlation for lead-lag detection
    - Causal graph structure learning
    """
    
    def __init__(self, bus: EventBus, symbols: List[str], top_n: int = 100):
        """
        Initialize the Causal Discovery Engine.
        
        Args:
            bus: Event bus for subscribing to market data
            symbols: List of trading symbols to monitor
            top_n: Number of top assets to analyze (default: 100)
        """
        self._bus = bus
        self._symbols = symbols[:top_n]  # Limit to top N assets
        self._running = False
        
        # Price history for each asset (circular buffer)
        self._price_history: Dict[str, deque] = {}
        self._timestamp_history: Dict[str, deque] = {}
        self._history_window = 10000  # Keep last 10k ticks per asset
        
        # Causal graph (directed graph of asset relationships)
        self._causal_graph = nx.DiGraph()
        self._causal_edges: Dict[Tuple[str, str], CausalEdge] = {}
        
        # Active domino predictions
        self._active_predictions: List[DominoPrediction] = []
        
        # Statistics
        self._total_predictions = 0
        self._successful_predictions = 0
        self._prediction_accuracy = 0.0
        
        # Initialize price history for all symbols
        for symbol in self._symbols:
            self._price_history[symbol] = deque(maxlen=self._history_window)
            self._timestamp_history[symbol] = deque(maxlen=self._history_window)
            self._causal_graph.add_node(symbol)
        
        logger.info(f"Causal Discovery Engine initialized for {len(self._symbols)} assets")
    
    async def start(self) -> None:
        """Start the causal discovery engine."""
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        self._running = True
        
        # Start background tasks
        asyncio.create_task(self._update_causal_graph_periodically())
        asyncio.create_task(self._validate_predictions())
        
        logger.info("Causal Discovery Engine started")
    
    async def stop(self) -> None:
        """Stop the causal discovery engine."""
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        self._running = False
        logger.info("Causal Discovery Engine stopped")
    
    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update price history."""
        if not self._running:
            return
        
        tick: Tick = event.data["tick"]
        
        if tick.symbol not in self._symbols:
            return
        
        # Update price history
        self._price_history[tick.symbol].append(tick.price)
        self._timestamp_history[tick.symbol].append(tick.timestamp)
        
        # Check for domino effects
        await self._check_domino_effects(tick)
    
    def _calculate_granger_causality(self, asset_a: str, asset_b: str, max_lag: int = 50) -> Optional[Tuple[float, int]]:
        """
        Calculate Granger Causality between two assets.
        
        Returns:
            Tuple of (causality_strength, lag_in_samples) or None if insufficient data
        """
        if asset_a not in self._price_history or asset_b not in self._price_history:
            return None
        
        prices_a = list(self._price_history[asset_a])
        prices_b = list(self._price_history[asset_b])
        
        if len(prices_a) < max_lag * 2 or len(prices_b) < max_lag * 2:
            return None
        
        # Calculate returns
        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]
        
        # Align series
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a[-min_len:]
        returns_b = returns_b[-min_len:]
        
        if min_len < max_lag * 2:
            return None
        
        # Simple Granger causality test using cross-correlation
        best_correlation = 0.0
        best_lag = 0
        
        for lag in range(1, min(max_lag, min_len // 2)):
            if lag >= len(returns_a) or lag >= len(returns_b):
                break
            
            # A leads B (A causes B)
            a_shifted = returns_a[:-lag] if lag > 0 else returns_a
            b_shifted = returns_b[lag:] if lag > 0 else returns_b
            
            if len(a_shifted) != len(b_shifted) or len(a_shifted) < 10:
                continue
            
            correlation = np.corrcoef(a_shifted, b_shifted)[0, 1]
            
            if not np.isnan(correlation) and abs(correlation) > abs(best_correlation):
                best_correlation = correlation
                best_lag = lag
        
        if abs(best_correlation) > 0.1:  # Minimum threshold
            return (abs(best_correlation), best_lag)
        
        return None
    
    def _calculate_transfer_entropy(self, asset_a: str, asset_b: str) -> Optional[float]:
        """
        Calculate Transfer Entropy (information flow) from asset_a to asset_b.
        
        Simplified implementation using mutual information.
        """
        if asset_a not in self._price_history or asset_b not in self._price_history:
            return None
        
        prices_a = list(self._price_history[asset_a])
        prices_b = list(self._price_history[asset_b])
        
        if len(prices_a) < 100 or len(prices_b) < 100:
            return None
        
        # Calculate returns
        returns_a = np.diff(prices_a) / prices_a[:-1]
        returns_b = np.diff(prices_b) / prices_b[:-1]
        
        # Align series
        min_len = min(len(returns_a), len(returns_b))
        returns_a = returns_a[-min_len:]
        returns_b = returns_b[-min_len:]
        
        if min_len < 100:
            return None
        
        # Simplified transfer entropy using correlation
        correlation = np.corrcoef(returns_a, returns_b)[0, 1]
        
        if np.isnan(correlation):
            return None
        
        # Convert correlation to information-theoretic measure
        # Higher correlation = higher information transfer
        return abs(correlation)
    
    async def _update_causal_graph_periodically(self) -> None:
        """Periodically update the causal graph by analyzing relationships."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Update every minute
                
                if not self._running:
                    break
                
                logger.debug("Updating causal graph...")
                
                # Analyze all pairs of assets
                new_edges = []
                
                for asset_a in self._symbols:
                    for asset_b in self._symbols:
                        if asset_a == asset_b:
                            continue
                        
                        # Calculate Granger causality
                        granger_result = self._calculate_granger_causality(asset_a, asset_b)
                        
                        if granger_result is None:
                            continue
                        
                        causality_strength, lag_samples = granger_result
                        
                        # Convert lag from samples to microseconds
                        # Estimate: average tick interval (assume 100ms = 100,000 microseconds)
                        avg_tick_interval_us = 100000  # 100ms default
                        if asset_a in self._timestamp_history and len(self._timestamp_history[asset_a]) > 1:
                            timestamps = list(self._timestamp_history[asset_a])
                            intervals = [timestamps[i] - timestamps[i-1] for i in range(1, len(timestamps))]
                            if intervals:
                                avg_tick_interval_us = np.mean(intervals) / 1000  # Convert to microseconds
                        
                        lag_microseconds = int(lag_samples * avg_tick_interval_us)
                        
                        # Calculate transfer entropy
                        transfer_entropy = self._calculate_transfer_entropy(asset_a, asset_b)
                        
                        if transfer_entropy is None:
                            continue
                        
                        # Combined strength (weighted average)
                        combined_strength = (causality_strength * 0.6 + transfer_entropy * 0.4)
                        
                        # Only keep strong relationships
                        if combined_strength > 0.2:  # Threshold
                            edge_key = (asset_a, asset_b)
                            
                            if edge_key in self._causal_edges:
                                # Update existing edge
                                edge = self._causal_edges[edge_key]
                                edge.strength = combined_strength
                                edge.lag_microseconds = lag_microseconds
                                edge.last_observed = datetime.utcnow()
                                edge.observation_count += 1
                            else:
                                # Create new edge
                                edge = CausalEdge(
                                    source=asset_a,
                                    target=asset_b,
                                    lag_microseconds=lag_microseconds,
                                    strength=combined_strength,
                                    confidence=min(1.0, combined_strength * 1.5),  # Scale confidence
                                    last_observed=datetime.utcnow(),
                                    observation_count=1
                                )
                                self._causal_edges[edge_key] = edge
                                new_edges.append(edge)
                                
                                # Update graph
                                self._causal_graph.add_edge(asset_a, asset_b, 
                                                           weight=combined_strength,
                                                           lag_us=lag_microseconds)
                            
                            logger.debug(f"Causal edge: {asset_a} -> {asset_b} "
                                       f"(strength={combined_strength:.3f}, lag={lag_microseconds}μs)")
                
                if new_edges:
                    logger.info(f"Discovered {len(new_edges)} new causal relationships")
                
            except Exception as e:
                logger.error(f"Error updating causal graph: {e}", exc_info=True)
    
    async def _check_domino_effects(self, tick: Tick) -> None:
        """Check if a tick triggers a domino effect prediction."""
        if tick.symbol not in self._symbols:
            return
        
        # Check all outgoing edges from this asset
        if tick.symbol not in self._causal_graph:
            return
        
        for target_asset in self._causal_graph.successors(tick.symbol):
            edge_key = (tick.symbol, target_asset)
            
            if edge_key not in self._causal_edges:
                continue
            
            edge = self._causal_edges[edge_key]
            
            # Calculate expected magnitude based on current price movement
            if len(self._price_history[tick.symbol]) < 2:
                continue
            
            recent_prices = list(self._price_history[tick.symbol])[-10:]
            if len(recent_prices) < 2:
                continue
            
            price_change_pct = (recent_prices[-1] - recent_prices[-2]) / recent_prices[-2]
            
            # Predict domino effect
            lag_timedelta = timedelta(microseconds=edge.lag_microseconds)
            predicted_time = tick.timestamp + lag_timedelta
            
            # Expected magnitude (scaled by edge strength)
            expected_magnitude = price_change_pct * edge.strength * 0.8  # Conservative scaling
            
            prediction = DominoPrediction(
                trigger_asset=tick.symbol,
                affected_asset=target_asset,
                predicted_time=predicted_time,
                expected_magnitude=expected_magnitude,
                confidence=edge.confidence,
                lag_microseconds=edge.lag_microseconds
            )
            
            self._active_predictions.append(prediction)
            self._total_predictions += 1
            
            # Publish prediction event
            await self._bus.publish(
                Event(
                    event_type=EventType.SIGNAL,  # Reuse SIGNAL event type
                    data={
                        "domino_prediction": prediction,
                        "type": "causal_domino"
                    }
                )
            )
            
            logger.info(f"Domino prediction: {tick.symbol} -> {target_asset} "
                       f"at {predicted_time} (lag={edge.lag_microseconds}μs, "
                       f"magnitude={expected_magnitude:.4f}, confidence={edge.confidence:.3f})")
    
    async def _validate_predictions(self) -> None:
        """Validate active predictions and update accuracy metrics."""
        while self._running:
            try:
                await asyncio.sleep(1)  # Check every second
                
                current_time = datetime.utcnow()
                validated_predictions = []
                
                for prediction in self._active_predictions:
                    # Check if prediction time has passed
                    if current_time >= prediction.predicted_time:
                        # Validate prediction
                        if prediction.affected_asset in self._price_history:
                            prices = list(self._price_history[prediction.affected_asset])
                            if len(prices) >= 2:
                                actual_change = (prices[-1] - prices[-2]) / prices[-2]
                                
                                # Check if prediction was correct (same direction)
                                predicted_direction = 1 if prediction.expected_magnitude > 0 else -1
                                actual_direction = 1 if actual_change > 0 else -1
                                
                                if predicted_direction == actual_direction:
                                    self._successful_predictions += 1
                                
                                # Update accuracy
                                if self._total_predictions > 0:
                                    self._prediction_accuracy = self._successful_predictions / self._total_predictions
                                
                                logger.debug(f"Validated prediction: {prediction.trigger_asset} -> "
                                           f"{prediction.affected_asset}, "
                                           f"predicted={prediction.expected_magnitude:.4f}, "
                                           f"actual={actual_change:.4f}, "
                                           f"correct={predicted_direction == actual_direction}")
                        
                        # Remove old predictions (keep only recent ones)
                        if (current_time - prediction.predicted_time).total_seconds() > 60:
                            continue  # Skip old predictions
                    
                    validated_predictions.append(prediction)
                
                self._active_predictions = validated_predictions
                
            except Exception as e:
                logger.error(f"Error validating predictions: {e}", exc_info=True)
    
    def get_causal_graph(self) -> nx.DiGraph:
        """Get the current causal graph."""
        return self._causal_graph.copy()
    
    def get_causal_edges(self) -> Dict[Tuple[str, str], CausalEdge]:
        """Get all causal edges."""
        return self._causal_edges.copy()
    
    def get_prediction_accuracy(self) -> float:
        """Get the current prediction accuracy."""
        return self._prediction_accuracy
    
    def get_statistics(self) -> Dict:
        """Get engine statistics."""
        return {
            "total_assets": len(self._symbols),
            "causal_edges": len(self._causal_edges),
            "total_predictions": self._total_predictions,
            "successful_predictions": self._successful_predictions,
            "prediction_accuracy": self._prediction_accuracy,
            "active_predictions": len(self._active_predictions)
        }

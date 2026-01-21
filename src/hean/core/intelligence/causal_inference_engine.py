"""
Causal Inference Engine: Granger Causality + Transfer Entropy
Predicts Bybit moves by analyzing subtle 'pre-echoes' in global cross-asset orderflow.
"""

import asyncio
import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Tick, Signal
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CausalRelationship:
    """Represents a causal relationship between two assets."""
    source_symbol: str
    target_symbol: str
    granger_causality: float  # F-statistic (higher = stronger causality)
    transfer_entropy: float  # Bits (higher = stronger information flow)
    lag_period: int  # Number of periods source leads target
    p_value: float  # Statistical significance (lower = more significant)
    confidence: float  # Combined confidence score (0-1)
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PreEchoSignal:
    """Pre-echo signal detected from cross-asset orderflow."""
    target_symbol: str  # Symbol to predict (e.g., BTCUSDT on Bybit)
    source_symbol: str  # Source symbol showing pre-echo (e.g., BTC on another exchange)
    predicted_direction: str  # "buy" or "sell"
    predicted_magnitude: float  # Expected price move magnitude
    confidence: float  # Prediction confidence (0-1)
    lag_ms: int  # Time lag between source and target (milliseconds)
    granger_score: float
    transfer_entropy_score: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


class GrangerCausalityCalculator:
    """Calculate Granger Causality between time series."""
    
    @staticmethod
    def calculate(
        X: np.ndarray,
        Y: np.ndarray,
        max_lag: int = 10
    ) -> Tuple[float, float, int]:
        """
        Calculate Granger Causality: Does X Granger-cause Y?
        
        Returns:
            (F-statistic, p-value, optimal_lag)
            Higher F-statistic and lower p-value = stronger causality
        """
        if len(X) < max_lag * 2 or len(Y) < max_lag * 2:
            return 0.0, 1.0, 0
        
        # Align series
        min_len = min(len(X), len(Y))
        X = X[-min_len:]
        Y = Y[-min_len:]
        
        best_f = 0.0
        best_p = 1.0
        best_lag = 0
        
        # Try different lag values
        for lag in range(1, min(max_lag + 1, min_len // 4)):
            try:
                f_stat, p_val = GrangerCausalityCalculator._granger_test(X, Y, lag)
                
                if f_stat > best_f and p_val < best_p:
                    best_f = f_stat
                    best_p = p_val
                    best_lag = lag
            except Exception:
                continue
        
        return best_f, best_p, best_lag
    
    @staticmethod
    def _granger_test(X: np.ndarray, Y: np.ndarray, lag: int) -> Tuple[float, float]:
        """
        Simplified Granger Causality test using VAR (Vector Auto-Regression).
        
        Tests: H0: X does not Granger-cause Y
        """
        n = len(Y) - lag
        
        if n < lag * 2:
            return 0.0, 1.0
        
        # Build restricted model (Y regressed only on Y's past)
        Y_lag = np.zeros((n, lag))
        for i in range(n):
            for j in range(lag):
                Y_lag[i, j] = Y[i + lag - j - 1]
        
        # Add constant term
        X_restricted = np.column_stack([np.ones(n), Y_lag])
        Y_current = Y[lag:lag+n]
        
        try:
            # OLS regression: restricted model
            beta_restricted = np.linalg.lstsq(X_restricted, Y_current, rcond=None)[0]
            residuals_restricted = Y_current - X_restricted @ beta_restricted
            ssr_restricted = np.sum(residuals_restricted ** 2)
            
            # Build unrestricted model (Y regressed on both Y's and X's past)
            X_lag = np.zeros((n, lag))
            for i in range(n):
                for j in range(lag):
                    X_lag[i, j] = X[i + lag - j - 1]
            
            X_unrestricted = np.column_stack([np.ones(n), Y_lag, X_lag])
            beta_unrestricted = np.linalg.lstsq(X_unrestricted, Y_current, rcond=None)[0]
            residuals_unrestricted = Y_current - X_unrestricted @ beta_unrestricted
            ssr_unrestricted = np.sum(residuals_unrestricted ** 2)
            
            # F-statistic: F = ((SSR_r - SSR_ur) / p) / (SSR_ur / (n - k))
            # where p = number of restrictions, k = number of parameters in unrestricted
            p_restrictions = lag  # X lags
            k_unrestricted = 1 + 2 * lag  # constant + Y lags + X lags
            df_unrestricted = n - k_unrestricted
            
            if ssr_unrestricted > 0 and df_unrestricted > 0:
                f_stat = ((ssr_restricted - ssr_unrestricted) / p_restrictions) / \
                         (ssr_unrestricted / df_unrestricted)
                f_stat = max(0.0, f_stat)  # Ensure non-negative
                
                # Approximate p-value using F-distribution (simplified)
                # In production, would use scipy.stats.f.sf
                p_value = max(0.0, min(1.0, 1.0 / (1.0 + f_stat)))  # Simplified
            else:
                f_stat = 0.0
                p_value = 1.0
            
            return f_stat, p_value
            
        except Exception as e:
            logger.debug(f"Granger test error: {e}")
            return 0.0, 1.0


class TransferEntropyCalculator:
    """Calculate Transfer Entropy between time series."""
    
    @staticmethod
    def calculate(
        X: np.ndarray,
        Y: np.ndarray,
        lag: int = 1,
        bins: int = 10
    ) -> float:
        """
        Calculate Transfer Entropy: TE(X -> Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})
        
        Measures information flow from X to Y (in bits).
        Higher value = more information transferred.
        """
        if len(X) < lag * 2 or len(Y) < lag * 2:
            return 0.0
        
        # Align series
        min_len = min(len(X), len(Y))
        X = X[-min_len:]
        Y = Y[-min_len:]
        
        # Discretize to bins
        X_binned = TransferEntropyCalculator._bin_data(X, bins)
        Y_binned = TransferEntropyCalculator._bin_data(Y, bins)
        
        # Calculate conditional entropies
        try:
            # H(Y_t | Y_{t-1})
            h_y_given_y_prev = TransferEntropyCalculator._conditional_entropy(
                Y_binned[1:], Y_binned[:-1]
            )
            
            # H(Y_t | Y_{t-1}, X_{t-lag})
            if lag > 0:
                X_lagged = X_binned[:-lag] if lag < len(X_binned) else X_binned[:1]
                Y_current = Y_binned[lag:] if lag < len(Y_binned) else Y_binned[:1]
                Y_prev = Y_binned[lag-1:-1] if lag > 0 and lag < len(Y_binned) else Y_binned[:-1]
                
                # Align lengths
                min_len = min(len(Y_current), len(Y_prev), len(X_lagged))
                Y_current = Y_current[:min_len]
                Y_prev = Y_prev[:min_len]
                X_lagged = X_lagged[:min_len]
                
                # Combine Y_prev and X_lagged as condition
                condition = [(y, x) for y, x in zip(Y_prev, X_lagged)]
                h_y_given_y_prev_x = TransferEntropyCalculator._conditional_entropy(
                    Y_current, condition
                )
            else:
                h_y_given_y_prev_x = h_y_given_y_prev
            
            # Transfer Entropy = reduction in uncertainty
            te = max(0.0, h_y_given_y_prev - h_y_given_y_prev_x)
            
            return te
            
        except Exception as e:
            logger.debug(f"Transfer Entropy calculation error: {e}")
            return 0.0
    
    @staticmethod
    def _bin_data(data: np.ndarray, bins: int) -> np.ndarray:
        """Discretize continuous data into bins."""
        if len(data) == 0:
            return np.array([])
        
        min_val = np.min(data)
        max_val = np.max(data)
        
        if max_val == min_val:
            return np.zeros(len(data), dtype=int)
        
        bin_width = (max_val - min_val) / bins
        binned = ((data - min_val) / bin_width).astype(int)
        binned = np.clip(binned, 0, bins - 1)
        
        return binned
    
    @staticmethod
    def _entropy(values: List) -> float:
        """Calculate Shannon entropy."""
        if len(values) == 0:
            return 0.0
        
        # Count frequencies
        from collections import Counter
        counts = Counter(values)
        n = len(values)
        
        # Calculate entropy: H = -Σ p(x) * log2(p(x))
        entropy = 0.0
        for count in counts.values():
            p = count / n
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    @staticmethod
    def _conditional_entropy(X: List, Y: List | List[Tuple]) -> float:
        """Calculate conditional entropy H(X | Y)."""
        if len(X) != len(Y) or len(X) == 0:
            return 0.0
        
        # Group X values by Y condition
        from collections import defaultdict
        groups = defaultdict(list)
        
        for x, y in zip(X, Y):
            groups[y].append(x)
        
        # H(X|Y) = Σ p(y) * H(X|Y=y)
        conditional_entropy = 0.0
        n = len(X)
        
        for y, x_values in groups.items():
            p_y = len(x_values) / n
            h_x_given_y = TransferEntropyCalculator._entropy(x_values)
            conditional_entropy += p_y * h_x_given_y
        
        return conditional_entropy


class CausalInferenceEngine:
    """
    Causal Inference Engine using Granger Causality and Transfer Entropy.
    
    Analyzes cross-asset orderflow to detect 'pre-echoes' that predict Bybit moves.
    """
    
    def __init__(
        self,
        bus: EventBus,
        target_symbols: List[str] = None,  # Symbols to predict (e.g., BTCUSDT on Bybit)
        source_symbols: List[str] = None,  # Source symbols (other exchanges, indices, etc.)
        window_size: int = 500,
        min_causality_threshold: float = 0.3,  # Minimum F-statistic
        min_transfer_entropy: float = 0.1  # Minimum bits
    ):
        """Initialize the causal inference engine.
        
        Args:
            bus: Event bus for publishing signals
            target_symbols: Symbols to predict (Bybit trading pairs)
            source_symbols: Source symbols for pre-echo detection
            window_size: Rolling window size for analysis
            min_causality_threshold: Minimum Granger F-statistic for significance
            min_transfer_entropy: Minimum Transfer Entropy (bits) for significance
        """
        self._bus = bus
        self._target_symbols = target_symbols or ["BTCUSDT", "ETHUSDT"]
        self._source_symbols = source_symbols or []
        self._window_size = window_size
        self._min_causality_threshold = min_causality_threshold
        self._min_transfer_entropy = min_transfer_entropy
        
        # Time series storage (returns for causality analysis)
        self._returns_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Causal relationships
        self._causal_relationships: Dict[Tuple[str, str], CausalRelationship] = {}
        
        # Pre-echo signals
        self._pre_echo_signals: deque = deque(maxlen=100)
        
        self._running = False
        
        logger.info(
            f"Causal Inference Engine initialized: "
            f"targets={self._target_symbols}, "
            f"sources={len(self._source_symbols)}, "
            f"window={window_size}"
        )
    
    async def start(self) -> None:
        """Start the causal inference engine."""
        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        
        # Start periodic analysis
        asyncio.create_task(self._periodic_analysis())
        
        logger.info("Causal Inference Engine started")
    
    async def stop(self) -> None:
        """Stop the causal inference engine."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("Causal Inference Engine stopped")
    
    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update time series."""
        tick: Tick = event.data["tick"]
        symbol = tick.symbol
        
        # Update price history
        self._price_history[symbol].append(tick.price)
        
        # Calculate and store returns
        if len(self._price_history[symbol]) >= 2:
            prices = list(self._price_history[symbol])
            prev_price = prices[-2]
            if prev_price > 0:
                ret = (prices[-1] - prev_price) / prev_price
                self._returns_history[symbol].append(ret)
    
    async def _periodic_analysis(self) -> None:
        """Periodically analyze causal relationships and detect pre-echoes."""
        while self._running:
            await asyncio.sleep(10.0)  # Analyze every 10 seconds
            
            # Update causal relationships
            await self._update_causal_relationships()
            
            # Detect pre-echo signals
            await self._detect_pre_echoes()
    
    async def _update_causal_relationships(self) -> None:
        """Update causal relationships between source and target symbols."""
        for target in self._target_symbols:
            for source in self._source_symbols:
                if source == target:
                    continue
                
                # Get returns for both symbols
                target_returns = np.array(list(self._returns_history[target]))
                source_returns = np.array(list(self._returns_history[source]))
                
                if len(target_returns) < 50 or len(source_returns) < 50:
                    continue
                
                # Calculate Granger Causality: Does source cause target?
                f_stat, p_value, lag = GrangerCausalityCalculator.calculate(
                    source_returns, target_returns, max_lag=10
                )
                
                # Calculate Transfer Entropy
                te = TransferEntropyCalculator.calculate(
                    source_returns, target_returns, lag=lag, bins=10
                )
                
                # Check if relationship is significant
                is_significant = (
                    f_stat >= self._min_causality_threshold and
                    te >= self._min_transfer_entropy and
                    p_value < 0.05
                )
                
                if is_significant or (target, source) in self._causal_relationships:
                    # Calculate combined confidence
                    granger_score = min(1.0, f_stat / 10.0)  # Normalize F-stat
                    te_score = min(1.0, te / 5.0)  # Normalize TE (bits)
                    confidence = (granger_score * 0.6 + te_score * 0.4) * (1.0 - p_value)
                    
                    relationship = CausalRelationship(
                        source_symbol=source,
                        target_symbol=target,
                        granger_causality=f_stat,
                        transfer_entropy=te,
                        lag_period=lag,
                        p_value=p_value,
                        confidence=confidence,
                        last_updated=datetime.utcnow()
                    )
                    
                    self._causal_relationships[(source, target)] = relationship
                    
                    logger.debug(
                        f"Causal relationship: {source} -> {target}: "
                        f"F={f_stat:.3f}, TE={te:.3f} bits, lag={lag}, p={p_value:.4f}, "
                        f"confidence={confidence:.3f}"
                    )
    
    async def _detect_pre_echoes(self) -> None:
        """Detect pre-echo signals that predict target symbol moves."""
        for (source, target), relationship in list(self._causal_relationships.items()):
            if relationship.confidence < 0.5:  # Low confidence relationship
                continue
            
            # Get recent returns
            source_returns = list(self._returns_history[source])
            target_returns = list(self._returns_history[target])
            
            if len(source_returns) < relationship.lag_period + 10:
                continue
            
            # Recent source return (lag periods ago)
            recent_source_return = source_returns[-relationship.lag_period - 1] if \
                len(source_returns) > relationship.lag_period else 0.0
            
            # Recent target return (current)
            recent_target_return = target_returns[-1] if target_returns else 0.0
            
            # Predict based on source's past move
            predicted_direction = "buy" if recent_source_return > 0 else "sell"
            predicted_magnitude = abs(recent_source_return) * relationship.confidence
            
            # Only emit signal if prediction is strong enough
            if predicted_magnitude > 0.001:  # 0.1% minimum move
                pre_echo = PreEchoSignal(
                    target_symbol=target,
                    source_symbol=source,
                    predicted_direction=predicted_direction,
                    predicted_magnitude=predicted_magnitude,
                    confidence=relationship.confidence,
                    lag_ms=relationship.lag_period * 1000,  # Approximate (1 period ≈ 1 second)
                    granger_score=relationship.granger_causality,
                    transfer_entropy_score=relationship.transfer_entropy
                )
                
                self._pre_echo_signals.append(pre_echo)
                
                # Generate trading signal
                signal = Signal(
                    strategy_id="causal_inference",
                    symbol=target,
                    side=predicted_direction,
                    entry_price=0.0,  # Will be filled by executor
                    metadata={
                        "pre_echo": True,
                        "source_symbol": source,
                        "predicted_magnitude": predicted_magnitude,
                        "confidence": relationship.confidence,
                        "granger_score": relationship.granger_causality,
                        "transfer_entropy": relationship.transfer_entropy,
                        "lag_periods": relationship.lag_period
                    }
                )
                
                await self._bus.publish(
                    Event(
                        event_type=EventType.SIGNAL,
                        data={"signal": signal}
                    )
                )
                
                logger.info(
                    f"Pre-echo signal: {source} -> {target} {predicted_direction.upper()}, "
                    f"magnitude={predicted_magnitude:.4f}, confidence={relationship.confidence:.3f}"
                )
    
    def get_causal_relationships(self) -> Dict[Tuple[str, str], CausalRelationship]:
        """Get all detected causal relationships."""
        return self._causal_relationships.copy()
    
    def get_pre_echo_signals(self, limit: int = 10) -> List[PreEchoSignal]:
        """Get recent pre-echo signals."""
        return list(self._pre_echo_signals)[-limit:]
    
    def add_source_symbol(self, symbol: str) -> None:
        """Add a source symbol for pre-echo detection."""
        if symbol not in self._source_symbols:
            self._source_symbols.append(symbol)
            logger.info(f"Added source symbol: {symbol}")
    
    def remove_source_symbol(self, symbol: str) -> None:
        """Remove a source symbol."""
        if symbol in self._source_symbols:
            self._source_symbols.remove(symbol)
            # Remove related causal relationships
            keys_to_remove = [
                (s, t) for (s, t) in self._causal_relationships.keys()
                if s == symbol
            ]
            for key in keys_to_remove:
                del self._causal_relationships[key]
            logger.info(f"Removed source symbol: {symbol}")
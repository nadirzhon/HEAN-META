"""
Multimodal Liquidity Swarm: Process price, sentiment, on-chain whale movements,
and macro-economic data as a single, unified tensor.
"""

import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import asyncio

from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Signal, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MultimodalTensor:
    """Unified tensor representing all market data modalities."""
    timestamp: datetime
    symbol: str
    
    # Price data (1D)
    price_features: np.ndarray  # [price, returns, volatility, momentum, ...]
    
    # Social sentiment (1D)
    sentiment_features: np.ndarray  # [twitter_sentiment, reddit_sentiment, news_sentiment, ...]
    
    # On-chain data (1D)
    onchain_features: np.ndarray  # [whale_movements, exchange_flows, supply_change, ...]
    
    # Macro-economic data (1D)
    macro_features: np.ndarray  # [dxy, bond_yields, inflation_expectations, ...]
    
    # Unified tensor (concatenated)
    unified_tensor: np.ndarray  # Combined feature vector
    
    # Metadata
    confidence: float = 0.0
    modality_weights: Dict[str, float] = field(default_factory=dict)


@dataclass
class SentimentData:
    """Social sentiment data."""
    symbol: str
    twitter_sentiment: float = 0.0  # -1 (bearish) to +1 (bullish)
    reddit_sentiment: float = 0.0
    news_sentiment: float = 0.0
    volume_sentiment: float = 0.0  # Social volume
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class OnChainData:
    """On-chain whale movement and flow data."""
    symbol: str
    whale_inflow: float = 0.0  # Large wallet inflows (BTC equivalent)
    whale_outflow: float = 0.0  # Large wallet outflows
    exchange_inflow: float = 0.0  # Flow to exchanges (bearish)
    exchange_outflow: float = 0.0  # Flow from exchanges (bullish)
    supply_change: float = 0.0  # Supply change (for tokens)
    active_addresses: int = 0  # Active address count
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MacroData:
    """Macro-economic indicators."""
    dxy: float = 0.0  # Dollar Index
    bond_10y_yield: float = 0.0  # 10-year Treasury yield
    inflation_expectations: float = 0.0  # Inflation expectations
    vix: float = 0.0  # VIX (volatility index)
    gold_price: float = 0.0  # Gold price (USD)
    timestamp: datetime = field(default_factory=datetime.utcnow)


class MultimodalSwarm:
    """
    Multimodal Liquidity Swarm that processes all market data as a unified tensor.
    
    Features:
    - Price data: Standard market data (price, volume, volatility)
    - Social sentiment: Twitter, Reddit, news sentiment
    - On-chain data: Whale movements, exchange flows, supply metrics
    - Macro data: DXY, bond yields, inflation, VIX, gold
    - Unified tensor: All modalities combined with learned weights
    - Swarm intelligence: Multiple agents analyze different aspects of the tensor
    """
    
    def __init__(
        self,
        bus: EventBus,
        symbols: List[str] = None,
        window_size: int = 100,
        num_agents: int = 50
    ):
        """Initialize the multimodal swarm.
        
        Args:
            bus: Event bus for publishing signals
            symbols: Trading symbols to monitor
            window_size: Rolling window size for tensor history
            num_agents: Number of swarm agents analyzing the tensor
        """
        self._bus = bus
        self._symbols = symbols or ["BTCUSDT", "ETHUSDT"]
        self._window_size = window_size
        self._num_agents = num_agents
        
        # Data storage
        self._price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._sentiment_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._onchain_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._macro_history: deque = deque(maxlen=window_size)
        
        # Tensor storage
        self._tensor_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))

        # Latest swarm state per symbol (for dashboards)
        self._latest_state: Dict[str, Dict[str, Any]] = {}
        
        # Modality weights (learned from performance)
        self._modality_weights: Dict[str, float] = {
            'price': 0.4,      # Base weight
            'sentiment': 0.2,
            'onchain': 0.25,
            'macro': 0.15
        }
        
        # Swarm agents (each analyzes different aspects)
        self._agents: List[Dict[str, Any]] = []
        self._initialize_agents()
        
        self._running = False
        
        logger.info(
            f"Multimodal Swarm initialized: "
            f"symbols={self._symbols}, "
            f"window={window_size}, "
            f"agents={num_agents}"
        )
    
    def _initialize_agents(self) -> None:
        """Initialize swarm agents specializing in different aspects."""
        agent_types = [
            'price_momentum',
            'sentiment_contrarian',
            'whale_follower',
            'macro_correlator',
            'tensor_fusion'
        ]
        
        agents_per_type = self._num_agents // len(agent_types)
        
        for agent_type in agent_types:
            for i in range(agents_per_type):
                agent = {
                    'id': f"{agent_type}_{i}",
                    'type': agent_type,
                    'confidence': 0.5,
                    'performance_history': deque(maxlen=100),
                    'specialization': self._get_specialization(agent_type)
                }
                self._agents.append(agent)
    
    def _get_specialization(self, agent_type: str) -> Dict[str, float]:
        """Get modality weights for agent specialization."""
        if agent_type == 'price_momentum':
            return {'price': 0.8, 'sentiment': 0.1, 'onchain': 0.05, 'macro': 0.05}
        elif agent_type == 'sentiment_contrarian':
            return {'price': 0.2, 'sentiment': 0.7, 'onchain': 0.05, 'macro': 0.05}
        elif agent_type == 'whale_follower':
            return {'price': 0.2, 'sentiment': 0.1, 'onchain': 0.6, 'macro': 0.1}
        elif agent_type == 'macro_correlator':
            return {'price': 0.2, 'sentiment': 0.1, 'onchain': 0.1, 'macro': 0.6}
        else:  # tensor_fusion
            return self._modality_weights.copy()
    
    async def start(self) -> None:
        """Start the multimodal swarm."""
        self._running = True
        self._bus.subscribe(EventType.TICK, self._handle_tick)
        
        # Start data ingestion (in production, would connect to external APIs)
        asyncio.create_task(self._ingest_sentiment_data())
        asyncio.create_task(self._ingest_onchain_data())
        asyncio.create_task(self._ingest_macro_data())
        
        # Start tensor fusion
        asyncio.create_task(self._tensor_fusion_loop())
        
        # Start swarm analysis
        asyncio.create_task(self._swarm_analysis_loop())
        
        logger.info("Multimodal Swarm started")
    
    async def stop(self) -> None:
        """Stop the multimodal swarm."""
        self._running = False
        self._bus.unsubscribe(EventType.TICK, self._handle_tick)
        logger.info("Multimodal Swarm stopped")
    
    async def _handle_tick(self, event: Event) -> None:
        """Handle tick events to update price data."""
        tick: Tick = event.data["tick"]
        symbol = tick.symbol
        
        if symbol not in self._symbols:
            return
        
        # Update price history
        self._price_history[symbol].append({
            'price': tick.price,
            'volume': tick.volume,
            'timestamp': tick.timestamp
        })
        
        # Calculate price features
        price_features = self._extract_price_features(symbol)
        
        # Create price-only tensor (will be fused later)
        if price_features is not None:
            await self._update_tensor(symbol, price_features=price_features)
    
    async def _ingest_sentiment_data(self) -> None:
        """Ingest social sentiment data (simulated - in production, would use APIs)."""
        while self._running:
            await asyncio.sleep(30.0)  # Update every 30 seconds
            
            # Simulate sentiment data (in production, fetch from Twitter/Reddit APIs)
            for symbol in self._symbols:
                sentiment = SentimentData(
                    symbol=symbol,
                    twitter_sentiment=np.random.uniform(-0.5, 0.5),  # Simulated
                    reddit_sentiment=np.random.uniform(-0.3, 0.3),
                    news_sentiment=np.random.uniform(-0.4, 0.4),
                    volume_sentiment=np.random.uniform(0.0, 1.0)
                )
                
                self._sentiment_history[symbol].append(sentiment)
                
                # Extract sentiment features
                sentiment_features = self._extract_sentiment_features(symbol)
                if sentiment_features is not None:
                    await self._update_tensor(symbol, sentiment_features=sentiment_features)
    
    async def _ingest_onchain_data(self) -> None:
        """Ingest on-chain whale movement data (simulated)."""
        while self._running:
            await asyncio.sleep(60.0)  # Update every minute
            
            # Simulate on-chain data (in production, fetch from blockchain APIs)
            for symbol in self._symbols:
                # For BTCUSDT, simulate BTC on-chain metrics
                if symbol.startswith("BTC"):
                    onchain = OnChainData(
                        symbol=symbol,
                        whale_inflow=np.random.uniform(0, 1000),  # Simulated BTC
                        whale_outflow=np.random.uniform(0, 1000),
                        exchange_inflow=np.random.uniform(0, 500),
                        exchange_outflow=np.random.uniform(0, 500),
                        supply_change=np.random.uniform(-0.001, 0.001),
                        active_addresses=np.random.randint(800000, 1200000)
                    )
                else:
                    # For other symbols, use simplified metrics
                    onchain = OnChainData(
                        symbol=symbol,
                        whale_inflow=np.random.uniform(0, 100),
                        whale_outflow=np.random.uniform(0, 100),
                        exchange_inflow=np.random.uniform(0, 50),
                        exchange_outflow=np.random.uniform(0, 50)
                    )
                
                self._onchain_history[symbol].append(onchain)
                
                # Extract on-chain features
                onchain_features = self._extract_onchain_features(symbol)
                if onchain_features is not None:
                    await self._update_tensor(symbol, onchain_features=onchain_features)
    
    async def _ingest_macro_data(self) -> None:
        """Ingest macro-economic data (simulated)."""
        while self._running:
            await asyncio.sleep(300.0)  # Update every 5 minutes (macro data changes slower)
            
            # Simulate macro data (in production, fetch from economic data APIs)
            macro = MacroData(
                dxy=np.random.uniform(100, 110),
                bond_10y_yield=np.random.uniform(3.5, 4.5),
                inflation_expectations=np.random.uniform(2.0, 3.0),
                vix=np.random.uniform(15, 25),
                gold_price=np.random.uniform(1900, 2100)
            )
            
            self._macro_history.append(macro)
            
            # Extract macro features
            macro_features = self._extract_macro_features()
            if macro_features is not None:
                # Update all symbols with macro data
                for symbol in self._symbols:
                    await self._update_tensor(symbol, macro_features=macro_features)
    
    def _extract_price_features(self, symbol: str) -> Optional[np.ndarray]:
        """Extract price-based features."""
        if len(self._price_history[symbol]) < 10:
            return None
        
        prices = [d['price'] for d in list(self._price_history[symbol])]
        
        # Calculate features
        current_price = prices[-1]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        # Volatility (rolling std)
        volatility = np.std(returns[-20:]) * np.sqrt(252) if len(returns) >= 20 else 0.0
        
        # Momentum (rate of change)
        momentum = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0.0
        
        # Volume-weighted price (simplified)
        volumes = [d.get('volume', 0.0) for d in list(self._price_history[symbol])]
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else 1.0
        
        return np.array([
            current_price / 100000.0,  # Normalized price
            np.mean(returns[-10:]) if returns else 0.0,  # Mean return
            volatility,  # Volatility
            momentum,  # Momentum
            avg_volume / 1000000.0  # Normalized volume
        ])
    
    def _extract_sentiment_features(self, symbol: str) -> Optional[np.ndarray]:
        """Extract social sentiment features."""
        if len(self._sentiment_history[symbol]) == 0:
            return None
        
        latest = self._sentiment_history[symbol][-1]
        
        return np.array([
            latest.twitter_sentiment,
            latest.reddit_sentiment,
            latest.news_sentiment,
            latest.volume_sentiment
        ])
    
    def _extract_onchain_features(self, symbol: str) -> Optional[np.ndarray]:
        """Extract on-chain whale movement features."""
        if len(self._onchain_history[symbol]) == 0:
            return None
        
        latest = self._onchain_history[symbol][-1]
        
        # Net whale flow
        net_whale_flow = latest.whale_inflow - latest.whale_outflow
        
        # Net exchange flow (negative = bullish, positive = bearish)
        net_exchange_flow = latest.exchange_outflow - latest.exchange_inflow
        
        # Normalize by recent average (simplified)
        return np.array([
            net_whale_flow / 1000.0,  # Normalized
            net_exchange_flow / 500.0,
            latest.supply_change * 1000.0,  # Scaled
            latest.active_addresses / 1000000.0  # Normalized
        ])
    
    def _extract_macro_features(self) -> Optional[np.ndarray]:
        """Extract macro-economic features."""
        if len(self._macro_history) == 0:
            return None
        
        latest = self._macro_history[-1]
        
        return np.array([
            (latest.dxy - 105.0) / 10.0,  # Normalized DXY (centered around 105)
            (latest.bond_10y_yield - 4.0) / 2.0,  # Normalized yield
            (latest.inflation_expectations - 2.5) / 1.0,  # Normalized inflation
            (latest.vix - 20.0) / 10.0,  # Normalized VIX
            (latest.gold_price - 2000.0) / 200.0  # Normalized gold
        ])
    
    async def _update_tensor(
        self,
        symbol: str,
        price_features: Optional[np.ndarray] = None,
        sentiment_features: Optional[np.ndarray] = None,
        onchain_features: Optional[np.ndarray] = None,
        macro_features: Optional[np.ndarray] = None
    ) -> None:
        """Update unified tensor with new modality data."""
        # Get or create current tensor
        current_tensors = list(self._tensor_history[symbol])
        if current_tensors and current_tensors[-1].timestamp > datetime.utcnow() - timedelta(seconds=5):
            tensor = current_tensors[-1]
        else:
            # Create new tensor
            tensor = MultimodalTensor(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                price_features=np.zeros(5),
                sentiment_features=np.zeros(4),
                onchain_features=np.zeros(4),
                macro_features=np.zeros(5),
                unified_tensor=np.zeros(18)  # 5+4+4+5 = 18
            )
        
        # Update features
        if price_features is not None:
            tensor.price_features = price_features
        if sentiment_features is not None:
            tensor.sentiment_features = sentiment_features
        if onchain_features is not None:
            tensor.onchain_features = onchain_features
        if macro_features is not None:
            tensor.macro_features = macro_features
        
        # Create unified tensor with weighted combination
        unified = np.concatenate([
            tensor.price_features * self._modality_weights['price'],
            tensor.sentiment_features * self._modality_weights['sentiment'],
            tensor.onchain_features * self._modality_weights['onchain'],
            tensor.macro_features * self._modality_weights['macro']
        ])
        
        tensor.unified_tensor = unified
        tensor.modality_weights = self._modality_weights.copy()
        
        # Store tensor
        self._tensor_history[symbol].append(tensor)
    
    async def _tensor_fusion_loop(self) -> None:
        """Continuously fuse tensors from all modalities."""
        while self._running:
            await asyncio.sleep(5.0)  # Fuse every 5 seconds
            
            for symbol in self._symbols:
                # Get latest features
                price_features = self._extract_price_features(symbol)
                sentiment_features = self._extract_sentiment_features(symbol)
                onchain_features = self._extract_onchain_features(symbol)
                macro_features = self._extract_macro_features()
                
                # Update tensor if any features available
                if price_features is not None or sentiment_features is not None or \
                   onchain_features is not None or macro_features is not None:
                    await self._update_tensor(
                        symbol,
                        price_features=price_features,
                        sentiment_features=sentiment_features,
                        onchain_features=onchain_features,
                        macro_features=macro_features
                    )
    
    async def _swarm_analysis_loop(self) -> None:
        """Swarm agents analyze unified tensors and generate signals."""
        while self._running:
            await asyncio.sleep(10.0)  # Analyze every 10 seconds
            
            for symbol in self._symbols:
                if len(self._tensor_history[symbol]) == 0:
                    continue
                
                # Get latest unified tensor
                latest_tensor = self._tensor_history[symbol][-1]
                
                # Get agent votes
                buy_votes = 0
                sell_votes = 0
                total_confidence = 0.0
                
                for agent in self._agents:
                    # Agent analyzes tensor based on specialization
                    signal, confidence = self._agent_analyze(agent, latest_tensor)
                    
                    if signal == "buy":
                        buy_votes += 1
                    elif signal == "sell":
                        sell_votes += 1
                    
                    total_confidence += confidence
                    
                    # Update agent performance (simplified)
                    agent['performance_history'].append(confidence)
                    agent['confidence'] = np.mean(list(agent['performance_history'])) if agent['performance_history'] else 0.5
                
                # Consensus decision (>60% required)
                total_votes = buy_votes + sell_votes
                if total_votes > 0:
                    avg_confidence = total_confidence / total_votes
                    buy_percentage = buy_votes / total_votes
                    sell_percentage = sell_votes / total_votes

                    consensus_percentage = max(buy_percentage, sell_percentage) * 100.0
                    execution_signal_strength = avg_confidence * max(buy_percentage, sell_percentage)
                    consensus_reached = (
                        (buy_percentage > 0.6 or sell_percentage > 0.6)
                        and avg_confidence > 0.6
                    )

                    self._latest_state[symbol] = {
                        "symbol": symbol,
                        "consensus_percentage": consensus_percentage,
                        "buy_votes": buy_votes,
                        "sell_votes": sell_votes,
                        "total_agents": len(self._agents),
                        "average_confidence": avg_confidence,
                        "execution_signal_strength": execution_signal_strength,
                        "consensus_reached": consensus_reached,
                        "timestamp": datetime.utcnow(),
                    }
                    
                    if buy_percentage > 0.6 and avg_confidence > 0.6:
                        # Generate buy signal
                        signal = Signal(
                            strategy_id="multimodal_swarm",
                            symbol=symbol,
                            side="buy",
                            entry_price=0.0,  # Will be filled
                            metadata={
                                "multimodal": True,
                                "buy_votes": buy_votes,
                                "sell_votes": sell_votes,
                                "consensus_percentage": buy_percentage,
                                "confidence": avg_confidence,
                                "tensor_features": latest_tensor.unified_tensor.tolist()
                            }
                        )
                        
                        await self._bus.publish(
                            Event(
                                event_type=EventType.SIGNAL,
                                data={"signal": signal}
                            )
                        )
                        
                        logger.info(
                            f"Multimodal Swarm signal: {symbol} BUY "
                            f"(consensus={buy_percentage:.1%}, confidence={avg_confidence:.3f})"
                        )
                    
                    elif sell_percentage > 0.6 and avg_confidence > 0.6:
                        # Generate sell signal
                        signal = Signal(
                            strategy_id="multimodal_swarm",
                            symbol=symbol,
                            side="sell",
                            entry_price=0.0,
                            metadata={
                                "multimodal": True,
                                "buy_votes": buy_votes,
                                "sell_votes": sell_votes,
                                "consensus_percentage": sell_percentage,
                                "confidence": avg_confidence,
                                "tensor_features": latest_tensor.unified_tensor.tolist()
                            }
                        )
                        
                        await self._bus.publish(
                            Event(
                                event_type=EventType.SIGNAL,
                                data={"signal": signal}
                            )
                        )
                        
                        logger.info(
                            f"Multimodal Swarm signal: {symbol} SELL "
                            f"(consensus={sell_percentage:.1%}, confidence={avg_confidence:.3f})"
                        )
    
    def _agent_analyze(
        self,
        agent: Dict[str, Any],
        tensor: MultimodalTensor
    ) -> Tuple[str, float]:
        """Agent analyzes tensor based on specialization."""
        specialization = agent['specialization']
        
        # Weight tensor components by specialization
        weighted_tensor = np.concatenate([
            tensor.price_features * specialization['price'],
            tensor.sentiment_features * specialization['sentiment'],
            tensor.onchain_features * specialization['onchain'],
            tensor.macro_features * specialization['macro']
        ])
        
        # Simple analysis: positive weighted sum = buy, negative = sell
        signal_strength = np.sum(weighted_tensor)
        
        # Map to confidence (0-1)
        confidence = min(1.0, abs(signal_strength) / 2.0)
        
        signal = "buy" if signal_strength > 0 else "sell"

        return signal, confidence

    def get_latest_state(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return latest swarm state for a symbol (or any symbol)."""
        if not self._latest_state:
            return None
        if symbol and symbol in self._latest_state:
            return self._latest_state[symbol]
        # Return most recent by timestamp
        latest = max(self._latest_state.values(), key=lambda item: item.get("timestamp"))
        return latest
    
    def get_tensor_history(self, symbol: str, limit: int = 10) -> List[MultimodalTensor]:
        """Get recent tensor history for a symbol."""
        return list(self._tensor_history[symbol])[-limit:]
    
    def get_modality_weights(self) -> Dict[str, float]:
        """Get current modality weights."""
        return self._modality_weights.copy()
    
    def update_modality_weights(self, weights: Dict[str, float]) -> None:
        """Update modality weights (learned from performance)."""
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            self._modality_weights = {k: v / total for k, v in weights.items()}
            logger.info(f"Updated modality weights: {self._modality_weights}")

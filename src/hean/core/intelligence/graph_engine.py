"""Python wrapper for C++ Graph Engine with real-time adjacency matrix."""


try:
    # Try to import the compiled C++ module
    import graph_engine_py  # type: ignore
    _CPP_AVAILABLE = True
except ImportError:
    _CPP_AVAILABLE = False
    # Fallback to Python implementation
    from hean.core.intelligence.correlation_engine import CorrelationEngine


from hean.core.bus import EventBus
from hean.core.types import Event, EventType, Tick
from hean.logging import get_logger

logger = get_logger(__name__)


class GraphEngineWrapper:
    """Wrapper for C++ Graph Engine with real-time adjacency matrix and lead-lag detection."""

    def __init__(self, bus: EventBus, symbols: list[str] | None = None, window_size: int = 100):
        """Initialize the graph engine wrapper.

        Args:
            bus: Event bus for subscribing to ticks
            symbols: List of symbols to track (defaults to top 50+ crypto assets)
            window_size: Rolling window size for correlation calculation
        """
        self._bus = bus
        self._symbols = symbols or self._get_top_50_assets()
        self._window_size = window_size
        self._cpp_available = _CPP_AVAILABLE

        if self._cpp_available:
            try:
                self._engine = graph_engine_py.GraphEngine(window_size=window_size)
                logger.info("C++ Graph Engine initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize C++ engine: {e}. Falling back to Python.")
                self._cpp_available = False
                self._engine = CorrelationEngine(bus, self._symbols)
        else:
            logger.warning("C++ Graph Engine not available. Using Python CorrelationEngine.")
            self._engine = CorrelationEngine(bus, self._symbols)

        # Initialize assets in C++ engine
        if self._cpp_available:
            for symbol in self._symbols:
                self._engine.add_asset(symbol)
            logger.info(f"Graph Engine initialized with {len(self._symbols)} assets")

    def _get_top_50_assets(self) -> list[str]:
        """Get top 50+ crypto assets by market cap."""
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT",
            "DOTUSDT", "MATICUSDT", "SHIBUSDT", "LTCUSDT", "UNIUSDT",
            "ATOMUSDT", "ETCUSDT", "XLMUSDT", "BCHUSDT", "FILUSDT",
            "NEARUSDT", "APTUSDT", "HBARUSDT", "VETUSDT", "ICPUSDT",
            "OPUSDT", "ARBUSDT", "MKRUSDT", "INJUSDT", "ALGOUSDT",
            "AAVEUSDT", "GRTUSDT", "SANDUSDT", "MANAUSDT", "THETAUSDT",
            "AXSUSDT", "FLOWUSDT", "EGLDUSDT", "XTZUSDT", "EOSUSDT",
            "FTMUSDT", "CHZUSDT", "ENJUSDT", "ZILUSDT", "ROSEUSDT",
            "WAVESUSDT", "ZECUSDT", "BATUSDT", "ZENUSDT", "CRVUSDT",
            "1INCHUSDT", "SNXUSDT", "COMPUSDT", "YFIUSDT", "SUSHIUSDT"
        ]

    async def start(self) -> None:
        """Start the graph engine."""
        if self._cpp_available:
            self._bus.subscribe(EventType.TICK, self._handle_tick_cpp)
        else:
            # Use existing CorrelationEngine start method
            if hasattr(self._engine, 'start'):
                await self._engine.start()
        logger.info("Graph Engine started")

    async def stop(self) -> None:
        """Stop the graph engine."""
        if self._cpp_available:
            self._bus.unsubscribe(EventType.TICK, self._handle_tick_cpp)
        else:
            if hasattr(self._engine, 'stop'):
                await self._engine.stop()
        logger.info("Graph Engine stopped")

    async def _handle_tick_cpp(self, event: Event) -> None:
        """Handle tick events for C++ engine."""
        tick: Tick = event.data["tick"]

        if tick.symbol not in self._symbols:
            return

        # Update price in C++ engine
        import time
        timestamp_ns = int(time.time_ns())
        self._engine.update_price(tick.symbol, float(tick.price), timestamp_ns)

        # Recalculate matrix periodically (every 10 ticks per symbol)
        if not hasattr(self, '_tick_counts'):
            self._tick_counts = {}
        self._tick_counts[tick.symbol] = self._tick_counts.get(tick.symbol, 0) + 1

        if self._tick_counts[tick.symbol] % 10 == 0:
            self._engine.recalculate()

    def get_feature_vector(self, size: int = 5000) -> list[float]:
        """Get high-dimensional feature vector for neural network input.

        Returns flattened adjacency matrix + metadata as a feature vector.
        """
        if self._cpp_available:
            try:
                arr = self._engine.get_feature_vector(size)
                # Handle numpy array or list
                if hasattr(arr, 'tolist'):
                    return arr.tolist()
                return list(arr) if isinstance(arr, (list, tuple)) else [0.0] * size
            except Exception as e:
                logger.warning(f"Failed to get feature vector from C++ engine: {e}")
                return [0.0] * size
        else:
            # Fallback: construct from Python correlation engine
            if hasattr(self._engine, 'get_correlation_matrix'):
                matrix = self._engine.get_correlation_matrix()
                # Flatten matrix (simplified)
                features = []
                symbols = list(self._symbols)
                for i, sym_a in enumerate(symbols):
                    for _j, sym_b in enumerate(symbols[i:], start=i):
                        features.append(matrix.get((sym_a, sym_b), 0.0))
                return features[:size]
            return [0.0] * size

    def get_current_leader(self) -> str | None:
        """Get current market leader (asset that leads others)."""
        if self._cpp_available:
            leader = self._engine.get_current_leader()
            return leader if leader else None
        return None

    def get_laggards(self, max_count: int = 10) -> list[str]:
        """Get laggard assets (those following the leader)."""
        # For now, return empty list - C++ function needs array handling
        # This would require additional C++ binding work
        return []

    def get_correlation(self, symbol_a: str, symbol_b: str) -> float:
        """Get correlation between two assets."""
        if self._cpp_available:
            return self._engine.get_correlation(symbol_a, symbol_b)
        elif hasattr(self._engine, 'get_correlation'):
            return self._engine.get_correlation(symbol_a, symbol_b)
        return 0.0

    def get_lead_lag(self, symbol_a: str, symbol_b: str) -> float:
        """Get lead-lag relationship.

        Returns:
            Positive value = a leads b
            Negative value = b leads a
            Zero = no relationship
        """
        if self._cpp_available:
            return self._engine.get_lead_lag(symbol_a, symbol_b)
        return 0.0

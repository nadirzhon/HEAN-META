"""Physics-Aware Position Sizer.

Adjusts position sizes based on market thermodynamics:
- Phase alignment: boost when trading with the phase
- Temperature: reduce in extreme heat
- Entropy: reduce in chaos
- Volatility: inverse scaling for mean reversion
"""

from collections import deque

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class PhysicsAwarePositionSizer:
    """Calculate position size multipliers based on physics state."""

    def __init__(self, bus: EventBus, enabled: bool = True) -> None:
        """Initialize the physics-aware position sizer.

        Args:
            bus: Event bus for subscribing to PHYSICS_UPDATE events
            enabled: Whether physics-based sizing is enabled
        """
        self._bus = bus
        self._enabled = enabled
        self._physics_cache: dict[str, dict] = {}
        self._volatility_cache: dict[str, deque] = {}
        self._volatility_window = 100
        self._running = False

    async def start(self) -> None:
        """Start the physics position sizer."""
        if not self._enabled:
            logger.info("PhysicsAwarePositionSizer disabled (PHYSICS_SIZING_ENABLED=false)")
            return

        self._running = True
        self._bus.subscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        logger.info("PhysicsAwarePositionSizer started")

    async def stop(self) -> None:
        """Stop the physics position sizer."""
        if not self._enabled:
            return

        self._running = False
        self._bus.unsubscribe(EventType.PHYSICS_UPDATE, self._handle_physics_update)
        logger.info("PhysicsAwarePositionSizer stopped")

    async def _handle_physics_update(self, event: Event) -> None:
        """Cache physics state for each symbol."""
        if not self._running:
            return

        symbol = event.data.get("symbol")
        physics = event.data.get("physics")

        if symbol and physics:
            self._physics_cache[symbol] = physics

            # Track volatility for reference calculation
            temperature = physics.get("temperature", 0.0)
            if symbol not in self._volatility_cache:
                self._volatility_cache[symbol] = deque(maxlen=self._volatility_window)
            self._volatility_cache[symbol].append(temperature)

    def calculate_size_multiplier(
        self,
        symbol: str,
        signal_side: str,
        market_phase: str | None = None,
        temperature: float | None = None,
        entropy: float | None = None,
        volatility: float | None = None,
    ) -> float:
        """Calculate position size multiplier based on physics state.

        Args:
            symbol: Trading symbol
            signal_side: "buy" or "sell"
            market_phase: Market phase (accumulation/markup/distribution/markdown)
            temperature: Market temperature [0, 1]
            entropy: Market entropy [0, 1]
            volatility: Current volatility (if None, uses cached temperature)

        Returns:
            Size multiplier in range [0.25, 2.0]
        """
        if not self._enabled:
            return 1.0

        # Get physics state from cache if not provided
        physics = self._physics_cache.get(symbol, {})

        if market_phase is None:
            market_phase = physics.get("phase", "unknown")
        if temperature is None:
            temperature = physics.get("temperature", 0.5)
        if entropy is None:
            entropy = physics.get("entropy", 0.5)
        if volatility is None:
            volatility = temperature  # Use temperature as volatility proxy

        # Start with neutral multiplier
        multiplier = 1.0

        # 1. Phase alignment
        phase_mult = self._calculate_phase_multiplier(signal_side, market_phase)
        multiplier *= phase_mult

        # 2. Temperature scaling
        temp_mult = self._calculate_temperature_multiplier(temperature)
        multiplier *= temp_mult

        # 3. Entropy scaling (linear decay)
        entropy_mult = max(0.5, 1.0 - entropy * 0.5)  # 1.0 → 0.5 as entropy goes 0 → 1
        multiplier *= entropy_mult

        # 4. Volatility scaling (inverse for mean reversion)
        vol_mult = self._calculate_volatility_multiplier(symbol, volatility)
        multiplier *= vol_mult

        # Clamp to safe range
        multiplier = max(0.25, min(2.0, multiplier))

        logger.debug(
            f"[PhysicsPositionSizer] {symbol} {signal_side}: "
            f"phase={market_phase} ({phase_mult:.2f}x), "
            f"temp={temperature:.2f} ({temp_mult:.2f}x), "
            f"entropy={entropy:.2f} ({entropy_mult:.2f}x), "
            f"vol ({vol_mult:.2f}x) → final={multiplier:.2f}x"
        )

        return multiplier

    def _calculate_phase_multiplier(self, signal_side: str, market_phase: str) -> float:
        """Calculate size multiplier based on phase alignment.

        Buy signals:
        - accumulation: 1.5x (best for buying)
        - markup: 1.3x (trend continuation)
        - distribution: 0.6x (counter-trend)
        - markdown: 0.4x (worst for buying)

        Sell signals (inverse):
        - distribution: 1.5x (best for selling)
        - markdown: 1.3x (trend continuation)
        - accumulation: 0.6x (counter-trend)
        - markup: 0.4x (worst for selling)
        """
        phase_map_buy = {
            "accumulation": 1.5,
            "markup": 1.3,
            "distribution": 0.6,
            "markdown": 0.4,
            "unknown": 0.8,
        }

        phase_map_sell = {
            "distribution": 1.5,
            "markdown": 1.3,
            "accumulation": 0.6,
            "markup": 0.4,
            "unknown": 0.8,
        }

        if signal_side.lower() == "buy":
            return phase_map_buy.get(market_phase.lower(), 0.8)
        else:
            return phase_map_sell.get(market_phase.lower(), 0.8)

    def _calculate_temperature_multiplier(self, temperature: float) -> float:
        """Calculate size multiplier based on market temperature.

        Temperature regimes:
        - < 0.3 (COLD): 1.3x (stable, low risk)
        - 0.3-0.5 (WARM): 1.0x (normal)
        - 0.5-0.7 (HOT): 0.8x (elevated risk)
        - >= 0.7 (EXTREME): 0.5x (extreme volatility)
        """
        if temperature < 0.3:
            return 1.3
        elif temperature < 0.5:
            return 1.0
        elif temperature < 0.7:
            return 0.8
        else:
            return 0.5

    def _calculate_volatility_multiplier(self, symbol: str, volatility: float) -> float:
        """Calculate size multiplier based on volatility (inverse scaling).

        Uses reference volatility from recent history to scale inversely:
        - Low current vol vs reference → increase size
        - High current vol vs reference → decrease size

        This creates mean-reversion bias in sizing.
        """
        vol_history = self._volatility_cache.get(symbol)
        if not vol_history or len(vol_history) < 10:
            return 1.0

        # Calculate reference volatility (median of recent values)
        sorted_vols = sorted(vol_history)
        reference_vol = sorted_vols[len(sorted_vols) // 2]

        if reference_vol == 0:
            return 1.0

        # Inverse scaling: high vol → reduce size
        ratio = reference_vol / max(volatility, 0.01)

        # Clamp ratio to reasonable range [0.5, 1.5]
        ratio = max(0.5, min(1.5, ratio))

        return ratio

    def get_physics_state(self, symbol: str) -> dict | None:
        """Get cached physics state for a symbol."""
        return self._physics_cache.get(symbol)

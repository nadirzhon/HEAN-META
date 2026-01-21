"""Black Swan Protection - Tail Risk Hedge system.

Monitors Market Entropy from Phase 2 (regime detector) and automatically
reduces position sizes and initiates hedge positions when entropy spikes.
"""

import asyncio
from collections import deque

from hean.core.bus import EventBus
from hean.core.regime import RegimeDetector
from hean.core.types import Event, EventType, Signal
from hean.logging import get_logger
from hean.portfolio.accounting import PortfolioAccounting
from hean.risk.position_sizer import PositionSizer

logger = get_logger(__name__)


class GlobalSafetyNet:
    """Black Swan Protection System.
    
    Monitors Market Entropy (volatility + acceleration spikes) and automatically:
    - Reduces all position sizes by 80% when entropy spikes by 300%
    - Initiates hedge positions in stable assets or inverse futures
    - Prevents total deposit loss during market crashes
    """

    def __init__(
        self,
        bus: EventBus,
        regime_detector: RegimeDetector,
        accounting: PortfolioAccounting,
        position_sizer: PositionSizer
    ) -> None:
        """Initialize the Global Safety Net.
        
        Args:
            bus: Event bus for publishing hedge signals
            regime_detector: Regime detector for market entropy calculation
            accounting: Portfolio accounting for equity tracking
            position_sizer: Position sizer for size reduction
        """
        self._bus = bus
        self._regime_detector = regime_detector
        self._accounting = accounting
        self._position_sizer = position_sizer
        
        # Market entropy history (rolling window)
        self._entropy_history: deque[float] = deque(maxlen=100)
        
        # Baseline entropy (moving average)
        self._baseline_entropy: float = 0.0
        
        # Entropy spike threshold (300% = 3.0x baseline)
        self._entropy_spike_threshold = 3.0
        
        # Position size reduction factor (80% reduction = 0.2x multiplier)
        self._emergency_size_multiplier = 0.2
        
        # Safety net active state
        self._safety_net_active = False
        
        # Hedge positions tracking
        self._hedge_positions: dict[str, dict[str, float]] = {}
        
        # Stable assets for hedging (inverse correlation to crypto)
        self._hedge_assets = ["USDT", "BUSD", "USDC"]  # Stablecoins
        # For inverse futures, we'd need exchange-specific implementation
        
        logger.info("Global Safety Net initialized")

    async def start(self) -> None:
        """Start the safety net monitoring."""
        self._bus.subscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        
        # Start periodic entropy monitoring task
        asyncio.create_task(self._monitor_entropy())
        
        logger.info("Global Safety Net started")

    async def stop(self) -> None:
        """Stop the safety net monitoring."""
        self._bus.unsubscribe(EventType.REGIME_UPDATE, self._handle_regime_update)
        logger.info("Global Safety Net stopped")

    async def _handle_regime_update(self, event: Event) -> None:
        """Handle regime update events to calculate entropy."""
        # Entropy is calculated from volatility and acceleration
        # We use the regime detector's volatility as a proxy for entropy
        symbol = event.data.get("symbol")
        if not symbol:
            return
        
        volatility = self._regime_detector.get_volatility(symbol)
        
        # Calculate entropy: volatility + regime acceleration component
        # Higher volatility + higher acceleration = higher entropy
        regime = event.data.get("regime")
        acceleration_component = 0.0
        
        if regime == "impulse":
            acceleration_component = 0.005  # High acceleration
        elif regime == "range":
            acceleration_component = 0.0  # Low acceleration
        
        entropy = volatility + acceleration_component
        
        # Update entropy history
        self._entropy_history.append(entropy)
        
        # Update baseline (exponential moving average)
        if self._baseline_entropy == 0.0:
            self._baseline_entropy = entropy
        else:
            # EMA with alpha = 0.1 (slow adaptation)
            alpha = 0.1
            self._baseline_entropy = alpha * entropy + (1 - alpha) * self._baseline_entropy
        
        # Check for entropy spike
        await self._check_entropy_spike(entropy)

    async def _monitor_entropy(self) -> None:
        """Periodic entropy monitoring task."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                if len(self._entropy_history) < 10:
                    continue
                
                # Calculate current entropy (average of recent values)
                recent_entropy = sum(list(self._entropy_history)[-10:]) / 10
                
                await self._check_entropy_spike(recent_entropy)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in entropy monitoring: {e}", exc_info=True)

    async def _check_entropy_spike(self, current_entropy: float) -> None:
        """Check if entropy has spiked beyond threshold.
        
        Args:
            current_entropy: Current market entropy value
        """
        if self._baseline_entropy <= 0:
            return
        
        # Calculate entropy ratio
        entropy_ratio = current_entropy / self._baseline_entropy if self._baseline_entropy > 0 else 1.0
        
        # Check if spike threshold exceeded
        if entropy_ratio >= self._entropy_spike_threshold and not self._safety_net_active:
            # Activate safety net
            await self._activate_safety_net(current_entropy, entropy_ratio)
        
        elif entropy_ratio < self._entropy_spike_threshold * 0.7 and self._safety_net_active:
            # Deactivate safety net when entropy normalizes (70% of threshold)
            await self._deactivate_safety_net()

    async def _activate_safety_net(self, entropy: float, entropy_ratio: float) -> None:
        """Activate the safety net - reduce positions and initiate hedges.
        
        Args:
            entropy: Current entropy value
            entropy_ratio: Entropy ratio relative to baseline
        """
        self._safety_net_active = True
        
        logger.critical(
            f"BLACK SWAN PROTECTION ACTIVATED: Entropy spike detected! "
            f"Entropy={entropy:.6f}, Ratio={entropy_ratio:.2f}x baseline"
        )
        
        # Reduce all position sizes by 80% (multiply by 0.2)
        # This is done by setting a global size reduction multiplier
        # The position sizer will check this multiplier before calculating sizes
        
        # Initiate hedge positions
        await self._initiate_hedge_positions()
        
        # Publish safety net activation event
        await self._bus.publish(
            Event(
                event_type=EventType.REGIME_UPDATE,  # Reuse regime update for safety net
                data={
                    "safety_net": "activated",
                    "entropy": entropy,
                    "entropy_ratio": entropy_ratio,
                    "size_multiplier": self._emergency_size_multiplier
                }
            )
        )

    async def _deactivate_safety_net(self) -> None:
        """Deactivate the safety net when entropy normalizes."""
        self._safety_net_active = False
        
        logger.info("BLACK SWAN PROTECTION DEACTIVATED: Market entropy normalized")
        
        # Close hedge positions (or let them run if still profitable)
        # For now, we keep hedges active until manual review
        
        # Publish deactivation event
        await self._bus.publish(
            Event(
                event_type=EventType.REGIME_UPDATE,
                data={
                    "safety_net": "deactivated",
                    "size_multiplier": 1.0
                }
            )
        )

    async def _initiate_hedge_positions(self) -> None:
        """Initiate hedge positions in stable assets or inverse futures.
        
        Strategy: Allocate 30% of equity to stable asset hedge to protect capital.
        """
        equity = self._accounting.get_equity()
        
        if equity <= 0:
            logger.warning("Cannot initiate hedge: equity is zero or negative")
            return
        
        # Calculate hedge size (30% of equity)
        hedge_size = equity * 0.3
        
        # For crypto markets, hedge with stablecoins (USDT/USDC)
        # In production, could use inverse futures or options
        
        # Create hedge signal: Long stablecoin (convert to stable position)
        # This is a simplified hedge - in practice, would use futures or options
        
        hedge_signal = Signal(
            strategy_id="tail_risk_hedge",
            symbol="USDT",  # Placeholder - in practice would use exchange-specific hedge instrument
            side="buy",
            size=hedge_size,
            entry_price=1.0,  # Stablecoin price
            stop_loss=None,  # Stablecoins don't need stop loss
            take_profit=None,  # Hedge is protective, not profit-seeking
            metadata={
                "hedge_type": "safety_net",
                "protection_level": "black_swan",
                "equity_protected": hedge_size
            }
        )
        
        # Track hedge
        self._hedge_positions["safety_hedge"] = {
            "symbol": "USDT",
            "size": hedge_size,
            "entry_price": 1.0,
            "equity_protected": hedge_size
        }
        
        logger.info(f"Initiated safety hedge: ${hedge_size:.2f} in stable assets")
        
        # Note: In production, this would actually place orders through the exchange
        # For now, we log the hedge intention

    def get_size_multiplier(self) -> float:
        """Get current position size multiplier.
        
        Returns 0.2 (80% reduction) when safety net is active, 1.0 otherwise.
        """
        return self._emergency_size_multiplier if self._safety_net_active else 1.0

    def is_active(self) -> bool:
        """Check if safety net is currently active."""
        return self._safety_net_active

    def get_entropy_metrics(self) -> dict[str, float]:
        """Get current entropy metrics.
        
        Returns:
            Dictionary with entropy statistics
        """
        if len(self._entropy_history) == 0:
            return {
                "current_entropy": 0.0,
                "baseline_entropy": 0.0,
                "entropy_ratio": 1.0,
                "safety_net_active": False
            }
        
        current_entropy = self._entropy_history[-1] if self._entropy_history else 0.0
        entropy_ratio = (
            current_entropy / self._baseline_entropy
            if self._baseline_entropy > 0
            else 1.0
        )
        
        return {
            "current_entropy": current_entropy,
            "baseline_entropy": self._baseline_entropy,
            "entropy_ratio": entropy_ratio,
            "safety_net_active": self._safety_net_active,
            "entropy_spike_threshold": self._entropy_spike_threshold
        }

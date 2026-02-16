"""Component Registry for new advanced features.

Centralizes initialization and lifecycle management for:
- RL Risk Manager
- Dynamic Oracle Weighting
- Strategy Allocator
- (Future) TWAP Executor, Physics-aware sizers, Symbiont X Bridge

This module provides a clean integration point for main.py.
"""

from typing import Any

from hean.config import settings
from hean.core.bus import EventBus
from hean.logging import get_logger

logger = get_logger(__name__)

# Conditional imports (components may have optional dependencies)
try:
    from hean.risk.rl_risk_manager import RLRiskManager  # noqa: F401
    RL_RISK_AVAILABLE = True
except ImportError:
    RL_RISK_AVAILABLE = False
    logger.warning("RL Risk Manager not available (missing dependencies)")

try:
    from hean.core.intelligence.dynamic_oracle import DynamicOracleWeighting  # noqa: F401
    DYNAMIC_ORACLE_AVAILABLE = True
except ImportError:
    DYNAMIC_ORACLE_AVAILABLE = False
    logger.warning("Dynamic Oracle Weighting not available")

try:
    from hean.strategies.manager import StrategyAllocator  # noqa: F401
    STRATEGY_ALLOCATOR_AVAILABLE = True
except ImportError:
    STRATEGY_ALLOCATOR_AVAILABLE = False
    logger.warning("Strategy Allocator not available")


class ComponentRegistry:
    """Registry for advanced trading system components.

    Handles:
    - Component initialization
    - Lifecycle management (start/stop)
    - Dependency injection
    - Error handling and graceful degradation
    """

    def __init__(self, bus: EventBus):
        self._bus = bus

        # Components (initialized to None, created on demand)
        self.rl_risk_manager: Any = None
        self.oracle_weighting: Any = None
        self.strategy_allocator: Any = None

        # Component status tracking
        self._started_components: list[str] = []

    async def initialize_all(self, initial_capital: float = 10000.0) -> dict[str, bool]:
        """Initialize all available components.

        Args:
            initial_capital: Initial capital for allocator

        Returns:
            Dict mapping component names to success status
        """
        results = {}

        # 1. Initialize RL Risk Manager
        if settings.rl_risk_enabled and RL_RISK_AVAILABLE:
            try:
                from hean.risk.rl_risk_manager import RLRiskManager
                self.rl_risk_manager = RLRiskManager(
                    bus=self._bus,
                    model_path=settings.rl_risk_model_path,
                    adjustment_interval=settings.rl_risk_adjust_interval,
                    enabled=settings.rl_risk_enabled,
                )
                results["rl_risk_manager"] = True
                logger.info("RL Risk Manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RL Risk Manager: {e}")
                results["rl_risk_manager"] = False
        else:
            results["rl_risk_manager"] = False

        # 2. Initialize Dynamic Oracle Weighting
        if settings.oracle_dynamic_weighting and DYNAMIC_ORACLE_AVAILABLE:
            try:
                from hean.core.intelligence.dynamic_oracle import DynamicOracleWeighting
                self.oracle_weighting = DynamicOracleWeighting(bus=self._bus)
                results["oracle_weighting"] = True
                logger.info("Dynamic Oracle Weighting initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Dynamic Oracle Weighting: {e}")
                results["oracle_weighting"] = False
        else:
            results["oracle_weighting"] = False

        # 3. Initialize Strategy Allocator
        if STRATEGY_ALLOCATOR_AVAILABLE:
            try:
                from hean.strategies.manager import StrategyAllocator
                self.strategy_allocator = StrategyAllocator(
                    bus=self._bus,
                    initial_capital=initial_capital,
                    rebalance_interval=300,  # 5 minutes
                    min_allocation_pct=0.05,
                    max_allocation_pct=0.40,
                )
                results["strategy_allocator"] = True
                logger.info("Strategy Allocator initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Strategy Allocator: {e}")
                results["strategy_allocator"] = False
        else:
            results["strategy_allocator"] = False

        return results

    async def start_all(self) -> None:
        """Start all initialized components."""
        # Start RL Risk Manager
        if self.rl_risk_manager:
            try:
                await self.rl_risk_manager.start()
                self._started_components.append("rl_risk_manager")
                logger.info("✅ RL Risk Manager started")
            except Exception as e:
                logger.error(f"Failed to start RL Risk Manager: {e}")

        # Start Dynamic Oracle Weighting
        if self.oracle_weighting:
            try:
                await self.oracle_weighting.start()
                self._started_components.append("oracle_weighting")
                logger.info("✅ Dynamic Oracle Weighting started")
            except Exception as e:
                logger.error(f"Failed to start Dynamic Oracle Weighting: {e}")

        # Start Strategy Allocator
        if self.strategy_allocator:
            try:
                await self.strategy_allocator.start()
                self._started_components.append("strategy_allocator")
                logger.info("✅ Strategy Allocator started")
            except Exception as e:
                logger.error(f"Failed to start Strategy Allocator: {e}")

    async def stop_all(self) -> None:
        """Stop all started components in reverse order."""
        for component_name in reversed(self._started_components):
            component = getattr(self, component_name, None)
            if component:
                try:
                    await component.stop()
                    logger.info(f"✅ {component_name} stopped")
                except Exception as e:
                    logger.error(f"Failed to stop {component_name}: {e}")

        self._started_components.clear()

    def register_strategies(self, strategy_ids: list[str]) -> None:
        """Register strategies with the allocator.

        Args:
            strategy_ids: List of strategy IDs to register
        """
        if self.strategy_allocator:
            for strategy_id in strategy_ids:
                self.strategy_allocator.register_strategy(strategy_id)
            logger.info(f"Registered {len(strategy_ids)} strategies with allocator")

    def get_rl_risk_parameters(self) -> dict[str, float]:
        """Get current RL risk parameters.

        Returns:
            Dict with leverage, size_multiplier, stop_loss_pct
            Falls back to defaults if RL not available
        """
        if self.rl_risk_manager:
            return self.rl_risk_manager.get_risk_parameters()

        # Fallback defaults
        return {
            "leverage": settings.max_leverage,
            "size_multiplier": 1.0,
            "stop_loss_pct": 2.0,
        }

    def get_oracle_weights(self) -> dict[str, float]:
        """Get current oracle model weights.

        Returns:
            Dict with tcn, finbert, ollama, brain weights
            Falls back to fixed weights if dynamic weighting not available
        """
        if self.oracle_weighting:
            return self.oracle_weighting.get_weights()

        # Fallback fixed weights
        return {
            "tcn": 0.40,
            "finbert": 0.20,
            "ollama": 0.20,
            "brain": 0.20,
        }

    def fuse_oracle_signals(
        self,
        tcn_signal: float | None = None,
        finbert_signal: float | None = None,
        ollama_signal: float | None = None,
        brain_signal: float | None = None,
        min_confidence: float = 0.6,
    ) -> dict[str, Any] | None:
        """Fuse oracle signals with dynamic weights.

        Args:
            tcn_signal: TCN prediction [-1, 1]
            finbert_signal: FinBERT sentiment [-1, 1]
            ollama_signal: Ollama sentiment [-1, 1]
            brain_signal: Brain sentiment [-1, 1]
            min_confidence: Minimum confidence threshold

        Returns:
            Fused signal dict or None
        """
        if self.oracle_weighting:
            return self.oracle_weighting.fuse_signals(
                tcn_signal=tcn_signal,
                finbert_signal=finbert_signal,
                ollama_signal=ollama_signal,
                brain_signal=brain_signal,
                min_confidence=min_confidence,
            )

        # Fallback: simple average of available signals
        signals = []
        if tcn_signal is not None:
            signals.append(tcn_signal)
        if finbert_signal is not None:
            signals.append(finbert_signal)
        if ollama_signal is not None:
            signals.append(ollama_signal)
        if brain_signal is not None:
            signals.append(brain_signal)

        if not signals:
            return None

        avg_signal = sum(signals) / len(signals)
        confidence = abs(avg_signal)

        if confidence < min_confidence:
            return None

        return {
            "direction": "buy" if avg_signal > 0 else "sell",
            "confidence": confidence,
            "weighted_score": avg_signal,
            "sources_used": ["fallback_average"],
            "weights": {},
        }

    def get_strategy_allocation(self, strategy_id: str) -> float:
        """Get current capital allocation for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Allocated capital amount (0.0 if allocator not available)
        """
        if self.strategy_allocator:
            return self.strategy_allocator.get_allocation(strategy_id)
        return 0.0

    def get_strategy_performance(self, strategy_id: str) -> Any:
        """Get performance metrics for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            StrategyPerformance object or None
        """
        if self.strategy_allocator:
            return self.strategy_allocator.get_performance(strategy_id)
        return None

    def get_status(self) -> dict[str, Any]:
        """Get status of all components.

        Returns:
            Status dict with component states and metrics
        """
        status = {
            "components_started": len(self._started_components),
            "components": {},
        }

        # RL Risk Manager status
        if self.rl_risk_manager:
            params = self.rl_risk_manager.get_risk_parameters()
            status["components"]["rl_risk_manager"] = {
                "active": True,
                "leverage": params["leverage"],
                "size_multiplier": params["size_multiplier"],
                "stop_loss_pct": params["stop_loss_pct"],
            }
        else:
            status["components"]["rl_risk_manager"] = {"active": False}

        # Oracle Weighting status
        if self.oracle_weighting:
            weights = self.oracle_weighting.get_weights()
            status["components"]["oracle_weighting"] = {
                "active": True,
                "weights": weights,
            }
        else:
            status["components"]["oracle_weighting"] = {"active": False}

        # Strategy Allocator status
        if self.strategy_allocator:
            allocations = self.strategy_allocator.get_all_allocations()
            status["components"]["strategy_allocator"] = {
                "active": True,
                "num_strategies": len(allocations),
                "allocations": allocations,
            }
        else:
            status["components"]["strategy_allocator"] = {"active": False}

        return status


# Singleton instance (created by TradingSystem)
_registry: ComponentRegistry | None = None


def get_component_registry() -> ComponentRegistry | None:
    """Get the global component registry instance."""
    return _registry


def set_component_registry(registry: ComponentRegistry) -> None:
    """Set the global component registry instance."""
    global _registry
    _registry = registry

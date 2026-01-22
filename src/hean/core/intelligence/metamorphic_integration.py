"""
Metamorphic Engine Integration: Python wrapper for C++ Metamorphic Engine

This module provides a Python interface to the C++ Metamorphic Engine,
which profiles trading functions and triggers evolutionary cycles when
alpha decay is detected.
"""

import time

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)

# Try to import C++ engine
try:
    import graph_engine_py as cpp_engine
    CPP_ENGINE_AVAILABLE = True
except ImportError:
    logger.warning("C++ graph_engine_py not available. Metamorphic Engine will be disabled.")
    CPP_ENGINE_AVAILABLE = False


class MetamorphicIntegration:
    """
    Python integration for the Metamorphic Engine.

    This class wraps the C++ Metamorphic Engine and provides Python-friendly
    interfaces for profiling strategies and detecting alpha decay.
    """

    def __init__(self, bus: EventBus):
        """
        Initialize the Metamorphic Integration.

        Args:
            bus: Event bus for publishing evolution events
        """
        self._bus = bus
        self._enabled = CPP_ENGINE_AVAILABLE
        self._registered_strategies: dict[str, int] = {}  # strategy_id -> model_type

        if self._enabled:
            try:
                cpp_engine.metamorphic_engine_init()
                logger.info("Metamorphic Engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Metamorphic Engine: {e}")
                self._enabled = False

    async def start(self) -> None:
        """Start the Metamorphic Integration."""
        if not self._enabled:
            logger.warning("Metamorphic Engine not available, skipping start")
            return

        self._bus.subscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        logger.info("Metamorphic Integration started")

    async def stop(self) -> None:
        """Stop the Metamorphic Integration."""
        if not self._enabled:
            return

        self._bus.unsubscribe(EventType.ORDER_FILLED, self._handle_order_filled)
        logger.info("Metamorphic Integration stopped")

    def register_strategy(self, strategy_id: str, initial_model: int = 0) -> None:
        """
        Register a strategy for profiling.

        Args:
            strategy_id: Strategy identifier
            initial_model: Initial model type (0=LINEAR_REGRESSION, 1=NON_EUCLIDEAN_GEOMETRY, etc.)
        """
        if not self._enabled:
            return

        try:
            cpp_engine.metamorphic_engine_register_strategy(strategy_id, initial_model)
            self._registered_strategies[strategy_id] = initial_model
            logger.info(f"Registered strategy {strategy_id} with model type {initial_model}")
        except Exception as e:
            logger.error(f"Failed to register strategy {strategy_id}: {e}")

    async def _handle_order_filled(self, event: Event) -> None:
        """Handle order filled events to record trades."""
        if not self._enabled:
            return

        order = event.data.get("order")
        if not order or not order.strategy_id:
            return

        # Calculate PnL (simplified - should get from accounting)
        pnl = 0.0  # Will be updated from actual fill data
        if hasattr(order, 'filled_price') and hasattr(order, 'entry_price'):
            if order.side == "buy":
                pnl = (order.filled_price - order.entry_price) * order.filled_size
            else:
                pnl = (order.entry_price - order.filled_price) * order.filled_size

        is_win = pnl > 0
        timestamp_ns = int(time.time() * 1e9)

        try:
            cpp_engine.metamorphic_engine_record_trade(
                order.strategy_id,
                pnl,
                timestamp_ns,
                1 if is_win else 0
            )

            # Check for evolution
            evolution_status = cpp_engine.metamorphic_engine_get_evolution_status(
                order.strategy_id
            )

            if evolution_status.get('evolution_triggered', False):
                logger.warning(
                    f"Alpha decay detected for strategy {order.strategy_id}! "
                    f"Current model: {evolution_status['current_model']}, "
                    f"Proposed model: {evolution_status['proposed_model']}, "
                    f"Alpha decay rate: {evolution_status['alpha_decay_rate']:.2%}"
                )

                # Publish evolution event
                await self._bus.publish(
                    Event(
                        event_type=EventType.SIGNAL,  # Reuse SIGNAL type
                        data={
                            "type": "metamorphic_evolution",
                            "strategy_id": order.strategy_id,
                            "evolution_status": evolution_status
                        }
                    )
                )
        except Exception as e:
            logger.error(f"Error recording trade for Metamorphic Engine: {e}")

    def get_evolution_status(self, strategy_id: str) -> dict | None:
        """
        Get evolution status for a strategy.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Dictionary with evolution status or None if not available
        """
        if not self._enabled:
            return None

        try:
            return cpp_engine.metamorphic_engine_get_evolution_status(strategy_id)
        except Exception as e:
            logger.error(f"Error getting evolution status: {e}")
            return None

    def apply_evolution(self, strategy_id: str) -> None:
        """
        Apply evolutionary model change for a strategy.

        Args:
            strategy_id: Strategy identifier
        """
        if not self._enabled:
            return

        try:
            cpp_engine.metamorphic_engine_apply_evolution(strategy_id)
            logger.info(f"Applied evolution for strategy {strategy_id}")
        except Exception as e:
            logger.error(f"Error applying evolution: {e}")

    def get_system_evolution_level(self) -> float:
        """
        Get System Evolution Level (SEL) - overall system intelligence metric.

        Returns:
            SEL value between 0.0 and 1.0
        """
        if not self._enabled:
            return 0.0

        try:
            return cpp_engine.metamorphic_engine_get_sel()
        except Exception as e:
            logger.error(f"Error getting SEL: {e}")
            return 0.0

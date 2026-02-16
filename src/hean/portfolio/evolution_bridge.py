"""EvolutionBridge — Stub connecting MetaStrategyBrain to Symbiont X genome system.

When a strategy reaches TERMINATED state, this bridge encodes its parameters
as a genome and submits it to Symbiont X for genetic evolution. Evolved variants
can then be promoted back through the lifecycle.

This is a stub implementation — the full Symbiont X integration will use
the existing EvolutionEngine and SymbiontXBridge infrastructure.
"""

import time
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class EvolutionBridge:
    """Bridge between MetaStrategyBrain lifecycle and Symbiont X evolution.

    When strategies are terminated due to alpha decay, this bridge:
    1. Encodes the strategy's current parameters as a genome
    2. Submits to Symbiont X for tournament selection + crossover + mutation
    3. Receives evolved parameter sets
    4. Feeds them back to MetaStrategyBrain as candidate revivals

    Current implementation is a stub that logs evolution requests.
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._pending_evolutions: list[dict[str, Any]] = []
        self._evolved_candidates: list[dict[str, Any]] = []
        self._running = False

    async def start(self) -> None:
        """Start the evolution bridge."""
        self._running = True
        self._bus.subscribe(EventType.META_STRATEGY_UPDATE, self._on_meta_update)
        logger.info("EvolutionBridge started (stub mode)")

    async def stop(self) -> None:
        """Stop the evolution bridge."""
        self._running = False
        self._bus.unsubscribe(EventType.META_STRATEGY_UPDATE, self._on_meta_update)
        logger.info("EvolutionBridge stopped")

    async def _on_meta_update(self, event: Event) -> None:
        """Watch for TERMINATED transitions to trigger evolution."""
        transitions = event.data.get("transitions", [])
        for t in transitions:
            if t.get("to") == "terminated":
                await self._request_evolution(
                    strategy_id=t["strategy_id"],
                    fitness=t.get("fitness", 0.0),
                    reason=t.get("reason", "unknown"),
                )

    async def _request_evolution(
        self, strategy_id: str, fitness: float, reason: str
    ) -> None:
        """Request Symbiont X to evolve a terminated strategy.

        Stub implementation — logs the request for future integration.
        """
        request = {
            "strategy_id": strategy_id,
            "fitness": fitness,
            "reason": reason,
            "requested_at": time.time(),
            "status": "pending",
        }
        self._pending_evolutions.append(request)
        logger.info(
            "[EVOLUTION] Requested evolution for %s (fitness=%.3f, reason=%s)",
            strategy_id,
            fitness,
            reason,
        )

    def get_pending(self) -> list[dict[str, Any]]:
        """Get pending evolution requests."""
        return list(self._pending_evolutions)

    def get_evolved(self) -> list[dict[str, Any]]:
        """Get evolved candidates ready for promotion."""
        return list(self._evolved_candidates)

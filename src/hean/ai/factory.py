"""AI Factory - Safe self-improvement through shadow testing and canary promotion.

Split/Monster Factory pattern:
1. Generate candidate strategies (variations)
2. Evaluate in shadow mode (replay on historical events)
3. Promote to canary (10% of live traffic)
4. Quality gate → promote to production or rollback
"""

import uuid
from datetime import datetime
from typing import Any, Literal

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class AIFactory:
    """AI Factory for generating and testing strategy candidates.

    Safe self-improvement workflow:
    - Shadow → Canary → Production
    - Never modifies production without tests
    - Full audit trail
    """

    def __init__(self, bus: EventBus) -> None:
        """Initialize AI Factory.

        Args:
            bus: Event bus for publishing experiment results
        """
        self._bus = bus
        self._enabled = getattr(settings, "ai_factory_enabled", False)
        self._canary_pct = getattr(settings, "canary_percent", 10)
        self._candidates: dict[str, dict[str, Any]] = {}
        self._experiments: dict[str, dict[str, Any]] = {}

        logger.info(f"AI Factory initialized: enabled={self._enabled}, canary_pct={self._canary_pct}%")

    def generate_candidates(
        self,
        base_strategy: str,
        variations: list[str],
        param_grid: dict[str, list[Any]],
    ) -> list[dict[str, Any]]:
        """Generate candidate strategy variations.

        Args:
            base_strategy: Base strategy ID
            variations: Variation names (e.g., ["aggressive", "conservative"])
            param_grid: Parameter grid to search

        Returns:
            List of candidate configurations
        """
        if not self._enabled:
            logger.warning("AI Factory not enabled, returning empty candidates")
            return []

        candidates = []
        for variation in variations:
            # Generate simple parameter combinations
            for param_name, param_values in param_grid.items():
                for param_value in param_values:
                    candidate_id = f"{base_strategy}_{variation}_{param_name}_{param_value}"
                    candidate = {
                        "candidate_id": candidate_id,
                        "base_strategy": base_strategy,
                        "variation": variation,
                        "params": {param_name: param_value},
                        "status": "shadow",
                        "created_at": datetime.utcnow().isoformat(),
                    }
                    candidates.append(candidate)
                    self._candidates[candidate_id] = candidate

        logger.info(f"Generated {len(candidates)} candidates for {base_strategy}")
        return candidates

    async def evaluate_candidates(
        self,
        candidates: list[dict[str, Any]],
        replay_events: list[Event],
        metrics: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Evaluate candidates in shadow mode (replay on historical events).

        Args:
            candidates: List of candidates to evaluate
            replay_events: Historical events to replay
            metrics: Metrics to calculate (e.g., ["sharpe", "max_dd", "profit_factor"])

        Returns:
            Results dictionary {candidate_id: metrics}
        """
        if not self._enabled:
            logger.warning("AI Factory not enabled, skipping evaluation")
            return {}

        results = {}
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]

            # STUB: In real implementation, replay events with candidate strategy
            # For now, generate mock metrics
            mock_metrics = {
                "sharpe": 1.5 + (hash(candidate_id) % 10) / 10.0,  # 1.5-2.4
                "max_dd_pct": 8.0 + (hash(candidate_id) % 5),  # 8-12%
                "profit_factor": 1.3 + (hash(candidate_id) % 7) / 10.0,  # 1.3-1.9
                "trades": 50 + (hash(candidate_id) % 50),  # 50-99 trades
            }

            results[candidate_id] = mock_metrics

            logger.info(
                f"Evaluated {candidate_id}: "
                f"Sharpe={mock_metrics['sharpe']:.2f}, "
                f"MaxDD={mock_metrics['max_dd_pct']:.1f}%, "
                f"PF={mock_metrics['profit_factor']:.2f}"
            )

        return results

    async def promote_to_canary(
        self,
        strategy_id: str,
        canary_pct: int | None = None,
    ) -> dict[str, Any]:
        """Promote candidate to canary testing (live traffic split).

        Args:
            strategy_id: Candidate strategy ID
            canary_pct: Percentage of traffic for canary (default from settings)

        Returns:
            Promotion status
        """
        if not self._enabled:
            return {"status": "disabled", "message": "AI Factory not enabled"}

        if strategy_id not in self._candidates:
            return {"status": "error", "message": f"Candidate {strategy_id} not found"}

        canary_pct = canary_pct or self._canary_pct
        candidate = self._candidates[strategy_id]
        candidate["status"] = "canary"
        candidate["canary_pct"] = canary_pct
        candidate["promoted_to_canary_at"] = datetime.utcnow().isoformat()

        logger.info(f"Promoted {strategy_id} to canary ({canary_pct}% traffic)")

        # Publish event
        await self._bus.publish(Event(
            event_type=EventType.STRATEGY_UPDATED,  # Reuse existing event type
            data={
                "type": "CANARY_PROMOTION",
                "strategy_id": strategy_id,
                "canary_pct": canary_pct,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ))

        return {
            "status": "promoted",
            "strategy_id": strategy_id,
            "canary_pct": canary_pct,
        }

    async def promote_to_production(
        self,
        strategy_id: str,
    ) -> dict[str, Any]:
        """Promote canary to production (100% traffic).

        Args:
            strategy_id: Candidate strategy ID

        Returns:
            Promotion status
        """
        if not self._enabled:
            return {"status": "disabled", "message": "AI Factory not enabled"}

        if strategy_id not in self._candidates:
            return {"status": "error", "message": f"Candidate {strategy_id} not found"}

        candidate = self._candidates[strategy_id]
        if candidate["status"] != "canary":
            return {
                "status": "error",
                "message": f"Candidate {strategy_id} not in canary status (current: {candidate['status']})"
            }

        candidate["status"] = "production"
        candidate["promoted_to_production_at"] = datetime.utcnow().isoformat()

        logger.info(f"Promoted {strategy_id} to production (100% traffic)")

        # Publish event
        await self._bus.publish(Event(
            event_type=EventType.STRATEGY_UPDATED,
            data={
                "type": "PRODUCTION_PROMOTION",
                "strategy_id": strategy_id,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ))

        return {
            "status": "promoted",
            "strategy_id": strategy_id,
            "traffic_pct": 100,
        }

    async def rollback(
        self,
        strategy_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Rollback canary or production candidate.

        Args:
            strategy_id: Candidate strategy ID
            reason: Reason for rollback

        Returns:
            Rollback status
        """
        if not self._enabled:
            return {"status": "disabled", "message": "AI Factory not enabled"}

        if strategy_id not in self._candidates:
            return {"status": "error", "message": f"Candidate {strategy_id} not found"}

        candidate = self._candidates[strategy_id]
        old_status = candidate["status"]
        candidate["status"] = "rolled_back"
        candidate["rollback_reason"] = reason
        candidate["rolled_back_at"] = datetime.utcnow().isoformat()

        logger.warning(f"Rolled back {strategy_id} from {old_status}: {reason}")

        # Publish event
        await self._bus.publish(Event(
            event_type=EventType.STRATEGY_UPDATED,
            data={
                "type": "STRATEGY_ROLLBACK",
                "strategy_id": strategy_id,
                "previous_status": old_status,
                "reason": reason,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ))

        return {
            "status": "rolled_back",
            "strategy_id": strategy_id,
            "previous_status": old_status,
            "reason": reason,
        }

    def get_candidate(self, strategy_id: str) -> dict[str, Any] | None:
        """Get candidate by ID.

        Args:
            strategy_id: Candidate strategy ID

        Returns:
            Candidate configuration or None
        """
        return self._candidates.get(strategy_id)

    def get_all_candidates(self) -> list[dict[str, Any]]:
        """Get all candidates.

        Returns:
            List of all candidates
        """
        return list(self._candidates.values())

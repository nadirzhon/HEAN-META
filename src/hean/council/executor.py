"""Executes approved council recommendations."""

import logging
from typing import Any

from hean.core.bus import EventBus
from hean.council.review import Category, Recommendation

logger = logging.getLogger(__name__)

# Only these config params can be auto-applied by the council.
# Critical safety params (killswitch, leverage, API keys) are NEVER changed.
SAFE_CONFIG_PARAMS = frozenset({
    "brain_analysis_interval",
    "agent_generation_interval",
    "council_review_interval",
})


class CouncilExecutor:
    """Routes approved recommendations to appropriate systems."""

    def __init__(
        self,
        bus: EventBus,
        ai_factory: Any | None = None,
        improvement_catalyst: Any | None = None,
    ) -> None:
        self._bus = bus
        self._ai_factory = ai_factory
        self._catalyst = improvement_catalyst

    async def apply_recommendation(self, rec: Recommendation) -> dict[str, Any]:
        """Apply a single recommendation.

        Routes to:
        - Strategy param changes -> AIFactory shadow testing pipeline
        - Config param changes -> Direct application (safe whitelist only)
        - Everything else -> Queued for human review
        """
        try:
            if rec.category == Category.TRADING and rec.param_changes and rec.target_strategy:
                return await self._apply_strategy_param_change(rec)
            elif rec.category == Category.PERFORMANCE and rec.param_changes:
                return await self._apply_config_change(rec)
            else:
                return {"status": "queued", "message": "Requires human approval"}
        except Exception as e:
            logger.error(f"Failed to apply recommendation {rec.id}: {e}")
            return {"status": "error", "message": str(e)}

    async def _apply_strategy_param_change(self, rec: Recommendation) -> dict[str, Any]:
        """Route strategy param changes through AIFactory shadow->canary pipeline."""
        if not self._ai_factory:
            return {"status": "skipped", "message": "AI Factory not available"}

        try:
            candidates = self._ai_factory.generate_candidates(
                base_strategy=rec.target_strategy,
                variations=["council_recommended"],
                param_grid={k: [v] for k, v in (rec.param_changes or {}).items()},
            )

            if not candidates:
                return {"status": "no_candidates", "message": "No candidates generated"}

            results = await self._ai_factory.evaluate_candidates(candidates, sim_days=7)

            best_id = None
            best_score = 0.0
            for cid, result in results.items():
                if result.get("error"):
                    continue
                rate = result.get("signal_rate_per_day", 0)
                if rate > best_score:
                    best_score = rate
                    best_id = cid

            if best_id and best_score > 0:
                await self._ai_factory.promote_to_canary(best_id)
                logger.info(
                    f"Council: Promoted candidate {best_id} to canary "
                    f"(score={best_score:.2f})"
                )
                return {
                    "status": "applied",
                    "pipeline": "shadow_canary",
                    "candidate_id": best_id,
                    "score": best_score,
                }

            return {"status": "no_improvement", "message": "Candidates did not improve performance"}

        except Exception as e:
            logger.error(f"Strategy param change failed: {e}")
            return {"status": "error", "message": str(e)}

    async def _apply_config_change(self, rec: Recommendation) -> dict[str, Any]:
        """Apply safe config parameter changes.

        SAFETY: Only allows a strict whitelist of safe-to-change parameters.
        """
        if not rec.param_changes:
            return {"status": "skipped", "message": "No param changes"}

        applied: dict[str, Any] = {}
        skipped: list[str] = []

        from hean.config import settings

        for param, value in rec.param_changes.items():
            if param not in SAFE_CONFIG_PARAMS:
                logger.warning(f"Council: Refusing to change unsafe param '{param}'")
                skipped.append(param)
                continue
            if hasattr(settings, param):
                old_value = getattr(settings, param)
                try:
                    object.__setattr__(settings, param, value)
                    applied[param] = {"old": old_value, "new": value}
                    logger.info(f"Council: Config change {param}: {old_value} -> {value}")
                except Exception as e:
                    logger.error(f"Council: Failed to set {param}: {e}")
                    skipped.append(param)
            else:
                skipped.append(param)

        result: dict[str, Any] = {"status": "applied" if applied else "no_changes"}
        if applied:
            result["changes"] = applied
        if skipped:
            result["skipped"] = skipped
        return result

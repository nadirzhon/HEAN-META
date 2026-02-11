"""AI Factory - Safe self-improvement through shadow testing and canary promotion.

Split/Monster Factory pattern:
1. Generate candidate strategies (variations)
2. Evaluate in shadow mode (replay on historical events)
3. Promote to canary (10% of live traffic)
4. Quality gate → promote to production or rollback
"""

from datetime import datetime, timedelta
from typing import Any

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
        replay_events: list[Event] | None = None,
        metrics: list[str] | None = None,
        sim_days: int = 7,
    ) -> dict[str, dict[str, Any]]:
        """Evaluate candidates using EventSimulator shadow replay.

        Creates an isolated EventBus + EventSimulator per candidate,
        instantiates the strategy with candidate params, counts signals,
        and returns basic metrics.

        Args:
            candidates: List of candidates to evaluate
            replay_events: Historical events (unused, kept for API compat)
            metrics: Metrics to calculate (unused, kept for API compat)
            sim_days: Days of simulated data to replay (default 7)

        Returns:
            Results dictionary {candidate_id: metrics}
        """
        if not self._enabled:
            logger.warning("AI Factory not enabled, skipping evaluation")
            return {}

        from hean.backtest.event_sim import EventSimulator

        results = {}
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            base_strategy = candidate.get("base_strategy", "")
            params = candidate.get("params", {})

            try:
                result = await self._evaluate_single_candidate(
                    candidate_id, base_strategy, params, sim_days
                )
                results[candidate_id] = result
            except Exception as e:
                logger.warning(f"[AI_FACTORY] Evaluation failed for {candidate_id}: {e}")
                results[candidate_id] = {
                    "signal_count": 0,
                    "buy_signals": 0,
                    "sell_signals": 0,
                    "signal_rate_per_day": 0.0,
                    "buy_sell_ratio": 0.0,
                    "error": str(e),
                }

        return results

    async def _evaluate_single_candidate(
        self,
        candidate_id: str,
        base_strategy: str,
        params: dict[str, Any],
        sim_days: int,
    ) -> dict[str, Any]:
        """Evaluate a single candidate with an isolated EventSimulator.

        Returns:
            Metrics dict with signal_count, buy/sell ratio, signal_rate_per_day
        """
        from hean.backtest.event_sim import EventSimulator

        # Create isolated bus for this candidate
        sim_bus = EventBus()
        symbols = ["BTCUSDT", "ETHUSDT"]
        start_date = datetime.utcnow() - timedelta(days=sim_days)

        # Create simulator
        sim = EventSimulator(
            bus=sim_bus,
            symbols=symbols,
            start_date=start_date,
            days=sim_days,
        )

        # Create strategy instance with candidate params
        strategy = self._create_strategy(base_strategy, sim_bus, symbols, params)
        if strategy is None:
            return {
                "signal_count": 0,
                "error": f"Unknown base strategy: {base_strategy}",
            }

        # Track signals emitted by the strategy
        signal_counts = {"buy": 0, "sell": 0, "total": 0}

        async def _count_signal(event: Event) -> None:
            signal = event.data.get("signal")
            if signal:
                signal_counts["total"] += 1
                if signal.side == "buy":
                    signal_counts["buy"] += 1
                else:
                    signal_counts["sell"] += 1

        sim_bus.subscribe(EventType.SIGNAL, _count_signal)

        # Run simulation
        await sim.start()
        await strategy.start()
        await sim.run()
        await strategy.stop()
        await sim.stop()

        # Calculate metrics
        total = signal_counts["total"]
        buy = signal_counts["buy"]
        sell = signal_counts["sell"]

        return {
            "signal_count": total,
            "buy_signals": buy,
            "sell_signals": sell,
            "signal_rate_per_day": total / max(sim_days, 1),
            "buy_sell_ratio": buy / max(sell, 1),
            "sim_days": sim_days,
            "params": params,
        }

    def _create_strategy(
        self,
        base_strategy: str,
        bus: EventBus,
        symbols: list[str],
        params: dict[str, Any],
        code: str | None = None,
    ):
        """Create a strategy instance from base_strategy name or generated code.

        Args:
            base_strategy: Strategy name
            bus: EventBus
            symbols: Symbols list
            params: Parameter overrides
            code: Optional generated Python code (if provided, loads dynamically)

        Returns:
            Strategy instance or None if unknown strategy
        """
        # Dynamic code loading (for LLM-generated agents)
        if code is not None:
            return self._create_strategy_from_code(code, bus, symbols)

        try:
            if base_strategy in ("impulse_engine", "ImpulseEngine"):
                from hean.strategies.impulse_engine import ImpulseEngine
                strategy = ImpulseEngine(bus, symbols)
            elif base_strategy in ("funding_harvester", "FundingHarvester"):
                from hean.strategies.funding_harvester import FundingHarvester
                strategy = FundingHarvester(bus, symbols)
            elif base_strategy in ("basis_arbitrage", "BasisArbitrage"):
                from hean.strategies.basis_arbitrage import BasisArbitrage
                strategy = BasisArbitrage(bus, symbols)
            else:
                logger.warning(f"[AI_FACTORY] Unknown strategy: {base_strategy}")
                return None

            # Apply param overrides
            for param_name, param_value in params.items():
                if hasattr(strategy, param_name):
                    setattr(strategy, param_name, param_value)
                elif hasattr(strategy, f"_{param_name}"):
                    setattr(strategy, f"_{param_name}", param_value)

            return strategy
        except Exception as e:
            logger.warning(f"[AI_FACTORY] Failed to create {base_strategy}: {e}")
            return None

    def _create_strategy_from_code(
        self,
        code: str,
        bus: EventBus,
        symbols: list[str],
    ):
        """Create a strategy instance from generated Python code.

        Executes the code, finds the BaseStrategy subclass, and instantiates it.
        """
        import ast

        try:
            ast.parse(code)
            namespace: dict[str, Any] = {}
            exec(code, namespace)  # noqa: S102

            from hean.strategies.base import BaseStrategy
            strategy_class = None
            for obj in namespace.values():
                if isinstance(obj, type) and issubclass(obj, BaseStrategy) and obj is not BaseStrategy:
                    strategy_class = obj
                    break

            if strategy_class is None:
                logger.warning("[AI_FACTORY] No BaseStrategy subclass found in generated code")
                return None

            try:
                return strategy_class(bus=bus, symbols=symbols)
            except TypeError:
                return strategy_class(bus, symbols)

        except Exception as e:
            logger.warning(f"[AI_FACTORY] Failed to create strategy from code: {e}")
            return None

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
            event_type=EventType.STRATEGY_PARAMS_UPDATED,  # Reuse existing event type
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
            event_type=EventType.STRATEGY_PARAMS_UPDATED,
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
            event_type=EventType.STRATEGY_PARAMS_UPDATED,
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

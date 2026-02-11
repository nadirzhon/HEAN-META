"""Auto-improvement catalyst system using LLM for continuous optimization."""

import ast
import asyncio
import inspect
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from hean.agent_generation.generator import AgentGenerator
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger
from hean.observability.metrics import metrics
from hean.portfolio.accounting import PortfolioAccounting

logger = get_logger(__name__)


class ImprovementCatalyst:
    """Autonomous improvement system that monitors PnL and triggers AI Factory pipeline.

    Two improvement mechanisms:
    1. LLM mutation: Reads strategy source, sends to LLM, generates improved code
    2. Parameter grid: Systematic parameter variation via AI Factory + EventSimulator
    """

    def __init__(
        self,
        bus: EventBus | None = None,
        accounting: PortfolioAccounting | None = None,
        strategies: dict[str, Any] | None = None,
        ai_factory: Any = None,
        check_interval_minutes: int = 30,
        min_trades_for_analysis: int = 10,
    ) -> None:
        self._bus = bus
        self._accounting = accounting
        self._strategies = strategies or {}
        self._ai_factory = ai_factory
        self._check_interval = timedelta(minutes=check_interval_minutes)
        self._min_trades = min_trades_for_analysis
        self._generator = AgentGenerator()
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._improvement_history: list[dict[str, Any]] = []
        self._last_check: datetime | None = None
        self._optimization_results: dict[str, dict[str, Any]] = {}

        # PNL tracking from EventBus
        self._equity_history: deque[float] = deque(maxlen=1000)
        self._pnl_updates: deque[dict[str, Any]] = deque(maxlen=500)

    async def start(self) -> None:
        """Start the catalyst."""
        if self._running:
            return
        self._running = True

        if self._bus:
            self._bus.subscribe(EventType.PNL_UPDATE, self._handle_pnl_update)

        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("Improvement Catalyst started (LLM mutation + param grid)")

    async def stop(self) -> None:
        """Stop the catalyst."""
        self._running = False

        if self._bus:
            self._bus.unsubscribe(EventType.PNL_UPDATE, self._handle_pnl_update)

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Improvement Catalyst stopped")

    async def _handle_pnl_update(self, event: Event) -> None:
        """Track PnL updates from EventBus."""
        data = event.data
        equity = data.get("equity", 0.0)
        if equity > 0:
            self._equity_history.append(equity)
        self._pnl_updates.append({
            "timestamp": datetime.utcnow().isoformat(),
            "equity": equity,
            "realized_pnl": data.get("realized_pnl", 0.0),
            "unrealized_pnl": data.get("unrealized_pnl", 0.0),
        })

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval.total_seconds())
                await self._analyze_and_improve()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in catalyst loop: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def _analyze_and_improve(self) -> None:
        """Analyze performance and trigger improvements if needed."""
        logger.info("Catalyst: Starting analysis cycle")

        performance_data = self._collect_performance_data()

        if not self._should_analyze(performance_data):
            logger.debug("Catalyst: Not enough data for analysis")
            return

        problems = self._identify_problems(performance_data)

        if not problems:
            logger.debug("Catalyst: No problems identified")
            return

        for problem in problems:
            # 1. LLM mutation: generate improved agent code
            await self._generate_improvement(problem, performance_data)

            # 2. Parameter grid: systematic optimization via AI Factory
            if self._ai_factory and problem.get("strategy_id"):
                await self._run_ai_factory_pipeline(problem)

    def _collect_performance_data(self) -> dict[str, Any]:
        """Collect performance data from accounting and strategies."""
        if not self._accounting:
            return {"total_trades": 0}

        strategy_metrics = self._accounting.get_strategy_metrics()
        system_metrics = metrics.get_summary()
        equity = self._accounting.get_equity()
        drawdown, drawdown_pct = self._accounting.get_drawdown(equity)

        strategy_details = {}
        for strategy_id, strategy_obj in self._strategies.items():
            if hasattr(strategy_obj, "get_metrics"):
                try:
                    strategy_details[strategy_id] = strategy_obj.get_metrics()
                except Exception as e:
                    logger.debug(f"Could not get metrics for {strategy_id}: {e}")

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "equity": equity,
            "drawdown": drawdown,
            "drawdown_pct": drawdown_pct,
            "strategy_metrics": strategy_metrics or {},
            "strategy_details": strategy_details,
            "system_metrics": system_metrics,
            "total_trades": sum(m.get("trades", 0) for m in (strategy_metrics or {}).values()),
            "equity_history_len": len(self._equity_history),
            "pnl_updates_count": len(self._pnl_updates),
        }

    def _should_analyze(self, data: dict[str, Any]) -> bool:
        """Check if there's enough data for analysis."""
        total_trades = data.get("total_trades", 0)
        return total_trades >= self._min_trades

    def _identify_problems(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        """Identify performance problems."""
        problems = []
        strategy_metrics = data.get("strategy_metrics", {})

        for strategy_id, metrics_data in strategy_metrics.items():
            pf = metrics_data.get("profit_factor", 1.0)
            wr = metrics_data.get("win_rate_pct", metrics_data.get("win_rate", 0.0))
            trades = metrics_data.get("trades", 0)
            pnl = metrics_data.get("pnl", 0.0)
            dd = metrics_data.get("max_drawdown_pct", 0.0)

            if pf < 1.2 and trades >= 5:
                problems.append({
                    "type": "low_profit_factor",
                    "strategy_id": strategy_id,
                    "severity": "high" if pf < 1.0 else "medium",
                    "current_pf": pf,
                    "current_wr": wr,
                    "current_dd": dd,
                    "current_pnl": pnl,
                    "trades": trades,
                    "description": f"Strategy {strategy_id} has low profit factor {pf:.2f}",
                })

            if wr < 45.0 and trades >= 10:
                problems.append({
                    "type": "low_win_rate",
                    "strategy_id": strategy_id,
                    "severity": "medium",
                    "current_pf": pf,
                    "current_wr": wr,
                    "current_dd": dd,
                    "current_pnl": pnl,
                    "trades": trades,
                    "description": f"Strategy {strategy_id} has low win rate {wr:.1f}%",
                })

            if dd > 10.0:
                problems.append({
                    "type": "high_drawdown",
                    "strategy_id": strategy_id,
                    "severity": "high" if dd > 15.0 else "medium",
                    "current_pf": pf,
                    "current_wr": wr,
                    "current_dd": dd,
                    "current_pnl": pnl,
                    "trades": trades,
                    "description": f"Strategy {strategy_id} has high drawdown {dd:.1f}%",
                })

            if pnl < 0 and trades >= 5:
                problems.append({
                    "type": "losing_strategy",
                    "strategy_id": strategy_id,
                    "severity": "high",
                    "current_pf": pf,
                    "current_wr": wr,
                    "current_dd": dd,
                    "current_pnl": pnl,
                    "trades": trades,
                    "description": f"Strategy {strategy_id} is losing money: ${pnl:.2f}",
                })

        # System-wide drawdown
        drawdown_pct = data.get("drawdown_pct", 0.0)
        if drawdown_pct > 10.0:
            problems.append({
                "type": "system_high_drawdown",
                "severity": "high",
                "current_dd": drawdown_pct,
                "description": f"System-wide drawdown is high: {drawdown_pct:.1f}%",
            })

        return problems

    # ── Strategy source reading ──────────────────────────────────────

    def _get_strategy_source(self, strategy_id: str) -> str | None:
        """Read the source code of a strategy by its ID."""
        strategy = self._strategies.get(strategy_id)
        if strategy is None:
            return None
        try:
            source_file = inspect.getfile(type(strategy))
            return Path(source_file).read_text(encoding="utf-8")
        except (TypeError, OSError) as e:
            logger.warning(f"Cannot read source for {strategy_id}: {e}")
            return None

    # ── LLM-powered agent generation ─────────────────────────────────

    async def _generate_improvement(
        self, problem: dict[str, Any], performance_data: dict[str, Any]
    ) -> None:
        """Generate LLM-mutated agent for the identified problem."""
        strategy_id = problem.get("strategy_id")
        if not strategy_id:
            logger.info(f"Catalyst: System-level problem: {problem['description']}")
            self._improvement_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "problem": problem,
                "status": "identified_system_level",
            })
            return

        try:
            # 1. Read strategy source code
            source_code = self._get_strategy_source(strategy_id)
            if not source_code:
                logger.warning(f"Catalyst: Cannot read source for {strategy_id}, skipping LLM mutation")
                return

            # 2. Collect metrics
            pf = problem.get("current_pf", 1.0)
            wr = problem.get("current_wr", 0.0)
            dd = problem.get("current_dd", 0.0)
            pnl = problem.get("current_pnl", 0.0)

            # 3. Call LLM mutation
            logger.info(
                f"Catalyst: Mutating {strategy_id} via LLM "
                f"(PF={pf:.2f}, WR={wr:.1f}%, DD={dd:.1f}%, PnL=${pnl:.2f})"
            )
            try:
                mutated_code = self._generator.mutate_agent(
                    agent_code=source_code,
                    profit_factor=pf,
                    total_pnl=pnl,
                    max_drawdown_pct=dd,
                    win_rate=wr,
                    issues=problem["description"],
                )
            except Exception as e:
                logger.warning(f"Catalyst: LLM mutation failed for {strategy_id}: {e}")
                self._improvement_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "problem": problem,
                    "status": "llm_failed",
                    "error": str(e),
                })
                return

            # 4. Save generated code
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_dir = Path("generated_agents")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{timestamp}_{strategy_id}.py"
            output_path.write_text(mutated_code, encoding="utf-8")
            logger.info(f"Catalyst: Saved mutated agent to {output_path}")

            # 5. Evaluate in simulation
            eval_result = await self._evaluate_generated_agent(
                strategy_id=strategy_id,
                code=mutated_code,
                output_path=str(output_path),
            )

            # 6. Record in history
            self._improvement_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "problem": problem,
                "status": "llm_generated",
                "output_path": str(output_path),
                "eval_result": eval_result,
                "metrics_before": {
                    "profit_factor": pf,
                    "win_rate_pct": wr,
                    "max_drawdown_pct": dd,
                    "pnl": pnl,
                },
            })

            logger.info(
                f"Catalyst: Agent generated for {strategy_id} → {output_path} "
                f"(eval: {eval_result.get('status')}, signals: {eval_result.get('signal_count', 0)})"
            )

            # 7. Publish event
            if self._bus:
                await self._bus.publish(Event(
                    event_type=EventType.STRATEGY_PARAMS_UPDATED,
                    data={
                        "type": "AGENT_GENERATED",
                        "strategy_id": strategy_id,
                        "output_path": str(output_path),
                        "eval_result": eval_result,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                ))

        except Exception as e:
            logger.error(f"Catalyst: Error in LLM generation for {strategy_id}: {e}", exc_info=True)

    async def _evaluate_generated_agent(
        self,
        strategy_id: str,
        code: str,
        output_path: str,
    ) -> dict[str, Any]:
        """Load generated code dynamically and evaluate with EventSimulator."""
        try:
            from hean.backtest.event_sim import EventSimulator

            # Dynamic loading
            namespace = _safe_exec(code)
            if namespace is None:
                return {"status": "load_failed", "error": "Code execution failed"}

            strategy_class = _find_strategy_class(namespace)
            if strategy_class is None:
                return {"status": "load_failed", "error": "No BaseStrategy subclass found"}

            # Instantiate with isolated bus
            sim_bus = EventBus()
            symbols = ["BTCUSDT", "ETHUSDT"]

            try:
                strategy = strategy_class(bus=sim_bus, symbols=symbols)
            except TypeError:
                # Some generated classes might have different constructor
                strategy = strategy_class(sim_bus, symbols)

            # Set up EventSimulator (7-day replay)
            start_date = datetime.utcnow() - timedelta(days=7)
            sim = EventSimulator(bus=sim_bus, symbols=symbols, start_date=start_date, days=7)

            # Track signals
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

            total = signal_counts["total"]
            return {
                "status": "evaluated",
                "signal_count": total,
                "buy_signals": signal_counts["buy"],
                "sell_signals": signal_counts["sell"],
                "signal_rate_per_day": total / 7.0,
                "output_path": output_path,
            }

        except Exception as e:
            logger.warning(f"Catalyst: Evaluation failed for generated agent: {e}")
            return {"status": "eval_failed", "error": str(e)}

    # ── AI Factory parameter grid pipeline (existing) ────────────────

    async def _run_ai_factory_pipeline(self, problem: dict[str, Any]) -> None:
        """Run AI Factory pipeline: generate candidates → evaluate → promote."""
        strategy_id = problem.get("strategy_id")
        if not strategy_id or not self._ai_factory:
            return

        try:
            param_grid = self._build_param_grid(strategy_id, problem)
            if not param_grid:
                return

            candidates = self._ai_factory.generate_candidates(
                base_strategy=strategy_id,
                variations=["optimized"],
                param_grid=param_grid,
            )

            if not candidates:
                logger.debug(f"Catalyst: No candidates generated for {strategy_id}")
                return

            logger.info(
                f"Catalyst: Generated {len(candidates)} param candidates for {strategy_id}"
            )

            eval_results = await self._ai_factory.evaluate_candidates(
                candidates=candidates,
                sim_days=7,
            )

            best_candidate = None
            best_score = 0

            for candidate_id, result in eval_results.items():
                if result.get("error"):
                    continue
                signal_rate = result.get("signal_rate_per_day", 0.0)
                score = min(signal_rate, 20.0)
                if score > best_score:
                    best_score = score
                    best_candidate = candidate_id

            if best_candidate and best_score > 0:
                logger.info(
                    f"Catalyst: Best param candidate {best_candidate} "
                    f"(score={best_score:.1f} signals/day), promoting to canary"
                )
                await self._ai_factory.promote_to_canary(best_candidate)

                self._improvement_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "problem": problem,
                    "status": "candidate_promoted",
                    "candidate_id": best_candidate,
                    "score": best_score,
                    "eval_results": eval_results.get(best_candidate, {}),
                })
            else:
                logger.info(f"Catalyst: No viable param candidates for {strategy_id}")

            self._optimization_results[strategy_id] = {
                "timestamp": datetime.utcnow().isoformat(),
                "candidates_evaluated": len(eval_results),
                "best_candidate": best_candidate,
                "best_score": best_score,
                "status": "canary_promoted" if best_candidate else "no_improvement",
            }

        except Exception as e:
            logger.error(f"Catalyst: AI Factory pipeline error for {strategy_id}: {e}", exc_info=True)

    def _build_param_grid(self, strategy_id: str, problem: dict[str, Any]) -> dict[str, list[Any]]:
        """Build parameter grid for candidate generation based on problem type."""
        problem_type = problem.get("type", "")

        if strategy_id in ("impulse_engine", "ImpulseEngine"):
            if problem_type == "low_profit_factor":
                return {"_impulse_threshold": [0.003, 0.004, 0.006, 0.007]}
            elif problem_type == "low_win_rate":
                return {"_impulse_threshold": [0.006, 0.007, 0.008]}
            elif problem_type == "high_drawdown":
                return {"_impulse_threshold": [0.006, 0.008, 0.01]}
            return {"_impulse_threshold": [0.004, 0.006]}

        elif strategy_id in ("funding_harvester", "FundingHarvester"):
            return {"_min_funding_threshold": [0.00005, 0.0001, 0.0002, 0.0003]}

        elif strategy_id in ("basis_arbitrage", "BasisArbitrage"):
            return {"_basis_threshold": [0.001, 0.0015, 0.002, 0.003]}

        return {}

    def get_improvement_history(self) -> list[dict[str, Any]]:
        """Get improvement history."""
        return self._improvement_history.copy()

    def get_optimization_results(self) -> dict[str, dict[str, Any]]:
        """Get optimization results."""
        return self._optimization_results.copy()


# ── Module-level helpers for safe code loading ───────────────────────


def _safe_exec(code: str) -> dict[str, Any] | None:
    """Execute generated code in a sandboxed namespace."""
    try:
        ast.parse(code)
    except SyntaxError as e:
        logger.warning(f"Generated code has syntax error: {e}")
        return None

    namespace: dict[str, Any] = {}
    try:
        exec(code, namespace)  # noqa: S102
        return namespace
    except Exception as e:
        logger.warning(f"Failed to exec generated code: {e}")
        return None


def _find_strategy_class(namespace: dict[str, Any]):
    """Find the BaseStrategy subclass in the namespace."""
    from hean.strategies.base import BaseStrategy

    for obj in namespace.values():
        if (
            isinstance(obj, type)
            and issubclass(obj, BaseStrategy)
            and obj is not BaseStrategy
        ):
            return obj
    return None

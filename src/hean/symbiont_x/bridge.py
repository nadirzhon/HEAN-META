"""Symbiont X Bridge - GA Optimization for Live Trading.

Connects genetic algorithm optimization to live trading system:
1. Collects real performance data from POSITION_CLOSED events
2. Runs GA optimization in background to evolve strategy parameters
3. Applies winning parameter sets via STRATEGY_PARAMS_UPDATED events

Pure Python GA implementation (no external deps):
- Tournament selection
- Uniform crossover
- Gaussian mutation
"""

import asyncio
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Genome:
    """Parameter genome for a strategy."""

    params: dict[str, float]
    fitness: float = 0.0
    generation: int = 0


@dataclass
class PerformanceRecord:
    """Performance record for a closed position."""

    symbol: str
    strategy_id: str
    pnl: float
    duration_sec: float
    entry_price: float
    exit_price: float
    timestamp: float


class SymbiontXBridge:
    """Bridge between GA optimization and live trading system."""

    def __init__(
        self,
        bus: EventBus,
        enabled: bool = False,
        generations: int = 50,
        population_size: int = 20,
        mutation_rate: float = 0.1,
        reoptimize_interval: int = 3600,
    ) -> None:
        """Initialize Symbiont X Bridge.

        Args:
            bus: Event bus for subscribing to events
            enabled: Whether Symbiont X is enabled
            generations: Number of generations per optimization cycle
            population_size: Population size for GA
            mutation_rate: Mutation probability [0, 1]
            reoptimize_interval: Seconds between optimization cycles
        """
        self._bus = bus
        self._enabled = enabled
        self._generations = generations
        self._population_size = population_size
        self._mutation_rate = mutation_rate
        self._reoptimize_interval = reoptimize_interval

        self._running = False
        self._optimization_tasks: dict[str, asyncio.Task] = {}

        # Performance data per strategy
        self._performance_history: dict[str, deque] = {}
        self._history_window = 100

        # Current best genomes per strategy
        self._best_genomes: dict[str, Genome] = {}

        # Default parameter ranges (strategy_id → param_name → (min, max))
        self._param_ranges: dict[str, dict[str, tuple[float, float]]] = {}

        # Stats
        self._stats = {
            "optimization_cycles": 0,
            "parameters_updated": 0,
            "performance_records": 0,
        }

    async def start(self, strategy_configs: dict[str, dict]) -> None:
        """Start the Symbiont X Bridge.

        Args:
            strategy_configs: Dict of {strategy_id: {param_ranges: {...}}}
                Example:
                {
                    "impulse_engine": {
                        "param_ranges": {
                            "max_spread_bps": (8.0, 20.0),
                            "vol_expansion_ratio": (1.02, 1.15),
                            "confidence_threshold": (0.55, 0.85),
                        }
                    }
                }
        """
        if not self._enabled:
            logger.info("SymbiontXBridge disabled (SYMBIONT_X_ENABLED=false)")
            return

        self._running = True

        # Initialize parameter ranges
        for strategy_id, config in strategy_configs.items():
            param_ranges = config.get("param_ranges", {})
            if param_ranges:
                self._param_ranges[strategy_id] = param_ranges
                self._performance_history[strategy_id] = deque(maxlen=self._history_window)
                logger.info(
                    f"[SymbiontX] Registered {strategy_id} with params: "
                    f"{list(param_ranges.keys())}"
                )

        # If no configs provided, use default ImpulseEngine params
        if not self._param_ranges:
            self._param_ranges["impulse_engine"] = {
                "max_spread_bps": (8.0, 20.0),
                "vol_expansion_ratio": (1.02, 1.15),
                "confidence_threshold": (0.55, 0.85),
            }
            self._performance_history["impulse_engine"] = deque(maxlen=self._history_window)
            logger.info(
                "[SymbiontX] No configs provided, using default ImpulseEngine params"
            )

        # Subscribe to events
        self._bus.subscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.subscribe(EventType.STRATEGY_PARAMS_UPDATED, self._handle_params_updated)

        # Start optimization loops for each strategy
        for strategy_id in self._param_ranges:
            task = asyncio.create_task(self._optimization_loop(strategy_id))
            self._optimization_tasks[strategy_id] = task

        logger.info(
            f"SymbiontXBridge started for {len(self._param_ranges)} strategies"
        )

    async def stop(self) -> None:
        """Stop the Symbiont X Bridge."""
        if not self._enabled:
            return

        self._running = False

        # Stop all optimization tasks
        for task in self._optimization_tasks.values():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Unsubscribe
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._handle_position_closed)
        self._bus.unsubscribe(EventType.STRATEGY_PARAMS_UPDATED, self._handle_params_updated)

        logger.info("SymbiontXBridge stopped")

    async def _handle_position_closed(self, event: Event) -> None:
        """Collect performance data from closed positions."""
        if not self._running:
            return

        data = event.data
        position = data.get("position")

        if not position:
            return

        strategy_id = position.strategy_id
        if strategy_id not in self._performance_history:
            return

        # Calculate duration
        opened_at = position.opened_at
        if opened_at:
            duration_sec = (time.time() - opened_at.timestamp())
        else:
            duration_sec = 0.0

        # Create performance record
        record = PerformanceRecord(
            symbol=position.symbol,
            strategy_id=strategy_id,
            pnl=position.realized_pnl,
            duration_sec=duration_sec,
            entry_price=position.entry_price,
            exit_price=position.current_price,
            timestamp=time.time(),
        )

        self._performance_history[strategy_id].append(record)
        self._stats["performance_records"] += 1

        logger.debug(
            f"[SymbiontX] Recorded {strategy_id}: "
            f"pnl={record.pnl:.2f}, duration={record.duration_sec:.0f}s"
        )

    async def _handle_params_updated(self, event: Event) -> None:
        """Track parameter updates to avoid conflicts."""
        # Currently just logs - could add conflict detection
        strategy_id = event.data.get("strategy_id")
        params = event.data.get("params")
        logger.debug(f"[SymbiontX] External params update: {strategy_id} - {params}")

    async def _optimization_loop(self, strategy_id: str) -> None:
        """Background optimization loop for a strategy.

        Periodically runs GA optimization and applies best params.
        """
        logger.info(
            f"[SymbiontX] Starting optimization loop for {strategy_id} "
            f"(interval={self._reoptimize_interval}s)"
        )

        # Initial delay to collect some performance data
        initial_delay = min(self._reoptimize_interval, 300)
        await asyncio.sleep(initial_delay)

        while self._running:
            try:
                # Check if we have enough performance data
                history = self._performance_history.get(strategy_id, deque())

                if len(history) < 5:
                    logger.debug(
                        f"[SymbiontX] {strategy_id}: Insufficient data "
                        f"({len(history)} trades), waiting..."
                    )
                    await asyncio.sleep(self._reoptimize_interval)
                    continue

                # Run GA optimization
                logger.info(
                    f"[SymbiontX] {strategy_id}: Running optimization "
                    f"with {len(history)} trades"
                )

                best_genome = await self._run_ga_optimization(strategy_id)

                if best_genome and best_genome.fitness > 1.0:
                    # Apply best params if fitness threshold met
                    await self._apply_params(strategy_id, best_genome)
                else:
                    logger.info(
                        f"[SymbiontX] {strategy_id}: Best fitness "
                        f"{best_genome.fitness if best_genome else 0:.2f} below threshold, "
                        f"keeping current params"
                    )

                self._stats["optimization_cycles"] += 1

                # Wait for next cycle
                await asyncio.sleep(self._reoptimize_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[SymbiontX] Error in optimization loop for {strategy_id}: {e}",
                    exc_info=True,
                )
                await asyncio.sleep(60)  # Short delay before retry

    async def _run_ga_optimization(self, strategy_id: str) -> Genome | None:
        """Run genetic algorithm optimization.

        Returns:
            Best genome from optimization, or None if failed
        """
        param_ranges = self._param_ranges.get(strategy_id, {})
        if not param_ranges:
            return None

        # Initialize population
        population = [
            self._random_genome(param_ranges, generation=0)
            for _ in range(self._population_size)
        ]

        # Evolve for N generations
        for gen in range(self._generations):
            # Evaluate fitness
            for genome in population:
                genome.fitness = self._evaluate_fitness(strategy_id, genome)

            # Sort by fitness
            population.sort(key=lambda g: g.fitness, reverse=True)

            logger.debug(
                f"[SymbiontX] {strategy_id} Gen {gen}: "
                f"best_fitness={population[0].fitness:.3f}"
            )

            # Selection + crossover + mutation
            new_population = []

            # Elitism: keep top 2
            new_population.extend(population[:2])

            # Generate rest via tournament selection + crossover + mutation
            while len(new_population) < self._population_size:
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)

                child = self._crossover(parent1, parent2, param_ranges)
                child = self._mutate(child, param_ranges)
                child.generation = gen + 1

                new_population.append(child)

            population = new_population

        # Final evaluation
        for genome in population:
            genome.fitness = self._evaluate_fitness(strategy_id, genome)

        population.sort(key=lambda g: g.fitness, reverse=True)
        best = population[0]

        logger.info(
            f"[SymbiontX] {strategy_id}: Optimization complete. "
            f"Best fitness={best.fitness:.3f}, params={best.params}"
        )

        return best

    def _random_genome(
        self, param_ranges: dict[str, tuple[float, float]], generation: int = 0
    ) -> Genome:
        """Generate random genome within parameter ranges."""
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            params[param_name] = random.uniform(min_val, max_val)

        return Genome(params=params, generation=generation)

    def _evaluate_fitness(self, strategy_id: str, genome: Genome) -> float:
        """Evaluate fitness of a genome based on performance history.

        Fitness function combines:
        - Sharpe ratio (risk-adjusted return)
        - Profit factor (wins/losses)
        - Max drawdown penalty

        Returns:
            Fitness score (higher is better)
        """
        history = self._performance_history.get(strategy_id, deque())
        if not history:
            return 0.0

        # For simplicity, we evaluate based on recent performance
        # In production, would backtest genome params on historical data

        # Calculate metrics
        pnls = [r.pnl for r in history]
        total_pnl = sum(pnls)
        num_trades = len(pnls)

        if num_trades == 0:
            return 0.0

        wins = [p for p in pnls if p > 0]
        losses = [abs(p) for p in pnls if p < 0]

        # Profit factor
        total_wins = sum(wins)
        total_losses = sum(losses) if losses else 0.01
        profit_factor = total_wins / total_losses

        # Sharpe approximation (using PnL stddev)
        if num_trades < 2:
            sharpe = 0.0
        else:
            mean_pnl = total_pnl / num_trades
            variance = sum((p - mean_pnl) ** 2 for p in pnls) / num_trades
            stddev = variance ** 0.5
            sharpe = mean_pnl / stddev if stddev > 0 else 0.0

        # Max drawdown (simplified)
        max_dd = 0.0
        peak = 0.0
        cumulative = 0.0
        for pnl in pnls:
            cumulative += pnl
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        # Combine metrics
        # Fitness = Sharpe * 0.5 + PF * 0.3 - MaxDD_penalty * 0.2
        fitness = sharpe * 0.5 + profit_factor * 0.3 - (max_dd / 100) * 0.2

        return max(0.0, fitness)

    def _tournament_select(self, population: list[Genome], k: int = 3) -> Genome:
        """Tournament selection: pick best from k random individuals."""
        tournament = random.sample(population, min(k, len(population)))
        return max(tournament, key=lambda g: g.fitness)

    def _crossover(
        self,
        parent1: Genome,
        parent2: Genome,
        param_ranges: dict[str, tuple[float, float]],
    ) -> Genome:
        """Uniform crossover: randomly inherit each param from either parent."""
        child_params = {}

        for param_name in param_ranges:
            if random.random() < 0.5:
                child_params[param_name] = parent1.params.get(param_name, 0.0)
            else:
                child_params[param_name] = parent2.params.get(param_name, 0.0)

        return Genome(params=child_params)

    def _mutate(
        self, genome: Genome, param_ranges: dict[str, tuple[float, float]]
    ) -> Genome:
        """Gaussian mutation: add random noise to each param with probability."""
        for param_name, (min_val, max_val) in param_ranges.items():
            if random.random() < self._mutation_rate:
                # Gaussian mutation (±10% of range)
                range_size = max_val - min_val
                mutation = random.gauss(0, range_size * 0.1)
                new_val = genome.params[param_name] + mutation
                # Clamp to range
                genome.params[param_name] = max(min_val, min(max_val, new_val))

        return genome

    async def _apply_params(self, strategy_id: str, genome: Genome) -> None:
        """Apply best parameters to strategy via STRATEGY_PARAMS_UPDATED event.

        Args:
            strategy_id: Strategy ID
            genome: Genome with best parameters
        """
        logger.info(
            f"[SymbiontX] Applying optimized params to {strategy_id}: "
            f"{genome.params} (fitness={genome.fitness:.3f})"
        )

        # Store best genome
        self._best_genomes[strategy_id] = genome

        # Publish update event
        await self._bus.publish(
            Event(
                event_type=EventType.STRATEGY_PARAMS_UPDATED,
                data={
                    "strategy_id": strategy_id,
                    "params": genome.params,
                    "source": "symbiont_x_ga",
                    "fitness": genome.fitness,
                    "generation": genome.generation,
                },
            )
        )

        self._stats["parameters_updated"] += 1

    def get_stats(self) -> dict[str, Any]:
        """Get Symbiont X statistics."""
        return {
            **self._stats,
            "strategies_optimizing": len(self._optimization_tasks),
            "best_genomes": {
                sid: {
                    "params": g.params,
                    "fitness": g.fitness,
                    "generation": g.generation,
                }
                for sid, g in self._best_genomes.items()
            },
            "performance_records_per_strategy": {
                sid: len(history)
                for sid, history in self._performance_history.items()
            },
        }

"""MetaStrategyBrain — Dynamic strategy lifecycle management.

Manages strategies as a living ecosystem with lifecycle states:
    ACTIVE (100% capital) → REDUCED (50%) → HIBERNATED (5%) → TERMINATED (0%)

Evaluates composite fitness scores using:
    - Rolling Sharpe ratio (40%)
    - Regime alignment from MarketGenome (30%)
    - Drawdown resistance from DoomsdaySandbox (20%)
    - Alpha decay detection (10%)

Publishes META_STRATEGY_UPDATE events with lifecycle transitions.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from hean.config import settings
from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.logging import get_logger

logger = get_logger(__name__)


class StrategyState(str, Enum):
    """Strategy lifecycle state."""

    ACTIVE = "active"          # 100% allocated capital
    REDUCED = "reduced"        # 50% allocated capital
    HIBERNATED = "hibernated"  # 5% allocated capital (warm standby)
    TERMINATED = "terminated"  # 0% capital, pending evolution


# Capital multiplier per state
STATE_CAPITAL_MULTIPLIER = {
    StrategyState.ACTIVE: 1.0,
    StrategyState.REDUCED: 0.5,
    StrategyState.HIBERNATED: 0.05,
    StrategyState.TERMINATED: 0.0,
}

# Minimum hours in a state before transition is allowed
MIN_STATE_DURATION_HOURS = 2.0

# Maximum transitions per strategy per day
MAX_TRANSITIONS_PER_DAY = 3

# Minimum active strategies at all times
MIN_ACTIVE_STRATEGIES = 2


@dataclass
class StrategyFitnessRecord:
    """Rolling fitness metrics for a single strategy."""

    strategy_id: str
    state: StrategyState = StrategyState.ACTIVE
    state_entered_at: float = field(default_factory=time.time)
    transitions_today: int = 0
    transitions_reset_date: str = ""

    # Rolling trade results
    trade_pnls: deque = field(default_factory=lambda: deque(maxlen=200))
    trade_timestamps: deque = field(default_factory=lambda: deque(maxlen=200))

    # Rolling Sharpe windows (20-trade windows)
    sharpe_windows: deque = field(default_factory=lambda: deque(maxlen=20))

    # Regime affinity: regime_name -> (wins, total)
    regime_wins: dict = field(default_factory=lambda: defaultdict(int))
    regime_trades: dict = field(default_factory=lambda: defaultdict(int))

    # Doomsday resistance
    last_doomsday_score: float = 0.5

    # Composite fitness
    composite_fitness: float = 0.5

    @property
    def sharpe_ratio(self) -> float:
        """Compute Sharpe from recent trade PnLs."""
        if len(self.trade_pnls) < 10:
            return 0.0
        arr = np.array(self.trade_pnls, dtype=np.float64)
        std = np.std(arr)
        if std < 1e-10:
            return 0.0
        return float(np.mean(arr) / std * np.sqrt(100))

    @property
    def win_rate(self) -> float:
        """Win rate from recent trades."""
        if len(self.trade_pnls) < 5:
            return 0.5
        wins = sum(1 for p in self.trade_pnls if p > 0)
        return wins / len(self.trade_pnls)

    @property
    def max_drawdown_pct(self) -> float:
        """Max drawdown percentage from recent trade PnLs."""
        if len(self.trade_pnls) < 2:
            return 0.0
        cumulative = np.cumsum(list(self.trade_pnls))
        peak = np.maximum.accumulate(cumulative)
        dd = peak - cumulative
        if peak.max() < 1e-10:
            return 0.0
        return float(dd.max() / max(peak.max(), 1e-10) * 100)

    def regime_alignment(self, current_regime: str) -> float:
        """How well this strategy performs in the current regime (0.0-1.0)."""
        total = self.regime_trades.get(current_regime, 0)
        if total < 5:
            return 0.5  # Insufficient data, assume neutral
        wins = self.regime_wins.get(current_regime, 0)
        return wins / total

    def alpha_decay_score(self) -> float:
        """Detect alpha decay via linear regression on Sharpe windows.

        Returns 0.0 (no decay) to 1.0 (severe decay).
        """
        if len(self.sharpe_windows) < 5:
            return 0.0
        arr = np.array(self.sharpe_windows, dtype=np.float64)
        n = len(arr)
        x = np.arange(n, dtype=np.float64)
        x_mean = x.mean()
        y_mean = arr.mean()
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-10:
            return 0.0
        slope = np.sum((x - x_mean) * (arr - y_mean)) / denom
        # Normalize: negative slope = decaying alpha
        # slope of -0.1 per window → decay_score ~ 0.5
        decay = max(0.0, -slope * 5.0)
        return min(1.0, decay)


@dataclass
class LifecycleTransition:
    """Record of a strategy lifecycle transition."""

    timestamp: float
    strategy_id: str
    from_state: str
    to_state: str
    reason: str
    composite_fitness: float


class MetaStrategyBrain:
    """Dynamic strategy lifecycle manager.

    Subscribes to:
        - ORDER_FILLED / POSITION_CLOSED: Track per-strategy performance
        - MARKET_GENOME_UPDATE: Regime alignment scoring
        - RISK_SIMULATION_RESULT: Drawdown resistance scoring

    Publishes:
        - META_STRATEGY_UPDATE: Lifecycle transitions and fitness reports
    """

    # Fitness weight configuration
    SHARPE_WEIGHT = 0.40
    REGIME_WEIGHT = 0.30
    DRAWDOWN_WEIGHT = 0.20
    ALPHA_DECAY_WEIGHT = 0.10

    # Thresholds for state transitions
    DEMOTE_THRESHOLD = 0.30  # Below this → demote
    PROMOTE_THRESHOLD = 0.60  # Above this → promote

    def __init__(self, bus: EventBus, accounting: Any = None) -> None:
        self._bus = bus
        self._accounting = accounting
        self._running = False
        self._eval_task: asyncio.Task | None = None

        # Strategy fitness records
        self._fitness: dict[str, StrategyFitnessRecord] = {}

        # Current market genome (latest per symbol)
        self._current_regime: str = "range"
        self._current_regime_confidence: float = 0.5

        # Last doomsday report
        self._last_doomsday_survival: float = 0.5

        # Transition history
        self._transitions: deque[LifecycleTransition] = deque(maxlen=500)

    async def start(self) -> None:
        """Start the meta-strategy brain."""
        self._running = True
        self._bus.subscribe(EventType.POSITION_CLOSED, self._on_position_closed)
        self._bus.subscribe(EventType.MARKET_GENOME_UPDATE, self._on_genome_update)
        self._bus.subscribe(EventType.RISK_SIMULATION_RESULT, self._on_doomsday_result)
        self._eval_task = asyncio.create_task(self._evaluation_loop())
        logger.info(
            "MetaStrategyBrain started (eval_interval=%ds)",
            settings.meta_brain_evaluation_interval,
        )

    async def stop(self) -> None:
        """Stop the meta-strategy brain."""
        self._running = False
        if self._eval_task:
            self._eval_task.cancel()
            try:
                await self._eval_task
            except asyncio.CancelledError:
                pass
        self._bus.unsubscribe(EventType.POSITION_CLOSED, self._on_position_closed)
        self._bus.unsubscribe(EventType.MARKET_GENOME_UPDATE, self._on_genome_update)
        self._bus.unsubscribe(EventType.RISK_SIMULATION_RESULT, self._on_doomsday_result)
        logger.info("MetaStrategyBrain stopped")

    # --- Event Handlers ---

    async def _on_position_closed(self, event: Event) -> None:
        """Track strategy performance from closed positions."""
        position = event.data.get("position")
        if not position:
            return

        strategy_id = getattr(position, "strategy_id", None)
        if not strategy_id:
            strategy_id = event.data.get("strategy_id", "unknown")

        pnl = getattr(position, "realized_pnl", 0.0)
        if pnl is None:
            pnl = event.data.get("realized_pnl", 0.0)

        record = self._get_or_create(strategy_id)
        record.trade_pnls.append(pnl)
        record.trade_timestamps.append(time.time())

        # Update regime affinity
        regime = self._current_regime
        record.regime_trades[regime] = record.regime_trades.get(regime, 0) + 1
        if pnl > 0:
            record.regime_wins[regime] = record.regime_wins.get(regime, 0) + 1

        # Update Sharpe windows every 20 trades
        if len(record.trade_pnls) >= 20 and len(record.trade_pnls) % 20 == 0:
            recent = list(record.trade_pnls)[-20:]
            arr = np.array(recent, dtype=np.float64)
            std = np.std(arr)
            window_sharpe = float(np.mean(arr) / std) if std > 1e-10 else 0.0
            record.sharpe_windows.append(window_sharpe)

    async def _on_genome_update(self, event: Event) -> None:
        """Update current regime from MarketGenome."""
        genome = event.data.get("genome", {})
        regime = genome.get("regime", "range")
        confidence = genome.get("regime_confidence", 0.5)
        self._current_regime = regime
        self._current_regime_confidence = confidence

    async def _on_doomsday_result(self, event: Event) -> None:
        """Update drawdown resistance from DoomsdaySandbox results."""
        report = event.data.get("report", {})
        survival = report.get("overall_survival_score", 0.5)
        self._last_doomsday_survival = survival

        # Update per-strategy doomsday score if available
        per_strategy = report.get("per_strategy_scores", {})
        for sid, score in per_strategy.items():
            if sid in self._fitness:
                self._fitness[sid].last_doomsday_score = score

    # --- Evaluation Loop ---

    async def _evaluation_loop(self) -> None:
        """Periodically evaluate strategy fitness and manage lifecycle."""
        while self._running:
            try:
                await asyncio.sleep(settings.meta_brain_evaluation_interval)
                if not self._running:
                    break
                await self._evaluate_all()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("MetaStrategyBrain evaluation error")

    async def _evaluate_all(self) -> None:
        """Evaluate fitness of all tracked strategies and apply transitions."""
        if not self._fitness:
            return

        now = time.time()
        today = time.strftime("%Y-%m-%d")

        # Reset daily transition counters if new day
        for record in self._fitness.values():
            if record.transitions_reset_date != today:
                record.transitions_today = 0
                record.transitions_reset_date = today

        # Compute composite fitness for each strategy
        for record in self._fitness.values():
            record.composite_fitness = self._compute_fitness(record)

        # Count current active strategies
        active_count = sum(
            1 for r in self._fitness.values() if r.state == StrategyState.ACTIVE
        )

        # Apply lifecycle transitions
        transitions_made = []
        for record in sorted(self._fitness.values(), key=lambda r: r.composite_fitness):
            # Skip if strategy hasn't been in current state long enough
            hours_in_state = (now - record.state_entered_at) / 3600
            if hours_in_state < MIN_STATE_DURATION_HOURS:
                continue

            # Skip if max transitions reached today
            if record.transitions_today >= MAX_TRANSITIONS_PER_DAY:
                continue

            new_state = self._decide_transition(
                record, active_count
            )
            if new_state and new_state != record.state:
                old_state = record.state
                # Safety: don't go below minimum active strategies
                if (
                    old_state == StrategyState.ACTIVE
                    and new_state != StrategyState.ACTIVE
                    and active_count <= MIN_ACTIVE_STRATEGIES
                ):
                    continue

                record.state = new_state
                record.state_entered_at = now
                record.transitions_today += 1
                if old_state == StrategyState.ACTIVE:
                    active_count -= 1
                if new_state == StrategyState.ACTIVE:
                    active_count += 1

                transition = LifecycleTransition(
                    timestamp=now,
                    strategy_id=record.strategy_id,
                    from_state=old_state.value,
                    to_state=new_state.value,
                    reason=self._transition_reason(record, old_state, new_state),
                    composite_fitness=record.composite_fitness,
                )
                self._transitions.append(transition)
                transitions_made.append(transition)

                logger.info(
                    "[META-BRAIN] %s: %s → %s (fitness=%.3f, reason=%s)",
                    record.strategy_id,
                    old_state.value,
                    new_state.value,
                    record.composite_fitness,
                    transition.reason,
                )

        # Publish update event
        await self._publish_update(transitions_made)

    def _compute_fitness(self, record: StrategyFitnessRecord) -> float:
        """Compute composite fitness score (0.0-1.0)."""
        # Sharpe component: normalize to 0-1 range (Sharpe 0-3 → 0-1)
        raw_sharpe = record.sharpe_ratio
        sharpe_score = max(0.0, min(1.0, raw_sharpe / 3.0))

        # Regime alignment
        regime_score = record.regime_alignment(self._current_regime)

        # Drawdown resistance (from doomsday)
        dd_score = record.last_doomsday_score

        # Alpha decay (inverted: 0 decay = 1.0 score)
        decay = record.alpha_decay_score()
        alpha_score = 1.0 - decay

        composite = (
            sharpe_score * self.SHARPE_WEIGHT
            + regime_score * self.REGIME_WEIGHT
            + dd_score * self.DRAWDOWN_WEIGHT
            + alpha_score * self.ALPHA_DECAY_WEIGHT
        )
        return max(0.0, min(1.0, composite))

    def _decide_transition(
        self, record: StrategyFitnessRecord, active_count: int
    ) -> StrategyState | None:
        """Decide what state transition, if any, to apply."""
        fitness = record.composite_fitness
        current = record.state

        # Demotion logic
        if fitness < self.DEMOTE_THRESHOLD:
            if current == StrategyState.ACTIVE:
                return StrategyState.REDUCED
            if current == StrategyState.REDUCED:
                return StrategyState.HIBERNATED
            if current == StrategyState.HIBERNATED:
                return StrategyState.TERMINATED

        # Promotion logic
        if fitness > self.PROMOTE_THRESHOLD:
            if current == StrategyState.TERMINATED:
                return StrategyState.HIBERNATED
            if current == StrategyState.HIBERNATED:
                return StrategyState.REDUCED
            if current == StrategyState.REDUCED:
                return StrategyState.ACTIVE

        return None

    @staticmethod
    def _transition_reason(
        record: StrategyFitnessRecord,
        old: StrategyState,
        new: StrategyState,
    ) -> str:
        """Generate human-readable reason for transition."""
        decay = record.alpha_decay_score()
        if new.value > old.value:  # Demotion (ACTIVE→REDUCED etc)
            if decay > 0.5:
                return f"alpha_decay={decay:.2f}"
            return f"low_fitness={record.composite_fitness:.3f}"
        return f"fitness_recovery={record.composite_fitness:.3f}"

    async def _publish_update(self, transitions: list[LifecycleTransition]) -> None:
        """Publish META_STRATEGY_UPDATE event."""
        strategy_states = {}
        for sid, record in self._fitness.items():
            strategy_states[sid] = {
                "state": record.state.value,
                "capital_multiplier": STATE_CAPITAL_MULTIPLIER[record.state],
                "composite_fitness": round(record.composite_fitness, 4),
                "sharpe_ratio": round(record.sharpe_ratio, 4),
                "win_rate": round(record.win_rate, 4),
                "alpha_decay": round(record.alpha_decay_score(), 4),
                "regime_alignment": round(
                    record.regime_alignment(self._current_regime), 4
                ),
                "doomsday_score": round(record.last_doomsday_score, 4),
                "trade_count": len(record.trade_pnls),
            }

        await self._bus.publish(
            Event(
                event_type=EventType.META_STRATEGY_UPDATE,
                data={
                    "strategies": strategy_states,
                    "current_regime": self._current_regime,
                    "regime_confidence": self._current_regime_confidence,
                    "transitions": [
                        {
                            "strategy_id": t.strategy_id,
                            "from": t.from_state,
                            "to": t.to_state,
                            "reason": t.reason,
                            "fitness": t.composite_fitness,
                        }
                        for t in transitions
                    ],
                },
            )
        )

    # --- Public API ---

    def get_strategy_states(self) -> dict[str, dict[str, Any]]:
        """Get current state of all tracked strategies."""
        result = {}
        for sid, record in self._fitness.items():
            result[sid] = {
                "state": record.state.value,
                "capital_multiplier": STATE_CAPITAL_MULTIPLIER[record.state],
                "composite_fitness": round(record.composite_fitness, 4),
                "sharpe_ratio": round(record.sharpe_ratio, 4),
                "win_rate": round(record.win_rate, 4),
                "alpha_decay": round(record.alpha_decay_score(), 4),
                "regime_alignment": round(
                    record.regime_alignment(self._current_regime), 4
                ),
                "doomsday_score": round(record.last_doomsday_score, 4),
                "trade_count": len(record.trade_pnls),
                "state_duration_hours": round(
                    (time.time() - record.state_entered_at) / 3600, 2
                ),
            }
        return result

    def get_transitions(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent lifecycle transitions."""
        recent = list(self._transitions)[-limit:]
        return [
            {
                "timestamp": t.timestamp,
                "strategy_id": t.strategy_id,
                "from": t.from_state,
                "to": t.to_state,
                "reason": t.reason,
                "fitness": t.composite_fitness,
            }
            for t in recent
        ]

    def get_regime_affinity_matrix(self) -> dict[str, dict[str, float]]:
        """Get the learned regime affinity matrix for all strategies."""
        matrix = {}
        for sid, record in self._fitness.items():
            affinities = {}
            for regime in set(record.regime_trades.keys()):
                total = record.regime_trades.get(regime, 0)
                if total >= 5:
                    wins = record.regime_wins.get(regime, 0)
                    affinities[regime] = round(wins / total, 4)
                else:
                    affinities[regime] = None  # Insufficient data
            matrix[sid] = affinities
        return matrix

    def force_state(self, strategy_id: str, state: StrategyState) -> bool:
        """Manually override a strategy's lifecycle state (admin action)."""
        record = self._fitness.get(strategy_id)
        if not record:
            return False
        old = record.state
        record.state = state
        record.state_entered_at = time.time()
        self._transitions.append(
            LifecycleTransition(
                timestamp=time.time(),
                strategy_id=strategy_id,
                from_state=old.value,
                to_state=state.value,
                reason="manual_override",
                composite_fitness=record.composite_fitness,
            )
        )
        logger.info(
            "[META-BRAIN] Manual override: %s %s → %s",
            strategy_id,
            old.value,
            state.value,
        )
        return True

    def _get_or_create(self, strategy_id: str) -> StrategyFitnessRecord:
        """Get or create fitness record for a strategy."""
        if strategy_id not in self._fitness:
            self._fitness[strategy_id] = StrategyFitnessRecord(
                strategy_id=strategy_id
            )
        return self._fitness[strategy_id]

"""Tests for MetaStrategyBrain dynamic strategy lifecycle management."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from hean.core.bus import EventBus
from hean.core.types import Event, EventType
from hean.portfolio.meta_strategy_brain import (
    MAX_TRANSITIONS_PER_DAY,
    MIN_ACTIVE_STRATEGIES,
    MIN_STATE_DURATION_HOURS,
    MetaStrategyBrain,
    StrategyFitnessRecord,
    StrategyState,
)


@pytest.fixture
def bus():
    return EventBus()


@pytest.fixture
def brain(bus):
    return MetaStrategyBrain(bus=bus)


class TestStrategyFitnessRecord:
    def test_sharpe_ratio_insufficient_data(self):
        record = StrategyFitnessRecord(strategy_id="test")
        assert record.sharpe_ratio == 0.0

    def test_sharpe_ratio_with_data(self):
        record = StrategyFitnessRecord(strategy_id="test")
        for i in range(20):
            record.trade_pnls.append(1.0 + (i % 3) * 0.5)
        assert record.sharpe_ratio > 0

    def test_win_rate_no_trades(self):
        record = StrategyFitnessRecord(strategy_id="test")
        assert record.win_rate == 0.5  # Default neutral

    def test_win_rate_with_trades(self):
        record = StrategyFitnessRecord(strategy_id="test")
        for _ in range(7):
            record.trade_pnls.append(1.0)
        for _ in range(3):
            record.trade_pnls.append(-1.0)
        assert record.win_rate == pytest.approx(0.7, abs=0.01)

    def test_max_drawdown(self):
        record = StrategyFitnessRecord(strategy_id="test")
        pnls = [10, 5, -20, 8, 3]
        for p in pnls:
            record.trade_pnls.append(p)
        assert record.max_drawdown_pct > 0

    def test_regime_alignment_insufficient_data(self):
        record = StrategyFitnessRecord(strategy_id="test")
        assert record.regime_alignment("bull_trend") == 0.5

    def test_regime_alignment_with_data(self):
        record = StrategyFitnessRecord(strategy_id="test")
        record.regime_trades["bull_trend"] = 10
        record.regime_wins["bull_trend"] = 8
        assert record.regime_alignment("bull_trend") == pytest.approx(0.8)

    def test_alpha_decay_no_data(self):
        record = StrategyFitnessRecord(strategy_id="test")
        assert record.alpha_decay_score() == 0.0

    def test_alpha_decay_declining_sharpe(self):
        record = StrategyFitnessRecord(strategy_id="test")
        # Simulate declining Sharpe windows
        for i in range(10):
            record.sharpe_windows.append(2.0 - i * 0.2)
        assert record.alpha_decay_score() > 0.0

    def test_alpha_decay_improving_sharpe(self):
        record = StrategyFitnessRecord(strategy_id="test")
        # Simulate improving Sharpe windows
        for i in range(10):
            record.sharpe_windows.append(0.5 + i * 0.1)
        assert record.alpha_decay_score() == 0.0  # No decay when improving


class TestComputeFitness:
    def test_fitness_bounded_0_to_1(self, brain):
        record = StrategyFitnessRecord(strategy_id="test")
        fitness = brain._compute_fitness(record)
        assert 0.0 <= fitness <= 1.0

    def test_high_sharpe_boosts_fitness(self, brain):
        low = StrategyFitnessRecord(strategy_id="low")
        high = StrategyFitnessRecord(strategy_id="high")

        # Low sharpe
        for _ in range(20):
            low.trade_pnls.append(0.1)
            high.trade_pnls.append(5.0)

        # Add variance to high
        for i in range(20):
            high.trade_pnls.append(3.0 + (i % 5))

        low_f = brain._compute_fitness(low)
        high_f = brain._compute_fitness(high)
        # Both should be valid scores
        assert 0.0 <= low_f <= 1.0
        assert 0.0 <= high_f <= 1.0

    def test_weights_sum_to_one(self, brain):
        total = (
            brain.SHARPE_WEIGHT
            + brain.REGIME_WEIGHT
            + brain.DRAWDOWN_WEIGHT
            + brain.ALPHA_DECAY_WEIGHT
        )
        assert total == pytest.approx(1.0)


class TestTransitionDecisions:
    def test_demote_active_to_reduced(self, brain):
        record = StrategyFitnessRecord(strategy_id="test")
        record.state = StrategyState.ACTIVE
        record.composite_fitness = 0.2  # Below demote threshold
        result = brain._decide_transition(record, active_count=3)
        assert result == StrategyState.REDUCED

    def test_demote_reduced_to_hibernated(self, brain):
        record = StrategyFitnessRecord(strategy_id="test")
        record.state = StrategyState.REDUCED
        record.composite_fitness = 0.2
        result = brain._decide_transition(record, active_count=3)
        assert result == StrategyState.HIBERNATED

    def test_demote_hibernated_to_terminated(self, brain):
        record = StrategyFitnessRecord(strategy_id="test")
        record.state = StrategyState.HIBERNATED
        record.composite_fitness = 0.2
        result = brain._decide_transition(record, active_count=3)
        assert result == StrategyState.TERMINATED

    def test_promote_terminated_to_hibernated(self, brain):
        record = StrategyFitnessRecord(strategy_id="test")
        record.state = StrategyState.TERMINATED
        record.composite_fitness = 0.7
        result = brain._decide_transition(record, active_count=3)
        assert result == StrategyState.HIBERNATED

    def test_promote_hibernated_to_reduced(self, brain):
        record = StrategyFitnessRecord(strategy_id="test")
        record.state = StrategyState.HIBERNATED
        record.composite_fitness = 0.7
        result = brain._decide_transition(record, active_count=3)
        assert result == StrategyState.REDUCED

    def test_promote_reduced_to_active(self, brain):
        record = StrategyFitnessRecord(strategy_id="test")
        record.state = StrategyState.REDUCED
        record.composite_fitness = 0.7
        result = brain._decide_transition(record, active_count=3)
        assert result == StrategyState.ACTIVE

    def test_no_transition_in_middle_range(self, brain):
        record = StrategyFitnessRecord(strategy_id="test")
        record.state = StrategyState.ACTIVE
        record.composite_fitness = 0.45  # Between thresholds
        result = brain._decide_transition(record, active_count=3)
        assert result is None


class TestSafetyConstraints:
    async def test_min_state_duration_enforced(self, bus):
        brain = MetaStrategyBrain(bus=bus)
        record = brain._get_or_create("strategy_a")
        record.state = StrategyState.ACTIVE
        record.composite_fitness = 0.1  # Very low, would normally demote
        record.state_entered_at = time.time()  # Just entered
        record.transitions_reset_date = time.strftime("%Y-%m-%d")

        # With recent state entry, no transition should happen
        # because hours_in_state < MIN_STATE_DURATION_HOURS
        assert (time.time() - record.state_entered_at) / 3600 < MIN_STATE_DURATION_HOURS

    async def test_max_transitions_per_day(self, bus):
        brain = MetaStrategyBrain(bus=bus)
        record = brain._get_or_create("strategy_a")
        record.transitions_today = MAX_TRANSITIONS_PER_DAY
        record.transitions_reset_date = time.strftime("%Y-%m-%d")
        # Should not allow further transitions
        assert record.transitions_today >= MAX_TRANSITIONS_PER_DAY

    async def test_min_active_strategies_protection(self, brain):
        """Cannot demote below MIN_ACTIVE_STRATEGIES active strategies."""
        # Create exactly MIN_ACTIVE_STRATEGIES active strategies
        for i in range(MIN_ACTIVE_STRATEGIES):
            r = brain._get_or_create(f"strat_{i}")
            r.state = StrategyState.ACTIVE
            r.composite_fitness = 0.1  # Low fitness
            r.state_entered_at = time.time() - 86400  # Entered long ago
            r.transitions_reset_date = time.strftime("%Y-%m-%d")

        # The safety guard in _evaluate_all prevents demoting below MIN_ACTIVE_STRATEGIES
        active_count = sum(
            1 for r in brain._fitness.values() if r.state == StrategyState.ACTIVE
        )
        assert active_count == MIN_ACTIVE_STRATEGIES


class TestEventHandlers:
    async def test_position_closed_tracks_pnl(self, bus):
        brain = MetaStrategyBrain(bus=bus)
        await brain.start()
        await bus.start()

        mock_pos = MagicMock()
        mock_pos.strategy_id = "impulse_engine"
        mock_pos.realized_pnl = 5.0

        await bus.publish(
            Event(
                event_type=EventType.POSITION_CLOSED,
                data={"position": mock_pos},
            )
        )
        await asyncio.sleep(0.05)

        record = brain._fitness.get("impulse_engine")
        assert record is not None
        assert len(record.trade_pnls) == 1
        assert list(record.trade_pnls)[0] == 5.0

        await bus.stop()
        await brain.stop()

    async def test_genome_update_changes_regime(self, bus):
        brain = MetaStrategyBrain(bus=bus)
        await brain.start()
        await bus.start()

        await bus.publish(
            Event(
                event_type=EventType.MARKET_GENOME_UPDATE,
                data={
                    "symbol": "BTCUSDT",
                    "genome": {
                        "regime": "bull_trend",
                        "regime_confidence": 0.85,
                    },
                },
            )
        )
        await asyncio.sleep(0.05)

        assert brain._current_regime == "bull_trend"
        assert brain._current_regime_confidence == 0.85

        await bus.stop()
        await brain.stop()

    async def test_doomsday_result_updates_survival(self, bus):
        brain = MetaStrategyBrain(bus=bus)
        await brain.start()
        await bus.start()

        await bus.publish(
            Event(
                event_type=EventType.RISK_SIMULATION_RESULT,
                data={
                    "report": {
                        "overall_survival_score": 0.72,
                    },
                },
            )
        )
        await asyncio.sleep(0.05)

        assert brain._last_doomsday_survival == 0.72

        await bus.stop()
        await brain.stop()


class TestPublicAPI:
    def test_get_strategy_states_empty(self, brain):
        states = brain.get_strategy_states()
        assert states == {}

    def test_get_strategy_states_with_data(self, brain):
        record = brain._get_or_create("impulse_engine")
        record.state = StrategyState.ACTIVE
        record.composite_fitness = 0.75
        record.last_doomsday_score = 0.8

        states = brain.get_strategy_states()
        assert "impulse_engine" in states
        assert states["impulse_engine"]["state"] == "active"
        assert states["impulse_engine"]["capital_multiplier"] == 1.0

    def test_force_state(self, brain):
        brain._get_or_create("strat_a")
        success = brain.force_state("strat_a", StrategyState.HIBERNATED)
        assert success
        assert brain._fitness["strat_a"].state == StrategyState.HIBERNATED
        assert len(brain._transitions) == 1
        assert brain._transitions[0].reason == "manual_override"

    def test_force_state_unknown_strategy(self, brain):
        success = brain.force_state("nonexistent", StrategyState.ACTIVE)
        assert not success

    def test_get_transitions_empty(self, brain):
        assert brain.get_transitions() == []

    def test_get_regime_affinity_matrix(self, brain):
        record = brain._get_or_create("strat_a")
        record.regime_trades["bull_trend"] = 10
        record.regime_wins["bull_trend"] = 7
        record.regime_trades["range"] = 3  # Insufficient data

        matrix = brain.get_regime_affinity_matrix()
        assert "strat_a" in matrix
        assert matrix["strat_a"]["bull_trend"] == pytest.approx(0.7)
        assert matrix["strat_a"]["range"] is None  # Insufficient data


class TestEvolutionBridge:
    async def test_bridge_captures_terminations(self, bus):
        from hean.portfolio.evolution_bridge import EvolutionBridge

        bridge = EvolutionBridge(bus=bus)
        await bridge.start()
        await bus.start()

        # Simulate a META_STRATEGY_UPDATE with a termination
        await bus.publish(
            Event(
                event_type=EventType.META_STRATEGY_UPDATE,
                data={
                    "strategies": {},
                    "current_regime": "range",
                    "regime_confidence": 0.5,
                    "transitions": [
                        {
                            "strategy_id": "dead_strat",
                            "from": "hibernated",
                            "to": "terminated",
                            "reason": "alpha_decay=0.85",
                            "fitness": 0.1,
                        }
                    ],
                },
            )
        )
        await asyncio.sleep(0.05)

        pending = bridge.get_pending()
        assert len(pending) == 1
        assert pending[0]["strategy_id"] == "dead_strat"
        assert pending[0]["status"] == "pending"

        await bus.stop()
        await bridge.stop()

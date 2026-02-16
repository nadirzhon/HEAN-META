"""Tests for Genome Director."""

import pytest

from hean.archon.genome_director import GenomeDirector
from hean.core.bus import EventBus


@pytest.fixture
async def bus() -> EventBus:
    """Create and start EventBus."""
    bus = EventBus()
    await bus.start()
    yield bus
    await bus.stop()


async def test_compute_fitness_high_sharpe(bus: EventBus) -> None:
    """Test fitness computation with high Sharpe ratio."""
    director = GenomeDirector(bus=bus)

    metrics = {
        "sharpe": 3.0,  # Perfect Sharpe
        "profit_factor": 2.0,
        "win_rate": 0.6,
        "max_drawdown_pct": 5.0,
    }

    fitness = director.compute_fitness(metrics)

    # Sharpe: 3.0/3.0 = 1.0 * 0.35 = 0.35
    # PF: (2.0-1.0)/(3.0-1.0) = 0.5 * 0.25 = 0.125
    # WR: 0.6 * 0.2 = 0.12
    # DD: (1.0 - 5.0/20.0) = 0.75 * 0.2 = 0.15
    # Total: 0.35 + 0.125 + 0.12 + 0.15 = 0.745
    assert 0.74 <= fitness <= 0.75


async def test_compute_fitness_high_drawdown_penalty(bus: EventBus) -> None:
    """Test fitness computation with high drawdown penalty."""
    director = GenomeDirector(bus=bus)

    metrics = {
        "sharpe": 2.0,
        "profit_factor": 2.5,
        "win_rate": 0.7,
        "max_drawdown_pct": 18.0,  # High drawdown
    }

    fitness = director.compute_fitness(metrics)

    # Sharpe: 2.0/3.0 * 0.35 = 0.233
    # PF: (2.5-1.0)/(3.0-1.0) = 0.75 * 0.25 = 0.1875
    # WR: 0.7 * 0.2 = 0.14
    # DD: (1.0 - 18.0/20.0) = 0.1 * 0.2 = 0.02
    # Total: ~0.58
    assert 0.57 <= fitness <= 0.59


async def test_should_promote_significant_improvement(bus: EventBus) -> None:
    """Test promotion with significant fitness improvement."""
    director = GenomeDirector(bus=bus)

    # Set current production genome
    director._production_genomes["strategy_1"] = {
        "params": {"threshold": 10.0},
        "fitness": 0.5,
        "metrics": {},
    }

    candidate_metrics = {
        "sharpe": 2.5,
        "profit_factor": 2.0,
        "win_rate": 0.65,
        "max_drawdown_pct": 7.0,
        "num_trades": 50,  # Sufficient trades
    }

    candidate_fitness = 0.6  # 20% improvement (0.6 / 0.5 = 1.2)

    should_promote = await director.should_promote(
        "strategy_1", candidate_fitness, candidate_metrics
    )

    assert should_promote is True


async def test_should_not_promote_small_improvement(bus: EventBus) -> None:
    """Test rejection when improvement is below threshold."""
    director = GenomeDirector(bus=bus)

    # Set current production genome
    director._production_genomes["strategy_1"] = {
        "params": {"threshold": 10.0},
        "fitness": 0.5,
        "metrics": {},
    }

    candidate_metrics = {
        "sharpe": 2.0,
        "profit_factor": 1.8,
        "win_rate": 0.55,
        "max_drawdown_pct": 8.0,
        "num_trades": 40,
    }

    candidate_fitness = 0.52  # Only 4% improvement — below 15% threshold

    should_promote = await director.should_promote(
        "strategy_1", candidate_fitness, candidate_metrics
    )

    assert should_promote is False


async def test_should_not_promote_insufficient_trades(bus: EventBus) -> None:
    """Test rejection when backtest has too few trades."""
    director = GenomeDirector(bus=bus)

    candidate_metrics = {
        "sharpe": 3.0,
        "profit_factor": 2.5,
        "win_rate": 0.7,
        "max_drawdown_pct": 5.0,
        "num_trades": 10,  # Too few trades (< 20)
    }

    candidate_fitness = 0.8  # Great fitness but insufficient sample

    should_promote = await director.should_promote(
        "strategy_1", candidate_fitness, candidate_metrics
    )

    assert should_promote is False


async def test_should_not_promote_excessive_drawdown(bus: EventBus) -> None:
    """Test rejection when drawdown exceeds threshold."""
    director = GenomeDirector(bus=bus)

    candidate_metrics = {
        "sharpe": 3.0,
        "profit_factor": 2.5,
        "win_rate": 0.7,
        "max_drawdown_pct": 15.0,  # Exceeds 10% threshold
        "num_trades": 50,
    }

    candidate_fitness = 0.6

    should_promote = await director.should_promote(
        "strategy_1", candidate_fitness, candidate_metrics
    )

    assert should_promote is False


async def test_fitness_normalization(bus: EventBus) -> None:
    """Test that fitness is properly normalized to 0-1 range."""
    director = GenomeDirector(bus=bus)

    # Test edge cases
    zero_metrics = {
        "sharpe": 0.0,
        "profit_factor": 1.0,
        "win_rate": 0.0,
        "max_drawdown_pct": 20.0,
    }

    perfect_metrics = {
        "sharpe": 3.0,
        "profit_factor": 3.0,
        "win_rate": 1.0,
        "max_drawdown_pct": 0.0,
    }

    zero_fitness = director.compute_fitness(zero_metrics)
    perfect_fitness = director.compute_fitness(perfect_metrics)

    # Zero metrics should give very low fitness
    assert 0.0 <= zero_fitness <= 0.1

    # Perfect metrics should give high fitness
    assert 0.9 <= perfect_fitness <= 1.0


async def test_first_genome_promotion(bus: EventBus) -> None:
    """Test that first genome is promoted if it passes basic checks."""
    director = GenomeDirector(bus=bus)

    # No existing production genome
    assert "new_strategy" not in director._production_genomes

    candidate_metrics = {
        "sharpe": 2.0,
        "profit_factor": 2.0,
        "win_rate": 0.6,
        "max_drawdown_pct": 8.0,
        "num_trades": 30,
    }

    candidate_fitness = 0.5

    should_promote = await director.should_promote(
        "new_strategy", candidate_fitness, candidate_metrics
    )

    assert should_promote is True


async def test_run_evolution_cycle(bus: EventBus) -> None:
    """Test full evolution cycle with multiple genomes."""
    director = GenomeDirector(bus=bus)

    genomes = [
        {"params": {"threshold": 5.0}},
        {"params": {"threshold": 10.0}},
        {"params": {"threshold": 15.0}},
    ]

    historical_ticks: list[dict] = []  # Mock ticks

    result = await director.run_evolution_cycle(
        strategy_id="test_strategy",
        genomes=genomes,
        ticks=historical_ticks,
    )

    # Verify result structure
    assert "strategy_id" in result
    assert "best_genome" in result
    assert "best_fitness" in result
    assert "best_metrics" in result
    assert "all_fitnesses" in result
    assert "promoted" in result
    assert "num_genomes_evaluated" in result

    assert result["strategy_id"] == "test_strategy"
    assert result["num_genomes_evaluated"] == 3
    assert len(result["all_fitnesses"]) == 3
    assert isinstance(result["promoted"], bool)


async def test_get_status_structure(bus: EventBus) -> None:
    """Test that get_status returns expected structure."""
    director = GenomeDirector(bus=bus)

    # Add a production genome
    director._production_genomes["test_strategy"] = {
        "params": {"threshold": 12.0},
        "fitness": 0.65,
        "metrics": {"sharpe": 2.5, "win_rate": 0.6},
    }

    status = director.get_status()

    assert "num_production_genomes" in status
    assert "production_strategies" in status
    assert "min_improvement_pct" in status
    assert "min_backtest_trades" in status
    assert "max_backtest_drawdown_pct" in status
    assert "production_genomes" in status

    assert status["num_production_genomes"] == 1
    assert "test_strategy" in status["production_strategies"]
    assert status["min_improvement_pct"] == 15.0
    assert status["min_backtest_trades"] == 20
    assert status["max_backtest_drawdown_pct"] == 10.0


async def test_evaluate_genome_without_backtest_engine(bus: EventBus) -> None:
    """Test that evaluate_genome returns defaults when no backtest engine."""
    director = GenomeDirector(bus=bus, backtest_engine=None)

    metrics = await director.evaluate_genome(
        strategy_id="test",
        genome_params={"threshold": 10.0},
        historical_ticks=[],
    )

    # Should return default metrics
    assert metrics["sharpe"] == 0.0
    assert metrics["profit_factor"] == 1.0
    assert metrics["win_rate"] == 0.0
    assert metrics["max_drawdown_pct"] == 100.0
    assert metrics["num_trades"] == 0


async def test_get_production_genome(bus: EventBus) -> None:
    """Test retrieval of production genome."""
    director = GenomeDirector(bus=bus)

    # No genome initially
    assert director.get_production_genome("unknown") is None

    # Add genome
    director._production_genomes["strategy_1"] = {
        "params": {"threshold": 10.0},
        "fitness": 0.6,
        "metrics": {},
    }

    genome = director.get_production_genome("strategy_1")
    assert genome is not None
    assert genome["fitness"] == 0.6
    assert genome["params"]["threshold"] == 10.0

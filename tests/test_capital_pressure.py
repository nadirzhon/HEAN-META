"""Tests for active capital pressure layer."""

from datetime import date, datetime, timedelta

import pytest

from hean.core.regime import Regime
from hean.observability.metrics import metrics as system_metrics
from hean.portfolio.allocator import CapitalAllocator
from hean.portfolio.capital_pressure import CapitalPressure
from hean.portfolio.decision_memory import DecisionMemory


def _sample_context() -> tuple[str, str, str, str]:
    """Build a deterministic context using DecisionMemory helper."""
    dm = DecisionMemory()
    now = datetime.utcnow()
    return dm.build_context(
        regime=Regime.NORMAL,
        spread_bps=5.0,
        volatility=0.005,
        timestamp=now,
    )


def test_pressure_boost_on_rising_pf_and_stable_dd() -> None:
    """Short‑term PF rising with stable drawdown should trigger a boost."""
    system_metrics.reset()
    cp = CapitalPressure(pf_window_trades=10, boost_pct=0.2)

    strategy_id = "s1"
    ctx = _sample_context()

    # Record a series of mostly winning trades (high short‑term PF)
    pnls = [10.0, 12.0, 8.0, 15.0, 9.0, 11.0, 7.0, -5.0, 10.0, 9.0]
    for pnl in pnls:
        cp.record_trade(strategy_id=strategy_id, pnl=pnl, context=ctx)

    before = cp.get_multiplier(strategy_id)
    assert before == pytest.approx(1.0)

    # Drawdown is low and stable (no acceleration)
    cp.update_drawdown(strategy_id, 5.0)
    after = cp.get_multiplier(strategy_id)

    # Expect roughly +20% boost (bounded by config)
    assert after > before
    assert after == pytest.approx(1.2, rel=0.05)

    counters = system_metrics.get_counters()
    assert counters.get("pressure_boost_events", 0) >= 1


def test_immediate_cut_after_two_losses_same_context() -> None:
    """Two consecutive losses in the same context should trigger an immediate cut."""
    system_metrics.reset()
    cp = CapitalPressure(cut_multiplier=0.5)

    strategy_id = "s1"
    ctx = _sample_context()

    # First loss – should NOT cut yet
    cp.record_trade(strategy_id=strategy_id, pnl=-10.0, context=ctx)
    m1 = cp.get_multiplier(strategy_id)
    assert m1 == pytest.approx(1.0)

    # Second consecutive loss in the same context – should cut
    cp.record_trade(strategy_id=strategy_id, pnl=-15.0, context=ctx)
    m2 = cp.get_multiplier(strategy_id)

    assert m2 < m1
    assert m2 == pytest.approx(0.5, rel=0.05)

    counters = system_metrics.get_counters()
    assert counters.get("pressure_cut_events", 0) >= 1


def test_pressure_decays_toward_neutral() -> None:
    """Pressure multipliers should decay back toward 1.0 over time."""
    cp = CapitalPressure(pf_window_trades=5, boost_pct=0.2, decay_rate=0.5)
    strategy_id = "s1"
    ctx = _sample_context()

    # Create strong PF and trigger a boost
    for pnl in [10.0, 12.0, 9.0, 11.0, 8.0]:
        cp.record_trade(strategy_id=strategy_id, pnl=pnl, context=ctx)

    cp.update_drawdown(strategy_id, 3.0)
    boosted = cp.get_multiplier(strategy_id)
    assert boosted > 1.0

    # Apply decay multiple times – multiplier should approach 1.0
    for _ in range(4):
        cp.decay_all()

    decayed = cp.get_multiplier(strategy_id)
    assert decayed < boosted
    assert decayed > 1.0  # still above neutral but closer


def test_capital_pressure_influences_allocator_weights() -> None:
    """Capital pressure should tilt allocator weights before normalization."""
    allocator = CapitalAllocator()
    cp = allocator.get_capital_pressure()

    # Manually configure two synthetic strategies
    allocator._weights = {
        "strategy_good": 0.5,
        "strategy_neutral": 0.5,
    }

    ctx = _sample_context()

    # Good strategy: strong recent PF
    for pnl in [10.0, 12.0, 9.0, 11.0, 8.0]:
        cp.record_trade("strategy_good", pnl=pnl, context=ctx)

    # Neutral strategy: flat PF
    for pnl in [2.0, -2.0, 1.0, -1.0, 0.0]:
        cp.record_trade("strategy_neutral", pnl=pnl, context=ctx)

    # Stable, low drawdown for both
    strategy_metrics = {
        "strategy_good": {
            "profit_factor": 1.0,
            "max_drawdown_pct": 5.0,
            "pnl": 100.0,
            "wins": 5,
            "losses": 0,
        },
        "strategy_neutral": {
            "profit_factor": 1.0,
            "max_drawdown_pct": 5.0,
            "pnl": 0.0,
            "wins": 2,
            "losses": 2,
        },
    }

    allocator._last_rebalance_date = date.today() - timedelta(days=1)

    # Update weights – capital pressure should give "strategy_good" a relative tilt
    new_weights = allocator.update_weights(strategy_metrics)

    assert new_weights["strategy_good"] > new_weights["strategy_neutral"]








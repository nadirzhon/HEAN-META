"""Tests for decision memory (context-aware blocking and penalties)."""

from datetime import datetime, timedelta

from hean.config import settings
from hean.core.regime import Regime
from hean.portfolio.decision_memory import DecisionMemory


def test_context_blocks_after_loss_streak() -> None:
    """Context should be blocked after configured loss streak."""
    dm = DecisionMemory()
    strategy_id = "s1"
    regime = Regime.NORMAL

    # Use a fixed context for determinism
    now = datetime.utcnow()
    context_key = dm.build_context(regime, spread_bps=10.0, volatility=0.01, timestamp=now)

    # Record consecutive losing trades in this context using new API
    for _ in range(settings.memory_loss_streak):
        dm.record_close(
            strategy_id=strategy_id,
            context_key=context_key,
            pnl=-10.0,
            timestamp=now,
        )

    assert dm.blocked(strategy_id, context_key) is True


def test_block_expires_after_configured_time() -> None:
    """Context block should expire after memory_block_hours."""
    dm = DecisionMemory()
    strategy_id = "s1"
    regime = Regime.NORMAL
    now = datetime.utcnow()
    context_key = dm.build_context(regime, spread_bps=5.0, volatility=0.005, timestamp=now)

    # Trigger a block via loss streak using new API
    for _ in range(settings.memory_loss_streak):
        dm.record_close(
            strategy_id=strategy_id,
            context_key=context_key,
            pnl=-5.0,
            timestamp=now,
        )

    assert dm.blocked(strategy_id, context_key) is True

    # Manually rewind block_until to simulate expiry. Tests are allowed to
    # touch internals for determinism.
    stats_map = dm._stats
    stats_obj = stats_map[(strategy_id, context_key)]
    stats_obj.block_until = now - timedelta(hours=settings.memory_block_hours + 1)

    # Now the block should be considered expired
    assert dm.blocked(strategy_id, context_key) is False


def test_penalty_scales_size_correctly() -> None:
    """Penalty multiplier should reduce size for bad contexts and never increase it."""
    dm = DecisionMemory()
    strategy_id = "s1"
    regime = Regime.NORMAL
    now = datetime.utcnow()
    context_key = dm.build_context(regime, spread_bps=10.0, volatility=0.01, timestamp=now)

    # Mix of wins and losses giving PF < 1.0 and some drawdown using new API
    pnls = [-10.0, -5.0, -15.0, 5.0]
    for pnl in pnls:
        dm.record_close(
            strategy_id=strategy_id,
            context_key=context_key,
            pnl=pnl,
            timestamp=now,
        )

    multiplier = dm.penalty(strategy_id, context_key)

    # Multiplier should be in (0, 1.0] and less than 1.0 for a clearly bad context
    assert 0.0 <= multiplier <= 1.0
    assert multiplier < 1.0


def test_penalty_returns_zero_when_blocked() -> None:
    """Penalty should return 0.0 when context is blocked."""
    dm = DecisionMemory()
    strategy_id = "s1"
    regime = Regime.NORMAL
    now = datetime.utcnow()
    context_key = dm.build_context(regime, spread_bps=10.0, volatility=0.01, timestamp=now)

    # Trigger a block via loss streak
    for _ in range(settings.memory_loss_streak):
        dm.record_close(
            strategy_id=strategy_id,
            context_key=context_key,
            pnl=-10.0,
            timestamp=now,
        )

    # Context should be blocked
    assert dm.blocked(strategy_id, context_key) is True

    # Penalty should return 0.0 for blocked contexts
    penalty_value = dm.penalty(strategy_id, context_key)
    assert penalty_value == 0.0



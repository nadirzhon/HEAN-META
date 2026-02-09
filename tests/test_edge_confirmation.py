"""Tests for edge confirmation loop (2-step impulse entries)."""

import pathlib
import sys
from datetime import datetime, timedelta

# Ensure local `src` package is importable as `hean`
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from hean.core.types import Signal, Tick
from hean.strategies.edge_confirmation import EdgeConfirmationLoop


def _make_signal(
    side: str = "buy",
    price: float = 100.0,
    strategy_id: str = "impulse_engine",
    symbol: str = "BTCUSDT",
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        entry_price=price,
        stop_loss=price * 0.99,
        take_profit=price * 1.01,
        take_profit_1=price * 1.005,
        metadata={},
    )


def _make_tick(
    price: float,
    ts: datetime,
    symbol: str = "BTCUSDT",
    bid: float | None = None,
    ask: float | None = None,
) -> Tick:
    return Tick(
        symbol=symbol,
        price=price,
        timestamp=ts,
        bid=bid,
        ask=ask,
    )


def test_first_impulse_creates_candidate_only() -> None:
    """First qualifying impulse should only create a candidate (no signal)."""
    loop = EdgeConfirmationLoop(timeout_sec=10)
    now = datetime.utcnow()

    signal = _make_signal(price=100.0)
    tick = _make_tick(price=100.0, ts=now, bid=99.99, ask=100.01)
    context = {
        "spread_bps": 4.0,
        "vol_short": 0.02,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices = [95.0, 97.0, 99.0, 100.0]

    result = loop.confirm_or_update(signal, tick, context, prices)
    assert result is None, "First impulse should only register a candidate"


def test_confirmation_by_spread_tightening() -> None:
    """Second impulse with tighter spread should confirm candidate."""
    loop = EdgeConfirmationLoop(timeout_sec=10, spread_tightening_ratio=0.7)
    now = datetime.utcnow()

    # Candidate
    signal = _make_signal(price=100.0)
    tick1 = _make_tick(price=100.0, ts=now, bid=99.98, ask=100.02)  # 8 bps
    context1 = {
        "spread_bps": 8.0,
        "vol_short": 0.01,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [98.0, 99.0, 100.0]
    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Confirmation: spread tightens well below 70% of 8 bps (i.e., < 5.6 bps)
    tick2 = _make_tick(price=100.5, ts=now + timedelta(seconds=1), bid=100.49, ask=100.51)  # 4 bps
    context2 = {
        "spread_bps": 4.0,
        "vol_short": 0.015,
        "vol_long": 0.01,
        "return_pct": 0.015,
    }
    prices2 = [98.0, 99.0, 100.0, 100.5]

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is not None, "Spread tightening should confirm candidate"
    assert confirmed.strategy_id == signal.strategy_id
    assert confirmed.symbol == signal.symbol


def test_confirmation_by_volatility_expansion() -> None:
    """Volatility expansion beyond configured ratio should confirm."""
    loop = EdgeConfirmationLoop(timeout_sec=10)
    now = datetime.utcnow()

    # Candidate with modest expansion ratio 1.1
    signal = _make_signal(price=100.0)
    tick1 = _make_tick(price=100.0, ts=now)
    context1 = {
        "spread_bps": 5.0,
        "vol_short": 0.011,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [99.0, 99.5, 100.0]
    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Confirmation: short/long ratio jumps to 2.0 (0.02 / 0.01)
    tick2 = _make_tick(price=101.0, ts=now + timedelta(seconds=1))
    context2 = {
        "spread_bps": 5.0,
        "vol_short": 0.02,
        "vol_long": 0.01,
        "return_pct": 0.02,
    }
    prices2 = [99.0, 99.5, 100.0, 101.0]

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is not None, "Volatility expansion should confirm candidate"


def test_micro_pullback_then_resume_long_confirms() -> None:
    """Long-side micro pullback then resume should confirm."""
    loop = EdgeConfirmationLoop(timeout_sec=10, pullback_min_pct=0.0005, pullback_max_pct=0.003)
    now = datetime.utcnow()

    signal = _make_signal(side="buy", price=100.0)
    tick1 = _make_tick(price=100.0, ts=now)
    context1 = {
        "spread_bps": 5.0,
        "vol_short": 0.01,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [99.5, 100.0]
    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Price dips slightly (0.2% pullback) then resumes to entry
    prices2 = [99.5, 100.0, 99.8, 100.0]
    tick2 = _make_tick(price=100.0, ts=now + timedelta(seconds=2))
    context2 = {
        "spread_bps": 5.0,
        "vol_short": 0.012,
        "vol_long": 0.01,
        "return_pct": 0.012,
    }

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is not None, "Micro pullback then resume (long) should confirm"


def test_micro_pullback_too_large_does_not_confirm() -> None:
    """Adverse move larger than allowed micro pullback should not confirm."""
    loop = EdgeConfirmationLoop(timeout_sec=10, pullback_min_pct=0.0005, pullback_max_pct=0.003)
    now = datetime.utcnow()

    signal = _make_signal(side="buy", price=100.0)
    tick1 = _make_tick(price=100.0, ts=now)
    context1 = {
        "spread_bps": 5.0,
        "vol_short": 0.01,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [99.5, 100.0]
    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Pullback of 1% (0.01) is larger than pullback_max_pct (0.003)
    prices2 = [99.5, 100.0, 99.0, 100.0]
    tick2 = _make_tick(price=100.0, ts=now + timedelta(seconds=2))
    context2 = {
        "spread_bps": 5.0,
        "vol_short": 0.012,
        "vol_long": 0.01,
        "return_pct": 0.012,
    }

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is None, "Large pullback should NOT confirm as micro pullback"


def test_candidate_expires_and_is_replaced() -> None:
    """Candidate should expire after timeout and be replaced by a new one."""
    loop = EdgeConfirmationLoop(timeout_sec=1)
    now = datetime.utcnow()

    signal = _make_signal(price=100.0)
    tick1 = _make_tick(price=100.0, ts=now)
    context1 = {
        "spread_bps": 5.0,
        "vol_short": 0.01,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [99.0, 100.0]

    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Second impulse arrives after timeout -> should become new candidate, not confirm
    tick2 = _make_tick(price=101.0, ts=now + timedelta(seconds=2))
    context2 = {
        "spread_bps": 4.0,
        "vol_short": 0.02,
        "vol_long": 0.01,
        "return_pct": 0.02,
    }
    prices2 = [99.0, 100.0, 101.0]

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is None, "Impulse after timeout should create a fresh candidate"

"""Tests for edge confirmation loop (2-step impulse entries)."""

from datetime import datetime

from hean.core.types import Signal, Tick


def _make_signal(
    side: str = "buy",
    price: float = 100.0,
    strategy_id: str = "impulse_engine",
    symbol: str = "BTCUSDT",
) -> Signal:
    """Helper to build a realistic impulse-style signal."""
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        entry_price=price,
        stop_loss=price * 0.99,
        take_profit=price * 1.01,
        take_profit_1=price * 1.005,
        metadata={},
    )


def _make_tick(
    price: float,
    ts: datetime,
    symbol: str = "BTCUSDT",
    bid: float | None = None,
    ask: float | None = None,
) -> Tick:
    """Helper to build a tick with optional bid/ask."""
    return Tick(
        symbol=symbol,
        price=price,
        timestamp=ts,
        bid=bid,
        ask=ask,
    )


def test_first_impulse_creates_candidate_only() -> None:
    """First qualifying impulse should only create a candidate (no signal)."""
    loop = EdgeConfirmationLoop(timeout_sec=10)
    now = datetime.utcnow()

    signal = _make_signal(price=100.0)
    tick = _make_tick(price=100.0, ts=now, bid=99.99, ask=100.01)
    context = {
        "spread_bps": 4.0,
        "vol_short": 0.02,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices = [95.0, 97.0, 99.0, 100.0]

    result = loop.confirm_or_update(signal, tick, context, prices)
    assert result is None, "First impulse should only register a candidate"


def test_confirmation_by_spread_tightening() -> None:
    """Second impulse with tighter spread should confirm candidate."""
    loop = EdgeConfirmationLoop(timeout_sec=10, spread_tightening_ratio=0.7)
    now = datetime.utcnow()

    # Candidate
    signal = _make_signal(price=100.0)
    tick1 = _make_tick(price=100.0, ts=now, bid=99.98, ask=100.02)  # 8 bps
    context1 = {
        "spread_bps": 8.0,
        "vol_short": 0.01,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [98.0, 99.0, 100.0]
    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Confirmation: spread tightens well below 70% of 8 bps (i.e., < 5.6 bps)
    tick2 = _make_tick(price=100.5, ts=now + timedelta(seconds=1), bid=100.49, ask=100.51)  # 4 bps
    context2 = {
        "spread_bps": 4.0,
        "vol_short": 0.015,
        "vol_long": 0.01,
        "return_pct": 0.015,
    }
    prices2 = [98.0, 99.0, 100.0, 100.5]

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is not None, "Spread tightening should confirm candidate"
    assert confirmed.strategy_id == signal.strategy_id
    assert confirmed.symbol == signal.symbol


def test_confirmation_by_volatility_expansion() -> None:
    """Volatility expansion beyond configured ratio should confirm."""
    loop = EdgeConfirmationLoop(timeout_sec=10)
    now = datetime.utcnow()

    # Candidate with modest expansion ratio 1.1
    signal = _make_signal(price=100.0)
    tick1 = _make_tick(price=100.0, ts=now)
    context1 = {
        "spread_bps": 5.0,
        "vol_short": 0.011,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [99.0, 99.5, 100.0]
    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Confirmation: short/long ratio jumps to 2.0 (0.02 / 0.01)
    tick2 = _make_tick(price=101.0, ts=now + timedelta(seconds=1))
    context2 = {
        "spread_bps": 5.0,
        "vol_short": 0.02,
        "vol_long": 0.01,
        "return_pct": 0.02,
    }
    prices2 = [99.0, 99.5, 100.0, 101.0]

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is not None, "Volatility expansion should confirm candidate"


def test_micro_pullback_then_resume_long_confirms() -> None:
    """Long-side micro pullback then resume should confirm."""
    loop = EdgeConfirmationLoop(timeout_sec=10, pullback_min_pct=0.0005, pullback_max_pct=0.003)
    now = datetime.utcnow()

    signal = _make_signal(side="buy", price=100.0)
    tick1 = _make_tick(price=100.0, ts=now)
    context1 = {
        "spread_bps": 5.0,
        "vol_short": 0.01,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [99.5, 100.0]
    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Price dips slightly (0.2% pullback) then resumes to entry
    prices2 = [99.5, 100.0, 99.8, 100.0]
    tick2 = _make_tick(price=100.0, ts=now + timedelta(seconds=2))
    context2 = {
        "spread_bps": 5.0,
        "vol_short": 0.012,
        "vol_long": 0.01,
        "return_pct": 0.012,
    }

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is not None, "Micro pullback then resume (long) should confirm"


def test_micro_pullback_too_large_does_not_confirm() -> None:
    """Adverse move larger than allowed micro pullback should not confirm."""
    loop = EdgeConfirmationLoop(timeout_sec=10, pullback_min_pct=0.0005, pullback_max_pct=0.003)
    now = datetime.utcnow()

    signal = _make_signal(side="buy", price=100.0)
    tick1 = _make_tick(price=100.0, ts=now)
    context1 = {
        "spread_bps": 5.0,
        "vol_short": 0.01,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [99.5, 100.0]
    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Pullback of 1% (0.01) is larger than pullback_max_pct (0.003)
    prices2 = [99.5, 100.0, 99.0, 100.0]
    tick2 = _make_tick(price=100.0, ts=now + timedelta(seconds=2))
    context2 = {
        "spread_bps": 5.0,
        "vol_short": 0.012,
        "vol_long": 0.01,
        "return_pct": 0.012,
    }

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is None, "Large pullback should NOT confirm as micro pullback"


def test_candidate_expires_and_is_replaced() -> None:
    """Candidate should expire after timeout and be replaced by a new one."""
    loop = EdgeConfirmationLoop(timeout_sec=1)
    now = datetime.utcnow()

    signal = _make_signal(price=100.0)
    tick1 = _make_tick(price=100.0, ts=now)
    context1 = {
        "spread_bps": 5.0,
        "vol_short": 0.01,
        "vol_long": 0.01,
        "return_pct": 0.01,
    }
    prices1 = [99.0, 100.0]

    assert loop.confirm_or_update(signal, tick1, context1, prices1) is None

    # Second impulse arrives after timeout -> should become new candidate, not confirm
    tick2 = _make_tick(price=101.0, ts=now + timedelta(seconds=2))
    context2 = {
        "spread_bps": 4.0,
        "vol_short": 0.02,
        "vol_long": 0.01,
        "return_pct": 0.02,
    }
    prices2 = [99.0, 100.0, 101.0]

    confirmed = loop.confirm_or_update(signal, tick2, context2, prices2)
    assert confirmed is None, "Impulse after timeout should create a fresh candidate"




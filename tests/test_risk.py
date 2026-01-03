"""Tests for risk management."""

import pytest

from hean.config import settings
from hean.core.types import OrderRequest, Position
from hean.risk.killswitch import KillSwitch
from hean.risk.limits import RiskLimits
from hean.risk.position_sizer import PositionSizer


def test_risk_limits_max_positions() -> None:
    """Test max positions limit."""
    limits = RiskLimits()

    # Fill up to max
    for i in range(settings.max_open_positions):
        pos = Position(
            position_id=f"pos-{i}",
            symbol="BTCUSDT",
            side="long",
            size=0.1,
            entry_price=50000.0,
            current_price=50000.0,
            opened_at=None,  # type: ignore
            strategy_id="test",
        )
        limits.register_position(pos)

    # Next order should be rejected
    order_request = OrderRequest(
        signal_id="test",
        strategy_id="test",
        symbol="ETHUSDT",
        side="buy",
        size=0.1,
    )

    allowed, reason = limits.check_order_request(order_request, 10000.0)
    assert not allowed
    assert "max open positions" in reason.lower()


def test_position_sizer() -> None:
    """Test position sizing."""
    sizer = PositionSizer()

    from hean.core.types import Signal

    signal = Signal(
        strategy_id="test",
        symbol="BTCUSDT",
        side="buy",
        entry_price=50000.0,
        stop_loss=49000.0,  # 2% stop
    )

    size = sizer.calculate_size(signal, 10000.0, 50000.0)
    assert size > 0
    # Size should be such that risk is max_trade_risk_pct of equity
    risk_amount = 10000.0 * (settings.max_trade_risk_pct / 100.0)
    stop_distance = 50000.0 - 49000.0
    expected_size = risk_amount / stop_distance
    assert abs(size - expected_size) < 0.01  # Allow small rounding


@pytest.mark.asyncio
async def test_killswitch_drawdown() -> None:
    """Test killswitch drawdown trigger."""
    from hean.core.bus import EventBus

    bus = EventBus()
    killswitch = KillSwitch(bus)

    await bus.start()

    # Trigger drawdown
    initial_equity = 10000.0
    peak_equity = 10000.0
    current_equity = 9400.0  # 6% drawdown (exceeds 5% limit)

    triggered = await killswitch.check_drawdown(current_equity, peak_equity)
    assert triggered
    assert killswitch.is_triggered()

    await bus.stop()



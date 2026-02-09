"""Tests for per-strategy accounting."""

from datetime import datetime

import pytest

from hean.core.types import Order, OrderStatus
from hean.portfolio.accounting import PortfolioAccounting


def test_strategy_pnl_tracking() -> None:
    """Test that PnL is tracked per strategy."""
    accounting = PortfolioAccounting(10000.0)

    # Record PnL for different strategies
    accounting.record_realized_pnl(100.0, "funding_harvester")
    accounting.record_realized_pnl(-50.0, "funding_harvester")
    accounting.record_realized_pnl(200.0, "impulse_engine")

    metrics = accounting.get_strategy_metrics()

    assert "funding_harvester" in metrics
    assert "impulse_engine" in metrics
    assert metrics["funding_harvester"]["pnl"] == 50.0
    assert metrics["impulse_engine"]["pnl"] == 200.0


def test_strategy_trades_tracking() -> None:
    """Test that trades are tracked per strategy."""
    accounting = PortfolioAccounting(10000.0)

    # Record fills for different strategies
    order1 = Order(
        order_id="order1",
        strategy_id="funding_harvester",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        order_type="market",
        status=OrderStatus.FILLED,
        timestamp=datetime.utcnow(),
    )

    order2 = Order(
        order_id="order2",
        strategy_id="impulse_engine",
        symbol="ETHUSDT",
        side="buy",
        size=0.5,
        order_type="market",
        status=OrderStatus.FILLED,
        timestamp=datetime.utcnow(),
    )

    accounting.record_fill(order1, 50000.0, 5.0)
    accounting.record_fill(order2, 3000.0, 1.5)

    metrics = accounting.get_strategy_metrics()

    assert "funding_harvester" in metrics
    assert "impulse_engine" in metrics
    assert metrics["funding_harvester"]["trades"] == 1.0
    assert metrics["impulse_engine"]["trades"] == 1.0


def test_strategy_win_loss_tracking() -> None:
    """Test that wins and losses are tracked per strategy."""
    accounting = PortfolioAccounting(10000.0)

    # Record wins and losses
    accounting.record_realized_pnl(100.0, "funding_harvester")  # Win
    accounting.record_realized_pnl(50.0, "funding_harvester")  # Win
    accounting.record_realized_pnl(-30.0, "funding_harvester")  # Loss

    metrics = accounting.get_strategy_metrics()

    assert metrics["funding_harvester"]["wins"] == 2.0
    assert metrics["funding_harvester"]["losses"] == 1.0
    assert metrics["funding_harvester"]["win_rate_pct"] == pytest.approx(66.67, abs=0.1)


def test_strategy_metrics_in_backtest_report() -> None:
    """Test that backtest report includes per-strategy metrics."""
    from hean.backtest.metrics import BacktestMetrics

    accounting = PortfolioAccounting(10000.0)

    # Initialize strategy capital
    accounting._strategy_initial_capital["funding_harvester"] = 3333.0
    accounting._strategy_initial_capital["impulse_engine"] = 3333.0

    # Record some activity
    accounting.record_realized_pnl(100.0, "funding_harvester")
    accounting.record_realized_pnl(200.0, "impulse_engine")

    order = Order(
        order_id="test",
        strategy_id="funding_harvester",
        symbol="BTCUSDT",
        side="buy",
        size=0.1,
        order_type="market",
        status=OrderStatus.FILLED,
        timestamp=datetime.utcnow(),
    )
    accounting.record_fill(order, 50000.0, 5.0)

    metrics_calc = BacktestMetrics(accounting=accounting, strategies=None, allocator=None)
    metrics = metrics_calc.calculate()

    # Check that strategies key exists
    assert "strategies" in metrics
    assert isinstance(metrics["strategies"], dict)

    # Check that strategy metrics are present
    assert "funding_harvester" in metrics["strategies"]
    assert "impulse_engine" in metrics["strategies"]

    # Check required fields
    strat_metrics = metrics["strategies"]["funding_harvester"]
    assert "return_pct" in strat_metrics
    assert "trades" in strat_metrics
    assert "win_rate_pct" in strat_metrics
    assert "profit_factor" in strat_metrics
    assert "max_drawdown_pct" in strat_metrics


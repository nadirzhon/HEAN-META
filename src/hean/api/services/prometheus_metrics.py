"""Prometheus metrics exporter for HEAN trading system.

Defines all Gauge/Counter/Histogram metrics referenced by Grafana dashboards
and Prometheus alert rules. Updated periodically from engine_facade + trading_metrics.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram

from hean.logging import get_logger

if TYPE_CHECKING:
    from hean.api.engine_facade import EngineFacade
    from hean.api.services.trading_metrics import TradingMetrics
    from hean.risk.killswitch import KillSwitch

logger = get_logger(__name__)

# ─── Gauges (current value) ────────────────────────────────────────

hean_equity = Gauge("hean_equity", "Current portfolio equity in USDT")
hean_initial_capital = Gauge("hean_initial_capital", "Initial capital in USDT")
hean_drawdown_pct = Gauge("hean_drawdown_pct", "Current drawdown percentage")
hean_realized_pnl = Gauge("hean_realized_pnl", "Total realized PnL in USDT")
hean_unrealized_pnl = Gauge("hean_unrealized_pnl", "Total unrealized PnL in USDT")
hean_fees = Gauge("hean_fees", "Total fees paid in USDT")
hean_open_positions = Gauge("hean_open_positions", "Number of open positions")
hean_killswitch_triggered = Gauge("hean_killswitch_triggered", "1 if killswitch is triggered, 0 otherwise")
hean_risk_state = Gauge("hean_risk_state", "Risk governor state (0=NORMAL, 1=SOFT_BRAKE, 2=QUARANTINE, 3=HARD_STOP)")

# ─── Counters (monotonically increasing) ───────────────────────────

hean_signals_total = Counter("hean_signals_total", "Total signals generated")
hean_orders_total = Counter("hean_orders_total", "Total orders submitted")
hean_fills_total = Counter("hean_fills_total", "Total orders filled")

# ─── Histograms ────────────────────────────────────────────────────

hean_execution_latency = Histogram(
    "hean_execution_latency",
    "Order execution latency in seconds (signal to fill)",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ─── Internal tracking ─────────────────────────────────────────────

_prev_signals: int = 0
_prev_orders: int = 0
_prev_fills: int = 0

RISK_STATE_MAP = {
    "NORMAL": 0,
    "SOFT_BRAKE": 1,
    "QUARANTINE": 2,
    "HARD_STOP": 3,
}


async def update_metrics(
    facade: EngineFacade | None,
    metrics: TradingMetrics,
    killswitch: KillSwitch | None,
) -> None:
    """Update all Prometheus metrics from current system state."""
    global _prev_signals, _prev_orders, _prev_fills

    try:
        # Engine status (equity, pnl, drawdown)
        if facade and facade.is_running:
            status = await facade.get_status()
            if status.get("running"):
                equity = status.get("equity", 0.0)
                hean_equity.set(equity)
                hean_initial_capital.set(status.get("initial_capital", 0.0))
                hean_realized_pnl.set(status.get("realized_pnl", 0.0))
                hean_unrealized_pnl.set(status.get("unrealized_pnl", 0.0))
                hean_fees.set(status.get("total_fees", 0.0))

                # Drawdown calculation
                initial = status.get("initial_capital", 0.0)
                if initial > 0:
                    dd = max(0.0, (initial - equity) / initial * 100)
                    hean_drawdown_pct.set(dd)

        # Trading metrics (signals, orders, fills)
        m = await metrics.get_metrics()
        session = m.get("counters", {}).get("session", {})

        new_signals = int(session.get("signals_total", 0))
        new_orders = int(session.get("orders_created", 0))
        new_fills = int(session.get("orders_filled", 0))

        # Increment counters by delta since last update
        if new_signals > _prev_signals:
            hean_signals_total.inc(new_signals - _prev_signals)
            _prev_signals = new_signals
        if new_orders > _prev_orders:
            hean_orders_total.inc(new_orders - _prev_orders)
            _prev_orders = new_orders
        if new_fills > _prev_fills:
            hean_fills_total.inc(new_fills - _prev_fills)
            _prev_fills = new_fills

        # Open positions
        hean_open_positions.set(m.get("active_positions_count", 0))

        # Killswitch
        if killswitch:
            hean_killswitch_triggered.set(1 if killswitch.triggered else 0)

        # Risk state from engine facade
        if facade and facade.is_running and hasattr(facade, '_trading_system'):
            ts = facade._trading_system
            if ts and hasattr(ts, '_risk_manager'):
                rm = ts._risk_manager
                if hasattr(rm, 'state'):
                    state_name = getattr(rm.state, 'name', str(rm.state))
                    hean_risk_state.set(RISK_STATE_MAP.get(state_name, 0))

    except Exception as e:
        logger.warning(f"Failed to update Prometheus metrics: {e}")


async def metrics_updater_loop(
    facade: EngineFacade | None,
    metrics: TradingMetrics,
    killswitch: KillSwitch | None,
) -> None:
    """Background loop that updates Prometheus metrics every 10 seconds."""
    while True:
        await update_metrics(facade, metrics, killswitch)
        await asyncio.sleep(10)

"""Metrics exporter for Prometheus or file-based export."""

import json
import time
from pathlib import Path
from typing import Any

from hean.logging import get_logger
from hean.observability.metrics import metrics

logger = get_logger(__name__)


class MetricsExporter:
    """Exports metrics for monitoring dashboards."""

    def __init__(self, export_path: str | Path | None = None) -> None:
        """Initialize metrics exporter.

        Args:
            export_path: Path to export metrics file (if None, uses in-memory only)
        """
        self.export_path = Path(export_path) if export_path else None

    def export_metrics(
        self,
        equity: float,
        realized_pnl: float,
        unrealized_pnl: float,
        drawdown_pct: float,
        fees: float,
        funding: float,
        rewards: float,
        opp_cost: float,
        profit_illusion_flag: bool,
        health_score: float,
        actions_enabled: bool,
        snapshot_staleness_seconds: float | None = None,
        order_rejects: int = 0,
        slippage_bps: float = 0.0,
        maker_taker_ratio: float = 0.0,
        api_latency_ms: float = 0.0,
    ) -> dict[str, Any]:
        """Export key metrics for dashboards.

        Args:
            equity: Current equity
            realized_pnl: Realized PnL
            unrealized_pnl: Unrealized PnL
            drawdown_pct: Drawdown percentage
            fees: Total fees
            funding: Net funding payments
            rewards: Total rewards
            opp_cost: Opportunity cost
            profit_illusion_flag: Profit illusion flag
            health_score: Health score (0-1)
            actions_enabled: Whether actions are enabled
            snapshot_staleness_seconds: Snapshot staleness in seconds
            order_rejects: Number of order rejects
            slippage_bps: Average slippage in basis points
            maker_taker_ratio: Maker/taker fill ratio
            api_latency_ms: API latency in milliseconds

        Returns:
            Dictionary of exported metrics
        """
        export_data = {
            "timestamp": time.time(),
            "equity": equity,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "drawdown_pct": drawdown_pct,
            "fees": fees,
            "funding": funding,
            "rewards": rewards,
            "opp_cost": opp_cost,
            "profit_illusion_flag": profit_illusion_flag,
            "health_score": health_score,
            "actions_enabled": actions_enabled,
            "snapshot_staleness_seconds": snapshot_staleness_seconds,
            "order_rejects": order_rejects,
            "slippage_bps": slippage_bps,
            "maker_taker_ratio": maker_taker_ratio,
            "api_latency_ms": api_latency_ms,
            # Additional system metrics
            "system_metrics": metrics.get_summary(),
        }

        # Export to file if path provided
        if self.export_path:
            try:
                self.export_path.parent.mkdir(parents=True, exist_ok=True)
                self.export_path.write_text(
                    json.dumps(export_data, indent=2), encoding="utf-8"
                )
            except Exception as e:
                logger.warning(f"Failed to export metrics to file: {e}")

        return export_data

    def export_prometheus_format(
        self,
        equity: float,
        realized_pnl: float,
        unrealized_pnl: float,
        drawdown_pct: float,
        fees: float,
        funding: float,
        rewards: float,
        opp_cost: float,
        profit_illusion_flag: bool,
        health_score: float,
        actions_enabled: bool,
        snapshot_staleness_seconds: float | None = None,
        order_rejects: int = 0,
        slippage_bps: float = 0.0,
        maker_taker_ratio: float = 0.0,
        api_latency_ms: float = 0.0,
    ) -> str:
        """Export metrics in Prometheus format.

        Returns:
            Prometheus metrics text format
        """
        lines = [
            "# HELP hean_equity Current equity in USD",
            "# TYPE hean_equity gauge",
            f"hean_equity {equity}",
            "# HELP hean_realized_pnl Realized PnL in USD",
            "# TYPE hean_realized_pnl gauge",
            f"hean_realized_pnl {realized_pnl}",
            "# HELP hean_unrealized_pnl Unrealized PnL in USD",
            "# TYPE hean_unrealized_pnl gauge",
            f"hean_unrealized_pnl {unrealized_pnl}",
            "# HELP hean_drawdown_pct Drawdown percentage",
            "# TYPE hean_drawdown_pct gauge",
            f"hean_drawdown_pct {drawdown_pct}",
            "# HELP hean_fees Total fees in USD",
            "# TYPE hean_fees gauge",
            f"hean_fees {fees}",
            "# HELP hean_funding Net funding payments in USD",
            "# TYPE hean_funding gauge",
            f"hean_funding {funding}",
            "# HELP hean_rewards Total rewards in USD",
            "# TYPE hean_rewards gauge",
            f"hean_rewards {rewards}",
            "# HELP hean_opp_cost Opportunity cost in USD",
            "# TYPE hean_opp_cost gauge",
            f"hean_opp_cost {opp_cost}",
            "# HELP hean_profit_illusion_flag Profit illusion flag (1 if true, 0 if false)",
            "# TYPE hean_profit_illusion_flag gauge",
            f"hean_profit_illusion_flag {1 if profit_illusion_flag else 0}",
            "# HELP hean_health_score Health score (0-1)",
            "# TYPE hean_health_score gauge",
            f"hean_health_score {health_score}",
            "# HELP hean_actions_enabled Actions enabled flag (1 if true, 0 if false)",
            "# TYPE hean_actions_enabled gauge",
            f"hean_actions_enabled {1 if actions_enabled else 0}",
        ]

        if snapshot_staleness_seconds is not None:
            lines.extend([
                "# HELP hean_snapshot_staleness_seconds Snapshot staleness in seconds",
                "# TYPE hean_snapshot_staleness_seconds gauge",
                f"hean_snapshot_staleness_seconds {snapshot_staleness_seconds}",
            ])

        lines.extend([
            "# HELP hean_order_rejects Number of order rejects",
            "# TYPE hean_order_rejects counter",
            f"hean_order_rejects {order_rejects}",
            "# HELP hean_slippage_bps Average slippage in basis points",
            "# TYPE hean_slippage_bps gauge",
            f"hean_slippage_bps {slippage_bps}",
            "# HELP hean_maker_taker_ratio Maker/taker fill ratio",
            "# TYPE hean_maker_taker_ratio gauge",
            f"hean_maker_taker_ratio {maker_taker_ratio}",
            "# HELP hean_api_latency_ms API latency in milliseconds",
            "# TYPE hean_api_latency_ms gauge",
            f"hean_api_latency_ms {api_latency_ms}",
        ])

        return "\n".join(lines) + "\n"


# Global exporter instance
_global_exporter: MetricsExporter | None = None


def get_exporter(export_path: str | Path | None = None) -> MetricsExporter:
    """Get or create global metrics exporter.

    Args:
        export_path: Path to export metrics file

    Returns:
        MetricsExporter instance
    """
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = MetricsExporter(export_path=export_path)
    return _global_exporter


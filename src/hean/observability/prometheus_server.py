"""Prometheus metrics endpoint server."""

import asyncio
from aiohttp import web
from typing import Any

from hean.logging import get_logger
from hean.observability.metrics_exporter import get_exporter

logger = get_logger(__name__)


class PrometheusServer:
    """Simple HTTP server for Prometheus metrics scraping."""

    def __init__(self, port: int = 9090) -> None:
        """Initialize Prometheus server.

        Args:
            port: Port to listen on
        """
        self.port = port
        self.app = web.Application()
        self.app.router.add_get("/metrics", self.handle_metrics)
        self.app.router.add_get("/health", self.handle_health)
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Handle /metrics endpoint for Prometheus scraping."""
        # Get current metrics from system
        # This is a placeholder - in real implementation, would get from TradingSystem
        exporter = get_exporter()

        # Default values (should be updated from actual system state)
        prometheus_text = exporter.export_prometheus_format(
            equity=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            drawdown_pct=0.0,
            fees=0.0,
            funding=0.0,
            rewards=0.0,
            opp_cost=0.0,
            profit_illusion_flag=False,
            health_score=1.0,
            actions_enabled=False,
            snapshot_staleness_seconds=None,
            order_rejects=0,
            slippage_bps=0.0,
            maker_taker_ratio=0.0,
            api_latency_ms=0.0,
        )

        return web.Response(text=prometheus_text, content_type="text/plain")

    async def handle_health(self, request: web.Request) -> web.Response:
        """Handle /health endpoint."""
        return web.Response(text="OK", content_type="text/plain")

    async def start(self) -> None:
        """Start the Prometheus server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", self.port)
        await self.site.start()
        logger.info(f"Prometheus metrics server started on port {self.port}")

    async def stop(self) -> None:
        """Stop the Prometheus server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        logger.info("Prometheus metrics server stopped")

    def update_metrics(
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
    ) -> None:
        """Update metrics for export.

        This should be called periodically from TradingSystem.
        """
        # Store metrics for /metrics endpoint
        self._current_metrics = {
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
        }


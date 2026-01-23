"""
Prometheus Metrics Collector

Real-time monitoring and observability for HEAN trading system.

Metrics Categories:
- Trading: trades, PnL, win rate, positions
- ML Models: predictions, accuracy, inference time
- System: API latency, cache hit rate, errors
- Risk: drawdown, exposure, risk limits

Expected Benefits:
- Real-time visibility into system performance
- Early detection of issues (model drift, API failures)
- Performance optimization insights
- Historical analysis and debugging

Author: HEAN Team
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

from loguru import logger

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Summary,
        start_http_server,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Install with: pip install prometheus-client")


@dataclass
class MetricsConfig:
    """Configuration for Prometheus metrics."""

    # HTTP server
    port: int = 8000
    host: str = "0.0.0.0"

    # Metric prefixes
    namespace: str = "hean"
    subsystem: str = "trading"

    # Collection intervals
    collect_interval_seconds: float = 1.0

    # Histogram buckets
    latency_buckets: tuple = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
    prediction_time_buckets: tuple = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


class MetricsCollector:
    """
    Prometheus metrics collector for HEAN trading system.

    Usage:
        # Initialize
        metrics = MetricsCollector()

        # Start metrics server (localhost:8000/metrics)
        metrics.start_server(port=8000)

        # Record trading metrics
        metrics.record_trade(
            symbol="BTC/USDT",
            side="BUY",
            size=0.1,
            pnl=150.0,
            is_win=True,
        )

        # Record ML predictions
        metrics.record_prediction(
            model="ensemble",
            prediction="UP",
            confidence=0.68,
            inference_time_ms=45.0,
        )

        # Record system metrics
        metrics.record_api_call("binance", latency_ms=120.0, success=True)
        metrics.record_cache_access(hit=True)

        # Update gauges
        metrics.update_account_balance(10500.0)
        metrics.update_position(symbol="BTC/USDT", size=0.5)
    """

    def __init__(self, config: Optional[MetricsConfig] = None) -> None:
        """Initialize metrics collector."""
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client required. Install with: pip install prometheus-client")

        self.config = config or MetricsConfig()
        self._server_started = False

        # Trading Metrics
        self.trades_total = Counter(
            "trades_total",
            "Total number of trades executed",
            ["symbol", "side", "strategy"],
            namespace=self.config.namespace,
        )

        self.trades_pnl = Counter(
            "trades_pnl_usd_total",
            "Cumulative PnL from trades (USD)",
            ["symbol", "strategy"],
            namespace=self.config.namespace,
        )

        self.trades_wins = Counter(
            "trades_wins_total",
            "Total winning trades",
            ["symbol", "strategy"],
            namespace=self.config.namespace,
        )

        self.trades_losses = Counter(
            "trades_losses_total",
            "Total losing trades",
            ["symbol", "strategy"],
            namespace=self.config.namespace,
        )

        self.account_balance = Gauge(
            "account_balance_usd",
            "Current account balance (USD)",
            namespace=self.config.namespace,
        )

        self.open_positions = Gauge(
            "open_positions",
            "Number of open positions",
            ["symbol"],
            namespace=self.config.namespace,
        )

        self.position_size = Gauge(
            "position_size",
            "Position size in units",
            ["symbol"],
            namespace=self.config.namespace,
        )

        self.unrealized_pnl = Gauge(
            "unrealized_pnl_usd",
            "Unrealized PnL (USD)",
            ["symbol"],
            namespace=self.config.namespace,
        )

        # ML Model Metrics
        self.predictions_total = Counter(
            "ml_predictions_total",
            "Total ML predictions made",
            ["model", "prediction"],
            namespace=self.config.namespace,
        )

        self.prediction_confidence = Summary(
            "ml_prediction_confidence",
            "ML prediction confidence distribution",
            ["model"],
            namespace=self.config.namespace,
        )

        self.inference_time = Histogram(
            "ml_inference_time_seconds",
            "ML model inference time",
            ["model"],
            buckets=self.config.prediction_time_buckets,
            namespace=self.config.namespace,
        )

        self.model_accuracy = Gauge(
            "ml_model_accuracy",
            "Current model accuracy (rolling window)",
            ["model"],
            namespace=self.config.namespace,
        )

        # System Metrics
        self.api_calls_total = Counter(
            "api_calls_total",
            "Total API calls made",
            ["exchange", "endpoint", "status"],
            namespace=self.config.namespace,
        )

        self.api_latency = Histogram(
            "api_latency_seconds",
            "API call latency",
            ["exchange", "endpoint"],
            buckets=self.config.latency_buckets,
            namespace=self.config.namespace,
        )

        self.cache_hits = Counter(
            "cache_hits_total",
            "Cache hit count",
            ["cache_type"],
            namespace=self.config.namespace,
        )

        self.cache_misses = Counter(
            "cache_misses_total",
            "Cache miss count",
            ["cache_type"],
            namespace=self.config.namespace,
        )

        self.errors_total = Counter(
            "errors_total",
            "Total errors encountered",
            ["component", "error_type"],
            namespace=self.config.namespace,
        )

        # Risk Metrics
        self.max_drawdown = Gauge(
            "max_drawdown_pct",
            "Maximum drawdown percentage",
            namespace=self.config.namespace,
        )

        self.current_drawdown = Gauge(
            "current_drawdown_pct",
            "Current drawdown percentage",
            namespace=self.config.namespace,
        )

        self.total_exposure = Gauge(
            "total_exposure_usd",
            "Total market exposure (USD)",
            namespace=self.config.namespace,
        )

        self.leverage = Gauge(
            "leverage_ratio",
            "Current leverage ratio",
            namespace=self.config.namespace,
        )

        self.risk_limit_breaches = Counter(
            "risk_limit_breaches_total",
            "Risk limit breach count",
            ["limit_type"],
            namespace=self.config.namespace,
        )

        # Performance Metrics
        self.sharpe_ratio = Gauge(
            "sharpe_ratio",
            "Current Sharpe ratio (rolling)",
            namespace=self.config.namespace,
        )

        self.win_rate = Gauge(
            "win_rate_pct",
            "Current win rate percentage",
            namespace=self.config.namespace,
        )

        self.profit_factor = Gauge(
            "profit_factor",
            "Profit factor (wins/losses)",
            namespace=self.config.namespace,
        )

        logger.info("MetricsCollector initialized", config=self.config)

    def start_server(self, port: Optional[int] = None, host: Optional[str] = None) -> None:
        """
        Start Prometheus metrics HTTP server.

        Args:
            port: HTTP port (default from config)
            host: Host address (default from config)
        """
        if self._server_started:
            logger.warning("Metrics server already started")
            return

        port = port or self.config.port
        host = host or self.config.host

        try:
            start_http_server(port, addr=host)
            self._server_started = True
            logger.info(f"Prometheus metrics server started at http://{host}:{port}/metrics")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise

    # Trading Metrics Methods

    def record_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        pnl: float,
        is_win: bool,
        strategy: str = "default",
    ) -> None:
        """Record a trade execution."""
        self.trades_total.labels(symbol=symbol, side=side, strategy=strategy).inc()
        self.trades_pnl.labels(symbol=symbol, strategy=strategy).inc(pnl)

        if is_win:
            self.trades_wins.labels(symbol=symbol, strategy=strategy).inc()
        else:
            self.trades_losses.labels(symbol=symbol, strategy=strategy).inc()

        logger.debug(f"Recorded trade: {symbol} {side} {size:.4f} PnL={pnl:.2f}")

    def update_account_balance(self, balance: float) -> None:
        """Update account balance gauge."""
        self.account_balance.set(balance)

    def update_position(self, symbol: str, size: float, unrealized_pnl: float = 0.0) -> None:
        """Update position metrics."""
        self.position_size.labels(symbol=symbol).set(size)
        self.unrealized_pnl.labels(symbol=symbol).set(unrealized_pnl)

        # Count open positions
        if size > 0:
            self.open_positions.labels(symbol=symbol).set(1)
        else:
            self.open_positions.labels(symbol=symbol).set(0)

    # ML Model Metrics Methods

    def record_prediction(
        self,
        model: str,
        prediction: str,
        confidence: float,
        inference_time_ms: float,
    ) -> None:
        """Record ML model prediction."""
        self.predictions_total.labels(model=model, prediction=prediction).inc()
        self.prediction_confidence.labels(model=model).observe(confidence)
        self.inference_time.labels(model=model).observe(inference_time_ms / 1000.0)

        logger.debug(f"Recorded prediction: {model} -> {prediction} (conf={confidence:.2%})")

    def update_model_accuracy(self, model: str, accuracy: float) -> None:
        """Update model accuracy gauge."""
        self.model_accuracy.labels(model=model).set(accuracy)

    # System Metrics Methods

    def record_api_call(
        self,
        exchange: str,
        latency_ms: float,
        success: bool,
        endpoint: str = "default",
    ) -> None:
        """Record API call."""
        status = "success" if success else "error"
        self.api_calls_total.labels(
            exchange=exchange,
            endpoint=endpoint,
            status=status,
        ).inc()

        if success:
            self.api_latency.labels(
                exchange=exchange,
                endpoint=endpoint,
            ).observe(latency_ms / 1000.0)

    def record_cache_access(self, hit: bool, cache_type: str = "default") -> None:
        """Record cache access."""
        if hit:
            self.cache_hits.labels(cache_type=cache_type).inc()
        else:
            self.cache_misses.labels(cache_type=cache_type).inc()

    def record_error(self, component: str, error_type: str) -> None:
        """Record error occurrence."""
        self.errors_total.labels(component=component, error_type=error_type).inc()

    # Risk Metrics Methods

    def update_drawdown(self, current: float, maximum: float) -> None:
        """Update drawdown metrics."""
        self.current_drawdown.set(current * 100)  # Convert to percentage
        self.max_drawdown.set(maximum * 100)

    def update_exposure(self, total_exposure: float, leverage: float = 1.0) -> None:
        """Update exposure and leverage."""
        self.total_exposure.set(total_exposure)
        self.leverage.set(leverage)

    def record_risk_breach(self, limit_type: str) -> None:
        """Record risk limit breach."""
        self.risk_limit_breaches.labels(limit_type=limit_type).inc()
        logger.warning(f"Risk limit breach: {limit_type}")

    # Performance Metrics Methods

    def update_performance(
        self,
        sharpe_ratio: float,
        win_rate: float,
        profit_factor: float,
    ) -> None:
        """Update performance metrics."""
        self.sharpe_ratio.set(sharpe_ratio)
        self.win_rate.set(win_rate * 100)  # Convert to percentage
        self.profit_factor.set(profit_factor)

    def get_metrics_summary(self) -> Dict[str, float]:
        """Get current metrics summary (for debugging)."""
        # Note: This is a simplified version. Prometheus metrics don't expose
        # their values easily. In production, query Prometheus API instead.
        return {
            "server_started": self._server_started,
            "port": self.config.port,
        }


# Convenience function
def start_metrics_server(port: int = 8000, host: str = "0.0.0.0") -> MetricsCollector:
    """
    Quick start for metrics server.

    Example:
        metrics = start_metrics_server(port=8000)
        metrics.record_trade("BTC/USDT", "BUY", 0.1, 150.0, True)
    """
    config = MetricsConfig(port=port, host=host)
    collector = MetricsCollector(config)
    collector.start_server()
    return collector


# Context manager for timing operations
class MetricsTimer:
    """
    Context manager for timing operations and recording to Prometheus.

    Usage:
        with MetricsTimer(metrics.inference_time.labels(model="lgb")):
            result = model.predict(X)
    """

    def __init__(self, histogram_metric) -> None:
        """Initialize timer."""
        self.histogram = histogram_metric
        self.start_time = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record."""
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time
            self.histogram.observe(elapsed)

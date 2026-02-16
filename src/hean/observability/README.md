# observability -- Metrics, Health, and Monitoring

System-wide observability: metrics collection, health scoring, signal rejection telemetry, no-trade reporting, latency tracking, and Prometheus integration.

## Architecture

Observability is spread across multiple singleton instances and utility classes. `SystemMetrics` provides counters, gauges, and histograms for general system instrumentation. `HealthScore` aggregates multiple indicators into a single 0-100 score with status levels (EXCELLENT/GOOD/DEGRADED/WARNING/CRITICAL). `NoTradeReport` tracks why signals are blocked through the pipeline, and `SignalRejectionTelemetry` provides detailed per-reason, per-strategy, per-symbol rejection analysis. `MetricsExporter` can export to files or Prometheus format. `HealthCheck` provides a simple HTTP endpoint for container liveness probes.

## Key Classes

- `SystemMetrics` (`metrics.py`) -- Counters, gauges, and histograms for system instrumentation. Provides `increment()`, `set_gauge()`, `record_histogram()`, `get_summary()`, and `reset()`.
- `HealthCheck` (`health.py`) -- Simple HTTP health check endpoint (uses aiohttp). Exposes `/health` endpoint on configurable port for container orchestration.
- `HealthScore` (`health_score.py`) -- Aggregated health score (0-100). Combines multiple `HealthComponent` objects with weighted scores. Levels: EXCELLENT (90-100), GOOD (70-89), DEGRADED (50-69), WARNING (30-49), CRITICAL (0-29).
- `MetricsExporter` (`metrics_exporter.py`) -- Exports system metrics (equity, PnL, drawdown, fees, health score) to file or Prometheus-compatible format.
- `NoTradeReport` (`no_trade_report.py`) -- Tracks reasons why signals are blocked: risk limits, cooldowns, edge requirements, regime gating, maker edge, spread, volatility, protection blocks. Extended with pipeline tracing counters for signals_emitted through order fills.
- `SignalRejectionTelemetry` (`signal_rejection_telemetry.py`) -- Detailed rejection tracking with categories (RISK, REGIME, EXECUTION, COOLDOWN, FILTER, ANOMALY, ORACLE, OFI, MULTI_FACTOR). Tracks per-strategy, per-symbol, and time-series rejection patterns.
- `LatencyHistogram` (`latency_histogram.py`) -- Tracks latency distributions for performance monitoring.
- `MoneyCriticalLog` (`money_critical_log.py`) -- Special logging for money-critical events (fills, PnL changes, balance discrepancies).
- `Phase1Metrics` / `Phase2Metrics` (`phase1_metrics.py`, `phase2_metrics.py`) -- Phased trading metrics for tracking system progression.
- `PrometheusServer` (`prometheus_server.py`) -- Serves Prometheus-format metrics.
- `SelfHealingMonitor` (`monitoring/self_healing.py`) -- Self-healing monitors that detect and attempt to fix common issues.

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| HEALTH_CHECK_PORT | 8001 | Port for the HTTP health check endpoint |

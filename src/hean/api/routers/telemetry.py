"""Telemetry and portfolio snapshot endpoints."""

import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request

import hean.api.state as state
from hean.api.telemetry import telemetry_service
from hean.config import settings
from hean.logging import get_logger
from hean.observability.health_score import health_score
from hean.observability.latency_histogram import latency_histograms
from hean.observability.money_critical_log import get_money_log
from hean.observability.signal_rejection_telemetry import signal_rejection_telemetry

logger = get_logger(__name__)
router = APIRouter(tags=["telemetry"])


def _ws_client_count() -> int:
    """Best-effort WebSocket client count (works with hean.api.main)."""
    try:
        from hean.api import main as api_main  # type: ignore

        return api_main.connection_manager.active_count()
    except Exception:
        return 0


@router.get("/telemetry/ping")
async def telemetry_ping() -> dict[str, Any]:
    """Lightweight smoke endpoint."""
    return {"status": "ok", "ts": datetime.now(UTC).isoformat()}


def _safe_summary_payload(*, duration_ms: float, error: Exception | None = None) -> dict[str, Any]:
    payload = {
        "available": error is None,
        "duration_ms": round(duration_ms, 2),
        "note": "Telemetry snapshot fetched" if error is None else str(error),
    }
    if error:
        payload["error"] = {"name": type(error).__name__, "message": str(error)}
    return payload


@router.get("/telemetry/summary")
async def telemetry_summary(request: Request) -> dict[str, Any]:
    """Return heartbeat-friendly telemetry snapshot."""
    start = time.time()
    engine_facade = state.get_engine_facade(request)
    engine_state = (
        getattr(engine_facade, "engine_state", telemetry_service.get_engine_state()) if engine_facade else telemetry_service.get_engine_state()
    )
    mode = "LIVE" if settings.is_live and not settings.dry_run else "PAPER"
    try:
        summary = telemetry_service.summary(ws_clients=_ws_client_count(), mode=mode)
        summary["engine_state"] = engine_state
        summary["last_heartbeat"] = telemetry_service.last_heartbeat()
        summary["available"] = True
        duration = (time.time() - start) * 1000
        summary.update(_safe_summary_payload(duration_ms=duration))
        logger.debug(
            "Telemetry summary served (duration=%.1fms, ws_clients=%d, events/sec=%.3f)",
            duration,
            summary.get("ws_clients", 0),
            summary.get("events_per_sec", 0.0),
        )
        return summary
    except Exception as exc:
        duration = (time.time() - start) * 1000
        logger.error("Failed to build telemetry summary: %s", exc, exc_info=True)
        response = {
            "engine_state": engine_state,
            "mode": mode,
            "ws_clients": _ws_client_count(),
            "events_per_sec": 0.0,
            "events_total": telemetry_service._events_total if hasattr(telemetry_service, "_events_total") else 0,
            "last_event_ts": telemetry_service.last_event_ts_iso(),
            "last_heartbeat": telemetry_service.last_heartbeat(),
        }
        response.update(_safe_summary_payload(duration_ms=duration, error=exc))
        response["available"] = False
        return response


@router.get("/portfolio/summary")
async def portfolio_summary(request: Request) -> dict[str, Any]:
    """Return minimal portfolio snapshot for UI fallback."""
    start = time.time()
    engine_facade = state.get_engine_facade(request)
    available = False
    account_state: dict[str, Any] | None = None
    note = "Engine is not running"

    if engine_facade and engine_facade.is_running:
        try:
            snapshot = await engine_facade.get_trading_state()
            account_state = snapshot.get("account_state")
            available = account_state is not None
            note = "Live account snapshot" if available else "Account state not available yet"
        except Exception as exc:
            duration = (time.time() - start) * 1000
            logger.error("Failed to fetch portfolio snapshot: %s", exc, exc_info=True)
            return {
                "available": False,
                "equity": None,
                "balance": None,
                "used_margin": 0.0,
                "free_margin": None,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
                "fees": 0.0,
                "note": f"Failed to fetch portfolio: {exc}",
                "duration_ms": round(duration, 2),
                "error": {"name": type(exc).__name__, "message": str(exc)},
            }

    equity = account_state.get("equity") if account_state else None
    balance = account_state.get("wallet_balance") if account_state else None
    used_margin = account_state.get("used_margin") if account_state else 0.0
    free_margin = account_state.get("available_balance") if account_state else None
    unrealized_pnl = account_state.get("unrealized_pnl") if account_state else 0.0
    realized_pnl = account_state.get("realized_pnl") if account_state else 0.0
    fees = account_state.get("fees") if account_state else 0.0

    # Derive free margin if missing but balance/equity known
    if free_margin is None and balance is not None and used_margin is not None:
        free_margin = max(balance - used_margin, 0.0)

    duration = (time.time() - start) * 1000
    logger.debug("Portfolio snapshot served (available=%s, duration=%.1fms)", available, duration)

    return {
        "available": available,
        "equity": equity,
        "balance": balance,
        "used_margin": used_margin,
        "free_margin": free_margin,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "fees": fees,
        "note": note,
        "duration_ms": round(duration, 2),
    }


@router.get("/telemetry/signal-rejections")
async def get_signal_rejections(minutes: int = 60) -> dict[str, Any]:
    """Get signal rejection telemetry.

    Args:
        minutes: Time window for statistics (default 60 minutes)

    Returns:
        Comprehensive rejection statistics
    """
    stats = signal_rejection_telemetry.get_stats(minutes)

    return {
        "time_window_minutes": minutes,
        "total_rejections": stats.total_rejections,
        "total_signals": stats.total_signals,
        "rejection_rate": round(stats.rejection_rate * 100, 2),  # As percentage
        "by_category": stats.by_category,
        "by_reason": stats.by_reason,
        "by_symbol": stats.by_symbol,
        "by_strategy": stats.by_strategy,
        "rates": {
            "1m": round(signal_rejection_telemetry.get_rejection_rate(1) * 100, 2),
            "5m": round(signal_rejection_telemetry.get_rejection_rate(5) * 100, 2),
            "15m": round(signal_rejection_telemetry.get_rejection_rate(15) * 100, 2),
            "1h": round(signal_rejection_telemetry.get_rejection_rate(60) * 100, 2),
        },
    }


@router.get("/telemetry/signal-rejections/recent")
async def get_recent_rejections(limit: int = 50) -> dict[str, Any]:
    """Get recent signal rejection events for debugging.

    Args:
        limit: Maximum number of events to return (default 50)

    Returns:
        List of recent rejection events
    """
    recent = signal_rejection_telemetry.get_recent_rejections(limit)

    return {
        "count": len(recent),
        "limit": limit,
        "events": recent,
    }


@router.get("/telemetry/signal-rejections/summary")
async def get_rejection_summary() -> dict[str, Any]:
    """Get full signal rejection telemetry summary.

    Returns:
        Complete telemetry data including all-time totals and rates
    """
    return signal_rejection_telemetry.get_summary()


@router.get("/telemetry/health")
async def get_health_score(request: Request) -> dict[str, Any]:
    """Get aggregated system health score (0-100).

    Returns a single health metric that combines:
    - Exchange connectivity (20%): API responsiveness, WebSocket health
    - Risk state (25%): Drawdown, killswitch status, position limits
    - Execution quality (15%): Slippage, latency, fill rates
    - Strategy health (20%): Signal generation, win rate, rejection rate
    - System resources (10%): Memory, CPU, disk
    - Data freshness (10%): Price staleness, heartbeat

    Returns:
        Health report with overall score, component breakdown, and recommendations
    """
    # Try to update health components from live system data
    engine_facade = state.get_engine_facade(request)

    if engine_facade and engine_facade.is_running:
        try:
            # Update from live system state
            trading_state = await engine_facade.get_trading_state()

            # Exchange health
            ws_connected = trading_state.get("ws_connected", False)
            api_response_time = trading_state.get("api_latency_ms", 100.0)
            api_error_rate = trading_state.get("api_error_rate", 0.0)
            health_score.update_exchange_health(ws_connected, api_response_time, api_error_rate)

            # Risk health
            risk_state = trading_state.get("risk_state", {})
            drawdown_pct = risk_state.get("drawdown_pct", 0.0)
            killswitch_triggered = risk_state.get("killswitch_triggered", False)
            risk_status = risk_state.get("status", "NORMAL")
            position_utilization = risk_state.get("position_utilization", 0.0)
            health_score.update_risk_health(
                drawdown_pct, killswitch_triggered, risk_status, position_utilization
            )

            # Strategy health
            strategy_stats = trading_state.get("strategy_stats", {})
            signals_per_hour = strategy_stats.get("signals_per_hour", 0.0)
            win_rate = strategy_stats.get("win_rate", 0.5)
            signal_rejection_rate = signal_rejection_telemetry.get_rejection_rate(60)
            active_strategies = strategy_stats.get("active_count", 0)
            health_score.update_strategy_health(
                signals_per_hour, win_rate, signal_rejection_rate, active_strategies
            )

            # Data health
            last_tick_age = trading_state.get("last_tick_age_seconds", 0.0)
            last_heartbeat_age = (
                time.time() - telemetry_service._last_heartbeat_ts
                if telemetry_service._last_heartbeat_ts > 0
                else 0.0
            )
            data_gaps_count = trading_state.get("data_gaps_count", 0)
            health_score.update_data_health(last_tick_age, last_heartbeat_age, data_gaps_count)

        except Exception as exc:
            logger.warning("Failed to update health from live data: %s", exc)

    return health_score.get_summary()


@router.get("/telemetry/health/components")
async def get_health_components() -> dict[str, Any]:
    """Get detailed health component breakdown.

    Returns:
        Individual component scores with details
    """
    report = health_score.get_report()

    return {
        "overall_score": round(report.overall_score, 1),
        "status": report.status.value,
        "components": {
            name: {
                "score": round(comp.score, 1),
                "status": comp.status.value,
                "weight": comp.weight,
                "details": comp.details,
                "last_updated": comp.last_updated.isoformat(),
                "stale": comp.is_stale(),
            }
            for name, comp in report.components.items()
        },
    }


@router.get("/telemetry/health/recommendations")
async def get_health_recommendations() -> dict[str, Any]:
    """Get actionable health recommendations.

    Returns:
        List of prioritized recommendations based on current health state
    """
    recommendations = health_score.get_recommendations()
    report = health_score.get_report()

    return {
        "overall_score": round(report.overall_score, 1),
        "status": report.status.value,
        "can_trade": report.status.value not in ("critical", "warning"),
        "recommendations": recommendations,
        "recommendation_count": len(recommendations),
    }


@router.get("/telemetry/latency")
async def get_latency_summary() -> dict[str, Any]:
    """Get latency histogram summary with P99.9 percentiles.

    Returns summary of all tracked latency histograms including:
    - api_response: API endpoint latency
    - order_execution: Order execution latency
    - websocket_message: WebSocket message delivery latency
    - signal_processing: Strategy signal processing time
    - event_bus: EventBus message delivery time

    Returns:
        Latency statistics for all histograms with alert status
    """
    return latency_histograms.get_summary()


@router.get("/telemetry/latency/{histogram_name}")
async def get_latency_histogram(histogram_name: str) -> dict[str, Any]:
    """Get detailed latency histogram for a specific metric.

    Args:
        histogram_name: Name of the histogram (e.g., "api_response", "order_execution")

    Returns:
        Detailed histogram statistics including percentiles and alert status
    """
    histogram = latency_histograms.get(histogram_name)

    if histogram is None:
        return {
            "error": f"Histogram '{histogram_name}' not found",
            "available_histograms": list(latency_histograms._histograms.keys()),
        }

    stats = histogram.get_stats()

    return {
        "name": histogram_name,
        "count": stats.count,
        "window_seconds": stats.window_seconds,
        "latency_ms": {
            "min": round(stats.min_ms, 3),
            "max": round(stats.max_ms, 3),
            "mean": round(stats.mean_ms, 3),
            "p50": round(stats.p50_ms, 3),
            "p90": round(stats.p90_ms, 3),
            "p95": round(stats.p95_ms, 3),
            "p99": round(stats.p99_ms, 3),
            "p999": round(stats.p999_ms, 3),
        },
        "thresholds": {
            "p999_warning_ms": histogram.p999_warning_ms,
            "p999_critical_ms": histogram.p999_critical_ms,
        },
        "alert_level": stats.alert_level.value,
    }


@router.get("/telemetry/latency/alerts/recent")
async def get_latency_alerts(limit: int = 20) -> dict[str, Any]:
    """Get recent latency alerts.

    Args:
        limit: Maximum number of alerts to return (default 20)

    Returns:
        Recent P99.9 threshold violations across all histograms
    """
    alerts = latency_histograms.get_all_alerts(limit_per_histogram=limit)

    return {
        "alerts": [alert.to_dict() for alert in alerts[:limit]],
        "count": len(alerts[:limit]),
        "limit": limit,
    }


@router.get("/telemetry/latency/prometheus")
async def get_latency_prometheus() -> str:
    """Export latency histograms in Prometheus text format.

    Returns:
        Prometheus exposition format metrics
    """
    from fastapi.responses import PlainTextResponse

    return latency_histograms.to_prometheus_text()


@router.get("/telemetry/money-log")
async def get_money_log_summary() -> dict[str, Any]:
    """Get money-critical log summary.

    Returns summary of the append-only audit log for money-affecting events:
    - SIGNAL, ORDER_REQUEST, ORDER_FILLED, ORDER_CANCELLED, ORDER_REJECTED
    - POSITION_OPENED, POSITION_CLOSED, PNL_UPDATE, RISK_ALERT

    Returns:
        Log statistics and recent entries
    """
    money_log = get_money_log()
    return money_log.get_summary()


@router.get("/telemetry/money-log/entries")
async def get_money_log_entries(
    limit: int = 100,
    event_type: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    """Get money-critical log entries with optional filtering.

    Args:
        limit: Maximum entries to return (default 100)
        event_type: Optional event type filter (e.g., "ORDER_FILLED")
        symbol: Optional symbol filter (e.g., "BTCUSDT")

    Returns:
        Filtered log entries
    """
    money_log = get_money_log()

    if event_type:
        entries = money_log.get_entries_by_type(event_type, limit)
    elif symbol:
        entries = money_log.get_entries_by_symbol(symbol, limit)
    else:
        entries = money_log.get_recent_entries(limit)

    return {
        "entries": [e.to_dict() for e in entries],
        "count": len(entries),
        "limit": limit,
        "filters": {
            "event_type": event_type,
            "symbol": symbol,
        },
    }


@router.get("/telemetry/money-log/chain/{correlation_id}")
async def get_money_log_chain(correlation_id: str) -> dict[str, Any]:
    """Get event chain by correlation ID for debugging.

    This allows tracing the full lifecycle of a trade:
    SIGNAL -> ORDER_REQUEST -> ORDER_FILLED -> POSITION_OPENED -> ... -> POSITION_CLOSED

    Args:
        correlation_id: Correlation ID linking related events

    Returns:
        Event chain with all related events
    """
    money_log = get_money_log()
    chain = money_log.get_chain(correlation_id)

    if chain is None:
        return {
            "error": f"Chain '{correlation_id}' not found",
            "correlation_id": correlation_id,
        }

    return {
        "chain": chain.get_summary(),
        "entries": [e.to_dict() for e in chain.entries],
    }


@router.get("/telemetry/money-log/stats")
async def get_money_log_stats() -> dict[str, Any]:
    """Get money-critical log statistics.

    Returns:
        Detailed statistics about the audit log
    """
    money_log = get_money_log()
    return money_log.get_stats()


@router.get("/telemetry/money-log/verify")
async def verify_money_log_integrity() -> dict[str, Any]:
    """Verify money-critical log integrity.

    Checks hash chain and entry integrity to ensure log has not been tampered with.

    Returns:
        Integrity verification result
    """
    money_log = get_money_log()
    is_valid, violations = money_log.verify_integrity()

    return {
        "is_valid": is_valid,
        "violation_count": len(violations),
        "violations": violations[:10] if violations else [],  # Limit to first 10
    }

"""Telemetry and portfolio snapshot endpoints."""

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request

import hean.api.state as state
from hean.api.telemetry import telemetry_service
from hean.config import settings

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
    return {"status": "ok", "ts": datetime.now(timezone.utc).isoformat()}


@router.get("/telemetry/summary")
async def telemetry_summary(request: Request) -> dict[str, Any]:
    """Return heartbeat-friendly telemetry snapshot."""
    engine_facade = state.get_engine_facade(request)
    engine_state = getattr(engine_facade, "engine_state", telemetry_service.get_engine_state()) if engine_facade else telemetry_service.get_engine_state()
    summary = telemetry_service.summary(ws_clients=_ws_client_count(), mode="LIVE" if settings.is_live and not settings.dry_run else "PAPER")
    summary["engine_state"] = engine_state
    summary["last_heartbeat"] = telemetry_service.last_heartbeat()
    summary["available"] = True
    return summary


@router.get("/portfolio/summary")
async def portfolio_summary(request: Request) -> dict[str, Any]:
    """Return minimal portfolio snapshot for UI fallback."""
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
            raise HTTPException(status_code=500, detail=f"Failed to fetch portfolio: {exc}")

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
    }

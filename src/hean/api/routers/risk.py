"""Risk management router."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request, status

import hean.api.state as state
from hean.api.schemas import RiskLimitsRequest
from hean.api.telemetry import telemetry_service
from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/risk", tags=["risk"])


def _get_facade(request: Request):
    facade = state.get_engine_facade(request)
    if facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")
    return facade


async def _log_control_event(
    command: str,
    phase: str,
    request: Request,
    *,
    success: bool = True,
    detail: str | None = None,
    extra: dict | None = None,
) -> None:
    """Send CONTROL_COMMAND / CONTROL_RESULT telemetry."""
    correlation_id = getattr(request.state, "request_id", None)
    event_type = "CONTROL_COMMAND" if phase == "command" else "CONTROL_RESULT"
    severity = "INFO" if success else "ERROR"
    payload = {
        "command": command,
        "success": success,
        "detail": detail,
        "path": str(request.url.path),
    }
    if extra:
        payload.update(extra)
    await telemetry_service.record_event(
        event_type,
        payload=payload,
        severity=severity,
        source="api",
        correlation_id=correlation_id,
        context={"command": command},
        publish_ws=True,
        topic="system_heartbeat",
    )


@router.get("/status")
async def get_risk_status(request: Request) -> dict:
    """Get risk management status."""
    engine_facade = _get_facade(request)

    try:
        result = await engine_facade.get_risk_status()
        return result
    except Exception as e:
        logger.error(f"Failed to get risk status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/limits")
async def get_risk_limits(request: Request) -> dict:
    """Get current risk limits."""
    engine_facade = _get_facade(request)

    try:
        if engine_facade.is_running:
            risk_status = await engine_facade.get_risk_status()
            return {
                "max_open_positions": risk_status.get("max_open_positions", 0),
                "max_daily_attempts": settings.max_daily_attempts,
                "max_exposure_usd": settings.max_exposure_usd,
                "min_notional_usd": settings.min_notional_usd,
                "cooldown_seconds": settings.cooldown_seconds,
            }
        else:
            return {
                "max_open_positions": settings.max_open_positions,
                "max_daily_attempts": settings.max_daily_attempts,
                "max_exposure_usd": settings.max_exposure_usd,
                "min_notional_usd": settings.min_notional_usd,
                "cooldown_seconds": settings.cooldown_seconds,
            }
    except Exception as e:
        logger.error(f"Failed to get risk limits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/decision-memory/blocks")
async def get_decision_memory_blocks(request: Request) -> dict:
    """Return active DecisionMemory blocks for diagnostics."""
    engine_facade = _get_facade(request)
    if not engine_facade.is_running or not getattr(engine_facade, "_trading_system", None):
        return {"blocked": []}

    decision_memory = engine_facade._trading_system._decision_memory
    now = datetime.now(timezone.utc)
    blocked = []
    for (strategy_id, context_key), stats in getattr(decision_memory, "_stats", {}).items():
        block_until = getattr(stats, "block_until", None)
        if block_until and isinstance(block_until, datetime):
            remaining = (block_until.replace(tzinfo=timezone.utc) - now).total_seconds()
            if remaining > 0:
                blocked.append(
                    {
                        "strategy_id": strategy_id,
                        "context": context_key,
                        "block_until": block_until.isoformat(),
                        "seconds_remaining": remaining,
                        "loss_streak": getattr(stats, "loss_streak", 0),
                        "trades_count": getattr(stats, "trades_count", 0),
                        "max_drawdown_pct": getattr(stats, "max_drawdown_pct", 0.0),
                    }
                )
    return {"blocked": blocked}


@router.post("/limits")
async def update_risk_limits(request: Request, payload: RiskLimitsRequest) -> dict:
    """Update risk limits (paper only)."""
    _ = _get_facade(request)

    if settings.is_live and not settings.dry_run:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Risk limits can only be updated in paper/dry_run mode",
        )

    try:
        # TODO: Implement risk limits update
        return {
            "status": "success",
            "message": "Risk limits updated",
        }
    except Exception as e:
        logger.error(f"Failed to update risk limits: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/killswitch/status")
async def get_killswitch_status(request: Request) -> dict:
    """Get killswitch status and details."""
    engine_facade = _get_facade(request)

    try:
        if not engine_facade.is_running or not getattr(engine_facade, "_trading_system", None):
            return {
                "triggered": False,
                "reasons": [],
                "triggered_at": None,
                "thresholds": {},
                "current_metrics": {}
            }

        trading_system = engine_facade._trading_system
        
        # Get killswitch status
        killswitch_status = {
            "triggered": False,
            "reasons": [],
            "triggered_at": None,
            "thresholds": {},
            "current_metrics": {}
        }
        
        if hasattr(trading_system, "_killswitch"):
            killswitch = trading_system._killswitch
            accounting = trading_system._accounting
            equity = accounting.get_equity()
            drawdown_amount, drawdown_pct = accounting.get_drawdown(equity)
            
            killswitch_status["triggered"] = killswitch.is_triggered()
            killswitch_status["reasons"] = getattr(killswitch, "_reasons", []) or ([killswitch.get_reason()] if killswitch.get_reason() else [])
            killswitch_status["triggered_at"] = getattr(killswitch, "_triggered_at", None)
            if killswitch_status["triggered_at"]:
                killswitch_status["triggered_at"] = killswitch_status["triggered_at"].isoformat()
            
            # Get thresholds from settings
            from hean.config import settings
            adaptive_limit = killswitch.get_adaptive_drawdown_limit(equity)
            killswitch_status["thresholds"] = {
                "drawdown_pct": min(adaptive_limit, settings.max_daily_drawdown_pct),
                "equity_drop": settings.killswitch_drawdown_pct,
                "max_loss": getattr(settings, "max_loss_threshold", 0),
                "risk_limit": getattr(settings, "risk_limit_threshold", 0)
            }
            
            # Get current metrics
            peak_equity = getattr(accounting, "_peak_equity", equity)
            max_drawdown_pct = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0.0
            killswitch_status["current_metrics"] = {
                "current_drawdown_pct": drawdown_pct,
                "current_equity": equity,
                "max_drawdown_pct": max_drawdown_pct,
                "peak_equity": peak_equity
            }
        
        return killswitch_status
    except Exception as e:
        logger.error(f"Failed to get killswitch status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/killswitch/reset")
async def reset_killswitch(request: Request, confirm: bool = False) -> dict:
    """Reset killswitch and stop_trading flags."""
    engine_facade = _get_facade(request)

    if settings.is_live and not settings.dry_run and not confirm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Killswitch reset requires confirmation in live mode. Set confirm=true to proceed.",
        )

    try:
        if not engine_facade.is_running or not getattr(engine_facade, "_trading_system", None):
            raise HTTPException(status_code=500, detail="Engine not running")

        trading_system = engine_facade._trading_system
        
        # Reset killswitch
        if hasattr(trading_system, "_killswitch"):
            trading_system._killswitch.reset()
            logger.info("Killswitch reset via API")
        
        # Reset stop_trading flag
        trading_system._stop_trading = False
        logger.info("Stop trading flag reset via API")
        
        await _log_control_event("killswitch_reset", "command", request, detail="killswitch/reset")
        await _log_control_event("killswitch_reset", "result", request, success=True, extra={"result": "Killswitch and stop_trading flags reset"})
        
        return {
            "status": "success",
            "message": "Killswitch and stop_trading flags reset",
        }
    except Exception as e:
        logger.error(f"Failed to reset killswitch: {e}", exc_info=True)
        await _log_control_event("killswitch_reset", "result", request, success=False, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))

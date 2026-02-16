"""Engine control router."""

import asyncio
import time

from fastapi import APIRouter, HTTPException, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

import hean.api.state as state
from hean.api.schemas import EnginePauseRequest, EngineStartRequest, EngineStopRequest
from hean.api.telemetry import telemetry_service
from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

# Persistent Bybit HTTP client and balance cache
_bybit_http: object | None = None
_bybit_balance_cache: dict | None = None
_bybit_balance_cache_ts: float = 0.0
_BYBIT_CACHE_TTL = 0.5  # seconds — sub-second freshness


async def _get_bybit_client():
    """Get or create persistent Bybit HTTP client."""
    global _bybit_http
    if _bybit_http is None:
        from hean.exchange.bybit.http import BybitHTTPClient
        _bybit_http = BybitHTTPClient()
        await _bybit_http.connect()
    return _bybit_http


async def _fetch_bybit_balance() -> dict | None:
    """Fetch real balance from Bybit with 2s cache for near real-time accuracy."""
    global _bybit_balance_cache, _bybit_balance_cache_ts, _bybit_http

    now = time.time()
    if _bybit_balance_cache is not None and now - _bybit_balance_cache_ts < _BYBIT_CACHE_TTL:
        return _bybit_balance_cache

    try:
        client = await _get_bybit_client()
        info = await asyncio.wait_for(client.get_account_info(), timeout=3.0)

        # BybitHTTPClient returns unwrapped result: {"list": [...]}
        accs = []
        if isinstance(info, dict):
            if "list" in info:
                accs = info["list"]
            elif info.get("retCode") == 0:
                accs = info.get("result", {}).get("list", [])

        if accs and len(accs) > 0:
            _bybit_balance_cache = {
                "equity": float(accs[0].get("totalEquity", 0)),
                "balance": float(accs[0].get("totalAvailableBalance", 0)),
            }
            _bybit_balance_cache_ts = now
            return _bybit_balance_cache
    except Exception as e:
        logger.warning(f"Bybit balance fetch failed: {e}")
        # Reset client on error so it reconnects next time
        _bybit_http = None

    return _bybit_balance_cache  # Return stale cache if fetch fails

router = APIRouter(prefix="/engine", tags=["engine"])
limiter = Limiter(key_func=get_remote_address)


def _get_facade(request: Request):
    """Fetch engine facade from app.state to avoid stale imports."""
    facade = state.get_engine_facade(request)
    if facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")
    return facade


def _check_live_trading(confirm_phrase: str | None) -> None:
    """Check if live trading is allowed."""
    # Testnet mode - no confirmation required
    if settings.bybit_testnet:
        return

    if not settings.is_live:
        return  # Paper trading, no check needed

    if not settings.live_confirm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="LIVE_CONFIRM must be true for live trading",
        )

    if settings.dry_run:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="DRY_RUN must be false for live trading",
        )

    if confirm_phrase != "I_UNDERSTAND_LIVE_TRADING":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid confirmation phrase for live trading",
        )


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


@router.post("/start")
@limiter.limit("5/minute")
async def start_engine(request: Request, payload: EngineStartRequest) -> dict:
    """Start the trading engine. Rate limited to 5 requests per minute."""
    engine_facade = _get_facade(request)

    _check_live_trading(payload.confirm_phrase)

    try:
        await _log_control_event("start", "command", request, detail="engine/start")
        result = await engine_facade.start()
        await _log_control_event("start", "result", request, success=True, extra={"result": result})
        return result
    except Exception as e:
        logger.error(f"Failed to start engine: {e}", exc_info=True)
        await _log_control_event("start", "result", request, success=False, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/stop")
@limiter.limit("5/minute")
async def stop_engine(request: Request, payload: EngineStopRequest) -> dict:
    """Stop the trading engine."""
    engine_facade = _get_facade(request)

    try:
        await _log_control_event("stop", "command", request, detail="engine/stop")
        result = await engine_facade.stop()
        await _log_control_event("stop", "result", request, success=True, extra={"result": result})
        # Standardized response
        return {
            "ok": True,
            "status": result.get("status", "stopped"),
            "engine_state": "STOPPED",
            "message": result.get("message", "Engine stopped successfully"),
        }
    except Exception as e:
        logger.error(f"Failed to stop engine: {e}", exc_info=True)
        await _log_control_event("stop", "result", request, success=False, detail=str(e))
        raise HTTPException(
            status_code=500,
            detail=str(e),
        ) from e


@router.post("/pause")
async def pause_engine(request: Request, payload: EnginePauseRequest) -> dict:
    """Pause the trading engine."""
    engine_facade = _get_facade(request)

    try:
        await _log_control_event("pause", "command", request, detail="engine/pause")
        result = await engine_facade.pause()
        await _log_control_event("pause", "result", request, success=True, extra={"result": result})
        return result
    except Exception as e:
        logger.error(f"Failed to pause engine: {e}", exc_info=True)
        await _log_control_event("pause", "result", request, success=False, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/resume")
async def resume_engine(request: Request) -> dict:
    """Resume the trading engine."""
    engine_facade = _get_facade(request)

    try:
        await _log_control_event("resume", "command", request, detail="engine/resume")
        result = await engine_facade.resume()
        await _log_control_event("resume", "result", request, success=True, extra={"result": result})
        # Standardized response
        return {
            "ok": True,
            "status": result.get("status", "resumed"),
            "engine_state": "RUNNING",
            "message": result.get("message", "Engine resumed successfully"),
        }
    except Exception as e:
        logger.error(f"Failed to resume engine: {e}", exc_info=True)
        await _log_control_event("resume", "result", request, success=False, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/kill")
async def kill_engine(request: Request) -> dict:
    """Emergency kill switch for the engine."""
    engine_facade = _get_facade(request)
    try:
        await _log_control_event("kill", "command", request, detail="engine/kill")
        result = await engine_facade.kill(reason="api_kill")
        await _log_control_event("kill", "result", request, success=True, extra={"result": result})
        return result
    except Exception as e:
        logger.error(f"Failed to kill engine: {e}", exc_info=True)
        await _log_control_event("kill", "result", request, success=False, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/restart")
async def restart_engine(request: Request) -> dict:
    """Restart the engine (stop then start)."""
    engine_facade = _get_facade(request)
    try:
        await _log_control_event("restart", "command", request, detail="engine/restart")
        result = await engine_facade.restart()
        await _log_control_event("restart", "result", request, success=True, extra={"result": result})
        # Standardized response
        start_result = result.get("start_result", {})
        return {
            "ok": True,
            "status": result.get("status", "restarted"),
            "engine_state": "RUNNING" if start_result.get("status") == "started" else "STOPPED",
            "message": result.get("message", start_result.get("message", "Engine restarted successfully")),
            "start_result": start_result,
        }
    except NotImplementedError as e:
        await _log_control_event("restart", "result", request, success=False, detail=str(e))
        raise HTTPException(status_code=501, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Failed to restart engine: {e}", exc_info=True)
        await _log_control_event("restart", "result", request, success=False, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/status")
async def get_engine_status(request: Request) -> dict:
    """Get engine status with real Bybit balance overlay."""
    facade = state.get_engine_facade(request)

    # If engine facade not initialized, return stopped status
    if facade is None:
        return {
            "status": "stopped",
            "running": False,
            "message": "Engine not initialized",
        }

    try:
        result = await facade.get_status()

        # Overlay real Bybit balance (fetched outside the engine lock)
        if settings.is_live and result.get("running"):
            bybit = await _fetch_bybit_balance()
            if bybit:
                result["equity"] = bybit["equity"]
                result["available_balance"] = bybit["balance"]

        return result
    except Exception as e:
        logger.error(f"Failed to get engine status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/lock-profit")
async def lock_profit(request: Request) -> dict:
    """Lock current profit by closing all positions and stopping engine."""
    engine_facade = _get_facade(request)

    try:
        await _log_control_event("lock_profit", "command", request, detail="engine/lock-profit")

        # Get current status
        status = await engine_facade.get_status()
        current_equity = status.get("equity", 0)
        initial_capital = status.get("initial_capital", 0)
        profit = current_equity - initial_capital if initial_capital > 0 else 0

        # Close all positions and stop engine
        result = await engine_facade.stop()

        await _log_control_event("lock_profit", "result", request, success=True, extra={
            "profit_locked": profit,
            "equity": current_equity,
            "initial_capital": initial_capital
        })

        return {
            "ok": True,
            "status": "profit_locked",
            "message": f"Profit locked: ${profit:.2f}",
            "profit_locked": profit,
            "equity": current_equity,
            "initial_capital": initial_capital,
            **result
        }
    except Exception as e:
        logger.error(f"Failed to lock profit: {e}", exc_info=True)
        await _log_control_event("lock_profit", "result", request, success=False, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e

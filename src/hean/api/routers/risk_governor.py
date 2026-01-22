"""Risk Governor API endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/risk/governor", tags=["risk"])


class ClearRequest(BaseModel):
    """Request to clear risk governor."""
    confirm: bool = False
    force: bool = False
    symbol: str | None = None


@router.get("/status")
async def get_risk_governor_status(request: Request) -> dict[str, Any]:
    """Get current risk governor status.

    Returns:
        Risk governor state with recommendations
    """
    try:
        # Try to get risk governor from engine facade
        facade = getattr(request.app.state, "engine_facade", None)
        if not facade or not facade.is_running:
            return {
                "risk_state": "NORMAL",
                "level": 0,
                "reason_codes": [],
                "metric": None,
                "value": None,
                "threshold": None,
                "recommended_action": "Engine not running",
                "clear_rule": "N/A",
                "quarantined_symbols": [],
                "blocked_at": None,
                "can_clear": True,
            }

        # Get risk governor instance
        trading_system = getattr(facade, "_trading_system", None)
        if not trading_system:
            return {
                "risk_state": "NORMAL",
                "level": 0,
                "reason_codes": [],
                "metric": None,
                "value": None,
                "threshold": None,
                "recommended_action": "Trading system not initialized",
                "clear_rule": "N/A",
                "quarantined_symbols": [],
                "blocked_at": None,
                "can_clear": True,
            }

        risk_governor = getattr(trading_system, "_risk_governor", None)
        if not risk_governor:
            return {
                "risk_state": "NORMAL",
                "level": 0,
                "reason_codes": [],
                "metric": None,
                "value": None,
                "threshold": None,
                "recommended_action": "Risk governor not initialized",
                "clear_rule": "N/A",
                "quarantined_symbols": [],
                "blocked_at": None,
                "can_clear": True,
            }

        return risk_governor.get_state()

    except Exception as e:
        logger.error(f"Failed to get risk governor status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
async def clear_risk_governor(request: Request, clear_request: ClearRequest) -> dict[str, Any]:
    """Clear risk governor blocks.

    Args:
        clear_request: Clear request with confirmation

    Returns:
        Status dictionary
    """
    try:
        facade = getattr(request.app.state, "engine_facade", None)
        if not facade or not facade.is_running:
            raise HTTPException(status_code=409, detail="Engine not running")

        trading_system = getattr(facade, "_trading_system", None)
        if not trading_system:
            raise HTTPException(status_code=409, detail="Trading system not initialized")

        risk_governor = getattr(trading_system, "_risk_governor", None)
        if not risk_governor:
            raise HTTPException(status_code=404, detail="Risk governor not initialized")

        # Check if live mode and confirmation required
        if settings.is_live and not clear_request.confirm:
            raise HTTPException(
                status_code=403,
                detail="Live mode requires confirm=true to clear risk governor"
            )

        # Clear specific symbol quarantine
        if clear_request.symbol:
            result = await risk_governor.clear_quarantine(symbol=clear_request.symbol)
            return result

        # Clear all blocks
        result = await risk_governor.clear_all(force=clear_request.force)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to clear risk governor: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/quarantine/{symbol}")
async def quarantine_symbol(request: Request, symbol: str, reason: str = "MANUAL") -> dict[str, Any]:
    """Quarantine a specific symbol (block trading).

    Args:
        symbol: Symbol to quarantine
        reason: Reason for quarantine

    Returns:
        Status dictionary
    """
    try:
        facade = getattr(request.app.state, "engine_facade", None)
        if not facade or not facade.is_running:
            raise HTTPException(status_code=409, detail="Engine not running")

        trading_system = getattr(facade, "_trading_system", None)
        if not trading_system:
            raise HTTPException(status_code=409, detail="Trading system not initialized")

        risk_governor = getattr(trading_system, "_risk_governor", None)
        if not risk_governor:
            raise HTTPException(status_code=404, detail="Risk governor not initialized")

        await risk_governor.quarantine_symbol(symbol=symbol, reason=reason)

        return {
            "status": "quarantined",
            "symbol": symbol,
            "reason": reason,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to quarantine symbol: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

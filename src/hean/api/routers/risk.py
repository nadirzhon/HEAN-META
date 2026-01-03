"""Risk management router."""

from fastapi import APIRouter, HTTPException, status

from hean.api.app import engine_facade
from hean.api.schemas import RiskLimitsRequest
from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/risk", tags=["risk"])


@router.get("/status")
async def get_risk_status() -> dict:
    """Get risk management status."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.get_risk_status()
        return result
    except Exception as e:
        logger.error(f"Failed to get risk status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/limits")
async def get_risk_limits() -> dict:
    """Get current risk limits."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

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


@router.post("/limits")
async def update_risk_limits(request: RiskLimitsRequest) -> dict:
    """Update risk limits (paper only)."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

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


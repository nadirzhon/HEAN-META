"""Engine control router."""

from fastapi import APIRouter, HTTPException, status

from hean.api.app import engine_facade
from hean.api.schemas import EnginePauseRequest, EngineStartRequest, EngineStopRequest
from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/engine", tags=["engine"])


def _check_live_trading(confirm_phrase: str | None) -> None:
    """Check if live trading is allowed."""
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


@router.post("/start")
async def start_engine(request: EngineStartRequest) -> dict:
    """Start the trading engine."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    _check_live_trading(request.confirm_phrase)

    try:
        result = await engine_facade.start()
        return result
    except Exception as e:
        logger.error(f"Failed to start engine: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop")
async def stop_engine(request: EngineStopRequest) -> dict:
    """Stop the trading engine."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.stop()
        return result
    except Exception as e:
        logger.error(f"Failed to stop engine: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pause")
async def pause_engine(request: EnginePauseRequest) -> dict:
    """Pause the trading engine."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.pause()
        return result
    except Exception as e:
        logger.error(f"Failed to pause engine: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume")
async def resume_engine() -> dict:
    """Resume the trading engine."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.resume()
        return result
    except Exception as e:
        logger.error(f"Failed to resume engine: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_engine_status() -> dict:
    """Get engine status."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.get_status()
        return result
    except Exception as e:
        logger.error(f"Failed to get engine status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


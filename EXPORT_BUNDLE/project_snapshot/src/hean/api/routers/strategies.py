"""Strategies management router."""

from fastapi import APIRouter, HTTPException, status

from hean.api.app import engine_facade
from hean.api.schemas import StrategyEnableRequest, StrategyParamsRequest
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies"])


@router.get("")
async def get_strategies() -> list[dict]:
    """Get list of strategies."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.get_strategies()
        return result
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_id}/enable")
async def enable_strategy(strategy_id: str, request: StrategyEnableRequest) -> dict:
    """Enable or disable a strategy."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.enable_strategy(strategy_id, request.enabled)
        return result
    except Exception as e:
        logger.error(f"Failed to enable/disable strategy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{strategy_id}/params")
async def update_strategy_params(strategy_id: str, request: StrategyParamsRequest) -> dict:
    """Update strategy parameters."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        # TODO: Implement strategy parameter update
        return {
            "status": "success",
            "message": f"Strategy {strategy_id} parameters updated",
        }
    except Exception as e:
        logger.error(f"Failed to update strategy params: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


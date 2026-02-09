"""Strategies management router."""

from fastapi import APIRouter, HTTPException, Request

import hean.api.state as state
from hean.api.schemas import StrategyEnableRequest, StrategyParamsRequest
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/strategies", tags=["strategies"])


def _get_facade(request: Request):
    facade = state.get_engine_facade(request)
    if facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")
    return facade


@router.get("")
async def get_strategies(request: Request) -> list[dict]:
    """Get list of strategies."""
    engine_facade = _get_facade(request)

    try:
        result = await engine_facade.get_strategies()
        return result
    except Exception as e:
        logger.error(f"Failed to get strategies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{strategy_id}/enable")
async def enable_strategy(strategy_id: str, request: Request, payload: StrategyEnableRequest) -> dict:
    """Enable or disable a strategy."""
    engine_facade = _get_facade(request)

    try:
        result = await engine_facade.enable_strategy(strategy_id, payload.enabled)
        return result
    except Exception as e:
        logger.error(f"Failed to enable/disable strategy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/{strategy_id}/params")
async def update_strategy_params(strategy_id: str, request: Request, payload: StrategyParamsRequest) -> dict:
    """Update strategy parameters.

    Allows updating runtime parameters for strategies with validation.
    Only allowlisted parameters can be updated for safety.

    Example payload:
    {
        "params": {
            "impulse_threshold": 0.006,
            "max_spread_bps": 10
        }
    }
    """
    engine_facade = _get_facade(request)

    try:
        # Validate strategy exists
        strategies = await engine_facade.get_strategies()
        strategy_names = [s.get("name") for s in strategies]

        if strategy_id not in strategy_names:
            raise HTTPException(
                status_code=404,
                detail=f"Strategy '{strategy_id}' not found. Available: {strategy_names}"
            )

        # Allowlist of updatable parameters per strategy
        PARAM_ALLOWLIST = {
            "impulse_engine": {
                "impulse_threshold", "spread_gate", "max_time_in_trade_sec",
                "cooldown_minutes", "maker_edge_threshold_bps"
            },
            "hf_scalping": {
                "entry_window_sec", "max_time_in_trade_sec", "min_move_bps", "tp_bps", "sl_bps"
            },
            "enhanced_grid": {
                "grid_spacing_pct", "num_levels"
            },
            "momentum_trader": {
                "window_size", "momentum_threshold"
            },
            "funding_harvester": set(),
            "basis_arbitrage": set(),
            "inventory_neutral_mm": set(),
            "correlation_arb": set(),
            "rebate_farmer": set(),
            "liquidity_sweep": set(),
            "sentiment_strategy": set(),
        }

        allowed_params = PARAM_ALLOWLIST.get(strategy_id, set())

        # Validate parameters
        params = payload.params or {}
        for param_name in params.keys():
            if param_name not in allowed_params:
                raise HTTPException(
                    status_code=400,
                    detail=f"Parameter '{param_name}' not allowed for strategy '{strategy_id}'. "
                           f"Allowed: {list(allowed_params)}"
                )

        # Publish parameter update event via EventBus
        from hean.core.types import Event, EventType
        event = Event(
            type=EventType.STRATEGY_PARAMS_UPDATED,
            data={
                "strategy_id": strategy_id,
                "params": params,
                "updated_at": __import__("datetime").datetime.utcnow().isoformat(),
            }
        )
        engine_facade._engine._bus.publish(event)

        logger.info(
            f"Strategy params updated: {strategy_id} with params {params}"
        )

        return {
            "status": "success",
            "message": f"Strategy '{strategy_id}' parameters updated",
            "strategy_id": strategy_id,
            "updated_params": params,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update strategy params: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e

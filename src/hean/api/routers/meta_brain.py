"""MetaStrategyBrain API Router — Strategy lifecycle endpoints."""

from fastapi import APIRouter, HTTPException, Query

from hean.api.engine_facade import get_facade
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/meta-brain", tags=["meta-brain"])


@router.get("/status")
async def get_meta_brain_status():
    """Get current MetaStrategyBrain status and all strategy states."""
    facade = get_facade()
    ts = getattr(facade, "_trading_system", None)
    brain = getattr(ts, "_meta_strategy_brain", None) if ts else None

    if not brain:
        return {
            "enabled": False,
            "message": "MetaStrategyBrain is not running",
            "strategies": {},
        }

    return {
        "enabled": True,
        "strategies": brain.get_strategy_states(),
        "current_regime": brain._current_regime,
        "regime_confidence": brain._current_regime_confidence,
    }


@router.get("/transitions")
async def get_transitions(limit: int = Query(default=50, ge=1, le=500)):
    """Get recent strategy lifecycle transitions."""
    facade = get_facade()
    ts = getattr(facade, "_trading_system", None)
    brain = getattr(ts, "_meta_strategy_brain", None) if ts else None

    if not brain:
        return {"transitions": []}

    return {"transitions": brain.get_transitions(limit=limit)}


@router.get("/affinity-matrix")
async def get_affinity_matrix():
    """Get the learned regime-strategy affinity matrix."""
    facade = get_facade()
    ts = getattr(facade, "_trading_system", None)
    brain = getattr(ts, "_meta_strategy_brain", None) if ts else None

    if not brain:
        return {"matrix": {}}

    return {"matrix": brain.get_regime_affinity_matrix()}


@router.post("/force-state")
async def force_strategy_state(strategy_id: str, state: str):
    """Manually override a strategy's lifecycle state (admin action)."""
    facade = get_facade()
    ts = getattr(facade, "_trading_system", None)
    brain = getattr(ts, "_meta_strategy_brain", None) if ts else None

    if not brain:
        raise HTTPException(status_code=503, detail="MetaStrategyBrain is not running")

    from hean.portfolio.meta_strategy_brain import StrategyState

    try:
        target_state = StrategyState(state)
    except ValueError as e:
        valid = [s.value for s in StrategyState]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid state '{state}'. Valid states: {valid}",
        ) from e

    success = brain.force_state(strategy_id, target_state)
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Strategy '{strategy_id}' not found in MetaStrategyBrain",
        )

    return {
        "success": True,
        "strategy_id": strategy_id,
        "new_state": state,
    }


@router.get("/evolution")
async def get_evolution_status():
    """Get Symbiont X evolution bridge status."""
    facade = get_facade()
    ts = getattr(facade, "_trading_system", None)
    bridge = getattr(ts, "_evolution_bridge", None) if ts else None

    if not bridge:
        return {"enabled": False, "pending": [], "evolved": []}

    return {
        "enabled": True,
        "pending": bridge.get_pending(),
        "evolved": bridge.get_evolved(),
    }

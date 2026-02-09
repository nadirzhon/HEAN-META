"""API router for Meta-Learning Engine."""


from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/meta-learning", tags=["meta-learning"])


class MetaLearningStateResponse(BaseModel):
    """Meta-learning state response."""
    total_scenarios_simulated: int
    scenarios_per_second: float
    failures_detected: int
    patches_applied: int
    performance_improvement: float
    last_simulation_time: str | None = None


class CodeWeightResponse(BaseModel):
    """Code weight response."""
    name: str
    file_path: str
    line_number: int
    current_value: float
    value_range: list[float]
    impact_score: float


class PatchHistoryResponse(BaseModel):
    """Patch history response."""
    timestamp: str
    weight: str
    old_value: float
    new_value: float
    scenario_id: str


@router.get("/state", response_model=MetaLearningStateResponse)
async def get_state(request: Request):
    """Get meta-learning engine state."""
    engine_facade = getattr(request.state, 'engine_facade', None)

    meta_engine = getattr(engine_facade, '_meta_learning_engine', None) if engine_facade else None
    if not meta_engine:
        raise HTTPException(
            status_code=503,
            detail="Meta-learning engine not initialized"
        )
    state = meta_engine.get_state()

    return MetaLearningStateResponse(
        total_scenarios_simulated=state.total_scenarios_simulated,
        scenarios_per_second=state.scenarios_per_second,
        failures_detected=state.failures_detected,
        patches_applied=state.patches_applied,
        performance_improvement=state.performance_improvement,
        last_simulation_time=state.last_simulation_time.isoformat() if state.last_simulation_time else None
    )


@router.get("/weights", response_model=list[CodeWeightResponse])
async def get_weights(request: Request):
    """Get all code weights."""
    engine_facade = getattr(request.state, 'engine_facade', None)

    meta_engine = getattr(engine_facade, '_meta_learning_engine', None) if engine_facade else None
    if not meta_engine:
        raise HTTPException(
            status_code=503,
            detail="Meta-learning engine not initialized"
        )
    weights = meta_engine.get_weights()

    return [
        CodeWeightResponse(
            name=w.name,
            file_path=w.file_path,
            line_number=w.line_number,
            current_value=w.current_value,
            value_range=list(w.value_range),
            impact_score=w.impact_score
        )
        for w in weights.values()
    ]


@router.get("/patches", response_model=list[PatchHistoryResponse])
async def get_patch_history(request: Request, limit: int = 10):
    """Get patch history."""
    engine_facade = getattr(request.state, 'engine_facade', None)

    meta_engine = getattr(engine_facade, '_meta_learning_engine', None) if engine_facade else None
    if not meta_engine:
        raise HTTPException(
            status_code=503,
            detail="Meta-learning engine not initialized"
        )

    patches = meta_engine.get_patch_history()

    return [
        PatchHistoryResponse(
            timestamp=p['timestamp'].isoformat(),
            weight=p['weight'],
            old_value=p['old_value'],
            new_value=p['new_value'],
            scenario_id=p['scenario_id']
        )
        for p in patches[-limit:]
    ]

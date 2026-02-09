"""API router for Multimodal Swarm."""


import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/multimodal-swarm", tags=["multimodal-swarm"])


class MultimodalTensorResponse(BaseModel):
    """Multimodal tensor response."""
    timestamp: str
    symbol: str
    price_features: list[float]
    sentiment_features: list[float]
    onchain_features: list[float]
    macro_features: list[float]
    unified_tensor: list[float]
    confidence: float
    modality_weights: dict[str, float]


@router.get("/stats")
async def get_stats(request: Request):
    """Get multimodal swarm statistics."""
    engine_facade = getattr(request.state, 'engine_facade', None)

    swarm = getattr(engine_facade, '_multimodal_swarm', None) if engine_facade else None
    if not swarm:
        raise HTTPException(
            status_code=503,
            detail="Multimodal swarm not initialized"
        )
    weights = swarm.get_modality_weights()

    return {
        "tensor_size": 18,  # Fixed size: 5+4+4+5
        "modality_weights": weights,
        "num_agents": len(swarm._agents) if hasattr(swarm, '_agents') else 0
    }


@router.get("/tensors/{symbol}", response_model=list[MultimodalTensorResponse])
async def get_tensors(request: Request, symbol: str, limit: int = 10):
    """Get recent tensor history for a symbol."""
    engine_facade = getattr(request.state, 'engine_facade', None)

    swarm = getattr(engine_facade, '_multimodal_swarm', None) if engine_facade else None
    if not swarm:
        raise HTTPException(
            status_code=503,
            detail="Multimodal swarm not initialized"
        )
    tensors = swarm.get_tensor_history(symbol, limit=limit)

    return [
        MultimodalTensorResponse(
            timestamp=t.timestamp.isoformat(),
            symbol=t.symbol,
            price_features=t.price_features.tolist() if isinstance(t.price_features, np.ndarray) else list(t.price_features),
            sentiment_features=t.sentiment_features.tolist() if isinstance(t.sentiment_features, np.ndarray) else list(t.sentiment_features),
            onchain_features=t.onchain_features.tolist() if isinstance(t.onchain_features, np.ndarray) else list(t.onchain_features),
            macro_features=t.macro_features.tolist() if isinstance(t.macro_features, np.ndarray) else list(t.macro_features),
            unified_tensor=t.unified_tensor.tolist() if isinstance(t.unified_tensor, np.ndarray) else list(t.unified_tensor),
            confidence=t.confidence,
            modality_weights=t.modality_weights
        )
        for t in tensors
    ]


@router.get("/modality-weights", response_model=dict[str, float])
async def get_modality_weights(request: Request):
    """Get current modality weights."""
    engine_facade = getattr(request.state, 'engine_facade', None)

    swarm = getattr(engine_facade, '_multimodal_swarm', None) if engine_facade else None
    if not swarm:
        raise HTTPException(
            status_code=503,
            detail="Multimodal swarm not initialized"
        )

    return swarm.get_modality_weights()


@router.post("/modality-weights")
async def update_modality_weights(request: Request, weights: dict[str, float]):
    """Update modality weights (learned from performance)."""
    engine_facade = getattr(request.state, 'engine_facade', None)

    swarm = getattr(engine_facade, '_multimodal_swarm', None) if engine_facade else None
    if not swarm:
        raise HTTPException(
            status_code=503,
            detail="Multimodal swarm not initialized"
        )

    swarm.update_modality_weights(weights)

    return {"status": "success", "weights": swarm.get_modality_weights()}

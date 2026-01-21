"""API router for Multimodal Swarm."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/multimodal-swarm", tags=["multimodal-swarm"])


class MultimodalTensorResponse(BaseModel):
    """Multimodal tensor response."""
    timestamp: str
    symbol: str
    price_features: List[float]
    sentiment_features: List[float]
    onchain_features: List[float]
    macro_features: List[float]
    unified_tensor: List[float]
    confidence: float
    modality_weights: Dict[str, float]


@router.get("/stats")
async def get_stats(request: Request):
    """Get multimodal swarm statistics."""
    engine_facade = request.state.engine_facade
    
    if not engine_facade or not hasattr(engine_facade, '_multimodal_swarm'):
        return {
            "tensor_size": 18,
            "modality_weights": {},
            "num_agents": 0
        }
    
    swarm = engine_facade._multimodal_swarm
    weights = swarm.get_modality_weights()
    
    return {
        "tensor_size": 18,  # Fixed size: 5+4+4+5
        "modality_weights": weights,
        "num_agents": len(swarm._agents) if hasattr(swarm, '_agents') else 0
    }


@router.get("/tensors/{symbol}", response_model=List[MultimodalTensorResponse])
async def get_tensors(request: Request, symbol: str, limit: int = 10):
    """Get recent tensor history for a symbol."""
    engine_facade = request.state.engine_facade
    
    if not engine_facade or not hasattr(engine_facade, '_multimodal_swarm'):
        return []
    
    swarm = engine_facade._multimodal_swarm
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


@router.get("/modality-weights", response_model=Dict[str, float])
async def get_modality_weights(request: Request):
    """Get current modality weights."""
    engine_facade = request.state.engine_facade
    
    if not engine_facade or not hasattr(engine_facade, '_multimodal_swarm'):
        return {
            "price": 0.4,
            "sentiment": 0.2,
            "onchain": 0.25,
            "macro": 0.15
        }
    
    swarm = engine_facade._multimodal_swarm
    return swarm.get_modality_weights()


@router.post("/modality-weights")
async def update_modality_weights(request: Request, weights: Dict[str, float]):
    """Update modality weights (learned from performance)."""
    engine_facade = request.state.engine_facade
    
    if not engine_facade or not hasattr(engine_facade, '_multimodal_swarm'):
        return {"status": "error", "message": "Multimodal swarm not available"}
    
    swarm = engine_facade._multimodal_swarm
    swarm.update_modality_weights(weights)
    
    return {"status": "success", "weights": swarm.get_modality_weights()}

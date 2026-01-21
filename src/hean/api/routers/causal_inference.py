"""API router for Causal Inference Engine."""

from fastapi import APIRouter, Request
from pydantic import BaseModel
from typing import Dict, List, Optional, Tuple

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/causal-inference", tags=["causal-inference"])


class CausalRelationshipResponse(BaseModel):
    """Causal relationship response."""
    source_symbol: str
    target_symbol: str
    granger_causality: float
    transfer_entropy: float
    lag_period: int
    p_value: float
    confidence: float
    last_updated: str


class PreEchoSignalResponse(BaseModel):
    """Pre-echo signal response."""
    target_symbol: str
    source_symbol: str
    predicted_direction: str
    predicted_magnitude: float
    confidence: float
    lag_ms: int
    granger_score: float
    transfer_entropy_score: float
    timestamp: str


@router.get("/stats")
async def get_stats(request: Request):
    """Get causal inference engine statistics."""
    engine_facade = request.state.engine_facade
    
    if not engine_facade or not hasattr(engine_facade, '_causal_inference_engine'):
        return {
            "relationships": {},
            "pre_echo_signals": []
        }
    
    causal_engine = engine_facade._causal_inference_engine
    
    # Get relationships
    relationships = causal_engine.get_causal_relationships()
    relationships_dict = {
        f"{k[0]}->{k[1]}": {
            "source_symbol": v.source_symbol,
            "target_symbol": v.target_symbol,
            "granger_causality": v.granger_causality,
            "transfer_entropy": v.transfer_entropy,
            "lag_period": v.lag_period,
            "p_value": v.p_value,
            "confidence": v.confidence,
            "last_updated": v.last_updated.isoformat()
        }
        for k, v in relationships.items()
    }
    
    # Get pre-echo signals
    pre_echo_signals = causal_engine.get_pre_echo_signals(limit=10)
    signals_list = [
        {
            "target_symbol": s.target_symbol,
            "source_symbol": s.source_symbol,
            "predicted_direction": s.predicted_direction,
            "predicted_magnitude": s.predicted_magnitude,
            "confidence": s.confidence,
            "lag_ms": s.lag_ms,
            "granger_score": s.granger_score,
            "transfer_entropy_score": s.transfer_entropy_score,
            "timestamp": s.timestamp.isoformat()
        }
        for s in pre_echo_signals
    ]
    
    return {
        "relationships": relationships_dict,
        "pre_echo_signals": signals_list
    }


@router.get("/relationships", response_model=List[CausalRelationshipResponse])
async def get_relationships(request: Request):
    """Get all causal relationships."""
    engine_facade = request.state.engine_facade
    
    if not engine_facade or not hasattr(engine_facade, '_causal_inference_engine'):
        return []
    
    causal_engine = engine_facade._causal_inference_engine
    relationships = causal_engine.get_causal_relationships()
    
    return [
        CausalRelationshipResponse(
            source_symbol=v.source_symbol,
            target_symbol=v.target_symbol,
            granger_causality=v.granger_causality,
            transfer_entropy=v.transfer_entropy,
            lag_period=v.lag_period,
            p_value=v.p_value,
            confidence=v.confidence,
            last_updated=v.last_updated.isoformat()
        )
        for v in relationships.values()
    ]


@router.get("/pre-echoes", response_model=List[PreEchoSignalResponse])
async def get_pre_echoes(request: Request, limit: int = 10):
    """Get recent pre-echo signals."""
    engine_facade = request.state.engine_facade
    
    if not engine_facade or not hasattr(engine_facade, '_causal_inference_engine'):
        return []
    
    causal_engine = engine_facade._causal_inference_engine
    signals = causal_engine.get_pre_echo_signals(limit=limit)
    
    return [
        PreEchoSignalResponse(
            target_symbol=s.target_symbol,
            source_symbol=s.source_symbol,
            predicted_direction=s.predicted_direction,
            predicted_magnitude=s.predicted_magnitude,
            confidence=s.confidence,
            lag_ms=s.lag_ms,
            granger_score=s.granger_score,
            transfer_entropy_score=s.transfer_entropy_score,
            timestamp=s.timestamp.isoformat()
        )
        for s in signals
    ]

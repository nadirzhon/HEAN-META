"""API endpoints for Singularity features: Metamorphic Engine, Causal Discovery, Atomic Execution."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any

from hean.api.engine_facade import EngineFacade
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["singularity"])


@router.get("/metamorphic/sel")
async def get_system_evolution_level() -> Dict[str, Any]:
    """
    Get System Evolution Level (SEL) - overall system intelligence metric.
    
    Returns:
        Dictionary with SEL value (0.0 to 1.0)
    """
    try:
        facade = EngineFacade()
        if not facade.is_running:
            return {"sel": 0.0, "status": "engine_not_running"}
    
        # Try to get SEL from Metamorphic Engine
        try:
            from hean.core.intelligence.metamorphic_integration import MetamorphicIntegration
            # This would need to be stored in the engine facade
            # For now, return mock data
            sel = 0.75  # Mock value
        except ImportError:
            sel = 0.0
        
        return {
            "sel": sel,
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"Error getting SEL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/causal/graph")
async def get_causal_graph() -> Dict[str, Any]:
    """
    Get the causal graph showing relationships between assets.
    
    Returns:
        Dictionary with nodes and edges of the causal graph
    """
    try:
        facade = EngineFacade()
        if not facade.is_running:
            return {"nodes": [], "edges": [], "status": "engine_not_running"}
        
        # Try to get causal graph from Causal Discovery Engine
        try:
            # This would need to be stored in the engine facade
            # For now, return mock data
            nodes = [
                {"id": "BTCUSDT", "x": 0, "y": 0, "z": 0},
                {"id": "ETHUSDT", "x": 1, "y": 0, "z": 0},
                {"id": "SOLUSDT", "x": 0.5, "y": 1, "z": 0},
            ]
            edges = [
                {"source": "BTCUSDT", "target": "ETHUSDT", "strength": 0.8, "lag_us": 50000},
                {"source": "BTCUSDT", "target": "SOLUSDT", "strength": 0.6, "lag_us": 75000},
            ]
        except Exception:
            nodes = []
            edges = []
        
        return {
            "nodes": nodes,
            "edges": edges,
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"Error getting causal graph: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/atomic/clusters")
async def get_atomic_clusters() -> Dict[str, Any]:
    """
    Get active atomic execution clusters.
    
    Returns:
        Dictionary with cluster information
    """
    try:
        facade = EngineFacade()
        if not facade.is_running:
            return {"clusters": [], "status": "engine_not_running"}
        
        # Try to get clusters from Atomic Executor
        # This would need to be stored in the engine facade
        return {
            "clusters": [],
            "statistics": {
                "total_clusters_created": 0,
                "active_clusters": 0,
                "total_orders_placed": 0
            },
            "status": "ok"
        }
    except Exception as e:
        logger.error(f"Error getting atomic clusters: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

"""API endpoints for Singularity features: Metamorphic Engine, Causal Discovery, Atomic Execution."""

from typing import Any

from fastapi import APIRouter, Request

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/singularity", tags=["singularity"])


@router.get("/metamorphic/sel")
async def get_system_evolution_level(request: Request) -> dict[str, Any]:
    """
    Get System Evolution Level (SEL) - overall system intelligence metric.

    Returns:
        Dictionary with SEL value (0.0 to 1.0)
    """
    engine_facade = getattr(request.state, 'engine_facade', None)

    if not engine_facade or not getattr(engine_facade, 'is_running', False):
        return {"sel": 0.0, "status": "engine_not_running"}

    # Try to get SEL from Metamorphic Engine if available
    meta_learning = getattr(engine_facade, '_meta_learning_engine', None)
    if meta_learning:
        try:
            state = meta_learning.get_state()
            sel = min(1.0, state.performance_improvement / 100.0) if state else 0.0
        except Exception as e:
            logger.warning(f"Failed to get SEL from meta-learning engine: {e}")
            sel = 0.0
    else:
        sel = 0.0

    return {
        "sel": sel,
        "status": "ok"
    }


@router.get("/causal/graph")
async def get_causal_graph(request: Request) -> dict[str, Any]:
    """
    Get the causal graph showing relationships between assets.

    Returns:
        Dictionary with nodes and edges of the causal graph
    """
    engine_facade = getattr(request.state, 'engine_facade', None)

    if not engine_facade or not getattr(engine_facade, 'is_running', False):
        return {"nodes": [], "edges": [], "status": "engine_not_running"}

    # Try to get causal graph from Causal Inference Engine
    causal_engine = getattr(engine_facade, '_causal_inference_engine', None)
    if causal_engine:
        try:
            relationships = causal_engine.get_causal_relationships()
            # Build nodes from unique symbols
            symbols = set()
            for source, target in relationships.keys():
                symbols.add(source)
                symbols.add(target)

            nodes = [{"id": s, "x": i * 0.5, "y": (i % 3) * 0.5, "z": 0} for i, s in enumerate(symbols)]
            edges = [
                {
                    "source": k[0],
                    "target": k[1],
                    "strength": v.confidence,
                    "lag_us": v.lag_period * 1000  # Convert to microseconds
                }
                for k, v in relationships.items()
            ]
        except Exception as e:
            logger.warning(f"Failed to get causal graph: {e}")
            nodes = []
            edges = []
    else:
        nodes = []
        edges = []

    return {
        "nodes": nodes,
        "edges": edges,
        "status": "ok"
    }


@router.get("/atomic/clusters")
async def get_atomic_clusters(request: Request) -> dict[str, Any]:
    """
    Get active atomic execution clusters.

    Returns:
        Dictionary with cluster information
    """
    engine_facade = getattr(request.state, 'engine_facade', None)

    if not engine_facade or not getattr(engine_facade, 'is_running', False):
        return {"clusters": [], "status": "engine_not_running"}

    # Atomic execution statistics would come from the execution router
    # For now, return empty but valid response
    return {
        "clusters": [],
        "statistics": {
            "total_clusters_created": 0,
            "active_clusters": 0,
            "total_orders_placed": 0
        },
        "status": "ok"
    }

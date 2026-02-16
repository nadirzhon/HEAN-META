"""ARCHON API endpoints."""

from fastapi import APIRouter, Request

router = APIRouter(prefix="/archon", tags=["archon"])


@router.get("/status")
async def archon_status(request: Request) -> dict:
    """Get ARCHON overall status."""
    facade = request.state.engine_facade
    ts = getattr(facade, "_trading_system", None)
    archon = getattr(ts, "_archon", None) if ts else None
    if not archon:
        return {"active": False, "error": "ARCHON not initialized"}
    return archon.get_status()


@router.get("/pipeline")
async def pipeline_status(request: Request) -> dict:
    """Get Signal Pipeline status with dead letters."""
    facade = request.state.engine_facade
    ts = getattr(facade, "_trading_system", None)
    archon = getattr(ts, "_archon", None) if ts else None
    if not archon or not archon.signal_pipeline:
        return {"active": False}
    return archon.signal_pipeline.get_status()


@router.get("/pipeline/trace/{correlation_id}")
async def signal_trace(request: Request, correlation_id: str) -> dict:
    """Get full trace for a specific signal."""
    facade = request.state.engine_facade
    ts = getattr(facade, "_trading_system", None)
    archon = getattr(ts, "_archon", None) if ts else None
    if not archon or not archon.signal_pipeline:
        return {"error": "Pipeline not active"}
    trace = archon.signal_pipeline.get_trace(correlation_id)
    return trace or {"error": "Trace not found"}


@router.get("/health")
async def health_matrix(request: Request) -> dict:
    """Get comprehensive health matrix."""
    facade = request.state.engine_facade
    ts = getattr(facade, "_trading_system", None)
    archon = getattr(ts, "_archon", None) if ts else None
    if not archon or not archon.health_matrix:
        return {"active": False}
    return await archon.health_matrix.get_full_status()


@router.get("/chronicle")
async def chronicle_query(
    request: Request,
    event_type: str | None = None,
    symbol: str | None = None,
    strategy_id: str | None = None,
    limit: int = 50,
) -> dict:
    """Query audit trail."""
    facade = request.state.engine_facade
    ts = getattr(facade, "_trading_system", None)
    archon = getattr(ts, "_archon", None) if ts else None
    if not archon or not archon.chronicle:
        return {"active": False}
    return {
        "entries": archon.chronicle.query(
            event_type=event_type,
            symbol=symbol,
            strategy_id=strategy_id,
            limit=limit,
        )
    }

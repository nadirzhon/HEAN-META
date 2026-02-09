"""Storage API Router - DuckDB query endpoints."""

from fastapi import APIRouter, Query

router = APIRouter(prefix="/storage", tags=["storage"])


@router.get("/ticks")
async def get_stored_ticks(
    symbol: str = Query(default="BTCUSDT"),
    limit: int = Query(default=100, le=1000),
    since: float | None = Query(default=None),
):
    """Query stored tick data."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_duckdb_store"):
        store = facade._duckdb_store
        if store:
            return store.query_ticks(symbol=symbol, limit=limit, since=since)

    return []


@router.get("/physics")
async def get_stored_physics(
    symbol: str = Query(default="BTCUSDT"),
    limit: int = Query(default=50, le=500),
):
    """Query stored physics snapshots."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_duckdb_store"):
        store = facade._duckdb_store
        if store:
            return store.query_physics(symbol=symbol, limit=limit)

    return []


@router.get("/brain")
async def get_stored_brain(limit: int = Query(default=20, le=100)):
    """Query stored brain analyses."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_duckdb_store"):
        store = facade._duckdb_store
        if store:
            return store.query_brain(limit=limit)

    return []

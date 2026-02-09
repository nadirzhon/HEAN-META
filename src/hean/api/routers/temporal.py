"""Temporal Stack API Router - Multi-timeframe Analysis Endpoints."""

from datetime import datetime

from fastapi import APIRouter, Query

router = APIRouter(prefix="/temporal", tags=["temporal"])


@router.get("/stack")
async def get_temporal_stack():
    """Get full temporal stack (all 5 levels)."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_temporal_stack"):
        stack = facade._temporal_stack
        if stack:
            return stack.get_stack_dict()

    return {
        "levels": {},
        "last_update": datetime.utcnow().isoformat(),
    }


@router.get("/impulse")
async def get_cross_market_impulse(limit: int = Query(default=20, le=100)):
    """Get cross-market impulse data."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_cross_market"):
        cross = facade._cross_market
        if cross:
            impulses = cross.get_recent_impulses(limit)
            stats = cross.get_propagation_stats()
            return {
                "impulses": [
                    {
                        "source": i.source_symbol,
                        "change_pct": i.source_price_change_pct,
                        "timestamp": i.timestamp,
                        "propagated_to": i.propagated_to,
                    }
                    for i in impulses
                ],
                "propagation_stats": [
                    {
                        "source": s.source,
                        "target": s.target,
                        "avg_delay_ms": s.avg_delay_ms,
                        "correlation": s.correlation,
                        "samples": s.sample_count,
                    }
                    for s in stats
                ],
            }

    return {"impulses": [], "propagation_stats": []}


@router.get("/sessions")
async def get_trading_sessions():
    """Get current trading session info."""
    now = datetime.utcnow()
    hour = now.hour

    if 0 <= hour < 8:
        session = "Asia"
        next_session = "London"
        hours_remaining = 8 - hour
    elif 8 <= hour < 16:
        session = "London"
        next_session = "New York"
        hours_remaining = 16 - hour
    else:
        session = "New York"
        next_session = "Asia"
        hours_remaining = 24 - hour

    return {
        "current_session": session,
        "next_session": next_session,
        "hours_remaining": hours_remaining,
        "utc_hour": hour,
        "timestamp": now.isoformat(),
    }

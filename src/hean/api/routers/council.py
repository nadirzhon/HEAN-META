"""AI Council API router."""

import logging
from datetime import datetime

from fastapi import APIRouter, Query

from hean.api.engine_facade import get_facade
from hean.council.review import ApprovalStatus

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/council", tags=["council"])


def _get_council():
    """Get council from facade."""
    facade = get_facade()
    if facade and hasattr(facade, "_council"):
        return facade._council
    return None


@router.get("/status")
async def get_council_status():
    """Get council health and last review times."""
    council = _get_council()
    if council:
        return council.get_status()
    return {"enabled": False, "message": "Council not active"}


@router.get("/reviews")
async def get_council_reviews(limit: int = Query(default=5, le=50)):
    """Get latest review sessions from all members."""
    council = _get_council()
    if council and council._sessions:
        sessions = list(council._sessions)[-limit:]
        return [s.model_dump() for s in sessions]
    return []


@router.get("/recommendations")
async def get_council_recommendations(
    limit: int = Query(default=50, le=200),
    pending_only: bool = Query(default=False),
):
    """Get prioritized improvement list."""
    council = _get_council()
    if council:
        if pending_only:
            return council.get_pending_recommendations()
        return council.get_all_recommendations(limit)
    return []


@router.post("/approve/{rec_id}")
async def approve_recommendation(rec_id: str):
    """Approve a pending recommendation for application."""
    council = _get_council()
    if not council:
        return {"status": "error", "message": "Council not active"}

    rec = council.approve_recommendation(rec_id)
    if not rec:
        return {
            "status": "not_found",
            "message": f"Recommendation {rec_id} not found or not pending",
        }

    result = await council._executor.apply_recommendation(rec)
    rec.applied_at = datetime.utcnow().isoformat()
    rec.apply_result = result
    if result.get("status") == "applied":
        rec.approval_status = ApprovalStatus.APPLIED

    return {
        "status": "approved",
        "recommendation": rec.model_dump(),
        "apply_result": result,
    }


@router.post("/trigger")
async def trigger_review():
    """Manually trigger a council review session."""
    council = _get_council()
    if not council:
        return {"status": "error", "message": "Council not active"}

    if not council._client:
        return {"status": "error", "message": "OpenRouter client not configured"}

    session = await council._run_review_session()
    council._sessions.append(session)
    return session.model_dump()

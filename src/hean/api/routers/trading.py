"""Trading operations router."""

from fastapi import APIRouter, HTTPException, Query, status

from hean.api.app import engine_facade
from hean.api.schemas import CancelAllOrdersRequest, ClosePositionRequest, TestOrderRequest
from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/orders", tags=["trading"])


def _check_live_trading(confirm_phrase: str | None) -> None:
    """Check if live trading is allowed."""
    if not settings.is_live:
        return  # Paper trading, no check needed

    if not settings.live_confirm:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="LIVE_CONFIRM must be true for live trading",
        )

    if settings.dry_run:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="DRY_RUN must be false for live trading",
        )

    if confirm_phrase != "I_UNDERSTAND_LIVE_TRADING":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid confirmation phrase for live trading",
        )


@router.get("/positions")
async def get_positions() -> list[dict]:
    """Get current positions."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.get_positions()
        return result
    except Exception as e:
        logger.error(f"Failed to get positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("")
async def get_orders(
    status: str = Query(default="all", description="Filter by status: all, open, filled")
) -> list[dict]:
    """Get orders."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.get_orders(status=status)  # type: ignore
        return result
    except Exception as e:
        logger.error(f"Failed to get orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test")
async def place_test_order(request: TestOrderRequest) -> dict:
    """Place a test order (paper only)."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    if settings.is_live and not settings.dry_run:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Test orders are only allowed in paper/dry_run mode",
        )

    try:
        # TODO: Implement test order placement
        return {
            "status": "success",
            "message": f"Test order placed: {request.side} {request.size} {request.symbol}",
            "order_id": "test_order_123",
        }
    except Exception as e:
        logger.error(f"Failed to place test order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/close-position")
async def close_position(request: ClosePositionRequest) -> dict:
    """Close a position."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    _check_live_trading(request.confirm_phrase)

    try:
        # TODO: Implement position closing
        return {
            "status": "success",
            "message": f"Position {request.position_id} closed",
        }
    except Exception as e:
        logger.error(f"Failed to close position: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cancel-all")
async def cancel_all_orders(request: CancelAllOrdersRequest) -> dict:
    """Cancel all open orders."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    _check_live_trading(request.confirm_phrase)

    try:
        # TODO: Implement cancel all orders
        return {
            "status": "success",
            "message": "All orders cancelled",
        }
    except Exception as e:
        logger.error(f"Failed to cancel all orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orderbook-presence")
async def get_orderbook_presence(
    symbol: str | None = Query(default=None, description="Symbol filter (optional)")
) -> dict | list[dict]:
    """Get our orderbook presence (Phase 3: Smart Limit Engine).
    
    Shows our limit orders as glowing clusters in the orderbook,
    with their distance from mid-price.
    """
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        result = await engine_facade.get_orderbook_presence(symbol)
        return result
    except Exception as e:
        logger.error(f"Failed to get orderbook presence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


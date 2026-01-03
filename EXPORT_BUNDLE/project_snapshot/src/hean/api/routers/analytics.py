"""Analytics router."""

from fastapi import APIRouter, HTTPException

from hean.api.app import engine_facade
from hean.api.schemas import AnalyticsSummary, BlockedSignalsAnalytics
from hean.api.services.job_queue import job_queue_service
from hean.api.schemas import BacktestRequest, EvaluateRequest
from hean.logging import get_logger
from hean.observability.no_trade_report import no_trade_report

logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
async def get_analytics_summary() -> AnalyticsSummary:
    """Get analytics summary."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        if not engine_facade.is_running:
            return AnalyticsSummary()

        # TODO: Calculate real analytics from trading history
        return AnalyticsSummary(
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            max_drawdown=0.0,
            max_drawdown_pct=0.0,
            avg_trade_duration_sec=0.0,
            trades_per_day=0.0,
            total_pnl=0.0,
            daily_pnl=0.0,
        )
    except Exception as e:
        logger.error(f"Failed to get analytics summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/blocks")
async def get_blocked_signals_analytics() -> BlockedSignalsAnalytics:
    """Get blocked signals analytics."""
    try:
        # Get data from no_trade_report
        report_data = no_trade_report.get_summary()

        return BlockedSignalsAnalytics(
            total_blocks=report_data.get("total_blocks", 0),
            top_reasons=report_data.get("top_reasons", []),
            blocks_by_hour=report_data.get("blocks_by_hour", {}),
            recent_blocks=report_data.get("recent_blocks", []),
        )
    except Exception as e:
        logger.error(f"Failed to get blocked signals analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backtest")
async def run_backtest(request: BacktestRequest) -> dict:
    """Run a backtest."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    async def backtest_task(job) -> dict:
        """Backtest task function."""
        # TODO: Implement actual backtest
        await asyncio.sleep(1)  # Simulate work
        return {
            "symbol": request.symbol,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "result": "success",
        }

    job_id = await job_queue_service.submit_job("backtest", request.model_dump(), backtest_task)
    return {"job_id": job_id, "status": "pending"}


@router.post("/evaluate")
async def run_evaluate(request: EvaluateRequest) -> dict:
    """Run an evaluation."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    async def evaluate_task(job) -> dict:
        """Evaluate task function."""
        # TODO: Implement actual evaluation
        import asyncio
        await asyncio.sleep(1)  # Simulate work
        return {
            "symbol": request.symbol,
            "days": request.days,
            "result": "success",
        }

    job_id = await job_queue_service.submit_job("evaluate", request.model_dump(), evaluate_task)
    return {"job_id": job_id, "status": "pending"}


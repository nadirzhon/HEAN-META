"""Analytics router."""

import asyncio
from typing import Any

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


# Phase 5: Statistical Arbitrage & Anti-Fragile Architecture endpoints
@router.get("/phase5/correlation-matrix")
async def get_correlation_matrix() -> dict[str, Any]:
    """Get correlation matrix for Phase 5 pair trading.
    
    Returns:
        Dictionary with correlation matrix data for visualization
    """
    if engine_facade is None or not engine_facade.is_running:
        raise HTTPException(status_code=500, detail="Engine not running")
    
    try:
        # Get correlation engine from trading system
        trading_system = engine_facade._trading_system
        if not trading_system or not hasattr(trading_system, '_correlation_engine'):
            return {"correlation_matrix": {}, "symbols": []}
        
        correlation_engine = trading_system._correlation_engine
        if correlation_engine is None:
            return {"correlation_matrix": {}, "symbols": []}
        
        # Get correlation matrix
        matrix = correlation_engine.get_correlation_matrix()
        
        # Format for UI: convert tuple keys to string keys
        formatted_matrix: dict[str, float] = {}
        symbols = set()
        
        for (symbol_a, symbol_b), correlation in matrix.items():
            key = f"{symbol_a}:{symbol_b}"
            formatted_matrix[key] = correlation
            symbols.add(symbol_a)
            symbols.add(symbol_b)
        
        return {
            "correlation_matrix": formatted_matrix,
            "symbols": sorted(list(symbols)),
            "min_correlation": 0.7,  # From config
            "threshold": 0.7
        }
    except Exception as e:
        logger.error(f"Failed to get correlation matrix: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phase5/profit-probability-curve")
async def get_profit_probability_curve() -> dict[str, Any]:
    """Get profit probability curve based on Kelly Criterion (Phase 5).
    
    Returns:
        Dictionary with profit probability data for visualization
    """
    if engine_facade is None or not engine_facade.is_running:
        raise HTTPException(status_code=500, detail="Engine not running")
    
    try:
        # Get strategy metrics and calculate Kelly fractions
        trading_system = engine_facade._trading_system
        if not trading_system or not hasattr(trading_system, '_accounting'):
            return {"strategies": [], "curve_points": []}
        
        accounting = trading_system._accounting
        strategy_metrics = accounting.get_strategy_metrics()
        
        # Calculate Kelly fractions for each strategy
        strategies_data = []
        curve_points = []
        
        for strategy_id, metrics_data in strategy_metrics.items():
            wins = metrics_data.get("wins", 0)
            losses = metrics_data.get("losses", 0)
            total = wins + losses
            
            if total < 10:
                continue
            
            win_rate = wins / total if total > 0 else 0.5
            avg_win = metrics_data.get("avg_win", 0.0)
            avg_loss = abs(metrics_data.get("avg_loss", 0.0))
            
            # Calculate Kelly edge
            if avg_loss > 0:
                odds_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
                kelly_fraction = (win_rate * odds_ratio - (1.0 - win_rate)) / odds_ratio if odds_ratio > 0 else 0.0
                fractional_kelly = kelly_fraction * 0.25  # Quarter Kelly
            else:
                kelly_fraction = 0.0
                fractional_kelly = 0.0
                odds_ratio = 0.0
            
            strategies_data.append({
                "strategy_id": strategy_id,
                "win_rate": win_rate,
                "odds_ratio": odds_ratio,
                "kelly_fraction": kelly_fraction,
                "fractional_kelly": fractional_kelly,
                "total_trades": total,
                "profit_factor": metrics_data.get("profit_factor", 1.0)
            })
            
            # Generate curve points (probability vs position size)
            for position_fraction in [0.1, 0.2, 0.3, 0.4, 0.5]:
                # Simplified: probability of profit = win_rate * position_fraction
                prob_profit = win_rate * position_fraction
                curve_points.append({
                    "strategy_id": strategy_id,
                    "position_fraction": position_fraction,
                    "probability_profit": prob_profit
                })
        
        return {
            "strategies": strategies_data,
            "curve_points": curve_points
        }
    except Exception as e:
        logger.error(f"Failed to get profit probability curve: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phase5/safety-net-status")
async def get_safety_net_status() -> dict[str, Any]:
    """Get Global Safety Net (Black Swan Protection) status (Phase 5).
    
    Returns:
        Dictionary with safety net status and entropy metrics
    """
    if engine_facade is None or not engine_facade.is_running:
        raise HTTPException(status_code=500, detail="Engine not running")
    
    try:
        trading_system = engine_facade._trading_system
        if not trading_system or not hasattr(trading_system, '_safety_net'):
            return {
                "active": False,
                "entropy_metrics": {},
                "hedge_positions": {}
            }
        
        safety_net = trading_system._safety_net
        if safety_net is None:
            return {
                "active": False,
                "entropy_metrics": {},
                "hedge_positions": {}
            }
        
        entropy_metrics = safety_net.get_entropy_metrics()
        
        return {
            "active": safety_net.is_active(),
            "entropy_metrics": entropy_metrics,
            "hedge_positions": safety_net._hedge_positions,
            "size_multiplier": safety_net.get_size_multiplier()
        }
    except Exception as e:
        logger.error(f"Failed to get safety net status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/phase5/system-health")
async def get_system_health() -> dict[str, Any]:
    """Get Self-Healing Middleware system health status (Phase 5).
    
    Returns:
        Dictionary with system health metrics
    """
    if engine_facade is None or not engine_facade.is_running:
        raise HTTPException(status_code=500, detail="Engine not running")
    
    try:
        trading_system = engine_facade._trading_system
        if not trading_system or not hasattr(trading_system, '_self_healing'):
            return {"status": "unknown", "metrics": {}}
        
        self_healing = trading_system._self_healing
        if self_healing is None:
            return {"status": "disabled", "metrics": {}}
        
        health_status = self_healing.get_health_status()
        
        return {
            "status": health_status["status"],
            "metrics": health_status,
            "healthy": self_healing.is_healthy()
        }
    except Exception as e:
        logger.error(f"Failed to get system health: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


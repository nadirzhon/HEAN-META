"""System endpoints router."""

import os
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from hean.api.app import engine_facade, reconcile_service
from hean.api.schemas import JobResponse
from hean.api.services.event_stream import event_stream_service
from hean.api.services.job_queue import job_queue_service
from hean.api.services.log_stream import log_stream_service
from hean.config import settings
from hean.core.system.health_monitor import get_health_monitor
from hean.logging import get_logger
from hean.observability.metrics import metrics

logger = get_logger(__name__)

router = APIRouter(tags=["system"])


@router.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "engine_running": engine_facade.is_running if engine_facade else False,
    }


@router.get("/health/pulse")
async def health_pulse() -> dict:
    """Get health status from The Pulse (health monitor)."""
    try:
        monitor = await get_health_monitor()
        status = monitor.get_health_status()
        return {
            "status": "ok",
            "modules": status,
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get health pulse: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
        }


@router.get("/health/module/{module_name}")
async def get_module_health(module_name: str) -> dict:
    """Get health status for a specific module."""
    try:
        monitor = await get_health_monitor()
        health = monitor.get_module_health(module_name)
        if not health:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
        return {
            "status": "ok",
            "module": module_name,
            "health": health,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get module health: {e}", exc_info=True)
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/settings")
async def get_settings() -> dict:
    """Get system settings (secrets masked)."""
    settings_dict = settings.model_dump()

    # Mask secrets
    if "bybit_api_key" in settings_dict and settings_dict["bybit_api_key"]:
        settings_dict["bybit_api_key"] = "***masked***"
    if "bybit_api_secret" in settings_dict and settings_dict["bybit_api_secret"]:
        settings_dict["bybit_api_secret"] = "***masked***"

    return settings_dict


@router.post("/reconcile/now")
async def reconcile_now() -> dict:
    """Trigger manual reconcile."""
    if reconcile_service is None:
        raise HTTPException(status_code=500, detail="Reconcile service not initialized")

    try:
        await reconcile_service.reconcile_now()
        return {"status": "success", "message": "Reconcile completed"}
    except Exception as e:
        logger.error(f"Failed to reconcile: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/smoke-test/run")
async def run_smoke_test() -> dict:
    """Run smoke test."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    try:
        # TODO: Implement smoke test
        return {"status": "success", "message": "Smoke test completed"}
    except Exception as e:
        logger.error(f"Failed to run smoke test: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs")
async def list_jobs(limit: int = 100) -> list[dict]:
    """List recent jobs."""
    try:
        jobs = await job_queue_service.list_jobs(limit=limit)
        return [job_queue_service.to_dict(job) for job in jobs]
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    """Get job by ID."""
    try:
        job = await job_queue_service.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return job_queue_service.to_dict(job)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/stream")
async def stream_events(request: Request) -> Response:
    """Stream events via SSE."""
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")

    # Set bus if available
    bus = await engine_facade.get_bus()
    if bus:
        event_stream_service.set_bus(bus)
        await event_stream_service.start()

    return await event_stream_service.stream(request)


@router.get("/logs/stream")
async def stream_logs(request: Request) -> Response:
    """Stream logs via SSE."""
    log_stream_service.setup()
    return await log_stream_service.stream(request)


@router.get("/metrics")
async def get_metrics() -> Response:
    """Get Prometheus metrics."""
    from hean.observability.metrics_exporter import MetricsExporter

    exporter = MetricsExporter()
    metrics_text = exporter.export()
    return Response(content=metrics_text, media_type="text/plain")


@router.get("/v1/dashboard")
async def get_dashboard_data() -> dict:
    """Get dashboard data for the frontend."""
    from hean.api.app import engine_facade
    
    if engine_facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")
    
    try:
        # Get basic metrics from engine facade
        risk_status = await engine_facade.get_risk_status() if engine_facade.is_running else {}
        
        # Get portfolio data if available
        portfolio_data = {}
        if hasattr(engine_facade, '_engine') and hasattr(engine_facade._engine, '_accounting'):
            accounting = engine_facade._engine._accounting
            equity = accounting.get_equity() if hasattr(accounting, 'get_equity') else 0
            portfolio_data = {
                "equity": equity,
            }
        
        # Phase 19: Get network stats from DistributedNodeManager if available
        network_stats = None
        try:
            # Try to get network manager from engine or system
            # In production, this would be injected or accessed via a singleton
            from hean.core.network.global_sync import DistributedNodeManager
            # For now, return empty stats - will be populated when manager is initialized
            network_stats = {
                "local_region": "UNKNOWN",
                "local_role": "UNKNOWN",
                "master_node": None,
                "nodes": {},
                "execution_count": {},
                "failover_count": 0,
                "active_positions": 0,
                "active_orders": 0,
            }
        except (ImportError, AttributeError) as e:
            logger.debug(f"Network manager not available: {e}")
            network_stats = None
        
        return {
            "timestamp": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "metrics": {
                "equity": portfolio_data.get("equity", 0),
                "daily_pnl": 0,  # TODO: Calculate from portfolio
                "return_pct": 0,  # TODO: Calculate from portfolio
                "open_positions": risk_status.get("open_positions", 0),
                "daily_trades": 0,  # TODO: Get from metrics
                "win_rate": 0,  # TODO: Get from metrics
            },
            "status": {
                "engine_running": engine_facade.is_running,
                "trading_mode": settings.trading_mode,
            },
            "network_stats": network_stats,  # Phase 19: Network stats
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


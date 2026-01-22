"""System endpoints router."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

import hean.api.state as state
from hean.api.services.event_stream import event_stream_service
from hean.api.services.job_queue import job_queue_service
from hean.api.services.log_stream import log_stream_service
from hean.config import settings
from hean.core.system.health_monitor import get_health_monitor
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["system"])


def _get_facade(request: Request | None = None):
    facade = state.get_engine_facade(request)
    if facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")
    return facade


def _get_reconcile_service(request: Request | None = None):
    reconcile = getattr(request.app.state, "reconcile_service", None) if request else None
    if reconcile is None:
        reconcile = state.reconcile_service
    return reconcile


def _default_network_stats() -> dict:
    """Provide a stable fallback payload for the network map UI."""
    return {
        "local_region": "TOKYO",
        "local_role": "STANDBY",
        "master_node": None,
        "nodes": {
            "TOKYO": {
                "region": "TOKYO",
                "role": "STANDBY",
                "is_healthy": False,
                "is_alive": False,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "network_latency_ms": 0.0,
                "exchange_latencies": {},
                "active_connections": 0,
            },
            "SINGAPORE": {
                "region": "SINGAPORE",
                "role": "STANDBY",
                "is_healthy": False,
                "is_alive": False,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "network_latency_ms": 0.0,
                "exchange_latencies": {},
                "active_connections": 0,
            },
            "FRANKFURT": {
                "region": "FRANKFURT",
                "role": "STANDBY",
                "is_healthy": False,
                "is_alive": False,
                "cpu_usage": 0.0,
                "memory_usage": 0.0,
                "network_latency_ms": 0.0,
                "exchange_latencies": {},
                "active_connections": 0,
            },
        },
        "execution_count": {},
        "failover_count": 0,
        "active_positions": 0,
        "active_orders": 0,
    }


def _resolve_current_symbol(trading_system: object | None) -> str:
    if trading_system:
        execution_router = getattr(trading_system, "_execution_router", None)
        if execution_router:
            current_prices = getattr(execution_router, "_current_prices", {})
            if current_prices:
                for preferred in ("BTCUSDT", "ETHUSDT"):
                    if preferred in current_prices:
                        return preferred
                return next(iter(current_prices.keys()))
    return settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"


def _default_swarm_state(symbol: str) -> dict:
    return {
        "symbol": symbol,
        "consensus_percentage": 0.0,
        "buy_votes": 0,
        "sell_votes": 0,
        "total_agents": 0,
        "average_confidence": 0.0,
        "execution_signal_strength": 0.0,
        "consensus_reached": False,
        "price_level_ofi": [],
    }


@router.get("/health/pulse")
async def health_pulse() -> dict:
    """Get health status from The Pulse (health monitor)."""
    try:
        monitor = await get_health_monitor()
        status = await monitor.get_health_status()
    except Exception as e:
        logger.error(f"Failed to get health pulse: {e}", exc_info=True)


@router.get("/exchange/health")
async def exchange_health(request: Request) -> dict:
    """Check Bybit API connectivity."""
    try:
        import httpx
        api_accessible = False
        try:
            base_url = "https://api-testnet.bybit.com" if settings.bybit_testnet else "https://api.bybit.com"
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{base_url}/v5/market/tickers", params={"category": "spot", "symbol": "BTCUSDT"})
                data = response.json()
                api_accessible = data.get("retCode") == 0
        except Exception as e:
            logger.warning(f"Bybit API ping failed: {e}")
            api_accessible = False

        # In PAPER mode, keys are not required
        keys_valid = True
        status = "ok" if api_accessible else "error"
        message = "Bybit API accessible" if status == "ok" else "Bybit API not accessible"

    except Exception as e:
        logger.error(f"Exchange health check failed: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Failed to get health pulse: {e}", exc_info=True)


@router.get("/health/module/{module_name}")
async def get_module_health(module_name: str) -> dict:
    """Get health status for a specific module."""
    try:
        monitor = await get_health_monitor()
        health = monitor.get_module_health(module_name)
        if not health:
            from fastapi import HTTPException
            raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
        return health
    except HTTPException:
        raise
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to get module health: {str(e)}")


@router.get("/dashboard")
async def get_dashboard(request: Request) -> dict:
    """Get dashboard data."""
    engine_facade = _get_facade(request)
    
    try:
        # Get engine status
        engine_status = await engine_facade.get_status()
        engine_running = engine_status.get("running", False)
        
        trading_state = await engine_facade.get_trading_state()
        account_state = trading_state.get("account_state") or cached_account_state or {
            "wallet_balance": default_equity,
            "available_balance": default_equity,
            "equity": default_equity,
            "used_margin": 0.0,
            "reserved_margin": 0.0,
            "unrealized_pnl": 0.0,
            "realized_pnl": 0.0,
            "fees": 0.0,
            "fees_24h": 0.0,
        }
        positions = trading_state.get("positions") or cached_positions or []
        orders = trading_state.get("orders") or cached_orders or []

        # Get portfolio data from engine status (includes equity)
        equity = engine_status.get("equity", default_equity)
        daily_pnl = engine_status.get("daily_pnl", 0.0)
        initial_capital = engine_status.get("initial_capital", default_equity)
        return_pct = ((equity - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0
        
        # Get risk status for positions count
        risk_status = await engine_facade.get_risk_status() if engine_running else {}
        open_positions = risk_status.get("current_positions", risk_status.get("open_positions", 0))
        
        # Get orders count for daily trades
        orders = await engine_facade.get_orders(status="filled")
        daily_trades = len(orders)  # Simple count for now
        
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
            # Network manager not available - this is expected in some configurations
            network_stats = None
        if network_stats is None:
            network_stats = _default_network_stats()

        trading_system = getattr(engine_facade, "_trading_system", None)
        current_symbol = _resolve_current_symbol(trading_system)

        swarm_state = None
        swarm = getattr(engine_facade, "_multimodal_swarm", None)
        if swarm and hasattr(swarm, "get_latest_state"):
            try:
                swarm_state = swarm.get_latest_state(current_symbol)
            except Exception as e:
                logger.debug(f"Failed to fetch swarm state: {e}")
        if not swarm_state:
            swarm_state = _default_swarm_state(current_symbol)

        execution_router = getattr(trading_system, "_execution_router", None)
        if execution_router and hasattr(execution_router, "_ofi"):
            try:
                ofi_monitor = execution_router._ofi
                if ofi_monitor:
                    ofi_result = ofi_monitor.calculate_ofi(current_symbol)
                    swarm_state["price_level_ofi"] = ofi_result.price_level_ofi
            except Exception as e:
                logger.debug(f"Failed to calculate OFI: {e}")

        iceberg_orders = []
        if execution_router and hasattr(execution_router, "get_iceberg_detections"):
            try:
                iceberg_orders = execution_router.get_iceberg_detections(limit=20)
            except Exception as e:
                logger.debug(f"Failed to get iceberg detections: {e}")
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}", exc_info=True)
        # Return default data instead of error to ensure dashboard always works
        from datetime import datetime, timezone
        current_symbol = _resolve_current_symbol(None)

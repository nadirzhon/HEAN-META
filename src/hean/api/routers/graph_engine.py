"""Graph Engine API endpoints for Eureka UI."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/graph-engine", tags=["graph-engine"])


@router.get("/state")
async def get_graph_state(request: Request) -> dict[str, Any]:
    """Get current graph engine state (assets, correlations, etc.)."""
    try:
        engine_facade = request.state.engine_facade

        if not engine_facade or not hasattr(engine_facade, '_trading_system'):
            raise HTTPException(
                status_code=503,
                detail="Graph engine not available - trading system not initialized"
            )

        trading_system = engine_facade._trading_system
        if not hasattr(trading_system, '_graph_engine'):
            raise HTTPException(
                status_code=503,
                detail="Graph engine not available - module not initialized"
            )

        graph_engine = trading_system._graph_engine

        # Get asset data
        assets = []
        symbols = graph_engine._symbols if hasattr(graph_engine, '_symbols') else []

        for symbol in symbols:
            asset_data = {
                "symbol": symbol,
                "price": 0.0,
                "correlation": {},
                "leader_score": 0.0
            }

            if hasattr(graph_engine, 'get_correlation'):
                asset_data["leader_score"] = graph_engine.get_lead_lag("BTCUSDT", symbol) if symbol != "BTCUSDT" else 1.0

            assets.append(asset_data)

        # Build correlation matrix
        correlations = {}
        for i, sym_a in enumerate(symbols):
            for sym_b in symbols[i+1:]:
                corr = graph_engine.get_correlation(sym_a, sym_b)
                correlations[f"{sym_a}-{sym_b}"] = corr
                correlations[f"{sym_b}-{sym_a}"] = corr

        return {
            "assets": assets,
            "correlations": correlations,
            "asset_count": len(assets)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting graph state: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving graph state: {str(e)}"
        )


@router.get("/leader")
async def get_leader(request: Request) -> dict[str, Any]:
    """Get current market leader."""
    try:
        engine_facade = request.state.engine_facade

        if not engine_facade or not hasattr(engine_facade, '_trading_system'):
            raise HTTPException(
                status_code=503,
                detail="Graph engine not available - trading system not initialized"
            )

        trading_system = engine_facade._trading_system
        if not hasattr(trading_system, '_graph_engine'):
            raise HTTPException(
                status_code=503,
                detail="Graph engine not available - module not initialized"
            )

        graph_engine = trading_system._graph_engine

        leader = graph_engine.get_current_leader()
        if not leader:
            raise HTTPException(
                status_code=404,
                detail="No market leader detected yet - insufficient data"
            )

        # Calculate leader score
        leader_score = 1.0
        if hasattr(graph_engine, 'get_lead_lag'):
            total_score = 0.0
            count = 0
            for symbol in graph_engine._symbols:
                if symbol != leader:
                    score = graph_engine.get_lead_lag(leader, symbol)
                    total_score += max(0, score)
                    count += 1
            if count > 0:
                leader_score = total_score / count

        return {
            "leader": leader,
            "leader_score": leader_score
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting leader: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving market leader: {str(e)}"
        )


@router.get("/feature-vector")
async def get_feature_vector(request: Request, size: int = 5000) -> dict[str, Any]:
    """Get high-dimensional feature vector for neural network."""
    try:
        engine_facade = request.state.engine_facade

        if not engine_facade or not hasattr(engine_facade, '_trading_system'):
            raise HTTPException(
                status_code=503,
                detail="Graph engine not available - trading system not initialized"
            )

        trading_system = engine_facade._trading_system
        if not hasattr(trading_system, '_graph_engine'):
            raise HTTPException(
                status_code=503,
                detail="Graph engine not available - module not initialized"
            )

        graph_engine = trading_system._graph_engine

        if hasattr(graph_engine, 'get_feature_vector'):
            feature_vector = graph_engine.get_feature_vector(size)
            return {
                "feature_vector": feature_vector[:size] if isinstance(feature_vector, list) else feature_vector.tolist()[:size],
                "size": len(feature_vector) if isinstance(feature_vector, list) else len(feature_vector)
            }

        raise HTTPException(
            status_code=501,
            detail="Graph engine does not support feature vector extraction"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting feature vector: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving feature vector: {str(e)}"
        )


@router.get("/topology/score")
async def get_topology_score(request: Request) -> dict[str, Any]:
    """Get market topology score (TDA-based structural stability)."""
    try:
        try:
            import graph_engine_py  # type: ignore
            fast_warden = graph_engine_py.FastWarden()
            market_score = fast_warden.get_market_topology_score()
            is_disconnected = fast_warden.is_market_disconnected()

            return {
                "market_topology_score": market_score,
                "is_disconnected": is_disconnected,
                "stability": "stable" if market_score > 0.7 else "unstable" if market_score > 0.4 else "collapsing",
            }
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="FastWarden not available - C++ graph engine module not loaded"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting topology score: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving topology score: {str(e)}"
        )


@router.get("/topology/manifold/{symbol}")
async def get_manifold_data(request: Request, symbol: str) -> dict[str, Any]:
    """Get 3D manifold data for visualization (topological holes overlay)."""
    try:
        engine_facade = request.state.engine_facade

        if not engine_facade or not hasattr(engine_facade, '_trading_system'):
            raise HTTPException(
                status_code=503,
                detail="TDA engine not available - trading system not initialized"
            )

        trading_system = engine_facade._trading_system

        # Try FastWarden for manifold data
        try:
            import graph_engine_py  # type: ignore
            fast_warden = graph_engine_py.FastWarden()
            manifold = fast_warden.get_manifold_data(symbol)
            return {
                "symbol": symbol,
                "point_cloud": manifold.get("points", []),
                "persistence_barcodes": manifold.get("barcodes", []),
                "num_holes": manifold.get("num_holes", 0),
                "topology_score": manifold.get("score", 0.0),
            }
        except (ImportError, AttributeError):
            raise HTTPException(
                status_code=503,
                detail="FastWarden not available for manifold data"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting manifold data for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving manifold data: {str(e)}"
        )


@router.get("/topology/watchdog")
async def get_watchdog_status(request: Request) -> dict[str, Any]:
    """Get Topological Watchdog status."""
    try:
        engine_facade = request.state.engine_facade

        if not engine_facade or not hasattr(engine_facade, '_trading_system'):
            raise HTTPException(
                status_code=503,
                detail="Topological watchdog not available - trading system not initialized"
            )

        trading_system = engine_facade._trading_system
        if hasattr(trading_system, '_topological_watchdog'):
            watchdog = trading_system._topological_watchdog
            status = watchdog.get_topology_status()
            return status

        # Fallback: check FastWarden directly
        try:
            import graph_engine_py  # type: ignore
            fast_warden = graph_engine_py.FastWarden()
            return {
                "halt_active": fast_warden.is_market_disconnected(),
                "topology_score": fast_warden.get_market_topology_score(),
                "is_disconnected": fast_warden.is_market_disconnected(),
            }
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Topological watchdog not available - FastWarden module not loaded"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting watchdog status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error retrieving watchdog status: {str(e)}"
        )

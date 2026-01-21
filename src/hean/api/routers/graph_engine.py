"""Graph Engine API endpoints for Eureka UI."""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/graph-engine", tags=["graph-engine"])


@router.get("/state")
async def get_graph_state(request: Request) -> dict[str, Any]:
    """Get current graph engine state (assets, correlations, etc.)."""
    try:
        engine_facade = request.state.engine_facade
        
        # Check if graph engine is available
        if not engine_facade or not hasattr(engine_facade, '_trading_system'):
            # Return mock data if engine not available
            return _get_mock_state()
        
        trading_system = engine_facade._trading_system
        if not hasattr(trading_system, '_graph_engine'):
            return _get_mock_state()
        
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
            
            # Try to get current price (would need to access price feed or cache)
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
    except Exception as e:
        logger.error(f"Error getting graph state: {e}")
        return _get_mock_state()


@router.get("/leader")
async def get_leader(request: Request) -> dict[str, Any]:
    """Get current market leader."""
    try:
        engine_facade = request.state.engine_facade
        
        if not engine_facade or not hasattr(engine_facade, '_trading_system'):
            return {"leader": "BTCUSDT", "leader_score": 1.0}
        
        trading_system = engine_facade._trading_system
        if not hasattr(trading_system, '_graph_engine'):
            return {"leader": "BTCUSDT", "leader_score": 1.0}
        
        graph_engine = trading_system._graph_engine
        
        leader = graph_engine.get_current_leader()
        if not leader:
            leader = "BTCUSDT"
        
        # Calculate leader score (simplified)
        leader_score = 1.0
        if hasattr(graph_engine, 'get_lead_lag'):
            # Sum of lead-lag scores relative to BTC
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
    except Exception as e:
        logger.error(f"Error getting leader: {e}")
        return {"leader": "BTCUSDT", "leader_score": 1.0}


@router.get("/feature-vector")
async def get_feature_vector(request: Request, size: int = 5000) -> dict[str, Any]:
    """Get high-dimensional feature vector for neural network."""
    try:
        engine_facade = request.state.engine_facade
        
        if not engine_facade or not hasattr(engine_facade, '_trading_system'):
            return {"feature_vector": [0.0] * min(size, 1000), "size": min(size, 1000)}
        
        trading_system = engine_facade._trading_system
        if not hasattr(trading_system, '_graph_engine'):
            return {"feature_vector": [0.0] * min(size, 1000), "size": min(size, 1000)}
        
        graph_engine = trading_system._graph_engine
        
        if hasattr(graph_engine, 'get_feature_vector'):
            feature_vector = graph_engine.get_feature_vector(size)
            return {
                "feature_vector": feature_vector[:size] if isinstance(feature_vector, list) else feature_vector.tolist()[:size],
                "size": len(feature_vector) if isinstance(feature_vector, list) else len(feature_vector)
            }
        
        return {"feature_vector": [0.0] * min(size, 1000), "size": min(size, 1000)}
    except Exception as e:
        logger.error(f"Error getting feature vector: {e}")
        return {"feature_vector": [0.0] * min(size, 1000), "size": min(size, 1000)}


@router.get("/topology/score")
async def get_topology_score(request: Request) -> dict[str, Any]:
    """Get market topology score (TDA-based structural stability)."""
    try:
        # Try to get from FastWarden if available
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
            logger.warning("FastWarden not available, using fallback topology score")
            return {
                "market_topology_score": 1.0,
                "is_disconnected": False,
                "stability": "stable",
            }
    except Exception as e:
        logger.error(f"Error getting topology score: {e}")
        return {
            "market_topology_score": 1.0,
            "is_disconnected": False,
            "stability": "unknown",
        }


@router.get("/topology/manifold/{symbol}")
async def get_manifold_data(request: Request, symbol: str) -> dict[str, Any]:
    """Get 3D manifold data for visualization (topological holes overlay).
    
    Returns point cloud data and persistence barcodes for 3D visualization.
    """
    try:
        # Mock manifold data structure for visualization
        # In production, this would come from FastWarden/TDA_Engine
        
        # Generate mock point cloud (price, size, time)
        import time
        import random
        
        num_points = 50
        points = []
        
        base_price = 50000.0 if "BTC" in symbol else 3000.0
        
        for i in range(num_points):
            # Price dimension (normalized)
            price_offset = (i / num_points - 0.5) * 0.02  # ±1% price range
            
            # Size dimension (normalized liquidity)
            size = random.uniform(0.1, 1.0)
            
            # Time dimension (normalized timestamp)
            time_dim = (i / num_points)
            
            points.append({
                "x": base_price * (1.0 + price_offset),
                "y": size,
                "z": time_dim,
            })
        
        # Mock persistence barcodes (topological holes)
        persistence_pairs = [
            {
                "birth": 0.1,
                "death": 0.3,
                "dimension": 0,  # Connected component
                "persistence": 0.2,
            },
            {
                "birth": 0.2,
                "death": 0.5,
                "dimension": 1,  # Loop (hole)
                "persistence": 0.3,
            },
        ]
        
        return {
            "symbol": symbol,
            "point_cloud": points,
            "persistence_barcodes": persistence_pairs,
            "num_holes": len([p for p in persistence_pairs if p["dimension"] == 1]),
            "topology_score": 0.85,  # Mock score
        }
    except Exception as e:
        logger.error(f"Error getting manifold data for {symbol}: {e}")
        return {
            "symbol": symbol,
            "point_cloud": [],
            "persistence_barcodes": [],
            "num_holes": 0,
            "topology_score": 1.0,
        }


@router.get("/topology/watchdog")
async def get_watchdog_status(request: Request) -> dict[str, Any]:
    """Get Topological Watchdog status."""
    try:
        engine_facade = request.state.engine_facade
        
        if not engine_facade or not hasattr(engine_facade, '_trading_system'):
            return {
                "halt_active": False,
                "topology_score": 1.0,
                "is_disconnected": False,
            }
        
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
            return {
                "halt_active": False,
                "topology_score": 1.0,
                "is_disconnected": False,
            }
    except Exception as e:
        logger.error(f"Error getting watchdog status: {e}")
        return {
            "halt_active": False,
            "topology_score": 1.0,
            "is_disconnected": False,
        }


def _get_mock_state() -> dict[str, Any]:
    """Return mock state for testing."""
    symbols = [
        "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
        "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "DOTUSDT", "LINKUSDT"
    ]
    
    assets = [
        {
            "symbol": symbol,
            "price": 50000.0 + i * 1000,
            "correlation": {},
            "leader_score": 1.0 if symbol == "BTCUSDT" else 0.5 - i * 0.05
        }
        for i, symbol in enumerate(symbols)
    ]
    
    correlations = {}
    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i+1:]:
            corr = 0.7 + (hash(f"{sym_a}-{sym_b}") % 30) / 100
            correlations[f"{sym_a}-{sym_b}"] = corr
            correlations[f"{sym_b}-{sym_a}"] = corr
    
    return {
        "assets": assets,
        "correlations": correlations,
        "asset_count": len(assets)
    }

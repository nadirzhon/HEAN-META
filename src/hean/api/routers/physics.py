"""Physics API Router - Market Thermodynamics Endpoints."""

from fastapi import APIRouter, Query

router = APIRouter(prefix="/physics", tags=["physics"])


@router.get("/state")
async def get_physics_state(symbol: str = Query(default="BTCUSDT")):
    """Get current physics state for a symbol."""
    # Import here to avoid circular imports
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_physics_engine"):
        engine = facade._physics_engine
        if engine:
            state = engine.get_state(symbol)
            if state:
                return state.to_dict()

    return {
        "symbol": symbol,
        "temperature": 0.0,
        "temperature_regime": "COLD",
        "entropy": 0.0,
        "entropy_state": "COMPRESSED",
        "phase": "unknown",
        "phase_confidence": 0.0,
        "szilard_profit": 0.0,
        "should_trade": False,
        "trade_reason": "Engine not running",
        "size_multiplier": 0.5,
    }


@router.get("/history")
async def get_physics_history(
    symbol: str = Query(default="BTCUSDT"),
    limit: int = Query(default=50, le=500),
):
    """Get historical physics readings."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_physics_engine"):
        engine = facade._physics_engine
        if engine:
            temp_history = engine._temperature.get_history(symbol, limit)
            entropy_history = engine._entropy.get_history(symbol, limit)
            phase_history = engine._phase_detector.get_history(symbol, limit)

            return {
                "symbol": symbol,
                "temperature": [
                    {"value": r.value, "regime": r.regime, "timestamp": r.timestamp}
                    for r in temp_history
                ],
                "entropy": [
                    {"value": r.value, "state": r.state, "timestamp": r.timestamp}
                    for r in entropy_history
                ],
                "phases": [
                    {
                        "phase": r.phase.value,
                        "confidence": r.confidence,
                        "timestamp": r.timestamp,
                    }
                    for r in phase_history
                ],
            }

    return {"symbol": symbol, "temperature": [], "entropy": [], "phases": []}


@router.get("/participants")
async def get_participants(symbol: str = Query(default="BTCUSDT")):
    """Get participant breakdown for a symbol."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_participant_classifier"):
        classifier = facade._participant_classifier
        if classifier:
            breakdown = classifier.get_breakdown(symbol)
            if breakdown:
                return breakdown.to_dict()

    return {
        "symbol": symbol,
        "mm_activity": 0.0,
        "institutional_flow": 0.0,
        "retail_sentiment": 0.5,
        "whale_activity": 0.0,
        "arb_pressure": 0.0,
        "dominant_player": "retail",
        "meta_signal": "No data",
    }


@router.get("/anomalies")
async def get_anomalies(limit: int = Query(default=20, le=100)):
    """Get recent market anomalies."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_anomaly_detector"):
        detector = facade._anomaly_detector
        if detector:
            anomalies = detector.get_recent(limit)
            return {
                "anomalies": [a.to_dict() for a in anomalies],
                "active_count": len(detector.get_active()),
            }

    return {"anomalies": [], "active_count": 0}

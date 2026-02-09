"""Brain API Router - AI Analysis Endpoints.

Reads from in-process brain client OR from Redis brain:signals stream
(published by the hean-brain microservice container).
"""

import time
from collections import deque

from fastapi import APIRouter, Query

from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/brain", tags=["brain"])

# Cache for Redis-sourced brain signals
_redis_signals: deque[dict] = deque(maxlen=200)
_last_redis_read: float = 0
_REDIS_POLL_INTERVAL = 2.0  # seconds


async def _read_brain_from_redis() -> list[dict]:
    """Read recent brain signals from Redis stream."""
    global _last_redis_read

    now = time.time()
    if now - _last_redis_read < _REDIS_POLL_INTERVAL and _redis_signals:
        return list(_redis_signals)

    try:
        import orjson
        from hean.core.system.redis_state import get_redis_state_manager

        manager = await get_redis_state_manager()
        if not manager or not manager._client:
            return list(_redis_signals)

        redis_client = manager._client

        # Read latest entries from brain:signals stream
        entries = await redis_client.xrevrange("brain:signals", count=50)
        if entries:
            new_signals = []
            for msg_id, msg_data in reversed(entries):
                data_bytes = msg_data.get(b"data") or msg_data.get("data")
                if data_bytes:
                    if isinstance(data_bytes, bytes):
                        signal = orjson.loads(data_bytes)
                    else:
                        signal = orjson.loads(data_bytes.encode())
                    new_signals.append(signal)

            if new_signals:
                _redis_signals.clear()
                _redis_signals.extend(new_signals)

        _last_redis_read = now

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Redis brain read: {e}")

    return list(_redis_signals)


def _signals_to_analysis(signals: list[dict]) -> dict:
    """Convert brain signals list to analysis format for the API."""
    if not signals:
        return {
            "timestamp": "",
            "thoughts": [],
            "forces": [],
            "signal": None,
            "summary": "Brain not active",
            "market_regime": "unknown",
        }

    latest = signals[-1]
    phase = latest.get("phase", "unknown")
    signal_type = latest.get("signal", "HOLD")
    confidence = latest.get("confidence", 0.5)
    symbol = latest.get("symbol", "BTCUSDT")
    temperature = latest.get("temperature", 0)
    entropy = latest.get("entropy", 0)
    ts_ms = latest.get("timestamp", 0)

    from datetime import datetime, timezone
    ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat() if ts_ms else ""

    thoughts = []
    # Physics thought
    thoughts.append({
        "id": f"phys-{ts_ms}",
        "timestamp": ts,
        "stage": "physics",
        "content": f"Phase: {phase.upper()}, T={temperature:.0f}, S={entropy:.2f}",
        "confidence": 0.85,
    })
    # Decision thought
    thoughts.append({
        "id": f"dec-{ts_ms}",
        "timestamp": ts,
        "stage": "decision",
        "content": f"{signal_type}: {phase.upper()} phase analysis",
        "confidence": confidence,
    })

    return {
        "timestamp": ts,
        "thoughts": thoughts,
        "forces": [],
        "signal": {
            "symbol": symbol,
            "action": signal_type,
            "confidence": confidence,
            "reason": f"{phase.upper()} phase, T={temperature:.0f}, S={entropy:.2f}",
        },
        "summary": f"{phase.upper()} phase, T={temperature:.0f}. Signal: {signal_type} ({confidence:.0%})",
        "market_regime": phase.lower(),
    }


def _signals_to_thoughts(signals: list[dict], limit: int = 50) -> list[dict]:
    """Convert brain signals to thoughts list."""
    from datetime import datetime, timezone
    thoughts = []
    for sig in signals[-limit:]:
        ts_ms = sig.get("timestamp", 0)
        ts = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat() if ts_ms else ""
        thoughts.append({
            "id": f"sig-{ts_ms}-{sig.get('symbol', '')}",
            "timestamp": ts,
            "stage": "decision",
            "content": f"{sig.get('symbol', '?')} {sig.get('signal', 'HOLD')} "
                       f"(conf={sig.get('confidence', 0):.2f}) "
                       f"phase={sig.get('phase', '?')}",
            "confidence": sig.get("confidence", 0.5),
        })
    return thoughts


@router.get("/analysis")
async def get_brain_analysis():
    """Get latest brain analysis."""
    from hean.api.engine_facade import get_facade

    # Try in-process brain client first
    facade = get_facade()
    if facade and hasattr(facade, "_brain_client"):
        client = facade._brain_client
        if client:
            analysis = client.get_latest_analysis()
            if analysis:
                return analysis

    # Fallback: read from Redis (hean-brain microservice)
    signals = await _read_brain_from_redis()
    return _signals_to_analysis(signals)


@router.get("/thoughts")
async def get_brain_thoughts(limit: int = Query(default=50, le=200)):
    """Get recent brain thoughts."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_brain_client"):
        client = facade._brain_client
        if client:
            return client.get_thoughts(limit)

    # Fallback: read from Redis
    signals = await _read_brain_from_redis()
    return _signals_to_thoughts(signals, limit)


@router.get("/history")
async def get_brain_history(limit: int = Query(default=20, le=50)):
    """Get brain analysis history."""
    from hean.api.engine_facade import get_facade

    facade = get_facade()
    if facade and hasattr(facade, "_brain_client"):
        client = facade._brain_client
        if client:
            return client.get_history(limit)

    # Fallback: build history from Redis signals (grouped by time)
    signals = await _read_brain_from_redis()
    if not signals:
        return []

    # Group signals into analysis snapshots (one per symbol batch)
    analyses = []
    seen = set()
    for sig in reversed(signals[-limit * 3:]):
        key = f"{sig.get('symbol', '')}-{sig.get('timestamp', 0) // 5000}"
        if key not in seen:
            seen.add(key)
            analyses.append(_signals_to_analysis([sig]))
            if len(analyses) >= limit:
                break

    return analyses

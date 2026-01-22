"""Market data endpoints (snapshot for UI priming)."""

from fastapi import APIRouter, Query

from hean.api.services.market_data_store import market_data_store
from hean.api.telemetry import telemetry_service
from hean.config import settings

router = APIRouter(prefix="/market", tags=["market"])


@router.get("/snapshot")
async def market_snapshot(
    symbol: str | None = Query(
        default=None, description="Trading symbol (defaults to first configured symbol)"
    ),
    timeframe: str = Query(default="1m", description="Timeframe for klines"),
    limit: int = Query(default=200, ge=1, le=1000, description="Number of klines"),
) -> dict:
    """Return recent market snapshot (klines + last tick)."""
    default_symbol = settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
    snapshot = await market_data_store.snapshot(symbol or default_symbol, timeframe=timeframe, limit=limit)
    snapshot["last_seq"] = telemetry_service.last_seq()
    return snapshot


@router.get("/ticker")
async def market_ticker(
    symbol: str | None = Query(
        default=None, description="Trading symbol (defaults to first configured symbol)"
    ),
) -> dict:
    """Return current ticker (last price, bid, ask, volume)."""
    default_symbol = settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
    tick = await market_data_store.latest_tick(symbol or default_symbol)
    if not tick:
        return {
            "symbol": symbol or default_symbol,
            "price": None,
            "bid": None,
            "ask": None,
            "volume": None,
            "timestamp": None,
        }
    return {
        "symbol": tick.get("symbol", symbol or default_symbol),
        "price": tick.get("price"),
        "bid": tick.get("bid"),
        "ask": tick.get("ask"),
        "volume": tick.get("volume"),
        "timestamp": tick.get("ts"),
    }


@router.get("/candles")
async def market_candles(
    symbol: str | None = Query(
        default=None, description="Trading symbol (defaults to first configured symbol)"
    ),
    timeframe: str = Query(default="1m", description="Timeframe for klines"),
    limit: int = Query(default=200, ge=1, le=1000, description="Number of klines"),
) -> dict:
    """Return candles (klines) for a symbol."""
    default_symbol = settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
    snapshot = await market_data_store.snapshot(symbol or default_symbol, timeframe=timeframe, limit=limit)
    return {
        "symbol": symbol or default_symbol,
        "timeframe": timeframe,
        "klines": snapshot.get("klines", []),
        "count": len(snapshot.get("klines", [])),
    }

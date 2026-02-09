"""Trading operations router."""

import asyncio
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, Request, status
from slowapi import Limiter
from slowapi.util import get_remote_address

import hean.api.state as state
from hean.api.schemas import (
    CancelAllOrdersRequest,
    ClosePositionRequest,
    TestOrderRequest,
    TestRoundtripRequest,
)
from hean.api.services.trading_metrics import trading_metrics
from hean.config import settings
from hean.core.types import Event, EventType, OrderStatus, Signal, Tick
from hean.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/orders", tags=["trading"])
why_router = APIRouter(prefix="/trading", tags=["trading"])
limiter = Limiter(key_func=get_remote_address)


def _get_facade(request: Request):
    facade = state.get_engine_facade(request)
    if facade is None:
        raise HTTPException(status_code=500, detail="Engine facade not initialized")
    return facade


def _check_live_trading(confirm_phrase: str | None) -> None:
    """Check if live trading is allowed."""
    # Testnet mode - no confirmation required
    if settings.bybit_testnet:
        return

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
async def get_positions(request: Request) -> dict:
    """Get current positions."""
    engine_facade = _get_facade(request)

    try:
        result = await engine_facade.get_positions()
        return {"positions": result}
    except Exception as e:
        logger.error(f"Failed to get positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/positions/monitor/stats")
async def get_position_monitor_stats(request: Request) -> dict:
    """Get position monitor statistics (force-close tracking)."""
    _get_facade(request)  # Validates facade exists

    try:
        # Get position monitor from trading system
        trading_system = getattr(request.app.state, "trading_system", None)
        if not trading_system or not hasattr(trading_system, "_position_monitor"):
            raise HTTPException(status_code=500, detail="Position monitor not available")

        stats = trading_system._position_monitor.get_statistics()
        return stats
    except Exception as e:
        logger.error(f"Failed to get position monitor stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("")
async def get_orders(
    request: Request,
    status: str = Query(default="all", description="Filter by status: all, open, filled")
) -> dict:
    """Get orders."""
    engine_facade = _get_facade(request)

    try:
        result = await engine_facade.get_orders(status=status)  # type: ignore
        return {"orders": result}
    except Exception as e:
        logger.error(f"Failed to get orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/test")
@limiter.limit("10/minute")
async def place_test_order(request: Request, payload: TestOrderRequest) -> dict:
    """Place a test order (paper only). Rate limited to 10 requests per minute."""
    engine_facade = _get_facade(request)
    bus = getattr(request.app.state, "bus", None)

    if settings.is_live and not settings.dry_run:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Test orders are only allowed in paper/dry_run mode",
        )

    if bus is None:
        raise HTTPException(status_code=500, detail="Event bus not initialized")

    try:
        # Resolve price
        price = payload.price
        if price is None and getattr(engine_facade, "_trading_system", None):
            try:
                router = engine_facade._trading_system._execution_router
                price = router._current_prices.get(payload.symbol)
            except Exception:
                price = None
        if price is None:
            raise HTTPException(
                status_code=400, detail="Price is required (no cached price available)"
            )

        # Build synthetic signal to exercise full pipeline
        side = payload.side.lower()
        tp = price * (1.003 if side == "buy" else 0.997)
        sl = price * (0.997 if side == "buy" else 1.003)
        signal = Signal(
            strategy_id="test_api",
            symbol=payload.symbol,
            side=side,
            entry_price=price,
            size=payload.size,
            stop_loss=sl,
            take_profit=tp,
            metadata={
                "entry_reason": "api_test_signal",
                "confidence": 1.0,
            },
        )

        await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))
        return {
            "status": "success",
            "message": f"Test signal published: {payload.side} {payload.size} {payload.symbol} @ {price}",
            "signal": signal.model_dump(),
        }
    except Exception as e:
        logger.error(f"Failed to place test order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/close-position")
@limiter.limit("10/minute")
async def close_position(request: Request, payload: ClosePositionRequest) -> dict:
    """Close a position."""
    engine_facade = _get_facade(request)

    _check_live_trading(payload.confirm_phrase)

    try:
        result = await engine_facade.close_position(payload.position_id, reason="api_close")
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Failed to close position"))
        return result
    except Exception as e:
        logger.error(f"Failed to close position: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/cancel")
@limiter.limit("30/minute")
async def cancel_order(request: Request, payload: dict) -> dict:
    """Cancel a single order by order_id."""
    engine_facade = _get_facade(request)
    order_id = payload.get("order_id")
    if not order_id:
        raise HTTPException(status_code=400, detail="order_id is required")

    try:
        trading_system = getattr(engine_facade, "_trading_system", None)
        if trading_system is None:
            raise HTTPException(status_code=500, detail="Engine not running")

        for order in trading_system._order_manager.get_open_orders():
            if getattr(order, "order_id", None) == order_id or str(getattr(order, "id", "")) == order_id:
                order.status = OrderStatus.CANCELLED
                await trading_system._bus.publish(
                    Event(event_type=EventType.ORDER_CANCELLED, data={"order": order})
                )
                return {"status": "success", "message": f"Order {order_id} cancelled"}

        raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel order: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/cancel-all")
@limiter.limit("5/minute")
async def cancel_all_orders(request: Request, payload: CancelAllOrdersRequest) -> dict:
    """Cancel all open orders."""
    engine_facade = _get_facade(request)

    _check_live_trading(payload.confirm_phrase)

    try:
        trading_system = getattr(engine_facade, "_trading_system", None)
        if trading_system is None:
            raise HTTPException(status_code=500, detail="Engine not running")

        cancelled = 0
        for order in trading_system._order_manager.get_open_orders():
            order.status = OrderStatus.CANCELLED
            await trading_system._bus.publish(
                Event(event_type=EventType.ORDER_CANCELLED, data={"order": order})
            )
            cancelled += 1

        return {
            "status": "success",
            "message": f"Cancelled {cancelled} open orders",
            "cancelled": cancelled,
        }
    except Exception as e:
        logger.error(f"Failed to cancel all orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/test_roundtrip")
@limiter.limit("10/minute")
async def test_roundtrip(request: Request, payload: TestRoundtripRequest) -> dict:
    """Open + close a paper position to validate full lifecycle."""
    engine_facade = _get_facade(request)
    bus = getattr(request.app.state, "bus", None)

    if settings.is_live and not settings.dry_run:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Roundtrip test only allowed in paper/dry_run mode",
        )
    if bus is None:
        raise HTTPException(status_code=500, detail="Event bus not initialized")

    trading_system = getattr(engine_facade, "_trading_system", None)
    if trading_system is None:
        raise HTTPException(status_code=500, detail="Engine not running")

    # Resolve price from router cache or fallback
    execution_router = trading_system._execution_router
    price = (
        execution_router._current_prices.get(payload.symbol)
        if hasattr(execution_router, "_current_prices")
        else None
    )
    if price is None or price <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"No price data available for {payload.symbol}. Wait for market data to arrive."
        )

    side = payload.side.lower()
    tp = price * (1 + payload.take_profit_pct / 100) if side == "buy" else price * (1 - payload.take_profit_pct / 100)
    sl = price * (1 - payload.stop_loss_pct / 100) if side == "buy" else price * (1 + payload.stop_loss_pct / 100)

    signal = Signal(
        strategy_id="test_roundtrip",
        symbol=payload.symbol,
        side=side,
        entry_price=price,
        size=payload.size,
        stop_loss=sl,
        take_profit=tp,
        metadata={
            "entry_reason": "roundtrip_smoke_test",
            "confidence": 1.0,
            "time_stop_seconds": payload.hold_timeout_sec,
        },
    )

    before_positions = len(trading_system._accounting.get_positions())

    await bus.publish(Event(event_type=EventType.SIGNAL, data={"signal": signal}))

    # Simulate a tick that should trigger TP quickly
    trigger_price = tp * (1.001 if side == "buy" else 0.999)
    tick = Tick(symbol=payload.symbol, price=trigger_price, timestamp=datetime.utcnow())
    await bus.publish(Event(event_type=EventType.TICK, data={"tick": tick}))

    # Wait for close or timeout
    for _ in range(12):
        await asyncio.sleep(0.5)
        current_positions = len(trading_system._accounting.get_positions())
        if current_positions <= before_positions:
            break

    snapshot = await engine_facade.get_trading_state()
    return {
        "status": "ok",
        "message": "Roundtrip executed",
        "account_state": snapshot.get("account_state"),
        "positions": snapshot.get("positions"),
        "orders": snapshot.get("orders"),
        "exit_decisions": trading_system._order_exit_decision_history[-5:],
    }


@router.post("/close-all-positions")
async def close_all_positions(request: Request) -> dict:
    """Close all open positions and cancel open orders."""
    engine_facade = _get_facade(request)
    try:
        result = await engine_facade.close_all_positions(reason="close_all_positions_api")
        return result
    except Exception as e:
        logger.error(f"Failed to close all positions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/close-position-by-symbol")
@limiter.limit("10/minute")
async def close_position_by_symbol(request: Request, payload: dict) -> dict:
    """Close a position by symbol name."""
    engine_facade = _get_facade(request)
    symbol = payload.get("symbol")
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol is required")

    try:
        positions = await engine_facade.get_positions()
        target = next((p for p in positions if p["symbol"] == symbol), None)
        if not target:
            return {"status": "not_found", "message": f"No open position for {symbol}"}

        result = await engine_facade.close_position(target["position_id"], reason="api_close_by_symbol")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to close position by symbol: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/paper/close_all")
async def panic_close_all(request: Request) -> dict:
    """Panic close all paper positions and cancel open orders (legacy endpoint)."""
    engine_facade = _get_facade(request)
    try:
        result = await engine_facade.close_all_positions(reason="panic_close_all_api")
        return result
    except Exception as e:
        logger.error(f"Failed to panic close all: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/paper/reset_state")
async def reset_paper_state(request: Request) -> dict:
    """Reset paper state (positions/orders/decisions)."""
    engine_facade = _get_facade(request)
    try:
        result = await engine_facade.reset_paper_state()
        return result
    except Exception as e:
        logger.error(f"Failed to reset paper state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/orderbook-presence")
async def get_orderbook_presence(
    request: Request,
    symbol: str | None = Query(default=None, description="Symbol filter (optional)")
) -> dict | list[dict]:
    """Get our orderbook presence (Phase 3: Smart Limit Engine).

    Shows our limit orders as glowing clusters in the orderbook,
    with their distance from mid-price.
    """
    engine_facade = _get_facade(request)

    try:
        result = await engine_facade.get_orderbook_presence(symbol)
        return result
    except Exception as e:
        logger.error(f"Failed to get orderbook presence: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@why_router.get("/why")
async def why_not_trading(request: Request) -> dict:
    """Explain why orders are not being created (transparency panel).

    Returns comprehensive diagnostics per AFO-Director spec:
    - engine_state, killswitch_state
    - last_tick_age_sec, last_signal_ts, last_decision_ts, last_order_ts, last_fill_ts
    - active_orders_count, active_positions_count
    - top_reason_codes_last_5m (top 10)
    - equity, balance, unreal_pnl, real_pnl, margin_used, margin_free
    - profit_capture_state (enabled, armed, triggered, cleared, mode, etc.)
    - execution_quality (ws_ok, rest_ok, avg_latency_ms, reject_rate_5m, slippage_est_5m)
    - multi_symbol (enabled, symbols_count, last_scanned_symbol, scan_cursor, scan_cycle_ts)
    """
    facade = _get_facade(request)

    try:
        # Get engine status
        engine_status = await facade.get_status() if facade.is_running else {}
        risk_status = await facade.get_risk_status() if facade.is_running else {}

        # Get killswitch status directly from risk router
        killswitch_status = {}
        try:
            if facade.is_running:
                from hean.api.routers.risk import (
                    get_killswitch_status as _get_killswitch_status_impl,
                )
                killswitch_status = await _get_killswitch_status_impl(request) if facade.is_running else {}
        except Exception as e:
            logger.debug(f"Could not get killswitch status: {e}")
            killswitch_status = {}

        # Get trading metrics
        metrics = await trading_metrics.get_metrics() if facade.is_running else {}

        # Get recent order decisions (last 5 minutes worth)
        from hean.api import main as api_main
        async with api_main.trading_state_lock:
            all_decisions = list(api_main.trading_state_cache.get("order_decisions", []))

        # Filter decisions from last 5 minutes
        now = datetime.utcnow()
        five_min_ago = now - timedelta(minutes=5)
        recent_decisions = []
        for decision in all_decisions:
            decision_ts = decision.get("timestamp")
            if decision_ts:
                try:
                    if isinstance(decision_ts, str):
                        decision_dt = datetime.fromisoformat(decision_ts.replace("Z", "+00:00"))
                    else:
                        decision_dt = decision_ts
                    if decision_dt >= five_min_ago:
                        recent_decisions.append(decision)
                except Exception:
                    pass

        # Analyze reasons from last 5 minutes
        reason_code_counts = {}
        for decision in recent_decisions:
            reason_code = decision.get("reason_code") or decision.get("decision") or "UNKNOWN"
            reason_code_counts[reason_code] = reason_code_counts.get(reason_code, 0) + 1

        # Top 10 reason codes from last 5m
        top_reason_codes_last_5m = [
            {"code": code, "count": count}
            for code, count in sorted(reason_code_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        # Analyze engine state
        engine_state = engine_status.get("status", "STOPPED") if facade.is_running else "STOPPED"
        killswitch_triggered = killswitch_status.get("triggered", False) or risk_status.get("killswitch_triggered", False)
        killswitch_state = {
            "triggered": killswitch_triggered,
            "reasons": killswitch_status.get("reasons", []),
            "triggered_at": killswitch_status.get("triggered_at"),
        }

        # Get last activity timestamps
        last_signal_ts = metrics.get("last_signal_ts")
        last_order_ts = metrics.get("last_order_ts")
        last_fill_ts = metrics.get("last_fill_ts")
        last_decision_ts = None
        if recent_decisions:
            last_decision = recent_decisions[-1]
            last_decision_ts = last_decision.get("timestamp")

        # Calculate last tick age
        last_tick_age_sec = None
        if facade.is_running:
            try:
                trading_system = getattr(facade, "_trading_system", None)
                if trading_system and hasattr(trading_system, "_last_tick_at"):
                    max_age = 0.0
                    for _symbol, last_tick in trading_system._last_tick_at.items():
                        age = (now - last_tick).total_seconds()
                        if age > max_age:
                            max_age = age
                    last_tick_age_sec = max_age if max_age > 0 else None
            except Exception as e:
                logger.debug(f"Could not check market data staleness: {e}")

        # Get account state for equity/pnl
        account_state = {}
        equity = None
        balance = None
        unreal_pnl = None
        real_pnl = None
        margin_used = None
        margin_free = None

        if facade.is_running:
            try:
                snapshot = await facade.get_trading_state()
                account_state = snapshot.get("account_state", {})
                equity = account_state.get("equity")
                balance = account_state.get("wallet_balance") or account_state.get("balance")
                unreal_pnl = account_state.get("unrealized_pnl", 0.0)
                real_pnl = account_state.get("realized_pnl", 0.0)
                margin_used = account_state.get("used_margin", 0.0)
                margin_free = account_state.get("available_balance") or account_state.get("free_margin")
                if margin_free is None and balance is not None and margin_used is not None:
                    margin_free = max(balance - margin_used, 0.0)
            except Exception as e:
                logger.debug(f"Could not get account state: {e}")

        # Profit capture state (will be implemented in B2)
        profit_capture_state = {
            "enabled": False,
            "armed": False,
            "triggered": False,
            "cleared": False,
            "mode": None,
            "start_equity": None,
            "peak_equity": None,
            "target_pct": None,
            "trail_pct": None,
            "after_action": None,
            "continue_risk_mult": None,
            "last_action": None,
            "last_reason": None,
        }
        if facade.is_running:
            try:
                trading_system = getattr(facade, "_trading_system", None)
                if trading_system and hasattr(trading_system, "_profit_capture"):
                    profit_capture_state = trading_system._profit_capture.get_state()
            except Exception as e:
                logger.debug(f"Could not get profit capture state: {e}")

        # Execution quality (AFO-Director / Execution Alpha v0)
        execution_quality = {
            "ws_ok": None,
            "rest_ok": None,
            "avg_latency_ms": None,
            "reject_rate_5m": None,
            "slippage_est_5m": None,
        }
        if facade.is_running:
            try:
                trading_system = getattr(facade, "_trading_system", None)
                if trading_system and hasattr(trading_system, "_execution_router"):
                    router = trading_system._execution_router
                    # Check WS connection status
                    if hasattr(router, "_bybit_ws_public"):
                        execution_quality["ws_ok"] = router._bybit_ws_public._connected if router._bybit_ws_public else None
                    if hasattr(router, "_bybit_http"):
                        execution_quality["rest_ok"] = router._bybit_http._connected if router._bybit_http else None
                    # Get execution diagnostics if available
                    if hasattr(router, "_diagnostics"):
                        diag = router._diagnostics
                        execution_quality["avg_latency_ms"] = diag.get_avg_latency_ms() if hasattr(diag, "get_avg_latency_ms") else None
                        execution_quality["reject_rate_5m"] = diag.get_reject_rate_5m() if hasattr(diag, "get_reject_rate_5m") else None
                        execution_quality["slippage_est_5m"] = diag.get_slippage_est_5m() if hasattr(diag, "get_slippage_est_5m") else None
            except Exception as e:
                logger.debug(f"Could not get execution quality: {e}")

        # Multi-symbol state (will be implemented in D2)
        multi_symbol = {
            "enabled": False,
            "symbols_count": 0,
            "last_scanned_symbol": None,
            "scan_cursor": None,
            "scan_cycle_ts": None,
        }
        if facade.is_running:
            try:
                trading_system = getattr(facade, "_trading_system", None)
                if trading_system and hasattr(trading_system, "_multi_symbol_scanner"):
                    multi_symbol = trading_system._multi_symbol_scanner.get_state()
            except Exception as e:
                logger.debug(f"Could not get multi-symbol state: {e}")
            # Fallback: check if multi-symbol is enabled via config
            from hean.config import settings
            if hasattr(settings, "multi_symbol_enabled") and settings.multi_symbol_enabled:
                multi_symbol["enabled"] = True
                multi_symbol["symbols_count"] = len(getattr(settings, "symbols", []))

        return {
            "engine_state": engine_state,
            "killswitch_state": killswitch_state,
            "last_tick_age_sec": last_tick_age_sec,
            "last_signal_ts": last_signal_ts,
            "last_decision_ts": last_decision_ts,
            "last_order_ts": last_order_ts,
            "last_fill_ts": last_fill_ts,
            "active_orders_count": metrics.get("active_orders_count", 0),
            "active_positions_count": metrics.get("active_positions_count", 0),
            "top_reason_codes_last_5m": top_reason_codes_last_5m,
            "equity": equity,
            "balance": balance,
            "unreal_pnl": unreal_pnl,
            "real_pnl": real_pnl,
            "margin_used": margin_used,
            "margin_free": margin_free,
            "profit_capture_state": profit_capture_state,
            "execution_quality": execution_quality,
            "multi_symbol": multi_symbol,
        }
    except Exception as e:
        logger.error(f"Failed to get why_not_trading: {e}", exc_info=True)
        return {
            "engine_state": "UNKNOWN",
            "killswitch_state": {"triggered": False, "reasons": []},
            "last_tick_age_sec": None,
            "last_signal_ts": None,
            "last_decision_ts": None,
            "last_order_ts": None,
            "last_fill_ts": None,
            "active_orders_count": 0,
            "active_positions_count": 0,
            "top_reason_codes_last_5m": [],
            "equity": None,
            "balance": None,
            "unreal_pnl": None,
            "real_pnl": None,
            "margin_used": None,
            "margin_free": None,
            "profit_capture_state": {"enabled": False},
            "execution_quality": {},
            "multi_symbol": {"enabled": False},
        }


@why_router.get("/equity-history")
async def get_equity_history(
    request: Request,
    limit: int = Query(default=50, description="Number of equity snapshots to return", ge=1, le=500),
) -> dict:
    """Return recent equity snapshots for performance sparkline chart."""
    facade = _get_facade(request)

    if not facade.is_running or not facade._trading_system:
        return {"snapshots": [], "count": 0}

    try:
        accounting = facade._trading_system._accounting
        history = accounting._equity_history[-limit:]
        snapshots = []
        for snap in history:
            snapshots.append({
                "timestamp": snap.timestamp.isoformat() if hasattr(snap.timestamp, "isoformat") else str(snap.timestamp),
                "equity": snap.equity,
            })
        return {"snapshots": snapshots, "count": len(snapshots)}
    except Exception as e:
        logger.error(f"Failed to get equity history: {e}", exc_info=True)
        return {"snapshots": [], "count": 0}


@why_router.get("/metrics")
async def get_trading_metrics(request: Request) -> dict:
    """Get aggregated trading funnel metrics."""
    try:
        metrics = await trading_metrics.get_metrics()
        return {
            "status": "ok",
            **metrics,
        }
    except Exception as e:
        logger.error(f"Failed to get trading metrics: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "counters": {"last_1m": {}, "last_5m": {}, "session": {}},
            "top_reasons_for_skip_block": [],
            "active_orders_count": 0,
            "active_positions_count": 0,
            "last_signal_ts": None,
            "last_order_ts": None,
            "last_fill_ts": None,
            "engine_state": "UNKNOWN",
            "mode": "unknown",
        }


@why_router.get("/state")
async def get_trading_state(request: Request) -> dict:
    """Get current trading state snapshot (positions, orders, fills, decisions)."""
    facade = _get_facade(request)

    try:
        # Get trading state from engine
        snapshot = await facade.get_trading_state() if facade.is_running else {}

        # Get recent decisions
        from hean.api import main as api_main
        async with api_main.trading_state_lock:
            recent_decisions = list(api_main.trading_state_cache.get("order_decisions", []))[-50:]
            recent_exit_decisions = list(api_main.trading_state_cache.get("order_exit_decisions", []))[-50:]

        # Get open positions
        positions = snapshot.get("positions", [])
        open_positions = [p for p in positions if p.get("status") != "closed"]

        # Get open orders
        orders = snapshot.get("orders", [])
        open_orders = [
            o for o in orders
            if o.get("status", "").upper() not in ("FILLED", "CANCELLED", "REJECTED")
        ]

        # Get recent fills (filled orders)
        recent_fills = [
            o for o in orders
            if o.get("status", "").upper() == "FILLED"
        ][-50:]

        return {
            "status": "ok",
            "open_positions": open_positions,
            "open_orders": open_orders,
            "recent_fills": recent_fills,
            "recent_decisions": recent_decisions + recent_exit_decisions,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Failed to get trading state: {e}", exc_info=True)
        return {
            "status": "error",
            "message": str(e),
            "open_positions": [],
            "open_orders": [],
            "recent_fills": [],
            "recent_decisions": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

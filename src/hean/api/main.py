"""Unified API Gateway for HEAN Trading System.

This is the bridge between Redis, C++ Core, and Next.js Frontend.
Provides WebSocket Pub/Sub with topic-based subscriptions for real-time data streaming.
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from hean.api.auth import setup_auth
from hean.api.engine_facade import EngineFacade
from hean.api.schemas import WebSocketMessage
from hean.api.services.market_data_store import market_data_store
from hean.api.services.trading_metrics import trading_metrics
from hean.api.services.websocket_service import get_websocket_service
from hean.api.telemetry import telemetry_service
from hean.config import settings
from hean.core.bus import EventBus
from hean.core.system.redis_state import get_redis_state_manager
from hean.core.types import Event, EventType
from hean.logging import get_logger
from hean.risk.killswitch import KillSwitch

logger = get_logger(__name__)

# Global instances
engine_facade: EngineFacade | None = None
bus: EventBus | None = None
redis_state_manager = None
websocket_service = None
killswitch: KillSwitch | None = None
heartbeat_task: asyncio.Task | None = None

# Cached trading state for quick WS priming and dashboard responses
trading_state_cache: dict[str, Any] = {
    "orders": [],
    "positions": [],
    "account_state": None,
    "order_decisions": [],
    "order_exit_decisions": [],
}
trading_state_lock = asyncio.Lock()
last_account_broadcast: float = 0.0

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections and topic subscriptions."""

    def __init__(self):
        self.active_connections: dict[str, WebSocket] = {}
        self.topic_subscriptions: dict[str, set[str]] = {}  # topic -> set of connection_ids
        self.connection_topics: dict[str, set[str]] = {}  # connection_id -> set of topics
        self.redis_pubsub_tasks: dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_topics[connection_id] = set()
        logger.info(f"WebSocket client connected: {connection_id}")
        await telemetry_service.record_event(
            "WS_CONNECT",
            {"connection_id": connection_id, "client": websocket.client.host if websocket.client else None},
            source="ws",
            context={"topic": "system_heartbeat"},
        )

        # Get actual state from system, but default to hardcoded values
        redis_status = "disconnected"
        try:
            if redis_state_manager:
                await redis_state_manager._client.ping()
                redis_status = "connected"
        except Exception as e:
            logger.warning(f"Redis connection check failed: {e}")
            redis_status = "disconnected"

        engine_running = engine_facade.is_running if engine_facade else False
        equity = settings.initial_capital

        # Try to get real equity if engine is running
        if engine_running and engine_facade:
            try:
                status = await engine_facade.get_status()
                equity = status.get("equity", settings.initial_capital)
            except Exception as e:
                logger.warning(f"Failed to get equity from engine: {e}")
                equity = settings.initial_capital

        # Send initial state immediately on connection
        await self.send_to_connection(connection_id, {
            "topic": "system_status",
            "data": {
                "type": "status_update",
                "engine": "running" if engine_running else "stopped",
                "redis": redis_status,
                "equity": equity,
                "timestamp": datetime.now(UTC).isoformat(),
            },
            "timestamp": datetime.now(UTC).isoformat(),
        })
        logger.info(f"✅ Sent initial state to {connection_id}: Engine={'RUNNING' if engine_running else 'STOPPED'}, Redis={redis_status.upper()}, Equity=${equity:.2f}")

    async def disconnect(self, connection_id: str) -> None:
        """Handle disconnection and cleanup."""
        if connection_id == "all":
            for cid in list(self.active_connections.keys()):
                await self.disconnect(cid)
            return

        if connection_id in self.active_connections:
            del self.active_connections[connection_id]

        # Unsubscribe from all topics
        if connection_id in self.connection_topics:
            for topic in list(self.connection_topics[connection_id]):
                self.unsubscribe(connection_id, topic)
            del self.connection_topics[connection_id]

        # Cancel Redis pubsub task if exists
        if connection_id in self.redis_pubsub_tasks:
            task = self.redis_pubsub_tasks[connection_id]
            task.cancel()
            del self.redis_pubsub_tasks[connection_id]

        logger.info(f"WebSocket client disconnected: {connection_id}")
        await telemetry_service.record_event(
            "WS_DISCONNECT",
            {"connection_id": connection_id},
            source="ws",
            context={"topic": "system_heartbeat"},
        )

    def subscribe(self, connection_id: str, topic: str) -> None:
        """Subscribe a connection to a topic."""
        if topic not in self.topic_subscriptions:
            self.topic_subscriptions[topic] = set()
        self.topic_subscriptions[topic].add(connection_id)

        if connection_id not in self.connection_topics:
            self.connection_topics[connection_id] = set()
        self.connection_topics[connection_id].add(topic)

        logger.info(f"Connection {connection_id} subscribed to topic: {topic}")

    def unsubscribe(self, connection_id: str, topic: str) -> None:
        """Unsubscribe a connection from a topic."""
        if topic in self.topic_subscriptions:
            self.topic_subscriptions[topic].discard(connection_id)
            if not self.topic_subscriptions[topic]:
                del self.topic_subscriptions[topic]

        if connection_id in self.connection_topics:
            self.connection_topics[connection_id].discard(topic)

        logger.debug(f"Connection {connection_id} unsubscribed from topic: {topic}")

    async def broadcast_to_topic(self, topic: str, data: dict[str, Any]) -> None:
        """Broadcast data to all subscribers of a topic."""
        if topic not in self.topic_subscriptions:
            return

        disconnected = []
        message = {
            "topic": topic,
            "data": data,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        for connection_id in list(self.topic_subscriptions[topic]):
            if connection_id in self.active_connections:
                try:
                    websocket = self.active_connections[connection_id]
                    # Use send_json with timeout to prevent blocking
                    await asyncio.wait_for(websocket.send_json(message), timeout=5.0)
                except TimeoutError:
                    logger.warning(f"Send timeout for {connection_id}, marking as disconnected")
                    disconnected.append(connection_id)
                except Exception as e:
                    logger.debug(f"Failed to send to {connection_id}: {e}")
                    disconnected.append(connection_id)
            else:
                disconnected.append(connection_id)

        # Clean up disconnected clients
        for connection_id in disconnected:
            # Remove from active connections if it's dead
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]
                logger.info(f"Removed dead connection {connection_id} from active_connections")
            # Unsubscribe from this topic
            self.unsubscribe(connection_id, topic)
            # Clean up all topics for this connection if it's completely gone
            if connection_id in self.connection_topics:
                for t in list(self.connection_topics[connection_id]):
                    if t != topic:  # Already unsubscribed from current topic above
                        self.unsubscribe(connection_id, t)
                del self.connection_topics[connection_id]
                logger.info(f"Cleaned up all subscriptions for dead connection {connection_id}")

    async def send_to_connection(self, connection_id: str, data: dict[str, Any]) -> None:
        """Send data to a specific connection."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                # Add timeout to prevent hanging on dead connections
                await asyncio.wait_for(websocket.send_json(data), timeout=5.0)
            except (TimeoutError, Exception) as e:
                logger.warning(f"Failed to send to {connection_id}: {e}, marking for cleanup")
                # Mark connection as dead and trigger cleanup
                await self.disconnect(connection_id)

    def active_count(self) -> int:
        """Return active WebSocket client count."""
        return len(self.active_connections)

# Global connection manager
connection_manager = ConnectionManager()
telemetry_service.set_broadcast(connection_manager.broadcast_to_topic)


async def emit_topic_event(
    *,
    topic: str,
    event_type: str,
    payload: dict[str, Any],
    source: str = "engine",
    severity: str = "INFO",
    correlation_id: str | None = None,
    context: dict[str, Any] | None = None,
    also_topics: list[str] | None = None,
) -> dict[str, Any]:
    """Create EventEnvelope with seq, store in telemetry, and broadcast to topic."""
    ctx = {"topic": topic}
    if context:
        ctx.update(context)
    envelope = await telemetry_service.record_event(
        event_type,
        payload=payload,
        severity=severity,
        source=source,
        correlation_id=correlation_id,
        context=ctx,
        publish_ws=True,
        topic=topic,
    )
    data = envelope.as_dict()
    if also_topics:
        for extra in also_topics:
            await connection_manager.broadcast_to_topic(extra, data)
    return data

async def update_trading_state_cache(
    account_state: dict[str, Any] | None = None,
    positions: list[dict[str, Any]] | None = None,
    orders: list[dict[str, Any]] | None = None,
) -> None:
    """Persist latest trading state locally and in Redis."""
    async with trading_state_lock:
        if account_state is not None:
            trading_state_cache["account_state"] = account_state
        if orders is not None:
            trading_state_cache["orders"] = orders
        if positions is not None:
            # Preserve closed positions already in cache
            existing = {
                p.get("position_id"): p for p in trading_state_cache.get("positions", [])
                if p.get("position_id")
            }
            merged: list[dict[str, Any]] = []
            for pos in positions:
                pos_id = pos.get("position_id")
                if pos_id and pos_id in existing:
                    base = existing[pos_id]
                    # Keep closed timestamps if present
                    merged.append({**base, **pos})
                    existing.pop(pos_id, None)
                else:
                    merged.append(pos)
            # Preserve previously closed positions not present in new snapshot
            for pos in existing.values():
                if pos.get("status") == "closed":
                    merged.append(pos)
            trading_state_cache["positions"] = merged

    if redis_state_manager:
        try:
            if account_state is not None:
                await redis_state_manager.set_state_atomic(
                    "account_state", account_state, namespace="state"
                )
            if orders is not None:
                await redis_state_manager.set_state_atomic(
                    "orders", orders, namespace="state"
                )
            if positions is not None:
                await redis_state_manager.set_state_atomic(
                    "positions", trading_state_cache["positions"], namespace="state"
                )
        except Exception as e:
            logger.debug(f"Failed to persist trading state to Redis: {e}")


async def broadcast_trading_state(
    account_state: dict[str, Any] | None = None,
    positions: list[dict[str, Any]] | None = None,
    orders: list[dict[str, Any]] | None = None,
    full_snapshot: bool = False,
) -> None:
    """Send trading state updates to WebSocket subscribers."""
    if account_state is not None:
        await emit_topic_event(
            topic="account_state",
            event_type="ACCOUNT_STATE",
            payload=account_state,
            source="engine",
            context={"type": "snapshot" if full_snapshot else "update"},
        )
    if positions is not None:
        await emit_topic_event(
            topic="positions",
            event_type="POSITIONS_SNAPSHOT" if full_snapshot else "POSITIONS_UPDATE",
            payload={"positions": positions, "type": "snapshot" if full_snapshot else "update"},
            source="engine",
        )
    if orders is not None and full_snapshot:
        await emit_topic_event(
            topic="orders_snapshot",
            event_type="ORDERS_SNAPSHOT",
            payload={"orders": orders, "type": "snapshot"},
            source="engine",
        )


async def sync_trading_state_from_engine(reason: str = "manual", full_snapshot: bool = True) -> None:
    """Refresh trading state from engine facade and fan out to consumers."""
    if not engine_facade:
        return
    try:
        snapshot = await engine_facade.get_trading_state()
        account_state = snapshot.get("account_state")
        positions = snapshot.get("positions") or []
        orders = snapshot.get("orders") or []
        await update_trading_state_cache(account_state, positions, orders)
        await broadcast_trading_state(account_state, positions, orders if full_snapshot else None, full_snapshot=full_snapshot)
        logger.debug(f"[TRADING_STATE_SYNC] Broadcast trading snapshot ({reason})")
    except Exception as e:
        logger.debug(f"Unable to sync trading state from engine ({reason}): {e}")


async def update_trading_state_from_snapshot(snapshot: dict[str, Any]) -> None:
    """Update cache/broadcast using a provided snapshot (e.g., from bus)."""
    account_state = snapshot.get("account_state")
    positions = snapshot.get("positions") or []
    orders = snapshot.get("orders") or []
    await update_trading_state_cache(account_state, positions, orders)
    await broadcast_trading_state(account_state, positions, None, full_snapshot=False)


async def record_order_decision(decision: dict[str, Any]) -> None:
    """Persist ORDER_DECISION telemetry for UI and diagnostics."""
    async with trading_state_lock:
        decisions = trading_state_cache.get("order_decisions", [])
        decisions.append(decision)
        trading_state_cache["order_decisions"] = decisions[-200:]

    if redis_state_manager:
        try:
            await redis_state_manager.set_state_atomic(
                "order_decisions",
                trading_state_cache["order_decisions"],
                namespace="state",
            )
        except Exception as e:
            logger.debug(f"Failed to persist order_decisions to Redis: {e}")


async def record_order_exit_decision(decision: dict[str, Any]) -> None:
    """Persist ORDER_EXIT_DECISION telemetry for UI and diagnostics."""
    async with trading_state_lock:
        decisions = trading_state_cache.get("order_exit_decisions", [])
        decisions.append(decision)
        trading_state_cache["order_exit_decisions"] = decisions[-200:]

    if redis_state_manager:
        try:
            await redis_state_manager.set_state_atomic(
                "order_exit_decisions",
                trading_state_cache["order_exit_decisions"],
                namespace="state",
            )
        except Exception as e:
            logger.debug(f"Failed to persist order_exit_decisions to Redis: {e}")


async def build_realtime_snapshot() -> dict[str, Any]:
    """Assemble trading + event snapshot for resync/reconnect."""
    async with trading_state_lock:
        snapshot = {
            "type": "snapshot",
            "account_state": trading_state_cache.get("account_state"),
            "positions": list(trading_state_cache.get("positions", [])),
            "orders": list(trading_state_cache.get("orders", [])),
            "order_decisions": list(trading_state_cache.get("order_decisions", [])),
            "order_exit_decisions": list(trading_state_cache.get("order_exit_decisions", [])),
        }
    snapshot["last_seq"] = telemetry_service.last_seq()
    snapshot["events"] = [env.as_dict() for env in telemetry_service.history(200)]
    try:
        snapshot["market"] = await market_data_store.snapshot(
            symbol=settings.trading_symbols[0] if settings.trading_symbols else None,
            timeframe="1m",
            limit=200,
        )
    except Exception as exc:
        logger.debug(f"Failed to build market snapshot: {exc}")
    return snapshot

async def safe_task_wrapper(coro, name: str):
    """
    Wrapper for background tasks to prevent silent failures.

    Catches exceptions and logs them instead of crashing silently.
    """
    try:
        await coro
    except asyncio.CancelledError:
        logger.info(f"Background task '{name}' cancelled")
        raise
    except Exception as e:
        logger.critical(
            f"Background task '{name}' crashed: {e}",
            exc_info=True,
            extra={"task_name": name, "error": str(e)},
        )
        # TODO: Emit alert to monitoring system
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global engine_facade, bus, redis_state_manager, websocket_service, killswitch, heartbeat_task

    # Initialize event bus
    logger.info("ENGINE_FACADE_INIT_START")
    bus = EventBus()
    await bus.start()
    telemetry_service.set_engine_state("STOPPED")

    # Initialize engine facade (share EventBus for WebSocket forwarding)
    try:
        engine_facade = EngineFacade(bus=bus)
        logger.info("ENGINE_FACADE_INIT_OK")
    except Exception:
        logger.error("ENGINE_FACADE_INIT_FAIL", exc_info=True)
        raise

    try:
        import hean.api.state as state
        state.engine_facade = engine_facade
        state.bind_app_state(app.state)
        app.state.engine_facade = engine_facade
        app.state.bus = bus
        app.state.market_data_store = market_data_store
        logger.info("ENGINE_FACADE_STATE_BOUND")
    except Exception as e:
        logger.error(f"Failed to bind engine facade into state: {e}", exc_info=True)
        raise

    # Initialize Redis state manager with retry logic
    redis_state_manager = None
    app.state.redis_state_manager = None
    logger.info("REDIS_CHECK_START")
    for attempt in range(3):
        try:
            redis_state_manager = await get_redis_state_manager()
            # Test connection
            await redis_state_manager._client.ping()
            app.state.redis_state_manager = redis_state_manager
            logger.info("REDIS_CHECK_OK: connected to redis:6379")
            break
        except Exception as e:
            if attempt < 2:
                logger.warning(f"REDIS_CHECK_RETRY {attempt + 1}/3 failed: {e}. Retrying...", exc_info=True)
                await asyncio.sleep(2.0)
            else:
                logger.error(f"REDIS_CHECK_FAIL after 3 attempts: {e}", exc_info=True)
                redis_state_manager = None

    # Initialize WebSocket service (Socket.io for compatibility)
    try:
        websocket_service = await get_websocket_service(bus=bus)
        logger.info("WebSocket service initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize WebSocket service: {e}")
        websocket_service = None

    # Initialize killswitch (will be connected to engine facade later)
    killswitch = KillSwitch(bus)

    # Start background tasks with safe wrappers (prevent silent failures)
    asyncio.create_task(safe_task_wrapper(forward_events_to_topics(), "forward_events"))

    # Start background task to forward Redis pub/sub to WebSocket topics
    if redis_state_manager:
        asyncio.create_task(safe_task_wrapper(forward_redis_to_topics(), "forward_redis"))

    # Start background task to periodically broadcast metrics
    asyncio.create_task(safe_task_wrapper(broadcast_metrics_periodically(), "broadcast_metrics"))

    # Start heartbeat loop (publishes to system_heartbeat topic)
    heartbeat_task = asyncio.create_task(safe_task_wrapper(heartbeat_loop(), "heartbeat"))

    # Start market ticks publisher (publishes market_ticks topic every 500ms-1s)
    asyncio.create_task(safe_task_wrapper(market_ticks_publisher_loop(), "market_ticks"))

    # Emit initial heartbeat for fast UI priming
    initial_state = "RUNNING" if (engine_facade and engine_facade.is_running) else telemetry_service.get_engine_state()
    await telemetry_service.emit_heartbeat(
        engine_state=initial_state,
        mode="LIVE" if settings.is_live and not settings.dry_run else "PAPER",
        ws_clients=connection_manager.active_count(),
        source="startup",
    )

    # AUTO-START ENGINE (for all modes including live testnet)
    if True:
        try:
            logger.info("AUTO-STARTING ENGINE...")
            result = await engine_facade.start()
            logger.info(f"ENGINE STARTED: mode={settings.trading_mode}, result={result}")

            # Wait a moment for engine to fully initialize
            await asyncio.sleep(0.5)

            # Push initial state immediately after engine starts
            if engine_facade.is_running:
                try:
                    status = await engine_facade.get_status()
                    equity = status.get("equity", settings.initial_capital)

                    # Broadcast initial balance to all WebSocket clients
                    await connection_manager.broadcast_to_topic(
                        "metrics",
                        {
                            "type": "equity",
                            "equity": equity,
                            "daily_pnl": status.get("daily_pnl", 0.0),
                            "return_pct": 0.0,
                            "open_positions": 0,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

                    # Broadcast engine status
                    await connection_manager.broadcast_to_topic(
                        "system_status",
                        {
                            "type": "engine_status",
                            "engine_running": True,
                            "trading_mode": settings.trading_mode,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

                    logger.info(f"✅ Initial state pushed: Equity=${equity:.2f}, Engine=RUNNING")
                    await sync_trading_state_from_engine(reason="startup")
                except Exception as e:
                    logger.warning(f"Failed to push initial state: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"❌ Failed to auto-start engine: {e}", exc_info=True)

    yield

    # Graceful shutdown with timeout
    logger.info("🛑 Starting graceful shutdown...")

    shutdown_tasks = []

    # 1. Cancel heartbeat task first
    if heartbeat_task:
        logger.info("Cancelling heartbeat task...")
        heartbeat_task.cancel()
        shutdown_tasks.append(heartbeat_task)

    # 2. Disconnect all WebSocket clients
    logger.info("Disconnecting WebSocket clients...")
    try:
        await asyncio.wait_for(
            connection_manager.disconnect("all"),
            timeout=5.0
        )
        logger.info("✅ WebSocket clients disconnected")
    except TimeoutError:
        logger.warning("⚠️  WebSocket disconnect timeout (5s)")

    # 3. Stop engine if running
    if engine_facade and engine_facade.is_running:
        logger.info("Stopping trading engine...")
        try:
            await asyncio.wait_for(
                engine_facade.stop(),
                timeout=10.0
            )
            logger.info("✅ Engine stopped")
        except TimeoutError:
            logger.warning("⚠️  Engine stop timeout (10s)")
        except Exception as e:
            logger.error(f"❌ Engine stop failed: {e}")

    # 4. Stop event bus
    if bus:
        logger.info("Stopping event bus...")
        try:
            await asyncio.wait_for(
                bus.stop(),
                timeout=5.0
            )
            logger.info("✅ Event bus stopped")
        except TimeoutError:
            logger.warning("⚠️  Event bus stop timeout (5s)")
        except Exception as e:
            logger.error(f"❌ Event bus stop failed: {e}")

    # 5. Disconnect Redis
    if redis_state_manager:
        logger.info("Disconnecting Redis...")
        try:
            await asyncio.wait_for(
                redis_state_manager.disconnect(),
                timeout=3.0
            )
            logger.info("✅ Redis disconnected")
        except TimeoutError:
            logger.warning("⚠️  Redis disconnect timeout (3s)")
        except Exception as e:
            logger.error(f"❌ Redis disconnect failed: {e}")

    # Wait for cancelled tasks to complete
    if shutdown_tasks:
        await asyncio.gather(*shutdown_tasks, return_exceptions=True)

    logger.info("✅ Graceful shutdown complete")

async def market_ticks_publisher_loop():
    """Publish market ticks to WebSocket clients every 500ms-1s for live chart updates."""
    global engine_facade
    default_symbol = settings.trading_symbols[0] if settings.trading_symbols else "BTCUSDT"
    last_price = None

    while True:
        try:
            await asyncio.sleep(0.5)  # Publish every 500ms for smooth chart updates

            # Get latest tick from market_data_store
            tick = await market_data_store.latest_tick(default_symbol)

            if tick and tick.get("price"):
                price = tick.get("price")
                # Only publish if price changed or it's been >1s since last publish
                if price != last_price or not last_price:
                    await emit_topic_event(
                        topic="market_ticks",
                        event_type="MARKET_TICK",
                        payload={
                            "kind": "tick",
                            "symbol": tick.get("symbol", default_symbol),
                            "price": price,
                            "bid": tick.get("bid"),
                            "ask": tick.get("ask"),
                            "volume": tick.get("volume", 0.0),
                            "ts": tick.get("ts"),
                            "ts_ms": tick.get("ts_ms"),
                            "source": "polling",
                        },
                        source="market",
                        context={"symbol": tick.get("symbol", default_symbol)},
                    )
                    last_price = price
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug(f"Market ticks publisher error: {exc}")
            await asyncio.sleep(1.0)  # Back off on error

async def heartbeat_loop():
    """Publish heartbeat to WebSocket clients once per second."""
    global engine_facade
    while True:
        try:
            mode = "LIVE" if settings.is_live and not settings.dry_run else "PAPER"
            state = telemetry_service.get_engine_state()
            if engine_facade and engine_facade.is_running:
                state = "RUNNING" if state != "PAUSED" else state
            await telemetry_service.emit_heartbeat(
                engine_state=state,
                mode=mode,
                ws_clients=connection_manager.active_count(),
                events_per_sec=telemetry_service.events_per_sec(),
                last_event_ts=telemetry_service.last_event_ts_iso(),
                source="heartbeat_loop",
            )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error(f"Heartbeat loop error: {exc}", exc_info=True)
        await asyncio.sleep(1.0)

async def forward_events_to_topics():
    """Forward events from EventBus to WebSocket topics."""
    if not bus:
        return

    async def capture_event(event: Event) -> None:
        """Record every bus event for telemetry stats."""
        try:
            await telemetry_service.record_event(
                event_type=event.event_type.value,
                payload=event.data or {},
                source="event_bus",
                context={"channel": "bus"},
                publish_ws=False,
                count_only=True,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(f"Telemetry capture failed for {event.event_type}: {exc}")

    for event_type in EventType:
        bus.subscribe(event_type, capture_event)

    async def handle_tick(event: Event) -> None:
        """Forward tick events to market_data topic and ticker_{symbol}."""
        if event.event_type != EventType.TICK:
            return

        tick_obj = event.data.get("tick")
        payload: dict[str, Any]
        symbol: str | None = None

        if tick_obj:
            symbol = getattr(tick_obj, "symbol", None)
            payload = {"kind": "tick", **(await market_data_store.record_tick(tick_obj))}
        else:
            symbol = event.data.get("symbol")
            ts_iso = event.data.get("timestamp") or datetime.now(UTC).isoformat()
            try:
                ts_ms = int(datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp() * 1000)
            except Exception:
                ts_ms = int(datetime.now(UTC).timestamp() * 1000)
            payload = {
                "kind": "tick",
                "symbol": symbol,
                "price": event.data.get("price"),
                "volume": event.data.get("volume", 0.0),
                "bid": event.data.get("bid"),
                "ask": event.data.get("ask"),
                "ts": ts_iso,
                "ts_ms": ts_ms,
            }

        if not symbol:
            return

        envelope = await emit_topic_event(
            topic="market_data",
            event_type="MARKET_TICK",
            payload=payload,
            source="market",
            context={"symbol": symbol},
        )
        await connection_manager.broadcast_to_topic(f"ticker_{symbol.lower()}", envelope)

    async def handle_candle(event: Event) -> None:
        """Handle aggregated candles from CandleAggregator."""
        if event.event_type != EventType.CANDLE:
            return
        candle = event.data.get("candle")
        timeframe = event.data.get("timeframe") or getattr(candle, "timeframe", None)
        if not candle or not timeframe:
            return
        candle_payload = {"kind": "kline", **(await market_data_store.record_candle(timeframe, candle))}
        await emit_topic_event(
            topic="market_data",
            event_type=f"KLINE_{timeframe}".upper(),
            payload=candle_payload,
            source="market",
            context={"symbol": candle.symbol, "timeframe": timeframe},
        )

    async def handle_signal(event: Event) -> None:
        """Forward signal events and emit SIGNAL_DETECTED for funnel."""
        if event.event_type != EventType.SIGNAL:
            return

        signal = event.data.get("signal")
        if signal is not None:
            payload = {
                "type": "signal",
                "symbol": signal.symbol,
                "side": signal.side,
                "strategy_id": signal.strategy_id,
                "size": signal.size,
                "entry_price": signal.entry_price,
                "metadata": signal.metadata,
            }
            # Extract signal metadata for funnel
            metadata = signal.metadata or {}
            signal_id = metadata.get("signal_id") or f"{signal.strategy_id}:{signal.symbol}:{signal.side}:{uuid.uuid4().hex[:8]}"
            confidence = metadata.get("confidence") or metadata.get("score")
            signal_type = metadata.get("signal_type") or metadata.get("type")
            features = {k: v for k, v in metadata.items() if k not in ("signal_id", "confidence", "score", "signal_type", "type")}
        else:
            payload = {
                "type": "signal",
                "symbol": event.data.get("symbol"),
                "side": event.data.get("side"),
                "strategy_id": event.data.get("strategy_id"),
                "metadata": event.data.get("metadata"),
            }
            metadata = event.data.get("metadata") or {}
            signal_id = metadata.get("signal_id") or f"{payload.get('strategy_id')}:{payload.get('symbol')}:{uuid.uuid4().hex[:8]}"
            confidence = metadata.get("confidence") or metadata.get("score")
            signal_type = metadata.get("signal_type") or metadata.get("type")
            features = {k: v for k, v in metadata.items() if k not in ("signal_id", "confidence", "score", "signal_type", "type")}

        envelope = await emit_topic_event(
            topic="signals",
            event_type="SIGNAL",
            payload=payload,
            source="engine",
            context={"symbol": payload.get("symbol")},
        )
        await connection_manager.broadcast_to_topic("strategy_events", envelope)

        # Emit SIGNAL_DETECTED for trading funnel observability
        signal_detected_payload = {
            "signal_id": signal_id,
            "symbol": payload.get("symbol"),
            "timeframe": metadata.get("timeframe") or "1m",
            "strategy_id": payload.get("strategy_id"),
            "score": confidence,
            "confidence": confidence,
            "signal_type": signal_type,
            "features": features,
        }
        await emit_topic_event(
            topic="trading_events",
            event_type="SIGNAL_DETECTED",
            payload=signal_detected_payload,
            source="engine",
            correlation_id=signal_id,
            context={
                "mode": settings.trading_mode,
                "account": settings.initial_capital,
                "symbol": payload.get("symbol"),
            },
        )

        # Record in metrics
        await trading_metrics.record_signal(
            symbol=payload.get("symbol", ""),
            strategy_id=payload.get("strategy_id", ""),
        )

        metadata = payload.get("metadata") or {}
        if metadata.get("triangular_cycle"):
            detection_time_ns = metadata.get("detection_time_ns")
            detection_time_ms = (
                int(detection_time_ns / 1_000_000)
                if isinstance(detection_time_ns, (int, float))
                else None
            )
            raw_stats = metadata.get("triangular_stats", {}) or {}
            stats_payload = {
                "total_cycles": raw_stats.get("cycles_found", 0),
                "completed_cycles": raw_stats.get("cycles_found", 0),
                "total_revenue": 0.0,
                "avg_execution_time_us": raw_stats.get("avg_latency_us", 0.0),
                "success_rate": 0.0,
            }

            cycle_payload = {
                "cycle": {
                    "id": metadata.get("cycle_id"),
                    "pair_a": metadata.get("pair_a"),
                    "pair_b": metadata.get("pair_b"),
                    "pair_c": metadata.get("pair_c"),
                    "asset_a": metadata.get("asset_a"),
                    "asset_b": metadata.get("asset_b"),
                    "asset_c": metadata.get("asset_c"),
                    "profit_bps": metadata.get("profit_bps"),
                    "profit_ratio": metadata.get("profit_ratio"),
                    "max_size": metadata.get("max_size"),
                    "detection_time": detection_time_ms,
                    "status": "detected",
                },
                "stats": stats_payload,
            }
            await emit_topic_event(
                topic="triangular_arb",
                event_type="TRIANGULAR_ARBITRAGE",
                payload=cycle_payload,
                source="engine",
                context={"symbol": payload.get("symbol")},
            )

    async def handle_order(event: Event) -> None:
        """Forward order events and refresh trading state."""
        if event.event_type not in {
            EventType.ORDER_PLACED,
            EventType.ORDER_FILLED,
            EventType.ORDER_CANCELLED,
        }:
            return

        order = event.data.get("order")
        if order is None:
            return
        metadata = getattr(order, "metadata", {}) or event.data.get("metadata") or {}
        price = (
            getattr(order, "avg_fill_price", None)
            or getattr(order, "price", None)
            or event.data.get("price")
        )
        quantity = getattr(order, "size", event.data.get("quantity"))
        try:
            leverage = max(1.0, float(metadata.get("applied_leverage", 1.0)))
        except (TypeError, ValueError):
            leverage = 1.0
        notional = (price or 0.0) * (quantity or 0.0)
        margin_used = notional / leverage if leverage else notional
        status_value = (
            order.status.value
            if hasattr(order, "status") and hasattr(order.status, "value")
            else str(getattr(order, "status", event.event_type.value))
        )

        timeline = metadata.get("status_timeline", [])
        if not timeline or timeline[-1].get("status") != status_value:
            timeline.append(
                {
                    "status": status_value,
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
        metadata["status_timeline"] = timeline

        payload = {
            "type": "order_event",
            "event": event.event_type.value,
            "order_id": getattr(order, "order_id", event.data.get("order_id")),
            "symbol": getattr(order, "symbol", event.data.get("symbol")),
            "side": getattr(order, "side", event.data.get("side")),
            "quantity": quantity,
            "price": price,
            "status": status_value,
            "created_at": getattr(order, "timestamp", datetime.now(UTC)).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "filled_qty": getattr(order, "filled_size", 0.0),
            "avg_fill_price": getattr(order, "avg_fill_price", price),
            "fee": event.data.get("fee") or metadata.get("fee"),
            "strategy_id": getattr(order, "strategy_id", event.data.get("strategy_id")),
            "signal_id": metadata.get("signal_id"),
            "rationale": metadata.get("rationale"),
            "exit_plan": metadata.get("exit_plan"),
            "expected_pnl": metadata.get("expected_pnl"),
            "status_timeline": metadata.get("status_timeline", []),
            "leverage": leverage,
            "margin_used": margin_used,
        }

        correlation = metadata.get("signal_id") or payload.get("order_id")
        envelope = await emit_topic_event(
            topic="orders",
            event_type=event.event_type.value.upper(),
            payload=payload,
            source="engine",
            correlation_id=correlation,
            context={"symbol": payload.get("symbol")},
        )
        if event.event_type == EventType.ORDER_FILLED:
            await connection_manager.broadcast_to_topic("order_filled", envelope)
        elif event.event_type == EventType.ORDER_CANCELLED:
            await connection_manager.broadcast_to_topic("order_cancelled", envelope)

        # Emit funnel events: ORDER_CREATED/ORDER_SUBMITTED, ORDER_UPDATE
        if event.event_type == EventType.ORDER_PLACED:
            order_created_payload = {
                "order_id": payload.get("order_id"),
                "symbol": payload.get("symbol"),
                "side": payload.get("side"),
                "qty": payload.get("quantity"),
                "price": payload.get("price"),
                "type": payload.get("status") or "LIMIT",
                "strategy_id": payload.get("strategy_id"),
                "signal_id": payload.get("signal_id"),
            }
            await emit_topic_event(
                topic="trading_events",
                event_type="ORDER_CREATED",
                payload=order_created_payload,
                source="engine",
                correlation_id=correlation,
                context={"symbol": payload.get("symbol")},
            )
            # Record in metrics
            await trading_metrics.record_order(
                event_type="ORDER_CREATED",
                symbol=payload.get("symbol"),
                strategy_id=payload.get("strategy_id"),
            )
        elif event.event_type in (EventType.ORDER_FILLED, EventType.ORDER_CANCELLED):
            order_update_payload = {
                "order_id": payload.get("order_id"),
                "status": status_value.upper(),
                "filled_qty": payload.get("filled_qty", 0.0),
                "avg_price": payload.get("avg_fill_price"),
                "last_fill_ts": payload.get("updated_at"),
                "reject_reason": None if status_value != "REJECTED" else payload.get("reason") or "UNKNOWN",
            }
            await emit_topic_event(
                topic="trading_events",
                event_type="ORDER_UPDATE",
                payload=order_update_payload,
                source="engine",
                correlation_id=correlation,
                context={"symbol": payload.get("symbol")},
            )
            # Record in metrics
            await trading_metrics.record_order(
                event_type=event.event_type.value.upper(),
                symbol=payload.get("symbol"),
                strategy_id=payload.get("strategy_id"),
            )

        await sync_trading_state_from_engine(reason=f"order_{event.event_type.value}")

    async def handle_pnl_update(event: Event) -> None:
        """Fan-out account/position snapshots."""
        if event.event_type != EventType.PNL_UPDATE:
            return
        snapshot = {
            "account_state": event.data.get("account_state"),
            "positions": event.data.get("positions") or [],
            "orders": event.data.get("orders") or [],
        }
        await update_trading_state_from_snapshot(snapshot)
        account_state = snapshot.get("account_state") or {}
        await emit_topic_event(
            topic="account_state",
            event_type="PNL_UPDATE",
            payload=account_state,
            source="engine",
            context={"reason": "pnl_update"},
        )
        if snapshot["positions"]:
            await emit_topic_event(
                topic="positions",
                event_type="PNL_UPDATE",
                payload={"positions": snapshot["positions"], "type": "pnl_update"},
                source="engine",
            )

        # Emit PNL_SNAPSHOT for funnel (every 2-5 seconds, throttled)
        global last_account_broadcast
        now = time.time()
        if now - last_account_broadcast >= 2.0:  # Throttle to every 2 seconds
            last_account_broadcast = now
            pnl_snapshot_payload = {
                "equity": account_state.get("equity", 0.0),
                "balance": account_state.get("wallet_balance", account_state.get("balance", 0.0)),
                "unrealized_pnl": account_state.get("unrealized_pnl", 0.0),
                "realized_pnl": account_state.get("realized_pnl", 0.0),
                "fees": account_state.get("fees", 0.0),
                "used_margin": account_state.get("used_margin", 0.0),
                "free_margin": account_state.get("available_balance", account_state.get("free_margin", 0.0)),
            }
            await emit_topic_event(
                topic="trading_events",
                event_type="PNL_SNAPSHOT",
                payload=pnl_snapshot_payload,
                source="engine",
                context={"reason": "periodic_snapshot"},
            )

            # Update metrics
            await trading_metrics.update_pnl(
                unrealized=pnl_snapshot_payload["unrealized_pnl"],
                realized=pnl_snapshot_payload["realized_pnl"],
                equity=pnl_snapshot_payload["equity"],
            )

    async def handle_position_event(event: Event) -> None:
        """Sync trading state when positions open/close/update."""
        if event.event_type not in {
            EventType.POSITION_OPENED,
            EventType.POSITION_CLOSED,
            EventType.POSITION_UPDATE,
        }:
            return
        position = event.data.get("position")
        if position is None:
            return
        payload = {
            "position_id": getattr(position, "position_id", None),
            "symbol": getattr(position, "symbol", None),
            "side": getattr(position, "side", None),
            "qty": getattr(position, "size", None),
            "entry_price": getattr(position, "entry_price", None),
            "mark_price": getattr(position, "current_price", None),
            "unrealized_pnl": getattr(position, "unrealized_pnl", None),
            "realized_pnl": getattr(position, "realized_pnl", None),
            "opened_at": getattr(position, "opened_at", None).isoformat()
            if getattr(position, "opened_at", None)
            else None,
            "strategy_id": getattr(position, "strategy_id", None),
            "exit_plan": getattr(position, "metadata", {}) or {},
            "status": "closed" if event.event_type == EventType.POSITION_CLOSED else "open",
            "closed_at": datetime.now(UTC).isoformat()
            if event.event_type == EventType.POSITION_CLOSED
            else None,
        }
        await update_trading_state_cache(positions=[payload])
        await emit_topic_event(
            topic="positions",
            event_type=event.event_type.value.upper(),
            payload={"positions": [payload], "type": event.event_type.value},
            source="engine",
            correlation_id=payload.get("position_id"),
            context={"symbol": payload.get("symbol")},
        )

        # Emit POSITION_UPDATE for funnel
        position_update_payload = {
            "symbol": payload.get("symbol"),
            "size": payload.get("qty") or getattr(position, "size", None),
            "entry_price": payload.get("entry_price"),
            "mark_price": payload.get("mark_price"),
            "unrealized_pnl": payload.get("unrealized_pnl", 0.0),
            "realized_pnl_delta": payload.get("realized_pnl", 0.0) if event.event_type == EventType.POSITION_CLOSED else None,
            "position_id": payload.get("position_id"),
        }
        await emit_topic_event(
            topic="trading_events",
            event_type="POSITION_UPDATE",
            payload=position_update_payload,
            source="engine",
            correlation_id=payload.get("position_id"),
            context={"symbol": payload.get("symbol")},
        )

        # Record in metrics
        await trading_metrics.record_position(
            event_type=event.event_type.value.upper(),
            symbol=payload.get("symbol"),
            strategy_id=payload.get("strategy_id"),
        )

        await sync_trading_state_from_engine(reason=f"position_{event.event_type.value}")

    async def handle_order_decision(event: Event) -> None:
        """Forward ORDER_DECISION telemetry to clients and Redis."""
        if event.event_type != EventType.ORDER_DECISION:
            return
        decision = event.data or {}
        await record_order_decision(decision)

        # Enhanced ORDER_DECISION for funnel
        decision_payload = {
            "decision": decision.get("decision") or decision.get("type", "UNKNOWN"),
            "signal_id": decision.get("signal_id"),
            "reason_codes": [decision.get("reason_code")] if decision.get("reason_code") else [],
            "confidence": decision.get("confidence") or decision.get("score"),
            "score": decision.get("score") or decision.get("confidence"),
            "intended_side": decision.get("side"),
            "intended_size": decision.get("computed_qty") or decision.get("size"),
            "gating": {
                "risk_ok": decision.get("reason_code") not in ("RISK_BLOCK", "DEPOSIT_PROTECTION", "PROTECTION_BLOCK"),
                "liquidity_ok": decision.get("reason_code") != "SPREAD_WIDE",
                "cooldown_ok": decision.get("reason_code") != "COOLDOWN",
                "max_orders_ok": decision.get("reason_code") != "LIMIT_REACHED",
                "paused": decision.get("reason_code") == "TRADING_DISABLED",
            },
            **decision,
        }

        envelope = await emit_topic_event(
            topic="order_decisions",
            event_type="ORDER_DECISION",
            payload={"decision": decision, "type": "order_decision"},
            source="engine",
            correlation_id=decision.get("signal_id") or decision.get("position_id"),
        )
        await connection_manager.broadcast_to_topic("strategy_events", envelope)

        # Also emit to trading_events for funnel
        await emit_topic_event(
            topic="trading_events",
            event_type="ORDER_DECISION",
            payload=decision_payload,
            source="engine",
            correlation_id=decision.get("signal_id") or decision.get("position_id"),
            context={
                "mode": settings.trading_mode,
                "symbol": decision.get("symbol"),
            },
        )

        # Record in metrics
        await trading_metrics.record_decision(
            decision=decision.get("decision") or decision.get("type", "UNKNOWN"),
            reason_code=decision.get("reason_code"),
            symbol=decision.get("symbol"),
            strategy_id=decision.get("strategy_id"),
        )

    async def handle_order_exit_decision(event: Event) -> None:
        """Forward ORDER_EXIT_DECISION telemetry to clients and Redis."""
        if event.event_type != EventType.ORDER_EXIT_DECISION:
            return
        decision = event.data or {}
        await record_order_exit_decision(decision)
        envelope = await emit_topic_event(
            topic="order_exit_decisions",
            event_type="ORDER_EXIT_DECISION",
            payload={"decision": decision, "type": "order_exit_decision"},
            source="engine",
            correlation_id=decision.get("position_id") or decision.get("order_id"),
        )
        await connection_manager.broadcast_to_topic("strategy_events", envelope)

    async def handle_ai_reasoning(event: Event) -> None:
        """Forward AI reasoning events."""
        if "reasoning" in event.data or "ai_reasoning" in event.data:
            await emit_topic_event(
                topic="ai_reasoning",
                event_type="AI_REASONING",
                payload=event.data,
                source="engine",
            )

    async def handle_killswitch(event: Event) -> None:
        """Forward killswitch events."""
        if event.event_type == EventType.KILLSWITCH_TRIGGERED:
            payload = {
                "type": "killswitch_triggered",
                "reason": event.data.get("reason"),
                "source": event.data.get("source"),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await emit_topic_event(
                topic="system_status",
                event_type="KILLSWITCH_TRIGGERED",
                payload=payload,
                source="engine",
                severity="ERROR",
            )
            await emit_topic_event(
                topic="risk_events",
                event_type="KILLSWITCH_TRIGGERED",
                payload=payload,
                source="engine",
                severity="ERROR",
            )

    async def handle_risk_event(event: Event) -> None:
        """Broadcast risk events to dedicated topic."""
        if event.event_type == EventType.RISK_BLOCKED:
            payload = {
                "type": "risk_blocked",
                "reason": event.data.get("reason"),
                "symbol": event.data.get("symbol"),
                "strategy_id": event.data.get("strategy_id"),
                "metadata": event.data.get("metadata", {}),
                "timestamp": datetime.now(UTC).isoformat(),
            }
            await emit_topic_event(
                topic="risk_events",
                event_type="RISK_BLOCKED",
                payload=payload,
                source="engine",
                severity="WARN",
            )

    async def handle_physics_update(event: Event) -> None:
        """Forward physics updates to WebSocket clients."""
        try:
            payload = event.data
            await emit_topic_event(
                topic="physics_update",
                event_type="PHYSICS_UPDATE",
                payload={
                    "temperature": payload.get("temperature", 0.0),
                    "entropy": payload.get("entropy", 0.5),
                    "phase": payload.get("phase", "WATER"),
                    "regime": payload.get("regime", "normal"),
                    "liquidations": payload.get("liquidations", []),
                    "players": payload.get("players", {
                        "marketMaker": 0,
                        "institutional": 0,
                        "arbBot": 0,
                        "retail": 0,
                        "whale": 0,
                    }),
                },
                source="physics_engine",
            )
        except Exception as exc:
            logger.debug(f"Failed to broadcast physics update: {exc}")

    async def handle_brain_analysis(event: Event) -> None:
        """Forward brain analysis to WebSocket clients."""
        try:
            payload = event.data
            await emit_topic_event(
                topic="brain_update",
                event_type="BRAIN_ANALYSIS",
                payload={
                    "stage": payload.get("stage", "analyze"),
                    "content": payload.get("content", ""),
                    "confidence": payload.get("confidence", 0.5),
                    "analysis": payload.get("analysis", {}),
                },
                source="brain",
            )
        except Exception as exc:
            logger.debug(f"Failed to broadcast brain analysis: {exc}")

    bus.subscribe(EventType.TICK, handle_tick)
    bus.subscribe(EventType.CANDLE, handle_candle)
    bus.subscribe(EventType.SIGNAL, handle_signal)
    bus.subscribe(EventType.ORDER_PLACED, handle_order)
    bus.subscribe(EventType.ORDER_FILLED, handle_order)
    bus.subscribe(EventType.ORDER_CANCELLED, handle_order)
    bus.subscribe(EventType.PNL_UPDATE, handle_pnl_update)
    bus.subscribe(EventType.POSITION_OPENED, handle_position_event)
    bus.subscribe(EventType.POSITION_CLOSED, handle_position_event)
    bus.subscribe(EventType.POSITION_UPDATE, handle_position_event)
    bus.subscribe(EventType.ORDER_DECISION, handle_order_decision)
    bus.subscribe(EventType.ORDER_EXIT_DECISION, handle_order_exit_decision)
    bus.subscribe(EventType.KILLSWITCH_TRIGGERED, handle_killswitch)
    bus.subscribe(EventType.RISK_BLOCKED, handle_risk_event)
    bus.subscribe(EventType.PHYSICS_UPDATE, handle_physics_update)
    bus.subscribe(EventType.BRAIN_ANALYSIS, handle_brain_analysis)

    # Keep running
    while True:
        await asyncio.sleep(1)

async def forward_redis_to_topics():
    """Forward Redis pub/sub messages to WebSocket topics."""
    if not redis_state_manager:
        return

    # Subscribe to Redis channels and forward to WebSocket topics
    # This allows C++ core to publish directly to Redis and have it streamed to frontend
    try:
        # In a real implementation, we'd subscribe to Redis pub/sub channels
        # For now, we'll use state updates
        pass
    except Exception as e:
        logger.error(f"Error forwarding Redis to topics: {e}")

async def broadcast_metrics_periodically():
    """Periodically broadcast engine metrics to WebSocket clients."""
    while True:
        try:
            await asyncio.sleep(1.0)  # Update every second

            # Check Redis connection status
            redis_status = "disconnected"
            if redis_state_manager:
                try:
                    await redis_state_manager._client.ping()
                    redis_status = "connected"
                except Exception as e:
                    logger.warning(f"Redis ping failed in broadcast loop: {e}")
                    redis_status = "disconnected"

            # Check engine status
            engine_running = engine_facade and engine_facade.is_running

            if engine_running:
                try:
                    status = await engine_facade.get_status()
                    risk_status = await engine_facade.get_risk_status()
                    equity = status.get("equity", settings.initial_capital)
                    daily_pnl = status.get("daily_pnl", 0.0)
                    initial_capital = status.get("initial_capital", settings.initial_capital)
                    return_pct = ((equity - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0
                    open_positions = risk_status.get("current_positions", 0)

                    # Broadcast metrics to all clients subscribed to "metrics" topic
                    await connection_manager.broadcast_to_topic(
                        "metrics",
                        {
                            "type": "metrics_update",
                            "equity": equity,
                            "daily_pnl": daily_pnl,
                            "return_pct": return_pct,
                            "open_positions": open_positions,
                            "engine_running": True,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )

                    # CRITICAL: Broadcast system_status message with exact format for UI sync
                    await connection_manager.broadcast_to_topic(
                        "system_status",
                        {
                            "type": "status_update",
                            "engine": "running",
                            "redis": redis_status,
                            "equity": equity,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                    await sync_trading_state_from_engine(reason="metrics_loop", full_snapshot=False)

                    # Update trading metrics state
                    open_orders = risk_status.get("current_orders", 0)
                    await trading_metrics.update_state(
                        engine_state=telemetry_service.get_engine_state(),
                        mode=settings.trading_mode,
                        active_orders=open_orders,
                        active_positions=open_positions,
                    )

                    # Broadcast trading_metrics every 2 seconds
                    metrics_data = await trading_metrics.get_metrics()
                    await connection_manager.broadcast_to_topic(
                        "trading_metrics",
                        {
                            "type": "trading_metrics_update",
                            **metrics_data,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error broadcasting metrics: {e}")
                    # Still send status_update even if metrics fail
                    await connection_manager.broadcast_to_topic(
                        "system_status",
                        {
                            "type": "status_update",
                            "engine": "running",
                            "redis": redis_status,
                            "equity": settings.initial_capital,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
            else:
                # Engine not running - still send status update and trading_metrics
                await connection_manager.broadcast_to_topic(
                    "system_status",
                    {
                        "type": "status_update",
                        "engine": "stopped",
                        "redis": redis_status,
                        "equity": settings.initial_capital,
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

                # Still broadcast trading_metrics even when engine is stopped
                try:
                    metrics_data = await trading_metrics.get_metrics()
                    await connection_manager.broadcast_to_topic(
                        "trading_metrics",
                        {
                            "type": "trading_metrics_update",
                            **metrics_data,
                            "timestamp": datetime.now(UTC).isoformat(),
                        }
                    )
                except Exception as e:
                    logger.debug(f"Error broadcasting trading_metrics when stopped: {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in metrics broadcast loop: {e}", exc_info=True)
            await asyncio.sleep(5.0)  # Back off on error

# Create rate limiter
limiter = Limiter(key_func=get_remote_address)

# Create the unified FastAPI app
app = FastAPI(
    title="HEAN Trading System - Unified API Gateway",
    description="Production-grade unified API gateway bridging Redis, C++ Core, and Next.js Frontend",
    version="1.0.0",
    lifespan=lifespan,
)

# Add rate limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware - SECURITY HARDENED
# Build CORS origins based on environment
ALLOWED_ORIGINS = []

if settings.environment == "development":
    # Development: allow all origins (iOS simulator, web dev servers, etc.)
    ALLOWED_ORIGINS.append("*")
elif settings.environment == "production":
    # Production: Load allowed origins from CORS_ALLOWED_ORIGINS env var
    import os as _os
    _cors_origins_env = _os.getenv("CORS_ALLOWED_ORIGINS", "")
    if _cors_origins_env:
        ALLOWED_ORIGINS.extend([
            origin.strip()
            for origin in _cors_origins_env.split(",")
            if origin.strip()
        ])
    else:
        logger.warning(
            "CORS_ALLOWED_ORIGINS not set in production mode. "
            "Set CORS_ALLOWED_ORIGINS environment variable."
        )

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=(settings.environment != "development"),  # No credentials with wildcard origins
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)

# Setup API authentication (enable via API_AUTH_ENABLED=true and API_AUTH_KEY=<key>)
setup_auth(app)

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# Include all existing routers from app.py
from hean.api.routers import (  # noqa: E402
    analytics,
    causal_inference,
    changelog,
    engine,
    graph_engine,
    market,
    meta_learning,
    multimodal_swarm,
    brain,
    physics,
    risk,
    risk_governor,
    singularity,
    storage,
    strategies,
    system,
    telemetry,
    temporal,
    trading,
)

API_PREFIX = "/api/v1"

app.include_router(engine.router, prefix=API_PREFIX)
app.include_router(trading.router, prefix=API_PREFIX)
app.include_router(trading.why_router, prefix=API_PREFIX)
app.include_router(strategies.router, prefix=API_PREFIX)
app.include_router(risk.router, prefix=API_PREFIX)
app.include_router(risk_governor.router, prefix=API_PREFIX)
app.include_router(analytics.router, prefix=API_PREFIX)
app.include_router(system.router, prefix=API_PREFIX)
app.include_router(graph_engine.router, prefix=API_PREFIX)
app.include_router(telemetry.router, prefix=API_PREFIX)
app.include_router(market.router, prefix=API_PREFIX)
app.include_router(changelog.router, prefix=API_PREFIX)
app.include_router(causal_inference.router, prefix=API_PREFIX)
app.include_router(meta_learning.router, prefix=API_PREFIX)
app.include_router(multimodal_swarm.router, prefix=API_PREFIX)
app.include_router(singularity.router, prefix=API_PREFIX)
app.include_router(physics.router, prefix=API_PREFIX)
app.include_router(temporal.router, prefix=API_PREFIX)
app.include_router(brain.router, prefix=API_PREFIX)
app.include_router(storage.router, prefix=API_PREFIX)

# WebSocket endpoint for Pub/Sub
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time Pub/Sub communication.

    Protocol:
    - Client sends: {"action": "subscribe", "topic": "ticker_btc"}
    - Client sends: {"action": "unsubscribe", "topic": "ticker_btc"}
    - Server sends: {"topic": "ticker_btc", "data": {...}, "timestamp": "..."}
    """
    connection_id = str(uuid.uuid4())
    client_host = websocket.client.host if websocket.client else "unknown"
    logger.info(f"[HEAN API] WebSocket connection attempt from {client_host}, connection_id: {connection_id}")
    await connection_manager.connect(websocket, connection_id)
    logger.info(f"[HEAN API] WebSocket connection established: {connection_id}")

    # Keepalive task to send periodic pings
    keepalive_task = None
    last_ping_time = time.time()
    PING_INTERVAL = 30.0  # Send ping every 30 seconds
    CONNECTION_TIMEOUT = 120.0  # Close connection if no activity for 2 minutes

    async def keepalive_loop():
        """Send periodic pings to keep connection alive."""
        nonlocal last_ping_time
        while connection_id in connection_manager.active_connections:
            try:
                await asyncio.sleep(PING_INTERVAL)
                if connection_id in connection_manager.active_connections:
                    try:
                        await websocket.send_json({
                            "type": "ping",
                            "timestamp": datetime.now(UTC).isoformat(),
                        })
                        last_ping_time = time.time()
                    except Exception as e:
                        logger.debug(f"Keepalive ping failed for {connection_id}: {e}")
                        break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"Keepalive loop error for {connection_id}: {e}")
                break

    keepalive_task = asyncio.create_task(keepalive_loop())

    try:
        while True:
            try:
                # Receive message from client with timeout
                # Use receive() directly to handle both text and binary frames gracefully
                ws_msg = await asyncio.wait_for(websocket.receive(), timeout=CONNECTION_TIMEOUT)
                if ws_msg.get("type") == "websocket.disconnect":
                    logger.info(f"Client {connection_id} sent disconnect frame")
                    break
                text = ws_msg.get("text") or (ws_msg.get("bytes", b"") or b"").decode("utf-8", errors="replace")
                if not text:
                    continue
                try:
                    raw_data = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    logger.debug(f"Non-JSON message from {connection_id}: {text[:100]}")
                    continue
                last_ping_time = time.time()  # Update last activity time

                # Validate incoming message with Pydantic
                try:
                    message = WebSocketMessage(**raw_data)
                    action = message.action.value
                    topic = message.topic.value if message.topic else None
                except ValidationError as e:
                    logger.warning(f"Invalid WebSocket message from {connection_id}: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "error": "Invalid message format",
                        "details": e.errors(),
                        "timestamp": datetime.now(UTC).isoformat(),
                    })
                    continue

            except TimeoutError:
                # Check if connection is still alive
                if time.time() - last_ping_time > CONNECTION_TIMEOUT:
                    logger.warning(f"WebSocket connection {connection_id} timed out (no activity for {CONNECTION_TIMEOUT}s)")
                    break
                # Send ping to check connection
                try:
                    await websocket.send_json({
                        "type": "ping",
                        "timestamp": datetime.now(UTC).isoformat(),
                    })
                    last_ping_time = time.time()
                except Exception:
                    logger.debug(f"Connection {connection_id} appears dead, closing")
                    break
                continue

            if action == "subscribe":
                if topic:
                    connection_manager.subscribe(connection_id, topic)
                    await websocket.send_json({
                        "type": "subscribed",
                        "topic": topic,
                        "connection_id": connection_id,
                    })
                    # Send immediate initial state when subscribing to system_status
                    if topic == "system_status":
                        redis_status = "disconnected"
                        if redis_state_manager:
                            try:
                                await redis_state_manager._client.ping()
                                redis_status = "connected"
                            except Exception as e:
                                logger.warning(f"Redis ping failed on system_status subscription: {e}")
                                redis_status = "disconnected"

                        engine_running = engine_facade and engine_facade.is_running
                        equity = settings.initial_capital

                        # Try to get real equity if engine is running
                        if engine_running:
                            try:
                                status = await engine_facade.get_status()
                                equity = status.get("equity", settings.initial_capital)
                            except Exception as e:
                                logger.warning(f"Failed to get equity on system_status subscription: {e}")
                                equity = settings.initial_capital

                        await connection_manager.send_to_connection(connection_id, {
                            "topic": "system_status",
                            "data": {
                                "type": "status_update",
                                "engine": "running" if engine_running else "stopped",
                                "redis": redis_status,
                                "equity": equity,
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                            "timestamp": datetime.now(UTC).isoformat(),
                        })
                    elif topic == "system_heartbeat":
                        heartbeat = telemetry_service.last_heartbeat()
                        await connection_manager.send_to_connection(
                            connection_id,
                            {
                                "topic": "system_heartbeat",
                                "data": heartbeat
                                or {
                                    "type": "HEARTBEAT",
                                    "engine_state": telemetry_service.get_engine_state(),
                                    "ws_clients": connection_manager.active_count(),
                                    "uptime_sec": telemetry_service.uptime_seconds(),
                                    "events_per_sec": telemetry_service.events_per_sec(),
                                    "timestamp": datetime.now(UTC).isoformat(),
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    elif topic in {"orders", "orders_snapshot"}:
                        cached_orders = trading_state_cache.get("orders", [])
                        await connection_manager.send_to_connection(
                            connection_id,
                            {
                                "topic": "orders_snapshot",
                                "data": {"orders": cached_orders, "type": "snapshot"},
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    elif topic == "positions":
                        cached_positions = trading_state_cache.get("positions", [])
                        await connection_manager.send_to_connection(
                            connection_id,
                            {
                                "topic": "positions",
                                "data": {"positions": cached_positions, "type": "snapshot"},
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    elif topic == "account_state":
                        cached_account = trading_state_cache.get("account_state")
                        if cached_account:
                            await connection_manager.send_to_connection(
                                connection_id,
                                {
                                    "topic": "account_state",
                                    "data": cached_account,
                                    "timestamp": datetime.now(UTC).isoformat(),
                                },
                            )
                    elif topic == "order_decisions":
                        cached_decisions = trading_state_cache.get("order_decisions", [])
                        await connection_manager.send_to_connection(
                            connection_id,
                            {
                                "topic": "order_decisions",
                                "data": {
                                    "type": "order_decisions_snapshot",
                                    "decisions": cached_decisions,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    elif topic == "trading_metrics":
                        # Send initial trading metrics snapshot
                        metrics_data = await trading_metrics.get_metrics()
                        await connection_manager.send_to_connection(
                            connection_id,
                            {
                                "topic": "trading_metrics",
                                "data": {
                                    "type": "trading_metrics_snapshot",
                                    **metrics_data,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    elif topic == "trading_events":
                        # Send recent trading events snapshot (last 50 from telemetry)
                        recent_events = [
                            env.as_dict()
                            for env in telemetry_service.history(50)
                            if env.type in (
                                "SIGNAL_DETECTED",
                                "ORDER_DECISION",
                                "ORDER_CREATED",
                                "ORDER_UPDATE",
                                "POSITION_UPDATE",
                                "PNL_SNAPSHOT",
                            )
                        ]
                        await connection_manager.send_to_connection(
                            connection_id,
                            {
                                "topic": "trading_events",
                                "data": {
                                    "type": "trading_events_snapshot",
                                    "events": recent_events,
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    elif topic == "snapshot":
                        snapshot = await build_realtime_snapshot()
                        await connection_manager.send_to_connection(
                            connection_id,
                            {
                                "topic": "snapshot",
                                "data": snapshot,
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    elif topic in {"risk_events", "strategy_events"}:
                        await connection_manager.send_to_connection(
                            connection_id,
                            {
                                "topic": topic,
                                "data": {
                                    "type": "snapshot",
                                    "items": [],
                                    "note": "No events yet",
                                    "timestamp": datetime.now(UTC).isoformat(),
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )
                    elif topic == "ai_catalyst":
                        # Send initial snapshot for AI Catalyst
                        await connection_manager.send_to_connection(
                            connection_id,
                            {
                                "topic": "ai_catalyst",
                                "data": {
                                    "type": "ai_catalyst_snapshot",
                                    "agents": [],  # Will be populated by agent registry
                                    "events": [],
                                    "note": "AI Catalyst topic subscribed",
                                },
                                "timestamp": datetime.now(UTC).isoformat(),
                            },
                        )

            elif action == "unsubscribe":
                if topic:
                    connection_manager.unsubscribe(connection_id, topic)
                    await websocket.send_json({
                        "type": "unsubscribed",
                        "topic": topic,
                    })

            elif action == "ping":
                # Heartbeat
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now(UTC).isoformat(),
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}",
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {connection_id} disconnected normally")
        if keepalive_task:
            keepalive_task.cancel()
        await connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}", exc_info=True)
        if keepalive_task:
            keepalive_task.cancel()
        await connection_manager.disconnect(connection_id)
    finally:
        if keepalive_task and not keepalive_task.done():
            keepalive_task.cancel()
            try:
                await keepalive_task
            except asyncio.CancelledError:
                pass

# Emergency Kill-Switch endpoint (high priority)
@app.post("/api/v1/emergency/killswitch", status_code=status.HTTP_200_OK)
async def trigger_killswitch(request: Request) -> dict[str, Any]:
    """Emergency kill-switch endpoint - HIGH PRIORITY.

    Immediately triggers the killswitch to halt all trading.
    Connected to C++ Emergency Kill-Switch via EventBus.
    """
    start_time = time.time()

    if not bus:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Event bus not initialized"
        )

    reason = f"Emergency killswitch triggered via API (request_id: {request.state.request_id})"

    try:
        # Trigger killswitch via event bus
        await bus.publish(
            Event(
                event_type=EventType.STOP_TRADING,
                data={"reason": reason, "source": "api_panic_button"},
            )
        )

        # Also trigger killswitch directly if available
        if killswitch:
            await killswitch._trigger(reason)

        # Broadcast to all WebSocket clients
        await connection_manager.broadcast_to_topic(
            "system_status",
            {
                "type": "killswitch_triggered",
                "reason": reason,
                "source": "panic_button",
                "timestamp": datetime.now(UTC).isoformat(),
            }
        )

        response_time_ms = (time.time() - start_time) * 1000

        logger.critical(f"EMERGENCY KILLSWITCH TRIGGERED via API: {reason} ({response_time_ms:.2f}ms)")

        return {
            "status": "success",
            "message": "Killswitch triggered",
            "reason": reason,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to trigger killswitch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger killswitch: {str(e)}"
        ) from e

# Health check endpoint
@app.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request) -> dict[str, Any]:
    """Health check endpoint. Rate limited to 60 requests per minute."""
    redis_status = "unknown"
    if redis_state_manager:
        try:
            await redis_state_manager._client.ping()
            redis_status = "connected"
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            redis_status = "disconnected"

    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "components": {
            "api": "healthy",
            "event_bus": "running" if bus else "stopped",
            "redis": redis_status,
            "engine": "running" if (engine_facade and engine_facade.is_running) else "stopped",
        },
    }


# Settings endpoint (secrets masked) - REQUIRES AUTHENTICATION
from hean.api.auth import verify_auth

@app.get("/settings")
@limiter.limit("30/minute")
async def get_settings(request: Request, authenticated: bool = Depends(verify_auth)) -> dict[str, Any]:
    """Get current settings with ALL secrets masked. Requires authentication."""

    def mask_secret(value: str | None) -> str | None:
        """Mask a secret value, showing only presence."""
        if not value:
            return None
        return "***" + value[-4:] if len(value) > 4 else "***"

    return {
        # Trading Mode
        "trading_mode": settings.trading_mode,
        "environment": settings.environment,
        "is_live": settings.is_live,

        # Exchange Settings (MASKED)
        "bybit_testnet": settings.bybit_testnet,
        "bybit_api_key": mask_secret(settings.bybit_api_key),
        "bybit_api_secret": mask_secret(settings.bybit_api_secret),

        # API Auth (MASKED)
        "api_auth_enabled": settings.api_auth_enabled,
        "api_auth_key": mask_secret(settings.api_auth_key),
        "jwt_secret": mask_secret(settings.jwt_secret),

        # LLM Keys (MASKED)
        "gemini_api_key": mask_secret(settings.gemini_api_key),

        # Capital & Risk (safe to show)
        "initial_capital": settings.initial_capital,
        "max_trade_risk_pct": settings.max_trade_risk_pct,
        "max_open_positions": settings.max_open_positions,
        "max_daily_drawdown_pct": settings.max_daily_drawdown_pct,
        "killswitch_drawdown_pct": settings.killswitch_drawdown_pct,

        # Trading Symbols
        "trading_symbols": settings.trading_symbols[:5] if len(settings.trading_symbols) > 5 else settings.trading_symbols,
        "total_symbols": len(settings.trading_symbols),

        # Observability
        "debug_mode": settings.debug_mode,
        "log_level": settings.log_level,

        # Strategy Flags
        "impulse_engine_enabled": settings.impulse_engine_enabled,
        "funding_harvester_enabled": settings.funding_harvester_enabled,
        "basis_arbitrage_enabled": settings.basis_arbitrage_enabled,
    }


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception [request_id={request_id}]: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "detail": str(exc) if settings.debug_mode else "Internal server error",
        },
    )

# Make engine_facade available to routers
@app.middleware("http")
async def inject_dependencies(request: Request, call_next):
    """Inject dependencies into request state."""
    global engine_facade, bus, redis_state_manager
    request.state.engine_facade = engine_facade
    request.state.bus = bus
    request.state.redis_state_manager = redis_state_manager
    return await call_next(request)

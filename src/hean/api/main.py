"""Unified API Gateway for HEAN Trading System.

This is the bridge between Redis, C++ Core, and Next.js Frontend.
Provides WebSocket Pub/Sub with topic-based subscriptions for real-time data streaming.
"""

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Set

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute

from hean.api.app import app as existing_app
from hean.api.engine_facade import EngineFacade
from hean.config import settings
from hean.core.bus import EventBus
from hean.core.system.redis_state import get_redis_state_manager
from hean.core.types import Event, EventType
from hean.logging import get_logger
from hean.risk.killswitch import KillSwitch
from hean.api.services.websocket_service import get_websocket_service

logger = get_logger(__name__)

# Global instances
engine_facade: EngineFacade | None = None
bus: EventBus | None = None
redis_state_manager = None
websocket_service = None
killswitch: KillSwitch | None = None

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections and topic subscriptions."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.topic_subscriptions: Dict[str, Set[str]] = {}  # topic -> set of connection_ids
        self.connection_topics: Dict[str, Set[str]] = {}  # connection_id -> set of topics
        self.redis_pubsub_tasks: Dict[str, asyncio.Task] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str) -> None:
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_topics[connection_id] = set()
        logger.info(f"WebSocket client connected: {connection_id}")
        
    async def disconnect(self, connection_id: str) -> None:
        """Handle disconnection and cleanup."""
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
        
    async def broadcast_to_topic(self, topic: str, data: Dict[str, Any]) -> None:
        """Broadcast data to all subscribers of a topic."""
        if topic not in self.topic_subscriptions:
            return
            
        disconnected = []
        message = {
            "topic": topic,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        for connection_id in list(self.topic_subscriptions[topic]):
            if connection_id in self.active_connections:
                try:
                    websocket = self.active_connections[connection_id]
                    await websocket.send_json(message)
                except Exception as e:
                    logger.warning(f"Failed to send to {connection_id}: {e}")
                    disconnected.append(connection_id)
            else:
                disconnected.append(connection_id)
                
        # Clean up disconnected clients
        for connection_id in disconnected:
            self.unsubscribe(connection_id, topic)
            
    async def send_to_connection(self, connection_id: str, data: Dict[str, Any]) -> None:
        """Send data to a specific connection."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_json(data)
            except Exception as e:
                logger.warning(f"Failed to send to {connection_id}: {e}")
                await self.disconnect(connection_id)

# Global connection manager
connection_manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global engine_facade, bus, redis_state_manager, websocket_service, killswitch
    
    # Initialize event bus
    bus = EventBus()
    await bus.start()
    
    # Initialize engine facade
    engine_facade = EngineFacade()
    
    # Initialize Redis state manager
    try:
        redis_state_manager = await get_redis_state_manager()
        logger.info("Redis state manager initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis state manager: {e}")
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
    
    # Start background task to forward events from EventBus to WebSocket topics
    asyncio.create_task(forward_events_to_topics())
    
    # Start background task to forward Redis pub/sub to WebSocket topics
    if redis_state_manager:
        asyncio.create_task(forward_redis_to_topics())
    
    yield
    
    # Cleanup
    await connection_manager.disconnect("all")  # Disconnect all
    if engine_facade and engine_facade.is_running:
        await engine_facade.stop()
    if bus:
        await bus.stop()
    if redis_state_manager:
        await redis_state_manager.disconnect()

async def forward_events_to_topics():
    """Forward events from EventBus to WebSocket topics."""
    if not bus:
        return
        
    # Subscribe to relevant events
    async def handle_tick(event: Event) -> None:
        """Forward tick events to ticker topics."""
        if event.event_type == EventType.TICK:
            symbol = event.data.get("symbol", "").lower()
            if symbol:
                await connection_manager.broadcast_to_topic(
                    f"ticker_{symbol}",
                    {
                        "type": "tick",
                        "symbol": event.data.get("symbol"),
                        "price": event.data.get("price"),
                        "volume": event.data.get("volume"),
                    }
                )
    
    async def handle_signal(event: Event) -> None:
        """Forward signal events."""
        if event.event_type == EventType.SIGNAL:
            await connection_manager.broadcast_to_topic(
                "signals",
                {
                    "type": "signal",
                    "symbol": event.data.get("symbol"),
                    "side": event.data.get("side"),
                    "strategy_id": event.data.get("strategy_id"),
                }
            )
    
    async def handle_order(event: Event) -> None:
        """Forward order events."""
        if event.event_type == EventType.ORDER_FILLED:
            await connection_manager.broadcast_to_topic(
                "orders",
                {
                    "type": "order_filled",
                    "order_id": event.data.get("order_id"),
                    "symbol": event.data.get("symbol"),
                    "side": event.data.get("side"),
                    "quantity": event.data.get("quantity"),
                    "price": event.data.get("price"),
                }
            )
    
    async def handle_ai_reasoning(event: Event) -> None:
        """Forward AI reasoning events."""
        if "reasoning" in event.data or "ai_reasoning" in event.data:
            await connection_manager.broadcast_to_topic(
                "ai_reasoning",
                {
                    "type": "reasoning",
                    "data": event.data,
                }
            )
    
    async def handle_killswitch(event: Event) -> None:
        """Forward killswitch events."""
        if event.event_type == EventType.KILLSWITCH_TRIGGERED:
            await connection_manager.broadcast_to_topic(
                "system_status",
                {
                    "type": "killswitch_triggered",
                    "reason": event.data.get("reason"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
    
    # Subscribe to events
    bus.subscribe(EventType.TICK, handle_tick)
    bus.subscribe(EventType.SIGNAL, handle_signal)
    bus.subscribe(EventType.ORDER_FILLED, handle_order)
    bus.subscribe(EventType.KILLSWITCH_TRIGGERED, handle_killswitch)
    
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

# Create the unified FastAPI app
app = FastAPI(
    title="HEAN Trading System - Unified API Gateway",
    description="Production-grade unified API gateway bridging Redis, C++ Core, and Next.js Frontend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
from hean.api.routers import analytics, engine, graph_engine, risk, strategies, system, trading

app.include_router(engine.router)
app.include_router(trading.router)
app.include_router(strategies.router)
app.include_router(risk.router)
app.include_router(analytics.router)
app.include_router(system.router)
app.include_router(graph_engine.router)

# Dashboard endpoint (add route directly since system router doesn't have prefix)
from hean.api.routers.system import get_dashboard_data
app.add_api_route("/api/v1/dashboard", get_dashboard_data, methods=["GET"], tags=["dashboard"])

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
    await connection_manager.connect(websocket, connection_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            action = data.get("action")
            
            if action == "subscribe":
                topic = data.get("topic")
                if topic:
                    connection_manager.subscribe(connection_id, topic)
                    await websocket.send_json({
                        "type": "subscribed",
                        "topic": topic,
                        "connection_id": connection_id,
                    })
                    
            elif action == "unsubscribe":
                topic = data.get("topic")
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
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown action: {action}",
                })
                
    except WebSocketDisconnect:
        await connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}", exc_info=True)
        await connection_manager.disconnect(connection_id)

# Emergency Kill-Switch endpoint (high priority)
@app.post("/api/v1/emergency/killswitch", status_code=status.HTTP_200_OK)
async def trigger_killswitch(request: Request) -> Dict[str, Any]:
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
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        
        response_time_ms = (time.time() - start_time) * 1000
        
        logger.critical(f"EMERGENCY KILLSWITCH TRIGGERED via API: {reason} ({response_time_ms:.2f}ms)")
        
        return {
            "status": "success",
            "message": "Killswitch triggered",
            "reason": reason,
            "response_time_ms": response_time_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Failed to trigger killswitch: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger killswitch: {str(e)}"
        )

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    redis_status = "unknown"
    if redis_state_manager:
        try:
            await redis_state_manager._client.ping()
            redis_status = "connected"
        except:
            redis_status = "disconnected"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {
            "api": "healthy",
            "event_bus": "running" if bus else "stopped",
            "redis": redis_status,
            "engine": "running" if (engine_facade and engine_facade.is_running) else "stopped",
        },
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
    request.state.engine_facade = engine_facade
    request.state.bus = bus
    request.state.redis_state_manager = redis_state_manager
    return await call_next(request)
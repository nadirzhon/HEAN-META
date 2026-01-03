"""FastAPI application for HEAN trading system."""

import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from hean.api.engine_facade import EngineFacade
from hean.api.reconcile import ReconcileService
from hean.api.routers import analytics, engine, risk, strategies, system, trading
from hean.api.services.event_stream import event_stream_service
from hean.api.services.log_stream import log_stream_service
from hean.config import settings
from hean.exchange.bybit.http import BybitHTTPClient
from hean.logging import get_logger
from hean.observability.metrics import metrics

logger = get_logger(__name__)

# Global instances (will be initialized in lifespan)
engine_facade: EngineFacade | None = None
reconcile_service: ReconcileService | None = None
bybit_client: BybitHTTPClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI app."""
    global engine_facade, reconcile_service, bybit_client

    # Initialize services
    engine_facade = EngineFacade()

    # Initialize Bybit client if credentials are available
    if settings.bybit_api_key and settings.bybit_api_secret:
        try:
            bybit_client = BybitHTTPClient()
            reconcile_service = ReconcileService(bybit_client)
        except Exception as e:
            logger.warning(f"Failed to initialize Bybit client: {e}")
            reconcile_service = ReconcileService()
    else:
        reconcile_service = ReconcileService()

    # Setup log stream
    log_stream_service.setup()

    yield

    # Cleanup
    await event_stream_service.stop()
    if engine_facade and engine_facade.is_running:
        await engine_facade.stop()


app = FastAPI(
    title="HEAN Trading System API",
    description="Production-grade event-driven crypto trading research system",
    version="0.1.0",
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


# Include routers
app.include_router(engine.router)
app.include_router(trading.router)
app.include_router(strategies.router)
app.include_router(risk.router)
app.include_router(analytics.router)
app.include_router(system.router)


# Make engine_facade available to routers
@app.middleware("http")
async def inject_engine_facade(request: Request, call_next):
    """Inject engine_facade into request state."""
    request.state.engine_facade = engine_facade
    request.state.reconcile_service = reconcile_service
    return await call_next(request)


# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception [request_id={request_id}]: {exc}", exc_info=True)
    metrics.increment("api_errors")

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request_id,
            "detail": str(exc) if settings.debug_mode else "Internal server error",
        },
    )

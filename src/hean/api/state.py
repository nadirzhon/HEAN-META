"""Global state for API application."""

from typing import TYPE_CHECKING, Any

from fastapi import Request

if TYPE_CHECKING:
    from hean.api.engine_facade import EngineFacade
    from hean.api.reconcile import ReconcileService
    from hean.exchange.bybit.http import BybitHTTPClient

# Global instances (will be initialized in app lifespan)
engine_facade: "EngineFacade | None" = None
reconcile_service: "ReconcileService | None" = None
bybit_client: "BybitHTTPClient | None" = None

# Remember app.state for runtime lookups by routers
app_state: Any | None = None

# Force engine_running flag - engine will auto-start on API initialization
engine_running: bool = True  # Default to True for paper trading mode


def bind_app_state(state: Any) -> None:
    """Bind FastAPI app.state for router access."""
    global app_state
    app_state = state


def get_engine_facade(request: Request | None = None):
    """Return the current engine facade using app.state as the single source of truth."""
    if request is not None:
        facade = getattr(request.app.state, "engine_facade", None)
        if facade:
            return facade

    if app_state is not None:
        facade = getattr(app_state, "engine_facade", None)
        if facade:
            return facade

    return engine_facade

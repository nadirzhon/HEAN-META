"""Health check HTTP endpoint."""

from aiohttp import web

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)


class HealthCheck:
    """Simple HTTP health check endpoint."""

    def __init__(self) -> None:
        """Initialize health check."""
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._running = False

    async def start(self) -> None:
        """Start the health check server."""
        self._app = web.Application()
        self._app.router.add_get("/health", self._health_handler)
        self._app.router.add_get("/", self._health_handler)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, "0.0.0.0", settings.health_check_port)
        await self._site.start()

        self._running = True
        logger.info(f"Health check server started on port {settings.health_check_port}")

    async def stop(self) -> None:
        """Stop the health check server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()
        self._running = False
        logger.info("Health check server stopped")

    async def _health_handler(self, request: web.Request) -> web.Response:
        """Handle health check requests."""
        return web.json_response(
            {
                "status": "healthy",
                "trading_mode": settings.trading_mode,
                "is_live": settings.is_live,
            }
        )

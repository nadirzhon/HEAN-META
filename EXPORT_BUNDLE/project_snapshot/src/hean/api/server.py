"""Server entrypoint for FastAPI application."""

import uvicorn

from hean.api.app import app
from hean.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "hean.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug_mode,
        log_level=settings.log_level.lower(),
    )


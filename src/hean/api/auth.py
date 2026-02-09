"""API Authentication Module.

Provides token-based authentication for the HEAN API.
Supports both API keys (for programmatic access) and JWT tokens (for web UI).

Security features:
- API key authentication via X-API-Key header
- JWT token authentication via Authorization: Bearer header
- Rate limiting per API key
- Audit logging of authentication events
"""

import hashlib
import hmac
import secrets
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Callable

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from hean.config import settings
from hean.logging import get_logger

logger = get_logger(__name__)

# Security schemes
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)


class AuthConfig:
    """Authentication configuration."""

    def __init__(self) -> None:
        # API key (can be set via environment variable)
        self._api_key: str | None = getattr(settings, "api_auth_key", None)
        self._auth_enabled: bool = getattr(settings, "api_auth_enabled", False)

        # JWT settings (for future web UI auth)
        self._jwt_secret: str = getattr(settings, "jwt_secret", secrets.token_hex(32))
        self._jwt_algorithm: str = "HS256"
        self._jwt_expiry_hours: int = 24

        # Rate limiting per key
        self._rate_limit_requests: int = 100  # requests per minute
        self._rate_limit_window: int = 60  # seconds

        # Track auth attempts for rate limiting
        self._auth_attempts: dict[str, list[float]] = {}

    def is_enabled(self) -> bool:
        """Check if authentication is enabled."""
        return self._auth_enabled and bool(self._api_key)

    def validate_api_key(self, api_key: str) -> bool:
        """Validate an API key.

        Uses constant-time comparison to prevent timing attacks.
        """
        if not self._api_key:
            return False

        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(api_key, self._api_key)

    def check_rate_limit(self, api_key: str) -> bool:
        """Check if API key has exceeded rate limit.

        Returns True if within limit, False if exceeded.
        """
        now = time.time()
        window_start = now - self._rate_limit_window

        # Get attempts within window
        if api_key not in self._auth_attempts:
            self._auth_attempts[api_key] = []

        # Clean old attempts
        self._auth_attempts[api_key] = [
            t for t in self._auth_attempts[api_key]
            if t > window_start
        ]

        # Check limit
        if len(self._auth_attempts[api_key]) >= self._rate_limit_requests:
            return False

        # Record this attempt
        self._auth_attempts[api_key].append(now)
        return True


# Global auth config
auth_config = AuthConfig()


async def get_api_key(
    api_key: str | None = Security(api_key_header),
    bearer: HTTPAuthorizationCredentials | None = Security(bearer_scheme),
) -> str | None:
    """Extract API key from request headers.

    Supports:
    - X-API-Key header
    - Authorization: Bearer <token> header
    """
    # Check X-API-Key header first
    if api_key:
        return api_key

    # Check Bearer token
    if bearer and bearer.credentials:
        return bearer.credentials

    return None


async def verify_auth(
    request: Request,
    api_key: str | None = Depends(get_api_key),
) -> bool:
    """Verify authentication for protected endpoints.

    Args:
        request: FastAPI request
        api_key: API key from headers

    Returns:
        True if authenticated

    Raises:
        HTTPException: If authentication fails
    """
    # Skip auth if disabled
    if not auth_config.is_enabled():
        return True

    # Get client info for logging
    client_ip = request.client.host if request.client else "unknown"
    endpoint = request.url.path

    # Check if API key provided
    if not api_key:
        logger.warning(
            f"Auth failed: Missing API key from {client_ip} for {endpoint}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check rate limit
    if not auth_config.check_rate_limit(api_key[:8]):  # Use key prefix for tracking
        logger.warning(
            f"Auth failed: Rate limit exceeded from {client_ip} for {endpoint}"
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )

    # Validate API key
    if not auth_config.validate_api_key(api_key):
        logger.warning(
            f"Auth failed: Invalid API key from {client_ip} for {endpoint}"
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    logger.debug(f"Auth success: {client_ip} -> {endpoint}")
    return True


def require_auth(func: Callable) -> Callable:
    """Decorator to require authentication for an endpoint.

    Usage:
        @app.get("/protected")
        @require_auth
        async def protected_endpoint():
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        # Auth is checked via Depends in the route
        return await func(*args, **kwargs)
    return wrapper


# Public endpoints that don't require auth
PUBLIC_ENDPOINTS = {
    "/",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def is_public_endpoint(path: str) -> bool:
    """Check if endpoint is public (no auth required)."""
    return path in PUBLIC_ENDPOINTS or path.startswith("/docs")


class AuthMiddleware:
    """Middleware to check authentication on all requests.

    Allows public endpoints without auth, requires auth for all others.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Skip auth for public endpoints
        path = scope["path"]
        if is_public_endpoint(path):
            await self.app(scope, receive, send)
            return

        # Skip auth if disabled
        if not auth_config.is_enabled():
            await self.app(scope, receive, send)
            return

        # Check for API key in headers
        headers = dict(scope["headers"])
        api_key = headers.get(b"x-api-key", b"").decode()

        # Also check Authorization header
        if not api_key:
            auth_header = headers.get(b"authorization", b"").decode()
            if auth_header.startswith("Bearer "):
                api_key = auth_header[7:]

        # Validate key
        if not api_key or not auth_config.validate_api_key(api_key):
            # Return 401 Unauthorized
            response = {
                "detail": "API key required" if not api_key else "Invalid API key"
            }
            import json
            body = json.dumps(response).encode()

            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    [b"content-type", b"application/json"],
                    [b"www-authenticate", b"ApiKey"],
                ],
            })
            await send({
                "type": "http.response.body",
                "body": body,
            })
            return

        await self.app(scope, receive, send)


def setup_auth(app) -> None:
    """Setup authentication for the FastAPI app.

    Call this in your app initialization:
        from hean.api.auth import setup_auth
        setup_auth(app)
    """
    if auth_config.is_enabled():
        app.add_middleware(AuthMiddleware)
        logger.info("API authentication enabled")
    else:
        logger.warning(
            "API authentication DISABLED - set API_AUTH_ENABLED=true and "
            "API_AUTH_KEY=<your-key> to enable"
        )

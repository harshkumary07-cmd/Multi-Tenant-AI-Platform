"""
Tenant context middleware.

Extracts and validates the X-User-Id header on every authenticated request.
Sets request.state.user_id so that route handlers and downstream middleware
can access the tenant identifier without re-parsing the header.

Exemptions:
    /health -- Docker/load-balancer probes must work without auth headers.

Failure behaviour:
    Missing or blank X-User-Id → 401 Unauthorized (JSON, not HTML).
    The 401 is returned directly from the middleware without invoking any
    route handler. ErrorHandlerMiddleware is not involved -- this is a
    transport-layer concern, not a domain-layer exception.

user_id validation:
    Phase 1: non-empty string check only.
    Phase 2: JWT signature verification replaces this middleware.
    The validation rule is a single function (_is_valid_user_id) so
    Phase 2 can replace it without changing the middleware structure.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.logging.logger import get_logger

logger = get_logger(__name__)

# Header name that carries the tenant identifier.
USER_ID_HEADER = "X-User-Id"

# Paths that do not require authentication.
# Equality check -- no wildcards. Adding a new public path is a conscious decision.
_UNAUTHENTICATED_PATHS: frozenset[str] = frozenset({
    "/health",
    "/user",   # Bootstrap: cannot require auth for the route that creates identities.
})


def _is_valid_user_id(user_id: str) -> bool:
    """
    Phase 1 user_id validation: non-empty string only.

    Phase 2 replacement: verify JWT signature, check expiry, decode claims.
    This function is the single replacement point.
    """
    return bool(user_id and user_id.strip())


class TenantContextMiddleware(BaseHTTPMiddleware):
    """
    Validates X-User-Id and stamps request.state.user_id on every request.

    Runs first in the middleware stack (registered last in main.py).
    All downstream middleware and route handlers can read request.state.user_id.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: object) -> Response:
        # Exempt unauthenticated paths
        if request.url.path in _UNAUTHENTICATED_PATHS:
            return await call_next(request)  # type: ignore[operator]

        raw_header = request.headers.get(USER_ID_HEADER, "")

        if not _is_valid_user_id(raw_header):
            logger.warning(
                "missing or blank X-User-Id header",
                extra={
                    "event": "AUTH_MISSING_HEADER",
                    "path": request.url.path,
                    "method": request.method,
                },
            )
            return JSONResponse(
                status_code=401,
                content={
                    "error_code": "UNAUTHORIZED",
                    "message": (
                        "X-User-Id header is required. "
                        "Provide a non-empty user identifier."
                    ),
                },
            )

        user_id = raw_header.strip()
        request.state.user_id = user_id

        logger.debug(
            "tenant context set",
            extra={
                "event": "TENANT_CONTEXT_SET",
                "user_id": user_id,
                "path": request.url.path,
            },
        )

        return await call_next(request)  # type: ignore[operator]

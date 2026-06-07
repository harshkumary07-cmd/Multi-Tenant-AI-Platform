"""
HTTP request and response logging middleware.

Responsibilities:
    1. Generate a unique request_id for every incoming request
    2. Bind request context (request_id, user_id placeholder) to contextvars
    3. Log request entry at DEBUG level
    4. Measure wall-clock latency for the complete request/response cycle
    5. Log request exit (method, path, status_code, latency_ms) at INFO
    6. Clean up request context in the finally block

What this middleware does NOT log:
    - Request bodies (PII / large binary uploads)
    - Response bodies (covered by query_service logging)
    - Query string parameters (may contain sensitive data)
    - Request headers (may contain auth tokens)

Health endpoint exclusion:
    GET /health is excluded. Docker/load balancer polling would produce
    thousands of meaningless log lines per day.

Middleware registration order (from main.py):
    The LAST registered middleware executes FIRST on incoming requests.
    RequestLoggerMiddleware is registered so it runs after
    TenantContextMiddleware (M9) fires. In M3, user_id is always
    "anonymous" -- full user_id logging requires M9.
"""

import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from app.logging.context import bind_request_context, clear_log_context
from app.logging.logger import get_logger
from app.logging.timing import elapsed_ms, start_timer

logger = get_logger(__name__)

_EXCLUDED_PATHS = frozenset({"/health"})


class RequestLoggerMiddleware(BaseHTTPMiddleware):
    """
    Logs every HTTP request and response as structured JSON.

    Log fields on REQUEST_END:
        event, method, path, status_code, latency_ms
        (plus request_id and user_id from LogContext)
    """

    def __init__(self, app: ASGIApp, **kwargs: Any) -> None:
        super().__init__(app, **kwargs)

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Process a single HTTP request through the logging lifecycle."""
        if request.url.path in _EXCLUDED_PATHS:
            response: Response = await call_next(request)
            return response

        request_id = f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id

        token = bind_request_context(request_id=request_id, user_id="anonymous")
        start = start_timer()
        status_code = 500

        try:
            logger.debug(
                "REQUEST_START",
                extra={
                    "event": "REQUEST_START",
                    "method": request.method,
                    "path": request.url.path,
                },
            )

            response = await call_next(request)
            status_code = response.status_code
            return response

        finally:
            latency = elapsed_ms(start)
            logger.info(
                "REQUEST_END",
                extra={
                    "event": "REQUEST_END",
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": status_code,
                    "latency_ms": latency,
                },
            )
            clear_log_context(token)

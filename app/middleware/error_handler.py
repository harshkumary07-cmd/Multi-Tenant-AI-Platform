"""
Error handler middleware.

Catches unhandled PlatformError subclasses that escape route handlers and
maps them to structured HTTP error responses.

HTTP mapping (from app/models/exceptions.py docstrings):
    ConfigurationError   → 500 Internal Server Error
    VectorStoreError     → 503 Service Unavailable
    InvalidFileTypeError → 400 Bad Request
    FileTooLargeError    → 413 Request Entity Too Large
    CorruptFileError     → 400 Bad Request
    CSVParseError        → 400 Bad Request
    EmptyDocumentError   → 400 Bad Request
    EmbeddingFailedError → 500 Internal Server Error
    LLMTimeoutError      → 504 Gateway Timeout
    LLMProviderError     → 502 Bad Gateway
    CacheError           → 503 Service Unavailable (degraded mode)
    UnauthorizedError    → 401 Unauthorized
    UserAlreadyExistsError → 409 Conflict
    UserNotFoundError    → 401 Unauthorized
    PlatformError (base) → 500 Internal Server Error (fallback)

Only PlatformError subclasses are caught here.
All other exceptions propagate to FastAPI's default handler (500).
This is intentional: AttributeError, KeyError, etc. are programming
errors and should be visible as unstructured 500s in development.

Error response envelope (consistent across all mapped exceptions):
    {
        "error_code": "EXCEPTION_CLASS_NAME_IN_SCREAMING_SNAKE_CASE",
        "message": "human-readable string (never a stack trace)"
    }

Stack traces are NEVER included in HTTP responses.
They are logged at ERROR level with the request_id for operator correlation.
"""

from __future__ import annotations

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

from app.logging.logger import get_logger
from app.models.exceptions import (
    CacheError,
    ConfigurationError,
    CorruptFileError,
    CSVParseError,
    EmbeddingFailedError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidFileTypeError,
    LLMProviderError,
    LLMTimeoutError,
    PlatformError,
    UnauthorizedError,
    UserAlreadyExistsError,
    UserNotFoundError,
    VectorStoreError,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Exception → HTTP status code mapping
# ---------------------------------------------------------------------------
_STATUS_MAP: dict[type[PlatformError], int] = {
    ConfigurationError: 500,
    VectorStoreError: 503,
    InvalidFileTypeError: 400,
    FileTooLargeError: 413,
    CorruptFileError: 400,
    CSVParseError: 400,
    EmptyDocumentError: 400,
    EmbeddingFailedError: 500,
    LLMTimeoutError: 504,
    LLMProviderError: 502,
    CacheError: 503,
    UnauthorizedError: 401,
    UserAlreadyExistsError: 409,
    UserNotFoundError: 401,
}

# Default for any PlatformError not in the map (should not occur in practice).
_DEFAULT_STATUS = 500


def _exception_to_error_code(exc: PlatformError) -> str:
    """
    Convert an exception class name to a screaming-snake-case error code.

    Example: LLMTimeoutError → "LLM_TIMEOUT_ERROR"
    """
    name = type(exc).__name__
    import re
    # Insert underscore before each uppercase letter that follows a lowercase
    code = re.sub(r"(?<=[a-z])(?=[A-Z])", "_", name)
    return code.upper()


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """
    Maps unhandled PlatformError exceptions to structured HTTP responses.

    Sits between TenantContextMiddleware (inner) and CORSMiddleware (outer)
    in the middleware stack. Catches domain exceptions that escape route
    handlers before they reach FastAPI's unstructured 500 handler.
    """

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: object) -> Response:
        try:
            return await call_next(request)  # type: ignore[operator]
        except PlatformError as exc:
            status_code = _STATUS_MAP.get(type(exc), _DEFAULT_STATUS)
            error_code = _exception_to_error_code(exc)
            message = str(exc.message) if hasattr(exc, "message") else str(exc)

            logger.error(
                "platform error mapped to http response",
                extra={
                    "event": "PLATFORM_ERROR",
                    "error_code": error_code,
                    "status_code": status_code,
                    "exception_type": type(exc).__name__,
                    "path": request.url.path,
                    "method": request.method,
                },
            )

            return JSONResponse(
                status_code=status_code,
                content={
                    "error_code": error_code,
                    "message": message,
                },
            )

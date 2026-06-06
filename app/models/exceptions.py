"""
Domain exceptions for the Multi-Tenant AI Platform.

Design principles:
    - Every exception carries enough context for a structured log entry.
    - HTTP status code mapping lives in ErrorHandlerMiddleware (Module 9),
      not here. Exceptions are domain concepts; HTTP codes are transport.
    - Exception names match the error_code strings returned in API responses.
      This makes log-to-API correlation straightforward.
    - All exceptions inherit from PlatformError so callers can catch the
      entire domain with a single except clause when needed.

Current exceptions (Module 2):
    ConfigurationError -- raised at startup when settings are invalid

Future exceptions added per module:
    Module 4: VectorStoreError
    Module 5: CorruptFileError, CSVParseError, InvalidFileTypeError,
              EmptyDocumentError, FileTooLargeError, EmbeddingFailedError
    Module 6: LLMTimeoutError, LLMProviderError, NoRelevantChunksError
    Module 8: CacheError
    Module 9: UnauthorizedError, UserAlreadyExistsError, UserNotFoundError
"""


class PlatformError(Exception):
    """
    Base class for all domain exceptions.

    All application-specific exceptions inherit from this class.
    Catching PlatformError catches every domain error without catching
    unrelated built-in exceptions.

    Attributes:
        message: Human-readable description of what went wrong.
        error_code: Machine-readable identifier matching the API error_code
                    field. Used by ErrorHandlerMiddleware for response mapping.
    """

    error_code: str = "PLATFORM_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return f"[{self.error_code}] {self.message}"


# ---------------------------------------------------------------------------
# Module 2 -- Configuration
# ---------------------------------------------------------------------------


class ConfigurationError(PlatformError):
    """
    Raised at application startup when configuration is invalid.

    This exception is raised by validate_startup_config() when the loaded
    settings are syntactically valid (pydantic accepted them) but
    semantically wrong for the current environment.

    Examples:
        - LLM_API_KEY is "changeme" in production
        - CHUNK_OVERLAP_TOKENS >= CHUNK_SIZE_TOKENS
        - RETRIEVAL_CONFIDENCE_THRESHOLD outside 0.0-1.0 range

    When raised during the lifespan startup hook, FastAPI will refuse
    to mark the application healthy and the process will exit.
    Operators see the error message directly in container logs.

    Attributes:
        message: Plain English description naming the field and fix.
        error_code: Always "CONFIGURATION_ERROR".

    Example:
        raise ConfigurationError(
            "LLM_API_KEY is set to the placeholder value 'changeme'. "
            "Set a real API key before deploying to production."
        )
    """

    error_code: str = "CONFIGURATION_ERROR"


# ---------------------------------------------------------------------------
# Module 4 -- Vector store (stub: filled in Module 4)
# ---------------------------------------------------------------------------


class VectorStoreError(PlatformError):
    """
    Raised when ChromaDB is unreachable or returns an unexpected error.

    HTTP mapping (Module 9): 503 Service Unavailable
    """

    error_code: str = "VECTOR_STORE_UNAVAILABLE"


# ---------------------------------------------------------------------------
# Module 5 -- Document ingestion (stubs: filled in Module 5)
# ---------------------------------------------------------------------------


class InvalidFileTypeError(PlatformError):
    """
    Raised when an uploaded file's type is not PDF or CSV.

    HTTP mapping (Module 9): 400 Bad Request
    """

    error_code: str = "INVALID_FILE_TYPE"


class FileTooLargeError(PlatformError):
    """
    Raised when an uploaded file exceeds MAX_UPLOAD_SIZE_MB.

    HTTP mapping (Module 9): 413 Request Entity Too Large
    """

    error_code: str = "FILE_TOO_LARGE"


class CorruptFileError(PlatformError):
    """
    Raised when a PDF cannot be parsed (invalid or corrupt bytes).

    HTTP mapping (Module 9): 400 Bad Request
    """

    error_code: str = "CORRUPT_FILE"


class CSVParseError(PlatformError):
    """
    Raised when a CSV file cannot be parsed (bad encoding, bad delimiters).

    HTTP mapping (Module 9): 400 Bad Request
    """

    error_code: str = "CSV_PARSE_ERROR"


class EmptyDocumentError(PlatformError):
    """
    Raised when a file produces zero usable text after parsing and cleaning.

    HTTP mapping (Module 9): 400 Bad Request
    """

    error_code: str = "EMPTY_DOCUMENT"


class EmbeddingFailedError(PlatformError):
    """
    Raised when the sentence-transformers model raises an error during encoding.

    HTTP mapping (Module 9): 500 Internal Server Error
    """

    error_code: str = "EMBEDDING_FAILED"


# ---------------------------------------------------------------------------
# Module 6 -- RAG / LLM (stubs: filled in Module 6)
# ---------------------------------------------------------------------------


class LLMTimeoutError(PlatformError):
    """
    Raised when the LLM provider does not respond within LLM_TIMEOUT_SECONDS.

    HTTP mapping (Module 9): 504 Gateway Timeout
    """

    error_code: str = "LLM_TIMEOUT"


class LLMProviderError(PlatformError):
    """
    Raised when the LLM API returns an error response (not a timeout).

    HTTP mapping (Module 9): 502 Bad Gateway
    """

    error_code: str = "LLM_PROVIDER_ERROR"


class NoRelevantChunksError(PlatformError):
    """
    Raised internally when retrieval returns zero chunks above the
    confidence threshold.

    Note: This does NOT map to an HTTP error. The query_service catches
    this and returns a structured 200 OK no-result response instead.
    It is a domain signal, not an HTTP error.
    """

    error_code: str = "NO_RELEVANT_CHUNKS"


# ---------------------------------------------------------------------------
# Module 8 -- Cache (stub: filled in Module 8)
# ---------------------------------------------------------------------------


class CacheError(PlatformError):
    """
    Raised when Redis returns an unexpected error (not a connection failure).

    Connection failures are handled silently as cache misses in cache_service.
    CacheError is for unexpected protocol or data errors.

    HTTP mapping (Module 9): degraded mode, not a user-facing error.
    """

    error_code: str = "CACHE_ERROR"


# ---------------------------------------------------------------------------
# Module 9 -- API / Auth (stubs: filled in Module 9)
# ---------------------------------------------------------------------------


class UnauthorizedError(PlatformError):
    """
    Raised when a request is missing or has an invalid X-User-Id header.

    HTTP mapping (Module 9): 401 Unauthorized
    """

    error_code: str = "UNAUTHORIZED"


class UserAlreadyExistsError(PlatformError):
    """
    Raised when POST /user is called with a user_id that already exists.

    HTTP mapping (Module 9): 409 Conflict
    """

    error_code: str = "USER_ALREADY_EXISTS"


class UserNotFoundError(PlatformError):
    """
    Raised when a user_id from the auth header does not exist in the registry.

    HTTP mapping (Module 9): 401 Unauthorized (not 404 -- never confirm
    whether a user_id exists to unauthenticated callers).
    """

    error_code: str = "USER_NOT_FOUND"

"""
Structured logger factory -- Module 3 full implementation.

Replaces the Module 1 scaffold (plain-text StreamHandler).
The get_logger() function signature is identical -- no call sites change.

What changed from M1:
    M1: standard Formatter with human-readable output
    M3: JSONFormatter with contextvars-enriched structured output

What stayed the same:
    - get_logger(__name__) call pattern
    - Logger name equals the module's __name__
    - Logs go to stdout (container runtime collects stdout)
    - No file I/O

Architecture:
    A single root logger is configured once at module import time via
    _configure_root_logger(). All module-level loggers obtained via
    get_logger(__name__) are children of the root logger and inherit:
        - The JSONFormatter (via the root handler)
        - The log level (from settings.LOG_LEVEL)
        - The stdout StreamHandler

    configure_logging() is the public function called from main.py's
    lifespan hook. It reconfigures the root logger with the correct
    level from the loaded Settings, then emits the startup log line.

    _configure_root_logger() sets a safe default (INFO) at import time
    so that any logging that happens before the lifespan runs (e.g.
    pydantic validation errors) is still structured JSON.

Health endpoint suppression:
    RequestLoggerMiddleware (M9) skips logging for GET /health.
    This is enforced at the middleware layer, not here.
    The logger itself has no knowledge of HTTP paths.
"""

import logging
import sys
from typing import Any

from app.logging.formatters import JSONFormatter


def _configure_root_logger(level: int = logging.INFO) -> None:
    """
    Configure the root logger with the JSON formatter and a stdout handler.

    Idempotent: calling multiple times with the same level has no effect
    beyond the first call. Calling with a new level updates the level.

    Called at module import time (INFO default) and again from
    configure_logging() once Settings are loaded (uses settings.LOG_LEVEL).

    Args:
        level: Python logging level integer (e.g. logging.INFO = 20).
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to avoid duplicate output
    # (e.g. if uvicorn or pytest has already added handlers)
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(JSONFormatter())
    root.addHandler(handler)


# Configure at import time with a safe default.
# configure_logging() will update the level from Settings.
_configure_root_logger(logging.INFO)


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a structured logger for the given module name.

    This is the single function all application code uses to obtain a logger.
    The returned logger automatically includes request_id and user_id in
    every log line via the JSONFormatter's contextvars integration.

    Args:
        name: Module name, always passed as __name__ at call sites.
              If None, returns the root logger.

    Returns:
        logging.Logger: A named logger inheriting the JSON formatter
                        and log level from the root configuration.

    Usage:
        from app.logging.logger import get_logger

        logger = get_logger(__name__)

        # Basic log
        logger.info("document ingested")

        # With structured extra fields (appear as top-level JSON keys)
        logger.info("upload complete", extra={
            "doc_id": "doc_9c4d1e2f",
            "chunks_stored": 24,
            "latency_ms": 8300,
        })

        # Exception logging (stack trace included automatically)
        try:
            risky_operation()
        except ValueError as exc:
            logger.error("operation failed", exc_info=True, extra={
                "doc_id": doc_id,
                "stage": "embedding",
            })
    """
    logger = logging.getLogger(name)
    # Child loggers propagate to root -- do not add duplicate handlers
    logger.propagate = True
    return logger


def configure_logging(log_level: str) -> None:
    """
    Apply the settings-derived log level to the root logger.

    Called from main.py's lifespan startup hook after Settings are loaded
    and validated. Updates the level that was defaulted to INFO at import.

    Args:
        log_level: Level name string from settings.LOG_LEVEL.
                   One of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

    Example (from main.py lifespan):
        from app.logging.logger import configure_logging
        configure_logging(settings.LOG_LEVEL)
    """
    numeric_level = logging.getLevelName(log_level)
    if not isinstance(numeric_level, int):
        # Unknown level string -- fall back to INFO and warn
        numeric_level = logging.INFO
        logging.getLogger(__name__).warning(
            "unknown log level",
            extra={"configured_value": log_level, "fallback": "INFO"},
        )
    _configure_root_logger(numeric_level)


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    **fields: Any,
) -> None:
    """
    Emit a structured log line with arbitrary key-value fields.

    Convenience wrapper for callers who prefer keyword arguments over
    the extra={} dict syntax. Both styles produce identical JSON output.

    Args:
        logger:  Logger instance obtained via get_logger(__name__).
        level:   Python logging level integer (e.g. logging.INFO).
        message: Log message string.
        **fields: Arbitrary key-value pairs added as JSON fields.

    Usage:
        log_with_context(
            logger, logging.INFO, "query complete",
            route="RETRIEVE",
            latency_ms=1240,
            cache_hit=False,
            chunks_retrieved=4,
        )
    """
    logger.log(level, message, extra=fields)

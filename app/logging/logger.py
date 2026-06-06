"""
Structured JSON logger factory.

Scaffold in Module 1. Fully implemented in Module 3.

Current state (M1):
    Returns a standard Python logging.Logger writing human-readable
    lines to stdout. Suitable for development.

Full implementation (M3) adds:
    - JSON formatter (every log line is a parseable JSON object)
    - contextvars integration (request_id + user_id on every line)
    - LOG_LEVEL read from settings
    - Health endpoint log suppression

Usage (identical in M1 and M3 -- no call sites change):
    from app.logging.logger import get_logger

    logger = get_logger(__name__)
    logger.info("upload complete", extra={"doc_id": doc_id, "chunks": 24})

Design:
    get_logger() accepts __name__ so log lines identify their source module.
    This mirrors Python's standard logging convention and is compatible
    with all log aggregation tools without additional configuration.
"""

import logging
import sys


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Args:
        name: Module name, typically passed as __name__.
              If None, returns the root logger.

    Returns:
        logging.Logger: Configured logger instance.

    Note:
        M1 format: human-readable lines for development.
        M3 format: structured JSON with request_id + user_id fields.
        The function signature is identical in both versions so no
        call sites need updating when M3 replaces the formatter.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if get_logger() is called multiple times
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    # M1: human-readable format for development.
    # M3 replaces this with a JSON formatter.
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    return logger

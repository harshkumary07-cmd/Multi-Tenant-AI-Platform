"""
Structured JSON logging package -- fully implemented in Module 3.

Public API (import from here, not from submodules):
    get_logger(name)         -- obtain a structured logger for a module
    configure_logging(level) -- apply settings-derived level at startup
    log_with_context(...)    -- convenience wrapper for keyword-style logging

Request context (used by middleware and tests):
    bind_request_context(request_id, user_id) -> Token
    update_user_id(user_id) -> Token
    clear_log_context(token) -> None
    get_log_context() -> LogContext

Timing utilities:
    start_timer() -> float
    elapsed_ms(start) -> int
    LatencyTracker

All application code should call:
    from app.logging.logger import get_logger
    logger = get_logger(__name__)
"""

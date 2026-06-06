"""
Cross-cutting HTTP concerns that execute on every request.

Middleware modules (added in M3 and M9):
    tenant_context.py   -- validates X-User-Id, injects request.state.user_id
    request_logger.py   -- stamps request_id, logs entry/exit with latency
    error_handler.py    -- maps domain exceptions to structured HTTP responses

TenantContextMiddleware is always registered last (executes first).
"""

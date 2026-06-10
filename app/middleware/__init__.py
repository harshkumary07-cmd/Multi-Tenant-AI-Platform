"""
Cross-cutting HTTP concerns that execute on every request.

Middleware modules (registered in main.py, executed in LIFO order):
    tenant_context.py   -- validates X-User-Id, injects request.state.user_id (M9)
    request_logger.py   -- stamps request_id, logs entry/exit with latency (M3)
    error_handler.py    -- maps PlatformError subclasses to structured HTTP responses (M9)

Execution order on each incoming request:
    1. TenantContextMiddleware  (registered last -> executes first)
    2. RequestLoggerMiddleware
    3. ErrorHandlerMiddleware
    4. CORSMiddleware           (registered first -> executes last)

TenantContextMiddleware always runs first so request.state.user_id
is available to all downstream middleware and route handlers.
"""

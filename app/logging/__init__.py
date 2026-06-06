"""
Structured JSON logger factory (scaffold in M1, fully implemented in M3).

All application code calls get_logger(__name__) from this package.
request_id and user_id propagated via Python contextvars.
"""

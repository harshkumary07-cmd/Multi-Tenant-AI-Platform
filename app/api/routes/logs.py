"""
Observability / logs route.

GET /logs -- retrieve aggregated operational metrics for the authenticated user.

STATUS: STUB -- returns 501 Not Implemented.
Logging infrastructure implemented in Module 3.
Route fully wired in Module 9 (API Layer).

Planned behaviour (approved architecture):
    Request:  ?user_id=u1&window=1h + X-User-Id header
    Response: aggregated metrics (NOT raw log lines):
        - request_metrics:  total_requests, avg_latency_ms, p95_latency_ms
        - cache_statistics: hit_rate_pct, cache_hits, cache_misses
        - route_decisions:  direct_count, retrieve_count
        - recent_events:    last 20 requests (sanitised, no stack traces)
        - error_summary:    error counts grouped by error_code
        - documents:        total_uploaded, total_chunks_stored
    Codes: 200 OK | 401 Unauthorized | 403 Forbidden

Design decisions (locked):
    - Returns AGGREGATED METRICS, not raw internal log lines
    - Stack traces never exposed through this endpoint
    - User can only retrieve their own logs
    - Requesting another user's logs returns 403 (not 404)
      -- existence of other users must not be confirmed or denied
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get(
    "",
    summary="Retrieve operational metrics",
    description="Returns aggregated metrics for the authenticated user. Not raw log lines.",
    status_code=501,
)
async def get_logs() -> JSONResponse:
    """
    Retrieve aggregated operational metrics.

    Not yet implemented. Returns 501 until Module 9.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error_code": "NOT_IMPLEMENTED",
            "message": "GET /logs will be implemented in Module 9.",
        },
    )

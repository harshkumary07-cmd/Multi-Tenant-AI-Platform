"""
Observability / logs route.

GET /logs -- retrieve aggregated operational metrics for the authenticated user.

Returns AGGREGATED METRICS, not raw log lines. Stack traces are never exposed.
Users can only retrieve their own metrics (enforced by TenantContextMiddleware).

Phase 1 (current):
    Returns structurally correct response with zero-value counters.
    No metrics persistence store exists yet.

Phase 2 (future):
    Populates counters from a metrics store (e.g. Prometheus, ClickHouse,
    or a dedicated metrics table).

HTTP codes:
    200 OK               -- metrics returned (may all be zero in Phase 1)
    401 Unauthorized     -- missing/blank X-User-Id header
    403 Forbidden        -- requesting another user's logs (Phase 2)
"""

from fastapi import APIRouter, Depends

from app.config.dependencies import get_current_user_id
from app.logging.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "",
    status_code=200,
    summary="Retrieve operational metrics",
    description=(
        "Returns aggregated metrics for the authenticated user. "
        "Not raw log lines. Phase 1: counters are zero (no persistence store). "
        "Phase 2: populated from metrics backend."
    ),
)
async def get_logs(
    user_id: str = Depends(get_current_user_id),
) -> dict:  # type: ignore[type-arg]
    """
    Retrieve aggregated operational metrics for the current user.

    Phase 1 returns structurally correct metrics with zero counters.
    """
    logger.info(
        "logs request",
        extra={
            "event": "LOGS_REQUEST",
            "user_id": user_id,
        },
    )

    return {
        "user_id": user_id,
        "note": (
            "Phase 1: metrics persistence not yet implemented. "
            "Counters reflect zero -- no historical data is stored. "
            "Phase 2 will populate from a metrics backend."
        ),
        "request_metrics": {
            "total_requests": 0,
            "avg_latency_ms": 0,
            "p95_latency_ms": 0,
        },
        "cache_statistics": {
            "hit_rate_pct": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        },
        "route_decisions": {
            "direct_count": 0,
            "retrieve_count": 0,
        },
        "documents": {
            "total_uploaded": 0,
            "total_chunks_stored": 0,
        },
        "recent_events": [],
        "error_summary": {},
    }

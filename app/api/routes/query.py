"""
Query route.

POST /query -- submit a natural language question for AI-powered answering.

The full pipeline is: cache lookup → router decision → (RETRIEVE or DIRECT) → response.
User identity always comes from request.state.user_id (set by TenantContextMiddleware).

HTTP codes:
    200 OK                   -- answer produced or no-result response
    401 Unauthorized         -- missing/blank X-User-Id header
    422 Unprocessable Entity -- request body fails validation
    502 Bad Gateway          -- LLM provider returned an error
    503 Service Unavailable  -- ChromaDB unavailable
    504 Gateway Timeout      -- LLM provider timed out
"""


from fastapi import APIRouter, Depends, Request

from app.config.dependencies import get_cached_query_service, get_current_user_id
from app.logging.logger import get_logger
from app.schemas.query_request import QueryRequest
from app.schemas.query_response import (
    QueryResponse,
    SourceReferenceSchema,
    TokenUsageSchema,
)
from app.services.cached_query_service import CachedQueryService

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=QueryResponse,
    status_code=200,
    summary="Submit a natural language query",
    description=(
        "Routes to DIRECT (general knowledge) or RETRIEVE (document search) "
        "based on query signals and document presence. "
        "Cache is checked before any pipeline work begins."
    ),
)
async def submit_query(
    request: Request,
    body: QueryRequest,
    user_id: str = Depends(get_current_user_id),
    service: CachedQueryService = Depends(get_cached_query_service),  # noqa: B008
) -> QueryResponse:
    """
    Submit a natural language query for AI-powered answering.

    Returns 200 OK for both answered queries and no-result responses.
    A no-result response has answer=null and no_result_reason set.
    """
    logger.info(
        "query request received",
        extra={
            "event": "QUERY_REQUEST",
            "user_id": user_id,
            "query_length": len(body.query),
            "top_k_override": body.top_k,
        },
    )

    result = service.query(
        user_id=user_id,
        query_text=body.query,
        top_k=body.top_k,
    )

    return QueryResponse(
        query=result.query,
        answer=result.answer,
        sources=[
            SourceReferenceSchema(
                doc_id=s.doc_id,
                source=s.source,
                chunk_count=s.chunk_count,
                top_score=s.top_score,
            )
            for s in result.sources
        ],
        route=result.route,
        chunks_retrieved=result.chunks_retrieved,
        chunks_used=result.chunks_used,
        token_usage=TokenUsageSchema(
            prompt_tokens=result.token_usage.prompt_tokens,
            completion_tokens=result.token_usage.completion_tokens,
            total_tokens=result.token_usage.total_tokens,
        ),
        latency_ms=result.latency_ms,
        no_result_reason=result.no_result_reason,
        cache_hit=result.cache_hit,
        timestamp=result.timestamp,
    )

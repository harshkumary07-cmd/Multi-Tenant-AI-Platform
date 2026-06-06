"""
Query route.

POST /query -- submit a natural language question for AI-powered answering.

STATUS: STUB -- returns 501 Not Implemented.
RAG engine implemented in Module 6.
Router Agent implemented in Module 7.
Cache layer implemented in Module 8.
Route fully wired in Module 9 (API Layer).

Planned behaviour (approved architecture):
    Request:  {"query": "...", "options": {...}} + X-User-Id header
    Response: {
        "answer": "...",
        "sources": [...],
        "cache_hit": bool,
        "route": "DIRECT|RETRIEVE",
        "latency_ms": N,
        "token_usage": {...}
    }
    Codes: 200 OK | 401 Unauthorized | 404 No Documents Found
           422 Validation Error | 504 LLM Timeout | 503 Service Degraded

Design decisions (locked):
    - Router Agent decides DIRECT vs RETRIEVE for each query
    - Cache checked before any pipeline work begins
    - RETRIEVE: embed -> ChromaDB search -> context assembly -> LLM
    - DIRECT: LLM with no document context
    - No fallback from RETRIEVE to DIRECT on low confidence or retrieval failure
    - No-result response (200 OK) when confidence threshold not met
    - user_id always from request.state (set by TenantContextMiddleware), never body
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post(
    "",
    summary="Submit a natural language query",
    description="Routes to DIRECT or RETRIEVE based on query signals and document presence.",
    status_code=501,
)
async def submit_query() -> JSONResponse:
    """
    Submit a natural language query for AI-powered answering.

    Not yet implemented. Returns 501 until Module 9.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error_code": "NOT_IMPLEMENTED",
            "message": "POST /query will be implemented in Module 9.",
        },
    )

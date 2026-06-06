"""
Document upload route.

POST /upload-doc -- upload a PDF or CSV document for ingestion.

STATUS: STUB -- returns 501 Not Implemented.
Ingestion pipeline implemented in Module 5.
Route fully wired in Module 9 (API Layer).

Planned behaviour (approved architecture):
    Request:  multipart/form-data with file binary + X-User-Id header
    Response: {"document_id": "...", "chunks_stored": N, "status": "complete"}
    Codes:    201 Created | 400 Bad Request | 401 Unauthorized
              413 Too Large | 500 Internal Error | 503 Unavailable | 504 Timeout

Design decisions (locked):
    - Synchronous -- full pipeline completes before 201 response is returned
    - 201 Created (not 202 Accepted) -- upload is done when response arrives
    - Maximum file size: MAX_UPLOAD_SIZE_MB (default 50MB), checked before read
    - Accepted types: PDF and CSV only
    - Cache invalidation runs before response is returned
    - Partial ChromaDB writes cleaned up on any pipeline failure
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post(
    "",
    summary="Upload a document for ingestion",
    description="Accepts PDF or CSV. Synchronous: returns 201 when fully stored in ChromaDB.",
    status_code=501,
)
async def upload_document() -> JSONResponse:
    """
    Upload and ingest a document.

    Not yet implemented. Returns 501 until Module 9.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error_code": "NOT_IMPLEMENTED",
            "message": "POST /upload-doc will be implemented in Module 9.",
        },
    )

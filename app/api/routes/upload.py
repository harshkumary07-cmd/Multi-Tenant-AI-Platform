"""
Document upload route.

POST /upload-doc -- upload a PDF or CSV document for ingestion.

Synchronous: the full pipeline (parse → chunk → embed → store → cache invalidate)
completes before 201 is returned. A 201 means the document is immediately queryable.

HTTP codes:
    201 Created              -- document successfully ingested
    400 Bad Request          -- invalid file type, corrupt file, or empty document
    401 Unauthorized         -- missing/blank X-User-Id header
    413 Request Entity Too Large -- file exceeds MAX_UPLOAD_SIZE_MB
    422 Unprocessable Entity -- multipart form missing required fields
    500 Internal Server Error -- embedding failure
    503 Service Unavailable  -- ChromaDB unavailable
"""

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile

from app.config.dependencies import get_current_user_id, get_document_service
from app.logging.logger import get_logger
from app.schemas.upload_response import UploadResponse
from app.services.document_service import DocumentService

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "",
    response_model=UploadResponse,
    status_code=201,
    summary="Upload a document for ingestion",
    description=(
        "Accepts PDF or CSV. Synchronous: returns 201 only after all chunks "
        "are confirmed stored in ChromaDB. Cache is invalidated automatically."
    ),
)
async def upload_document(
    request: Request,
    file: UploadFile = File(..., description="PDF or CSV file to ingest."),  # noqa: B008
    file_type: str = Form(..., description="Declared file type: 'pdf' or 'csv'."),
    user_id: str = Depends(get_current_user_id),
    service: DocumentService = Depends(get_document_service),  # noqa: B008
) -> UploadResponse:
    """
    Upload and ingest a document into the vector store.

    Returns 201 Created with document metadata when ingestion completes.
    """
    file_bytes = await file.read()
    filename = file.filename or "upload"

    logger.info(
        "upload request received",
        extra={
            "event": "UPLOAD_REQUEST",
            "user_id": user_id,
            "source_file": filename,
            "declared_type": file_type,
            "size_bytes": len(file_bytes),
        },
    )

    record = service.ingest(
        user_id=user_id,
        file_bytes=file_bytes,
        filename=filename,
        declared_type=file_type,
    )

    return UploadResponse(
        document_id=record.doc_id,
        user_id=record.user_id,
        filename=record.filename,
        file_type=record.file_type,
        chunks_stored=record.chunks_stored,
        status="complete",
        upload_timestamp=record.uploaded_at,
    )

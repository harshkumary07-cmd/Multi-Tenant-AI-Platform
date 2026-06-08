"""
Upload response schema.

Pydantic model for the 201 Created response from POST /upload-doc.
Built from a DocumentRecord domain model in the route handler (Module 9).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """
    Response body for a successful document upload.

    Returned with HTTP 201 Created after the full ingestion pipeline
    (parse → chunk → embed → store) completes successfully.

    The response is only sent after ChromaDB confirms all chunks are
    written. A 201 response means the document is immediately queryable.

    Attributes:
        document_id:   Unique document identifier assigned by the server.
        user_id:       Tenant identifier.
        filename:      Original filename as uploaded.
        file_type:     "pdf" or "csv".
        chunks_stored: Number of chunks written to ChromaDB.
        status:        Always "complete" in a 201 response.
        upload_timestamp: UTC timestamp of pipeline completion.
    """

    document_id: str = Field(description="Unique document identifier.")
    user_id: str = Field(description="Tenant identifier.")
    filename: str = Field(description="Original filename.")
    file_type: Literal["pdf", "csv"] = Field(description="File type.")
    chunks_stored: int = Field(description="Number of chunks stored in ChromaDB.")
    status: Literal["complete"] = Field(
        default="complete",
        description="Always 'complete' for a 201 response.",
    )
    upload_timestamp: datetime = Field(description="UTC timestamp of completion.")

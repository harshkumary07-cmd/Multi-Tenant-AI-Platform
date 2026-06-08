"""
Document domain model.

DocumentRecord is the internal representation of a successfully ingested
document. It is returned by DocumentService.ingest() and used by the
API layer (Module 9) to construct the HTTP 201 response.

It is not a pydantic model -- no HTTP validation overhead is needed for
an internal domain object. The pydantic response schema (Module 9) is
built from this dataclass.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal


@dataclass
class DocumentRecord:
    """
    Result of a completed document ingestion pipeline.

    Produced by DocumentService.ingest() after all chunks have been
    written to ChromaDB and the pipeline has completed successfully.

    Attributes:
        doc_id:           Unique document identifier. Format: "doc_<8hex>".
        user_id:          Tenant identifier.
        filename:         Original uploaded filename.
        file_type:        "pdf" or "csv".
        chunks_stored:    Number of chunks written to ChromaDB.
        status:           Always "complete" for a successful ingestion.
                          "failed" is represented by a raised exception,
                          never by a DocumentRecord with status="failed".
        uploaded_at:      UTC timestamp of pipeline completion.
    """

    doc_id: str
    user_id: str
    filename: str
    file_type: Literal["pdf", "csv"]
    chunks_stored: int
    status: Literal["complete"] = "complete"
    uploaded_at: datetime = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.uploaded_at is None:
            object.__setattr__(self, "uploaded_at", datetime.now(tz=UTC))

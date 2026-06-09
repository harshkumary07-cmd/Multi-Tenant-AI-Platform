"""
Document ingestion service.

Orchestrates the complete synchronous ingestion pipeline:
    validate → parse → chunk → embed → store → invalidate cache

This service is the single entry point for document upload processing.
Route handlers (Module 9) call ingest() and return its result as the
HTTP 201 response. The service never touches HTTP directly.

Pipeline stages:
    1. validate_upload()   -- file size, declared type, magic bytes
    2. parse()             -- extract text via PDF or CSV parser
    3. chunk_text()        -- split text into Chunk objects
    4. embed_chunks()      -- attach embedding vectors to chunks
    5. repository.add_chunks() -- write to ChromaDB in batches
    6. cache invalidation  -- clear user's cached query results (Module 8)

Partial write cleanup:
    If any stage after ChromaDB writes have begun raises an exception,
    repository.delete_document() is called to remove any partial data.
    This ensures the ChromaDB collection never contains orphaned chunks
    from a failed ingestion.

    The cleanup itself runs in a try/except -- a cleanup failure is
    logged at ERROR level but the original exception is still re-raised.
    An operator must manually verify and clean orphaned chunks if the
    cleanup step fails.

Dependency injection:
    DocumentService accepts ChromaRepository as a constructor argument.
    This makes it independently testable -- tests inject a mock repository
    without needing a running ChromaDB instance.
"""

import uuid
from typing import TYPE_CHECKING, Literal

from app.config.settings import Settings
from app.logging.logger import get_logger
from app.logging.timing import LatencyTracker
from app.models.document import DocumentRecord
from app.models.exceptions import (
    CorruptFileError,
    EmptyDocumentError,
    FileTooLargeError,
    InvalidFileTypeError,
)
from app.rag.parsers.csv_parser import parse_csv
from app.rag.parsers.pdf_parser import parse_pdf
from app.repositories.chroma_repository import ChromaRepository
from app.services.chunking_service import chunk_text
from app.services.embedding_service import embed_chunks

if TYPE_CHECKING:
    from app.cache.cache_service import CacheService

logger = get_logger(__name__)

ALLOWED_TYPES: frozenset[str] = frozenset({"pdf", "csv"})

# PDF magic bytes used for secondary validation
_PDF_MAGIC = b"%PDF-"


def _generate_doc_id() -> str:
    """Generate a unique document identifier. Format: 'doc_<8hex>'."""
    return f"doc_{uuid.uuid4().hex[:8]}"


def validate_upload(
    file_bytes: bytes,
    filename: str,
    declared_type: str,
    max_size_mb: int,
) -> Literal["pdf", "csv"]:
    """
    Validate an uploaded file before parsing begins.

    Checks file size, declared type, and magic bytes. Returns the
    validated file type on success.

    Args:
        file_bytes:    Raw file bytes.
        filename:      Original filename.
        declared_type: Type declared in the form field ("pdf" or "csv").
        max_size_mb:   Maximum allowed file size in megabytes.

    Returns:
        Literal["pdf", "csv"]: The validated file type.

    Raises:
        FileTooLargeError:    If file_bytes exceeds max_size_mb.
        InvalidFileTypeError: If declared_type is not "pdf" or "csv",
                              or if the file's magic bytes contradict
                              the declared type.
    """
    # Size check first -- avoids reading large files into memory for type checks
    max_bytes = max_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise FileTooLargeError(
            f"'{filename}' is {len(file_bytes) / (1024*1024):.1f}MB which exceeds "
            f"the {max_size_mb}MB limit. "
            "Split the file into smaller parts and upload each separately."
        )

    # Type check
    declared_lower = declared_type.lower()
    if declared_lower not in ALLOWED_TYPES:
        raise InvalidFileTypeError(
            f"File type '{declared_type}' is not supported. "
            "Accepted types: pdf, csv."
        )

    # Magic bytes check for PDFs
    # CSVs are text files with no universal magic bytes -- skip magic check
    if declared_lower == "pdf" and not file_bytes.startswith(_PDF_MAGIC):
        raise InvalidFileTypeError(
            f"'{filename}' was declared as PDF but does not begin with the "
            "PDF magic bytes ('%PDF-'). "
            "Ensure the file is a valid PDF and that the declared type is correct."
        )

    return declared_lower  # type: ignore[return-value]


class DocumentService:
    """
    Orchestrates the document ingestion pipeline.

    Accepts raw file bytes and produces a DocumentRecord after all chunks
    are confirmed stored in ChromaDB.

    Args:
        repository:    ChromaRepository instance for vector storage.
        settings:      Application settings for chunk size, model name, etc.
        cache_service: Optional CacheService. When provided, all cached query
                       results for the uploading user are invalidated after
                       successful ingestion so subsequent queries use the
                       freshly stored content. When absent, invalidation is
                       skipped silently (backward compatible with M1-M7 tests).
    """

    def __init__(
        self,
        repository: ChromaRepository,
        settings: Settings,
        cache_service: "CacheService | None" = None,
    ) -> None:
        self._repository = repository
        self._settings = settings
        self._cache_service = cache_service

    def ingest(
        self,
        user_id: str,
        file_bytes: bytes,
        filename: str,
        declared_type: str,
    ) -> DocumentRecord:
        """
        Run the complete ingestion pipeline for an uploaded file.

        Synchronous: returns only after all chunks are stored in ChromaDB.
        On any failure after writes have begun, attempts to clean up
        partial writes via delete_document().

        Args:
            user_id:       Tenant identifier.
            file_bytes:    Raw bytes of the uploaded file.
            filename:      Original filename.
            declared_type: File type declared by the client ("pdf" or "csv").

        Returns:
            DocumentRecord: Completed ingestion result with doc_id and
                            chunks_stored count.

        Raises:
            FileTooLargeError:    File exceeds configured size limit.
            InvalidFileTypeError: Unsupported or mismatched file type.
            CorruptFileError:     Parser cannot read the file.
            EmptyDocumentError:   No usable text after parsing/chunking.
            EmbeddingFailedError: Embedding model raised an error.
            VectorStoreError:     ChromaDB write failed.
        """
        tracker = LatencyTracker()
        doc_id = _generate_doc_id()
        writes_started = False

        logger.info(
            "ingestion started",
            extra={
                "event": "INGESTION_START",
                "user_id": user_id,
                "doc_id": doc_id,
                "source_file": filename,
                "file_size_bytes": len(file_bytes),
                "declared_type": declared_type,
            },
        )

        try:
            # ----------------------------------------------------------
            # Stage 1: Validate
            # ----------------------------------------------------------
            file_type = validate_upload(
                file_bytes=file_bytes,
                filename=filename,
                declared_type=declared_type,
                max_size_mb=self._settings.MAX_UPLOAD_SIZE_MB,
            )
            tracker.checkpoint("validate")

            # ----------------------------------------------------------
            # Stage 2: Parse
            # ----------------------------------------------------------
            if file_type == "pdf":
                text = parse_pdf(file_bytes, filename)
            else:
                text = parse_csv(file_bytes, filename)
            tracker.checkpoint("parse")

            # ----------------------------------------------------------
            # Stage 3: Chunk
            # ----------------------------------------------------------
            chunks = chunk_text(
                text=text,
                doc_id=doc_id,
                user_id=user_id,
                source=filename,
                chunk_size=self._settings.CHUNK_SIZE_TOKENS,
                chunk_overlap=self._settings.CHUNK_OVERLAP_TOKENS,
            )
            tracker.checkpoint("chunk")

            # ----------------------------------------------------------
            # Stage 4: Embed
            # ----------------------------------------------------------
            embed_chunks(chunks, batch_size=self._settings.EMBEDDING_BATCH_SIZE)
            tracker.checkpoint("embed")

            # ----------------------------------------------------------
            # Stage 5: Store
            # ----------------------------------------------------------
            writes_started = True
            self._repository.add_chunks(
                user_id=user_id,
                doc_id=doc_id,
                chunks=chunks,
            )
            tracker.checkpoint("store")

            # ----------------------------------------------------------
            # Stage 6: Cache invalidation (Module 8 wires this)
            # ----------------------------------------------------------
            # ----------------------------------------------------------
            # Stage 6: Cache invalidation
            # ----------------------------------------------------------
            # Invalidate all cached query results for this user so that
            # subsequent queries reflect the newly ingested document.
            # Skipped silently when no cache_service is configured.
            if self._cache_service is not None:
                self._cache_service.invalidate_user_cache(user_id)
            tracker.checkpoint("cache_invalidate")

        except (
            FileTooLargeError,
            InvalidFileTypeError,
            CorruptFileError,
            EmptyDocumentError,
        ):
            # User errors -- do not attempt cleanup (no writes occurred
            # for type/parse errors; chunking fails before writes start)
            raise

        except Exception:
            # Any failure after writes_started requires partial cleanup
            if writes_started:
                logger.error(
                    "ingestion failed after writes started -- cleaning up",
                    exc_info=True,
                    extra={
                        "event": "INGESTION_CLEANUP",
                        "user_id": user_id,
                        "doc_id": doc_id,
                        "source_file": filename,
                    },
                )
                try:
                    deleted = self._repository.delete_document(user_id, doc_id)
                    logger.info(
                        "partial write cleanup complete",
                        extra={
                            "user_id": user_id,
                            "doc_id": doc_id,
                            "chunks_cleaned": deleted,
                        },
                    )
                except Exception as cleanup_exc:
                    logger.error(
                        "partial write cleanup FAILED -- manual intervention required",
                        exc_info=True,
                        extra={
                            "user_id": user_id,
                            "doc_id": doc_id,
                            "cleanup_error": str(cleanup_exc),
                        },
                    )
            raise

        chunks_stored = len(chunks)

        logger.info(
            "ingestion complete",
            extra={
                "event": "INGESTION_COMPLETE",
                "user_id": user_id,
                "doc_id": doc_id,
                "source_file": filename,
                "file_type": file_type,
                "chunks_stored": chunks_stored,
                **tracker.to_log_fields(),
            },
        )

        return DocumentRecord(
            doc_id=doc_id,
            user_id=user_id,
            filename=filename,
            file_type=file_type,
            chunks_stored=chunks_stored,
        )

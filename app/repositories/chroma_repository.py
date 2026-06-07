"""
ChromaDB repository -- all vector database operations.

This is the only file that reads from and writes to ChromaDB.
No other module accesses the ChromaDB collection directly.

Tenant isolation contract:
    Every public method requires user_id as a non-optional parameter
    with no default value. This structural enforcement means it is
    impossible to call any method without a user_id -- a TypeError
    fires at call time if the argument is omitted.

    Every read operation applies: where={"user_id": {"$eq": user_id}}
    Every write operation includes user_id in chunk metadata.
    Every delete requires BOTH user_id AND doc_id.

Public methods (exactly four):
    add_chunks(user_id, doc_id, chunks)          -> None
    search_chunks(user_id, query_embedding, top_k) -> list[ChunkResult]
    delete_document(user_id, doc_id)             -> int
    count_documents(user_id)                     -> int

Error handling:
    All ChromaDB exceptions are caught and re-raised as VectorStoreError.
    Callers never see chromadb-specific exceptions.
"""

from typing import Any

from chromadb import Collection

from app.logging.logger import get_logger
from app.models.chunk import Chunk, ChunkResult
from app.models.exceptions import VectorStoreError
from app.vectorstore.tenant import (
    CHUNK_INDEX_FIELD,
    DOC_ID_FIELD,
    SOURCE_FIELD,
    USER_ID_FIELD,
    build_chunk_metadata,
)

logger = get_logger(__name__)

WRITE_BATCH_SIZE = 100


class ChromaRepository:
    """
    Tenant-isolated ChromaDB data access layer.

    All four public methods require user_id as a mandatory first argument.
    No method has a default value for user_id.
    """

    def __init__(self, collection: Collection) -> None:
        self._collection = collection

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        user_id: str,
        doc_id: str,
        chunks: list[Chunk],
    ) -> None:
        """
        Store embedded chunks in ChromaDB in batches.

        All chunks must have embeddings populated.
        Not atomic -- caller must call delete_document() on failure.

        Raises:
            ValueError:       If any chunk has an empty embedding vector.
            VectorStoreError: If ChromaDB raises any exception.
        """
        if not chunks:
            logger.debug(
                "add_chunks called with empty list -- no-op",
                extra={"user_id": user_id, "doc_id": doc_id},
            )
            return

        for chunk in chunks:
            if not chunk.has_embedding():
                raise ValueError(
                    f"Chunk {chunk.chunk_id} has no embedding vector. "
                    "Call embedding_service.embed() before add_chunks()."
                )

        total = len(chunks)
        stored = 0

        try:
            for batch_start in range(0, total, WRITE_BATCH_SIZE):
                batch = chunks[batch_start : batch_start + WRITE_BATCH_SIZE]

                self._collection.add(
                    ids=[c.chunk_id for c in batch],
                    embeddings=[c.embedding for c in batch],  # type: ignore[arg-type]
                    documents=[c.text for c in batch],
                    metadatas=[  # type: ignore[arg-type]
                        build_chunk_metadata(
                            user_id=user_id,
                            doc_id=doc_id,
                            chunk_index=c.chunk_index,
                            source=c.source,
                        )
                        for c in batch
                    ],
                )
                stored += len(batch)

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to store chunks for doc_id='{doc_id}': {exc}"
            ) from exc

        logger.info(
            "chunks stored",
            extra={
                "event": "CHUNKS_STORED",
                "user_id": user_id,
                "doc_id": doc_id,
                "chunks_stored": stored,
            },
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search_chunks(
        self,
        user_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ChunkResult]:
        """
        Search for the top_k most similar chunks for this user.

        The user_id filter is applied before distance computation.

        Returns:
            list[ChunkResult]: Chunks sorted by similarity score descending.
                               Empty list if the collection has no vectors.

        Raises:
            VectorStoreError: If ChromaDB raises any exception.
        """
        try:
            total_count = self._collection.count()
            if total_count == 0:
                return []

            raw: dict[str, Any] = self._collection.query(  # type: ignore[assignment]
                query_embeddings=[query_embedding],  # type: ignore[arg-type]
                n_results=min(top_k, total_count),
                where={USER_ID_FIELD: {"$eq": user_id}},  # type: ignore[arg-type]
                include=["documents", "metadatas", "distances"],
            )

        except Exception as exc:
            raise VectorStoreError(
                f"Similarity search failed for user_id='{user_id}': {exc}"
            ) from exc

        chunk_results: list[ChunkResult] = []

        ids: list[str] = (raw.get("ids") or [[]])[0]
        documents: list[str] = (raw.get("documents") or [[]])[0]
        metadatas: list[dict[str, Any]] = (raw.get("metadatas") or [[]])[0]
        distances: list[float] = (raw.get("distances") or [[]])[0]

        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=False
        ):
            if metadata is None:
                continue
            score = float(1.0 - distance)
            chunk_results.append(
                ChunkResult(
                    chunk_id=str(chunk_id),
                    doc_id=str(metadata.get(DOC_ID_FIELD, "")),
                    source=str(metadata.get(SOURCE_FIELD, "")),
                    chunk_index=int(metadata.get(CHUNK_INDEX_FIELD, 0)),
                    text=str(text),
                    score=score,
                )
            )

        logger.debug(
            "similarity search complete",
            extra={
                "event": "SIMILARITY_SEARCH",
                "user_id": user_id,
                "top_k": top_k,
                "results_returned": len(chunk_results),
                "top_score": round(chunk_results[0].score, 4) if chunk_results else None,
            },
        )

        return chunk_results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_document(
        self,
        user_id: str,
        doc_id: str,
    ) -> int:
        """
        Delete all chunks for the given user + document.

        A doc_id from another user's namespace silently returns 0.

        Returns:
            int: Number of chunks deleted. 0 if none found.

        Raises:
            VectorStoreError: If ChromaDB raises any exception.
        """
        try:
            existing: dict[str, Any] = self._collection.get(  # type: ignore[assignment]
                where={  # type: ignore[arg-type]
                    "$and": [
                        {USER_ID_FIELD: {"$eq": user_id}},
                        {DOC_ID_FIELD: {"$eq": doc_id}},
                    ]
                },
                include=[],
            )
            chunk_ids: list[str] = existing.get("ids", [])

            if not chunk_ids:
                return 0

            self._collection.delete(ids=chunk_ids)
            count = len(chunk_ids)

            logger.info(
                "document deleted",
                extra={
                    "event": "DOCUMENT_DELETED",
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "chunks_deleted": count,
                },
            )
            return count

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to delete doc_id='{doc_id}' for user_id='{user_id}': {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Count
    # ------------------------------------------------------------------

    def count_documents(self, user_id: str) -> int:
        """
        Count distinct documents stored for this user.

        Used by Router Agent to decide DIRECT vs RETRIEVE.

        Returns:
            int: Number of distinct doc_id values. 0 if none.

        Raises:
            VectorStoreError: If ChromaDB raises any exception.
        """
        try:
            result: dict[str, Any] = self._collection.get(  # type: ignore[assignment]
                where={USER_ID_FIELD: {"$eq": user_id}},  # type: ignore[arg-type]
                include=["metadatas"],
            )
            metadatas: list[dict[str, Any]] = result.get("metadatas") or []
            doc_ids = {
                str(m.get(DOC_ID_FIELD, ""))
                for m in metadatas
                if m and m.get(DOC_ID_FIELD)
            }
            count = len(doc_ids)

            logger.debug(
                "document count",
                extra={"user_id": user_id, "document_count": count},
            )
            return count

        except Exception as exc:
            raise VectorStoreError(
                f"Failed to count documents for user_id='{user_id}': {exc}"
            ) from exc

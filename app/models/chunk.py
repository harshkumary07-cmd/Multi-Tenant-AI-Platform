"""
Chunk domain models.

These are the internal domain objects passed between the ingestion pipeline
(Module 5), the embedding service (Module 5), and the repository layer
(Module 4). They have no HTTP concerns -- they are never serialised directly
to an API response.

Design:
    Chunk       -- a document chunk ready to be stored. Contains the text
                   and, after embedding, the float vector.
    ChunkResult -- a chunk returned by a similarity search. Contains the
                   text, the cosine similarity score, and source metadata.

Why dataclasses and not pydantic models:
    Pydantic models carry HTTP validation overhead that is unnecessary for
    purely internal objects. Plain dataclasses with type hints give mypy
    the static analysis it needs without the runtime cost.

    Pydantic is used only in app/schemas/ (HTTP request/response contracts).
    app/models/ uses Python dataclasses for domain objects.
"""

from dataclasses import dataclass, field


@dataclass
class Chunk:
    """
    A single text chunk ready for embedding and storage in ChromaDB.

    Produced by the chunking service (Module 5) and populated with an
    embedding vector by the embedding service (Module 5) before being
    passed to ChromaRepository.add_chunks().

    Attributes:
        chunk_id:    Unique identifier. Format: "{doc_id}_chunk_{index:03d}".
                     Built by tenant.build_chunk_id().
        doc_id:      Parent document identifier.
        user_id:     Tenant identifier. Must be present on every chunk.
        source:      Original filename (e.g. "q3_report.pdf").
        chunk_index: Zero-based position within the source document.
        text:        The raw text content of this chunk.
        embedding:   384-dimensional float vector from the embedding model.
                     Empty list until embedding_service populates it.
                     ChromaRepository.add_chunks() requires this to be
                     populated -- it raises ValueError if embedding is empty.
    """

    chunk_id: str
    doc_id: str
    user_id: str
    source: str
    chunk_index: int
    text: str
    embedding: list[float] = field(default_factory=list)

    def has_embedding(self) -> bool:
        """
        Return True if the embedding vector has been populated.

        ChromaRepository.add_chunks() calls this to validate chunks
        before attempting to write to ChromaDB.

        Returns:
            bool: True if embedding is a non-empty list of floats.
        """
        return len(self.embedding) > 0


@dataclass(frozen=True)
class ChunkResult:
    """
    A single chunk returned by a ChromaDB similarity search.

    Produced by ChromaRepository.search_chunks() and passed to the
    context assembler (Module 6). Frozen to prevent accidental mutation
    during context assembly.

    Attributes:
        chunk_id:    Unique identifier of the retrieved chunk.
        doc_id:      Parent document identifier.
        source:      Original filename. Used for source citations.
        chunk_index: Position within the source document.
        text:        The raw text content.
        score:       Cosine similarity score (0.0 to 1.0).
                     Higher is more relevant. Computed as 1.0 - distance.
                     Chunks below settings.RETRIEVAL_CONFIDENCE_THRESHOLD
                     are filtered out before context assembly.
    """

    chunk_id: str
    doc_id: str
    source: str
    chunk_index: int
    text: str
    score: float

    def is_above_threshold(self, threshold: float) -> bool:
        """
        Return True if this chunk's score meets the confidence threshold.

        Args:
            threshold: Minimum acceptable score (from settings).

        Returns:
            bool: True if score >= threshold.
        """
        return self.score >= threshold

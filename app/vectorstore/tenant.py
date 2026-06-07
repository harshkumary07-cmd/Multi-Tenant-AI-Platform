"""
Tenant isolation constants and metadata utilities for ChromaDB.

This module defines:
    - The canonical metadata field names used in every ChromaDB document
    - The ChunkMetadata TypedDict that documents the shape of stored metadata
    - Helper functions for constructing chunk IDs

Design principle:
    All field name strings are defined here as module-level constants.
    No other file in the codebase uses raw string literals for metadata
    field names. This means renaming a field requires changing exactly
    one line in this file -- not a grep across the codebase.

Tenant isolation contract:
    Every vector stored in ChromaDB MUST carry USER_ID_FIELD in its
    metadata. Every query MUST include a where clause using USER_ID_FIELD.
    These invariants are enforced at the ChromaRepository method signature
    level (user_id is a required parameter with no default) and verified
    by the tenant isolation test suite.
"""

from typing import TypedDict

# ---------------------------------------------------------------------------
# Metadata field name constants
# ---------------------------------------------------------------------------
# These strings are the keys in the metadata dict stored alongside each
# vector in ChromaDB. They are also the keys used in where= filter clauses.

USER_ID_FIELD = "user_id"
"""Tenant identifier. Present on every stored chunk. Used in every query filter."""

DOC_ID_FIELD = "doc_id"
"""Document identifier. Groups all chunks belonging to the same source file."""

CHUNK_ID_FIELD = "chunk_id"
"""Unique chunk identifier. Format: {doc_id}_chunk_{index:03d}."""

SOURCE_FIELD = "source"
"""Original filename (e.g. 'q3_report.pdf'). Used for source citations."""

CHUNK_INDEX_FIELD = "chunk_index"
"""Zero-based position of this chunk within its source document."""

# ---------------------------------------------------------------------------
# Distance metric
# ---------------------------------------------------------------------------

COSINE_DISTANCE = "cosine"
"""
Distance metric used when creating the documents collection.

Cosine distance measures the angle between vectors, making it invariant
to vector magnitude. This is correct for sentence-transformers embeddings
which encode semantic direction, not magnitude.

WARNING: This value is set at collection creation time and cannot be
changed without deleting and recreating the collection. If the collection
already exists with a different distance metric, ChromaRepository.__init__
raises VectorStoreError at startup.
"""


# ---------------------------------------------------------------------------
# Metadata TypedDict
# ---------------------------------------------------------------------------

class ChunkMetadata(TypedDict):
    """
    The complete metadata schema for every chunk stored in ChromaDB.

    This TypedDict documents the exact shape of the metadata dict passed
    to collection.add() and returned by collection.query(). Using TypedDict
    instead of a plain dict enables mypy to catch missing or misspelled
    metadata keys at analysis time rather than at runtime.

    All fields are required -- no chunk is stored without all five fields.

    Example:
        metadata: ChunkMetadata = {
            "user_id":     "u1",
            "doc_id":      "doc_9c4d1e2f",
            "chunk_id":    "doc_9c4d1e2f_chunk_003",
            "source":      "q3_report.pdf",
            "chunk_index": 3,
        }
    """

    user_id: str
    doc_id: str
    chunk_id: str
    source: str
    chunk_index: int


# ---------------------------------------------------------------------------
# Chunk ID construction
# ---------------------------------------------------------------------------

def build_chunk_id(doc_id: str, chunk_index: int) -> str:
    """
    Construct a canonical chunk_id from a doc_id and chunk index.

    Format: "{doc_id}_chunk_{index:03d}"
    Example: "doc_9c4d1e2f_chunk_003"

    The zero-padded three-digit index ensures that lexicographic sort
    order matches numeric sort order for documents with up to 999 chunks.
    Documents exceeding 999 chunks still work correctly -- the padding
    simply stops at three digits.

    Args:
        doc_id:      The document identifier (e.g. "doc_9c4d1e2f").
        chunk_index: Zero-based position of the chunk within the document.

    Returns:
        str: The canonical chunk_id string.

    Example:
        build_chunk_id("doc_9c4d1e2f", 3) -> "doc_9c4d1e2f_chunk_003"
        build_chunk_id("doc_9c4d1e2f", 0) -> "doc_9c4d1e2f_chunk_000"
    """
    return f"{doc_id}_chunk_{chunk_index:03d}"


def build_chunk_metadata(
    user_id: str,
    doc_id: str,
    chunk_index: int,
    source: str,
) -> ChunkMetadata:
    """
    Construct a complete ChunkMetadata dict for a single chunk.

    Centralising this construction ensures every stored chunk has all
    required metadata fields in the correct format. Callers cannot
    accidentally omit a field.

    Args:
        user_id:     Tenant identifier.
        doc_id:      Document identifier.
        chunk_index: Zero-based chunk position.
        source:      Original filename.

    Returns:
        ChunkMetadata: Complete metadata dict ready for ChromaDB storage.

    Example:
        meta = build_chunk_metadata("u1", "doc_abc", 0, "report.pdf")
        # {"user_id": "u1", "doc_id": "doc_abc",
        #  "chunk_id": "doc_abc_chunk_000", "source": "report.pdf",
        #  "chunk_index": 0}
    """
    return ChunkMetadata(
        user_id=user_id,
        doc_id=doc_id,
        chunk_id=build_chunk_id(doc_id, chunk_index),
        source=source,
        chunk_index=chunk_index,
    )

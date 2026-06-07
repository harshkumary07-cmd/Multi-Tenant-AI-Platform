"""
Unit tests for ChromaRepository using a mocked ChromaDB collection.

These tests verify the repository's logic without requiring a running
ChromaDB instance. They mock the Collection object and assert that the
repository calls it correctly.

Tests cover:
    - add_chunks passes correct IDs, embeddings, documents, metadatas
    - add_chunks raises ValueError on chunks with empty embeddings
    - add_chunks batches writes in groups of WRITE_BATCH_SIZE
    - add_chunks is a no-op on empty list
    - search_chunks applies user_id filter on every query
    - search_chunks converts distances to scores (score = 1 - distance)
    - search_chunks returns empty list when collection is empty
    - delete_document requires both user_id and doc_id in filter
    - delete_document returns 0 when no chunks found
    - delete_document returns correct count of deleted chunks
    - count_documents counts distinct doc_ids for user
    - count_documents returns 0 for user with no documents
    - All methods raise VectorStoreError on ChromaDB exceptions

No infrastructure required. Runs in under 1 second.
"""

from unittest.mock import MagicMock

import pytest

from app.models.chunk import Chunk, ChunkResult
from app.models.exceptions import VectorStoreError
from app.repositories.chroma_repository import WRITE_BATCH_SIZE, ChromaRepository
from app.vectorstore.tenant import (
    CHUNK_INDEX_FIELD,
    DOC_ID_FIELD,
    SOURCE_FIELD,
    USER_ID_FIELD,
    build_chunk_id,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_chunk(
    index: int = 0,
    doc_id: str = "doc_test",
    user_id: str = "u_test",
    source: str = "test.pdf",
    has_embedding: bool = True,
) -> Chunk:
    """Build a test Chunk with or without an embedding vector."""
    return Chunk(
        chunk_id=build_chunk_id(doc_id, index),
        doc_id=doc_id,
        user_id=user_id,
        source=source,
        chunk_index=index,
        text=f"chunk text {index}",
        embedding=[0.1] * 384 if has_embedding else [],
    )


def make_collection_mock() -> MagicMock:
    """Create a MagicMock for chromadb.Collection."""
    mock = MagicMock()
    # Default count() return -- tests override as needed
    mock.count.return_value = 10
    return mock


def make_repo(collection_mock: MagicMock | None = None) -> ChromaRepository:
    """Create a ChromaRepository with the given or a fresh mock collection."""
    if collection_mock is None:
        collection_mock = make_collection_mock()
    return ChromaRepository(collection=collection_mock)


# ---------------------------------------------------------------------------
# add_chunks
# ---------------------------------------------------------------------------


class TestAddChunks:
    """ChromaRepository.add_chunks() stores chunks with correct metadata."""

    def test_calls_collection_add_with_correct_ids(self) -> None:
        """add_chunks passes chunk_ids as ids to collection.add()."""
        mock = make_collection_mock()
        repo = make_repo(mock)
        chunks = [make_chunk(0, doc_id="doc_abc"), make_chunk(1, doc_id="doc_abc")]
        repo.add_chunks("u1", "doc_abc", chunks)
        call_kwargs = mock.add.call_args.kwargs
        assert call_kwargs["ids"] == [
            build_chunk_id("doc_abc", 0),
            build_chunk_id("doc_abc", 1),
        ]

    def test_calls_collection_add_with_embeddings(self) -> None:
        """add_chunks passes embedding vectors to collection.add()."""
        mock = make_collection_mock()
        repo = make_repo(mock)
        chunks = [make_chunk(0)]
        repo.add_chunks("u1", "doc_abc", chunks)
        call_kwargs = mock.add.call_args.kwargs
        assert len(call_kwargs["embeddings"]) == 1
        assert len(call_kwargs["embeddings"][0]) == 384

    def test_metadata_contains_user_id(self) -> None:
        """Each chunk's metadata includes the correct user_id."""
        mock = make_collection_mock()
        repo = make_repo(mock)
        chunks = [make_chunk(0)]
        repo.add_chunks("u1", "doc_abc", chunks)
        meta = mock.add.call_args.kwargs["metadatas"][0]
        assert meta[USER_ID_FIELD] == "u1"

    def test_metadata_contains_doc_id(self) -> None:
        """Each chunk's metadata includes the correct doc_id."""
        mock = make_collection_mock()
        repo = make_repo(mock)
        chunks = [make_chunk(0)]
        repo.add_chunks("u1", "doc_abc", chunks)
        meta = mock.add.call_args.kwargs["metadatas"][0]
        assert meta[DOC_ID_FIELD] == "doc_abc"

    def test_metadata_contains_source(self) -> None:
        """Each chunk's metadata includes the source filename."""
        mock = make_collection_mock()
        repo = make_repo(mock)
        chunks = [make_chunk(0, source="report.pdf")]
        repo.add_chunks("u1", "doc_abc", chunks)
        meta = mock.add.call_args.kwargs["metadatas"][0]
        assert meta[SOURCE_FIELD] == "report.pdf"

    def test_metadata_contains_chunk_index(self) -> None:
        """Each chunk's metadata includes the correct chunk_index."""
        mock = make_collection_mock()
        repo = make_repo(mock)
        chunks = [make_chunk(7)]
        repo.add_chunks("u1", "doc_abc", chunks)
        meta = mock.add.call_args.kwargs["metadatas"][0]
        assert meta[CHUNK_INDEX_FIELD] == 7

    def test_empty_list_is_noop(self) -> None:
        """add_chunks with empty list does not call collection.add()."""
        mock = make_collection_mock()
        repo = make_repo(mock)
        repo.add_chunks("u1", "doc_abc", [])
        mock.add.assert_not_called()

    def test_chunk_without_embedding_raises_value_error(self) -> None:
        """add_chunks raises ValueError if any chunk has no embedding."""
        repo = make_repo()
        chunk = make_chunk(0, has_embedding=False)
        with pytest.raises(ValueError, match="has no embedding vector"):
            repo.add_chunks("u1", "doc_abc", [chunk])

    def test_batching_on_large_input(self) -> None:
        """add_chunks calls collection.add() multiple times for large inputs."""
        mock = make_collection_mock()
        repo = make_repo(mock)
        # Create more chunks than WRITE_BATCH_SIZE
        chunks = [make_chunk(i) for i in range(WRITE_BATCH_SIZE + 10)]
        repo.add_chunks("u1", "doc_abc", chunks)
        # Should have been called twice: one full batch + one partial
        assert mock.add.call_count == 2

    def test_single_batch_for_small_input(self) -> None:
        """add_chunks calls collection.add() once for inputs <= WRITE_BATCH_SIZE."""
        mock = make_collection_mock()
        repo = make_repo(mock)
        chunks = [make_chunk(i) for i in range(5)]
        repo.add_chunks("u1", "doc_abc", chunks)
        assert mock.add.call_count == 1

    def test_chromadb_exception_raises_vector_store_error(self) -> None:
        """add_chunks wraps ChromaDB exceptions in VectorStoreError."""
        mock = make_collection_mock()
        mock.add.side_effect = RuntimeError("connection lost")
        repo = make_repo(mock)
        chunks = [make_chunk(0)]
        with pytest.raises(VectorStoreError, match="Failed to store chunks"):
            repo.add_chunks("u1", "doc_abc", chunks)


# ---------------------------------------------------------------------------
# search_chunks
# ---------------------------------------------------------------------------


class TestSearchChunks:
    """ChromaRepository.search_chunks() queries with user_id filter."""

    def _make_query_response(
        self,
        chunk_ids: list[str],
        texts: list[str],
        doc_ids: list[str],
        distances: list[float],
        source: str = "test.pdf",
    ) -> dict:
        """Build a realistic ChromaDB query() response dict."""
        metadatas = [
            {
                USER_ID_FIELD: "u1",
                DOC_ID_FIELD: doc_id,
                SOURCE_FIELD: source,
                CHUNK_INDEX_FIELD: i,
            }
            for i, doc_id in enumerate(doc_ids)
        ]
        return {
            "ids": [chunk_ids],
            "documents": [texts],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def test_applies_user_id_filter(self) -> None:
        """search_chunks always passes user_id in the where clause."""
        mock = make_collection_mock()
        mock.query.return_value = self._make_query_response(
            ["c1"], ["text"], ["doc_a"], [0.1]
        )
        repo = make_repo(mock)
        repo.search_chunks("u1", [0.1] * 384, top_k=5)
        where = mock.query.call_args.kwargs["where"]
        assert where == {USER_ID_FIELD: {"$eq": "u1"}}

    def test_converts_distance_to_score(self) -> None:
        """search_chunks returns score = 1.0 - distance."""
        mock = make_collection_mock()
        mock.query.return_value = self._make_query_response(
            ["c1"], ["text"], ["doc_a"], [0.1]
        )
        repo = make_repo(mock)
        results = repo.search_chunks("u1", [0.1] * 384, top_k=5)
        assert len(results) == 1
        assert abs(results[0].score - 0.9) < 1e-6

    def test_returns_chunk_result_objects(self) -> None:
        """search_chunks returns a list of ChunkResult instances."""
        mock = make_collection_mock()
        mock.query.return_value = self._make_query_response(
            ["c1"], ["hello world"], ["doc_a"], [0.2]
        )
        repo = make_repo(mock)
        results = repo.search_chunks("u1", [0.1] * 384, top_k=5)
        assert all(isinstance(r, ChunkResult) for r in results)

    def test_returns_empty_list_when_collection_empty(self) -> None:
        """search_chunks returns [] without querying if collection count is 0."""
        mock = make_collection_mock()
        mock.count.return_value = 0
        repo = make_repo(mock)
        results = repo.search_chunks("u1", [0.1] * 384, top_k=5)
        assert results == []
        mock.query.assert_not_called()

    def test_caps_n_results_at_collection_count(self) -> None:
        """search_chunks does not request more results than exist."""
        mock = make_collection_mock()
        mock.count.return_value = 3
        mock.query.return_value = self._make_query_response(
            ["c1", "c2", "c3"],
            ["t1", "t2", "t3"],
            ["d1", "d2", "d3"],
            [0.1, 0.2, 0.3],
        )
        repo = make_repo(mock)
        repo.search_chunks("u1", [0.1] * 384, top_k=10)
        n_results = mock.query.call_args.kwargs["n_results"]
        assert n_results == 3  # capped at collection count, not 10

    def test_chromadb_exception_raises_vector_store_error(self) -> None:
        """search_chunks wraps ChromaDB exceptions in VectorStoreError."""
        mock = make_collection_mock()
        mock.query.side_effect = RuntimeError("timeout")
        repo = make_repo(mock)
        with pytest.raises(VectorStoreError, match="Similarity search failed"):
            repo.search_chunks("u1", [0.1] * 384, top_k=5)

    def test_result_contains_correct_text(self) -> None:
        """search_chunks result text matches the stored document text."""
        mock = make_collection_mock()
        mock.query.return_value = self._make_query_response(
            ["c1"], ["exact chunk text"], ["doc_a"], [0.05]
        )
        repo = make_repo(mock)
        results = repo.search_chunks("u1", [0.1] * 384, top_k=5)
        assert results[0].text == "exact chunk text"

    def test_result_contains_correct_doc_id(self) -> None:
        """search_chunks result doc_id matches the stored metadata."""
        mock = make_collection_mock()
        mock.query.return_value = self._make_query_response(
            ["c1"], ["text"], ["doc_xyz"], [0.1]
        )
        repo = make_repo(mock)
        results = repo.search_chunks("u1", [0.1] * 384, top_k=5)
        assert results[0].doc_id == "doc_xyz"


# ---------------------------------------------------------------------------
# delete_document
# ---------------------------------------------------------------------------


class TestDeleteDocument:
    """ChromaRepository.delete_document() removes chunks by user+doc."""

    def test_returns_count_of_deleted_chunks(self) -> None:
        """delete_document returns the number of deleted chunks."""
        mock = make_collection_mock()
        mock.get.return_value = {"ids": ["c1", "c2", "c3"]}
        repo = make_repo(mock)
        count = repo.delete_document("u1", "doc_abc")
        assert count == 3

    def test_returns_zero_when_no_chunks_found(self) -> None:
        """delete_document returns 0 when no matching chunks exist."""
        mock = make_collection_mock()
        mock.get.return_value = {"ids": []}
        repo = make_repo(mock)
        count = repo.delete_document("u1", "doc_nonexistent")
        assert count == 0
        mock.delete.assert_not_called()

    def test_filter_includes_user_id(self) -> None:
        """delete_document filter includes user_id to prevent cross-tenant deletes."""
        mock = make_collection_mock()
        mock.get.return_value = {"ids": ["c1"]}
        repo = make_repo(mock)
        repo.delete_document("u1", "doc_abc")
        where = mock.get.call_args.kwargs["where"]
        # Verify user_id is in the $and filter
        conditions = where.get("$and", [])
        user_condition = next(
            (c for c in conditions if USER_ID_FIELD in c), None
        )
        assert user_condition is not None
        assert user_condition[USER_ID_FIELD] == {"$eq": "u1"}

    def test_filter_includes_doc_id(self) -> None:
        """delete_document filter includes doc_id."""
        mock = make_collection_mock()
        mock.get.return_value = {"ids": ["c1"]}
        repo = make_repo(mock)
        repo.delete_document("u1", "doc_abc")
        where = mock.get.call_args.kwargs["where"]
        conditions = where.get("$and", [])
        doc_condition = next(
            (c for c in conditions if DOC_ID_FIELD in c), None
        )
        assert doc_condition is not None
        assert doc_condition[DOC_ID_FIELD] == {"$eq": "doc_abc"}

    def test_chromadb_exception_raises_vector_store_error(self) -> None:
        """delete_document wraps ChromaDB exceptions in VectorStoreError."""
        mock = make_collection_mock()
        mock.get.side_effect = RuntimeError("connection error")
        repo = make_repo(mock)
        with pytest.raises(VectorStoreError, match="Failed to delete"):
            repo.delete_document("u1", "doc_abc")


# ---------------------------------------------------------------------------
# count_documents
# ---------------------------------------------------------------------------


class TestCountDocuments:
    """ChromaRepository.count_documents() counts distinct doc_ids."""

    def test_returns_correct_count(self) -> None:
        """count_documents counts distinct doc_id values in metadata."""
        mock = make_collection_mock()
        mock.get.return_value = {
            "metadatas": [
                {USER_ID_FIELD: "u1", DOC_ID_FIELD: "doc_a"},
                {USER_ID_FIELD: "u1", DOC_ID_FIELD: "doc_a"},  # same doc
                {USER_ID_FIELD: "u1", DOC_ID_FIELD: "doc_b"},
            ]
        }
        repo = make_repo(mock)
        count = repo.count_documents("u1")
        assert count == 2  # doc_a and doc_b -- not 3 chunks

    def test_returns_zero_for_empty_result(self) -> None:
        """count_documents returns 0 when user has no documents."""
        mock = make_collection_mock()
        mock.get.return_value = {"metadatas": []}
        repo = make_repo(mock)
        assert repo.count_documents("u1") == 0

    def test_applies_user_id_filter(self) -> None:
        """count_documents filters by user_id."""
        mock = make_collection_mock()
        mock.get.return_value = {"metadatas": []}
        repo = make_repo(mock)
        repo.count_documents("u1")
        where = mock.get.call_args.kwargs["where"]
        assert where == {USER_ID_FIELD: {"$eq": "u1"}}

    def test_chromadb_exception_raises_vector_store_error(self) -> None:
        """count_documents wraps ChromaDB exceptions in VectorStoreError."""
        mock = make_collection_mock()
        mock.get.side_effect = RuntimeError("timeout")
        repo = make_repo(mock)
        with pytest.raises(VectorStoreError, match="Failed to count"):
            repo.count_documents("u1")


# ---------------------------------------------------------------------------
# Tenant isolation contract -- structural enforcement
# ---------------------------------------------------------------------------


class TestTenantIsolationStructural:
    """Verify that user_id cannot be omitted from any public method."""

    def test_add_chunks_requires_user_id(self) -> None:
        """add_chunks raises TypeError if user_id is not provided."""
        repo = make_repo()
        with pytest.raises(TypeError):
            repo.add_chunks(doc_id="doc_abc", chunks=[])  # type: ignore[call-arg]

    def test_search_chunks_requires_user_id(self) -> None:
        """search_chunks raises TypeError if user_id is not provided."""
        repo = make_repo()
        with pytest.raises(TypeError):
            repo.search_chunks(query_embedding=[0.1], top_k=5)  # type: ignore[call-arg]

    def test_delete_document_requires_user_id(self) -> None:
        """delete_document raises TypeError if user_id is not provided."""
        repo = make_repo()
        with pytest.raises(TypeError):
            repo.delete_document(doc_id="doc_abc")  # type: ignore[call-arg]

    def test_count_documents_requires_user_id(self) -> None:
        """count_documents raises TypeError if user_id is not provided."""
        repo = make_repo()
        with pytest.raises(TypeError):
            repo.count_documents()  # type: ignore[call-arg]

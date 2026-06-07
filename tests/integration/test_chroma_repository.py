"""
Integration tests for ChromaRepository against a real ChromaDB instance.

These tests require ChromaDB to be running:
    docker compose up chromadb -d

They verify the actual database behaviour -- not mocks. In particular,
they verify the tenant isolation guarantee: User A's data is NEVER
returned when querying as User B.

The tenant isolation tests are a HARD CI GATE. No PR merges if any
isolation test fails.

All tests use the unique_user_id fixture from conftest.py to ensure
test data is isolated and does not pollute other tests.

Markers:
    @pytest.mark.integration  -- requires ChromaDB infrastructure
    @pytest.mark.isolation    -- tenant isolation tests (hard CI gate)
"""

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.models.chunk import Chunk
from app.repositories.chroma_repository import ChromaRepository
from app.vectorstore.tenant import COSINE_DISTANCE, build_chunk_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chroma_collection(tmp_path):  # type: ignore[no-untyped-def]
    """
    Create an ephemeral in-process ChromaDB collection for integration tests.

    Uses chromadb.EphemeralClient (in-memory) so tests do not require a
    running ChromaDB Docker container. The collection is unique per test
    via tmp_path, ensuring complete isolation between tests.
    """
    client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection_name = f"test_{tmp_path.name}"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": COSINE_DISTANCE},
    )
    return collection


@pytest.fixture
def repo(chroma_collection):  # type: ignore[no-untyped-def]
    """Create a ChromaRepository backed by the ephemeral test collection."""
    return ChromaRepository(collection=chroma_collection)


def make_chunk(
    index: int,
    doc_id: str,
    user_id: str,
    source: str = "test.pdf",
    text: str | None = None,
) -> Chunk:
    """Build a Chunk with a deterministic embedding vector."""
    # Use a distinct embedding per chunk_index so similarity search
    # returns predictable ordering in tests.
    embedding = [0.0] * 384
    embedding[index % 384] = 1.0  # one non-zero dimension per chunk

    return Chunk(
        chunk_id=build_chunk_id(doc_id, index),
        doc_id=doc_id,
        user_id=user_id,
        source=source,
        chunk_index=index,
        text=text or f"chunk {index} for {doc_id}",
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# Basic CRUD operations
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBasicOperations:
    """Basic store-and-retrieve operations against real ChromaDB."""

    def test_add_and_retrieve_single_chunk(
        self, repo: ChromaRepository, unique_user_id: str
    ) -> None:
        """A stored chunk is returned by search_chunks."""
        chunk = make_chunk(0, "doc_a", unique_user_id)
        repo.add_chunks(unique_user_id, "doc_a", [chunk])

        query = [0.0] * 384
        query[0] = 1.0  # same as chunk 0's embedding
        results = repo.search_chunks(unique_user_id, query, top_k=1)

        assert len(results) == 1
        assert results[0].chunk_id == chunk.chunk_id
        assert results[0].text == chunk.text

    def test_add_multiple_chunks_same_document(
        self, repo: ChromaRepository, unique_user_id: str
    ) -> None:
        """All chunks for a document are stored and retrievable."""
        chunks = [make_chunk(i, "doc_multi", unique_user_id) for i in range(5)]
        repo.add_chunks(unique_user_id, "doc_multi", chunks)

        count = repo.count_documents(unique_user_id)
        assert count == 1  # one document, five chunks

    def test_count_documents_increases_per_document(
        self, repo: ChromaRepository, unique_user_id: str
    ) -> None:
        """count_documents reflects distinct documents, not chunk count."""
        repo.add_chunks(unique_user_id, "doc_1", [make_chunk(0, "doc_1", unique_user_id)])
        repo.add_chunks(unique_user_id, "doc_2", [make_chunk(0, "doc_2", unique_user_id)])
        assert repo.count_documents(unique_user_id) == 2

    def test_delete_document_removes_all_chunks(
        self, repo: ChromaRepository, unique_user_id: str
    ) -> None:
        """delete_document removes all chunks for the given document."""
        chunks = [make_chunk(i, "doc_del", unique_user_id) for i in range(3)]
        repo.add_chunks(unique_user_id, "doc_del", chunks)

        deleted = repo.delete_document(unique_user_id, "doc_del")
        assert deleted == 3
        assert repo.count_documents(unique_user_id) == 0

    def test_delete_nonexistent_document_returns_zero(
        self, repo: ChromaRepository, unique_user_id: str
    ) -> None:
        """delete_document returns 0 for a doc_id that does not exist."""
        count = repo.delete_document(unique_user_id, "doc_ghost")
        assert count == 0

    def test_count_zero_for_new_user(
        self, repo: ChromaRepository, unique_user_id: str
    ) -> None:
        """A fresh user with no uploads has document count 0."""
        assert repo.count_documents(unique_user_id) == 0

    def test_search_returns_empty_for_user_with_no_chunks(
        self, repo: ChromaRepository, unique_user_id: str
    ) -> None:
        """search_chunks returns [] for a user with no stored documents."""
        results = repo.search_chunks(unique_user_id, [0.1] * 384, top_k=5)
        assert results == []

    def test_score_is_between_zero_and_one(
        self, repo: ChromaRepository, unique_user_id: str
    ) -> None:
        """Similarity scores from search_chunks are in [-1, 1] range."""
        chunk = make_chunk(0, "doc_score", unique_user_id)
        repo.add_chunks(unique_user_id, "doc_score", [chunk])

        results = repo.search_chunks(unique_user_id, [0.1] * 384, top_k=1)
        assert len(results) == 1
        assert -1.0 <= results[0].score <= 1.0

    def test_result_source_field_preserved(
        self, repo: ChromaRepository, unique_user_id: str
    ) -> None:
        """search_chunks results preserve the source filename."""
        chunk = make_chunk(0, "doc_src", unique_user_id, source="annual_report.pdf")
        repo.add_chunks(unique_user_id, "doc_src", [chunk])

        results = repo.search_chunks(unique_user_id, [0.1] * 384, top_k=1)
        assert results[0].source == "annual_report.pdf"


# ---------------------------------------------------------------------------
# Tenant isolation tests -- HARD CI GATE
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.isolation
class TestTenantIsolation:
    """
    Tenant isolation tests: User A cannot access User B's data.

    These tests are a HARD CI GATE. If any of these tests fail, the PR
    is blocked regardless of other test results.

    Each test writes data as one user and asserts that the same data is
    NOT returned when queried as a different user.
    """

    def test_search_returns_only_own_chunks(
        self,
        repo: ChromaRepository,
        unique_user_id: str,
    ) -> None:
        """User A's search does not return User B's chunks."""
        user_a = unique_user_id
        user_b = f"{unique_user_id}_b"

        # Store chunk for User A
        chunk_a = make_chunk(0, "doc_a", user_a, text="User A private document")
        repo.add_chunks(user_a, "doc_a", [chunk_a])

        # Query as User B -- must return empty
        query = [0.0] * 384
        query[0] = 1.0
        results_b = repo.search_chunks(user_b, query, top_k=5)
        assert results_b == [], (
            f"ISOLATION FAILURE: User B retrieved {len(results_b)} chunk(s) "
            f"belonging to User A. Cross-tenant data exposure."
        )

    def test_count_documents_isolated_per_user(
        self,
        repo: ChromaRepository,
        unique_user_id: str,
    ) -> None:
        """User B's document count is not affected by User A's uploads."""
        user_a = unique_user_id
        user_b = f"{unique_user_id}_b"

        repo.add_chunks(user_a, "doc_a", [make_chunk(0, "doc_a", user_a)])
        repo.add_chunks(user_a, "doc_b", [make_chunk(0, "doc_b", user_a)])

        # User B has no documents
        assert repo.count_documents(user_b) == 0

    def test_delete_cannot_affect_other_user(
        self,
        repo: ChromaRepository,
        unique_user_id: str,
    ) -> None:
        """User B cannot delete User A's document even with the doc_id."""
        user_a = unique_user_id
        user_b = f"{unique_user_id}_b"

        # User A stores a document
        chunk = make_chunk(0, "doc_sensitive", user_a)
        repo.add_chunks(user_a, "doc_sensitive", [chunk])

        # User B attempts to delete User A's doc_id
        deleted = repo.delete_document(user_b, "doc_sensitive")

        # Must return 0 -- User A's chunk is unaffected
        assert deleted == 0
        assert repo.count_documents(user_a) == 1

    def test_two_users_same_query_isolated_results(
        self,
        repo: ChromaRepository,
        unique_user_id: str,
    ) -> None:
        """Users A and B see only their own results for identical queries."""
        user_a = unique_user_id
        user_b = f"{unique_user_id}_b"

        # Store one document per user with identical structure
        chunk_a = make_chunk(0, "doc_a", user_a, text="User A content")
        chunk_b = make_chunk(0, "doc_b", user_b, text="User B content")
        repo.add_chunks(user_a, "doc_a", [chunk_a])
        repo.add_chunks(user_b, "doc_b", [chunk_b])

        query = [0.0] * 384
        query[0] = 1.0

        results_a = repo.search_chunks(user_a, query, top_k=5)
        results_b = repo.search_chunks(user_b, query, top_k=5)

        # Each user sees only their own chunk
        assert all(r.text == "User A content" for r in results_a)
        assert all(r.text == "User B content" for r in results_b)

        # Explicitly verify no cross-contamination
        a_chunk_ids = {r.chunk_id for r in results_a}
        b_chunk_ids = {r.chunk_id for r in results_b}
        assert a_chunk_ids.isdisjoint(b_chunk_ids), (
            "ISOLATION FAILURE: Users A and B share chunk IDs in search results."
        )

"""
Pytest shared fixtures -- available to all test modules automatically.

pytest discovers conftest.py; no imports required in test files.

Critical fixture -- unique_user_id:
    Every test that touches user-scoped data MUST use this fixture.
    It generates a UUID-based user_id per test invocation, ensuring:
        1. Tests cannot affect each other's data.
        2. Isolation tests cannot pass by coincidence.
        3. Tests are safe to run in parallel.

    Never hardcode user_id values like "u1" or "test" in any test.
"""

import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings
from fastapi.testclient import TestClient

from app.vectorstore.tenant import COSINE_DISTANCE
from main import app


@pytest.fixture
def unique_user_id() -> str:
    """
    Generate a unique user_id for each test invocation.

    Format: "test_<12-hex-char UUID fragment>"
    Example: "test_a3f8b2c1d4e5"

    Returns:
        str: A unique user identifier safe for use across all layers.
    """
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient for API tests.

    Patches the ChromaDB initialisation so that API/middleware tests
    run without requiring a live ChromaDB Docker container.
    The TestClient exercises the full middleware stack (logging,
    request_id generation, error handling) without infrastructure.

    Integration tests that need real ChromaDB use the chroma_collection
    fixture defined below, which creates an ephemeral in-memory client.

    Yields:
        TestClient: Configured test client for the application.
    """
    mock_collection = MagicMock()
    mock_collection.count.return_value = 0

    with patch(
        "main.initialise_chroma",
        return_value=mock_collection,
    ), patch(
        "main.close_chroma_client",
        return_value=None,
    ), patch(
        "main.initialise_embedding_model",
        return_value=None,
    ):
        with TestClient(app, raise_server_exceptions=False) as test_client:
            yield test_client


@pytest.fixture
def auth_headers(unique_user_id: str) -> dict[str, str]:
    """
    HTTP headers with a valid X-User-Id for authenticated requests.

    Args:
        unique_user_id: Injected by pytest from the unique_user_id fixture.

    Returns:
        dict[str, str]: Header dict with X-User-Id set to unique_user_id.
    """
    return {"X-User-Id": unique_user_id}


@pytest.fixture
def chroma_collection(tmp_path):  # type: ignore[no-untyped-def]
    """
    Ephemeral in-memory ChromaDB collection for integration tests.

    Uses chromadb.EphemeralClient (in-memory, no Docker required).
    Each test gets a unique collection name via tmp_path.

    Used directly by integration tests that need real ChromaDB behaviour.
    """
    client = chromadb.EphemeralClient(
        settings=ChromaSettings(anonymized_telemetry=False)
    )
    collection_name = f"test_{tmp_path.name[:20]}"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": COSINE_DISTANCE},
    )


# ---------------------------------------------------------------------------
# Module 5 -- Document ingestion fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    """
    Real binary PDF with known text content.

    Contains two pages with financial report text including the phrases:
    'revenue', '2.4 billion', 'cloud services', 'Asia-Pacific'.
    Used by ingestion pipeline integration tests.
    """
    return (FIXTURES_DIR / "sample.pdf").read_bytes()


@pytest.fixture
def sample_csv_bytes() -> bytes:
    """
    Real CSV with known financial data (25 rows, 5 columns).

    Columns: quarter, revenue_usd_millions, region, growth_pct, product.
    Contains 'Q3 2024' and other known values for assertion.
    """
    return (FIXTURES_DIR / "sample.csv").read_bytes()


@pytest.fixture
def corrupt_pdf_bytes() -> bytes:
    """
    File whose first bytes are NOT '%PDF-'.

    Used to test CORRUPT_FILE error handling and partial write cleanup.
    """
    return (FIXTURES_DIR / "corrupt.pdf").read_bytes()


# ---------------------------------------------------------------------------
# Future fixtures -- added in their respective modules
# ---------------------------------------------------------------------------
#
# Module 8 -- Redis:
#   @pytest.fixture
#   def redis_client() -> redis.Redis: ...

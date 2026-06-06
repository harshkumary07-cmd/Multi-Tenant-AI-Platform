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

import pytest
from fastapi.testclient import TestClient

from main import app


@pytest.fixture
def unique_user_id() -> str:
    """
    Generate a unique user_id for each test invocation.

    Format: "test_<12-hex-char UUID fragment>"
    Example: "test_a3f8b2c1d4e5"

    The "test_" prefix makes test-generated data immediately
    identifiable in logs and database inspection.

    Returns:
        str: A unique user identifier safe for use across all layers.
    """
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """
    FastAPI TestClient for API tests.

    Exercises the full FastAPI middleware stack without a real server.

    Usage:
        def test_health(client: TestClient) -> None:
            response = client.get("/health")
            assert response.status_code == 200

    Yields:
        TestClient: Configured test client for the application.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers(unique_user_id: str) -> dict[str, str]:
    """
    HTTP headers with a valid X-User-Id for authenticated requests.

    Combines with unique_user_id to provide isolated auth headers per test.
    Use this for all endpoint tests that require authentication.

    Usage:
        def test_upload(client: TestClient, auth_headers: dict) -> None:
            response = client.post("/upload-doc", headers=auth_headers, ...)

    Args:
        unique_user_id: Injected by pytest from the unique_user_id fixture.

    Returns:
        dict[str, str]: Header dict with X-User-Id set to unique_user_id.
    """
    return {"X-User-Id": unique_user_id}


# ---------------------------------------------------------------------------
# Future fixtures -- added in their respective modules
# ---------------------------------------------------------------------------
#
# Module 4 -- ChromaDB:
#   @pytest.fixture
#   def chroma_client() -> chromadb.HttpClient: ...
#
# Module 5 -- Document ingestion:
#   @pytest.fixture
#   def sample_pdf_bytes() -> bytes: ...   # reads tests/fixtures/sample.pdf
#   @pytest.fixture
#   def sample_csv_bytes() -> bytes: ...   # reads tests/fixtures/sample.csv
#   @pytest.fixture
#   def corrupt_pdf_bytes() -> bytes: ...  # reads tests/fixtures/corrupt.pdf
#
# Module 8 -- Redis:
#   @pytest.fixture
#   def redis_client() -> redis.Redis: ...

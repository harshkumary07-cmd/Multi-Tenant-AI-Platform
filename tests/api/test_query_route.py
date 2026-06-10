"""
Tests for POST /query.

Covers:
    - Authentication (401 on missing header)
    - Successful answer (200 with answer)
    - No-result response (200 with null answer)
    - Cache hit path (cache_hit=True in response)
    - Validation errors (422)
    - LLM timeout (504 via mocked exception)
    - VectorStore error (503 via mocked exception)
    - LLM provider error (502 via mocked exception)
    - Response schema completeness

All service calls are mocked -- no ChromaDB, Redis, or LLM required.
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.config.dependencies import get_cached_query_service
from app.models.exceptions import LLMProviderError, LLMTimeoutError, VectorStoreError
from app.models.query_result import QueryResult, SourceReference, TokenUsage
from main import app


def _make_query_result(
    answer: str | None = "Revenue was $2.4B in Q3.",
    route: str = "RETRIEVE",
    cache_hit: bool = False,
    no_result_reason: str | None = None,
) -> QueryResult:
    return QueryResult(
        query="What was the revenue?",
        user_id="u_test",
        answer=answer,
        sources=[
            SourceReference(
                doc_id="doc_abc",
                source="report.pdf",
                chunk_count=2,
                top_score=0.91,
            )
        ] if answer else [],
        route=route,  # type: ignore[arg-type]
        chunks_retrieved=5,
        chunks_used=2,
        token_usage=TokenUsage.from_counts(200, 80),
        latency_ms=1240,
        no_result_reason=no_result_reason,
        cache_hit=cache_hit,
        timestamp=datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


def _mock_service(result: QueryResult) -> MagicMock:
    mock = MagicMock()
    mock.query.return_value = result
    return mock


class TestQueryRouteAuthentication:

    def test_missing_user_id_header_returns_401(self, client: TestClient) -> None:
        response = client.post("/query", json={"query": "what is revenue?"})
        assert response.status_code == 401

    def test_missing_header_error_code(self, client: TestClient) -> None:
        response = client.post("/query", json={"query": "test"})
        assert response.json()["error_code"] == "UNAUTHORIZED"

    def test_blank_user_id_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/query",
            json={"query": "test"},
            headers={"X-User-Id": ""},
        )
        assert response.status_code == 401

    def test_whitespace_only_user_id_returns_401(self, client: TestClient) -> None:
        response = client.post(
            "/query",
            json={"query": "test"},
            headers={"X-User-Id": "   "},
        )
        assert response.status_code == 401


class TestQueryRouteHappyPath:

    def test_successful_query_returns_200(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result())
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query",
                json={"query": "What was the revenue?"},
                headers=auth_headers,
            )
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_response_contains_answer(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result(answer="Revenue was $2.4B."))
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query",
                json={"query": "What was the revenue?"},
                headers=auth_headers,
            )
            assert response.json()["answer"] == "Revenue was $2.4B."
        finally:
            app.dependency_overrides.clear()

    def test_response_contains_sources(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result())
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query",
                json={"query": "query"},
                headers=auth_headers,
            )
            body = response.json()
            assert "sources" in body
            assert isinstance(body["sources"], list)
        finally:
            app.dependency_overrides.clear()

    def test_response_contains_route_field(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result(route="RETRIEVE"))
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert response.json()["route"] == "RETRIEVE"
        finally:
            app.dependency_overrides.clear()

    def test_response_contains_cache_hit_field(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result(cache_hit=False))
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert "cache_hit" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_cache_hit_true_reflected_in_response(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result(cache_hit=True))
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert response.json()["cache_hit"] is True
        finally:
            app.dependency_overrides.clear()

    def test_response_contains_token_usage(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result())
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            body = response.json()
            assert "token_usage" in body
            assert "total_tokens" in body["token_usage"]
        finally:
            app.dependency_overrides.clear()

    def test_response_contains_latency_ms(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result())
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert "latency_ms" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_top_k_override_passed_to_service(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result())
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            client.post(
                "/query",
                json={"query": "q", "top_k": 7},
                headers=auth_headers,
            )
            call_kwargs = mock_svc.query.call_args.kwargs
            assert call_kwargs["top_k"] == 7
        finally:
            app.dependency_overrides.clear()


class TestQueryRouteNoResult:

    def test_no_result_returns_200(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        result = _make_query_result(
            answer=None,
            no_result_reason="NO_RELEVANT_CHUNKS",
        )
        mock_svc = _mock_service(result)
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert response.status_code == 200
        finally:
            app.dependency_overrides.clear()

    def test_no_result_answer_is_null(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        result = _make_query_result(answer=None, no_result_reason="NO_RELEVANT_CHUNKS")
        mock_svc = _mock_service(result)
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert response.json()["answer"] is None
        finally:
            app.dependency_overrides.clear()

    def test_no_result_reason_present(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        result = _make_query_result(answer=None, no_result_reason="NO_RELEVANT_CHUNKS")
        mock_svc = _mock_service(result)
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert response.json()["no_result_reason"] is not None
        finally:
            app.dependency_overrides.clear()


class TestQueryRouteValidation:

    def test_empty_query_returns_422(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result())
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": ""}, headers=auth_headers
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_missing_query_field_returns_422(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result())
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post("/query", json={}, headers=auth_headers)
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_top_k_below_range_returns_422(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result())
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q", "top_k": 0}, headers=auth_headers
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()

    def test_top_k_above_range_returns_422(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = _mock_service(_make_query_result())
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q", "top_k": 21}, headers=auth_headers
            )
            assert response.status_code == 422
        finally:
            app.dependency_overrides.clear()


class TestQueryRouteErrors:

    def test_llm_timeout_returns_504(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.query.side_effect = LLMTimeoutError("LLM timed out")
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert response.status_code == 504
        finally:
            app.dependency_overrides.clear()

    def test_llm_provider_error_returns_502(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.query.side_effect = LLMProviderError("provider failed")
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert response.status_code == 502
        finally:
            app.dependency_overrides.clear()

    def test_vector_store_error_returns_503(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.query.side_effect = VectorStoreError("ChromaDB unreachable")
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert response.status_code == 503
        finally:
            app.dependency_overrides.clear()

    def test_error_response_has_error_code(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.query.side_effect = LLMTimeoutError("timeout")
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert "error_code" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_error_response_has_message(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        mock_svc = MagicMock()
        mock_svc.query.side_effect = VectorStoreError("db down")
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert "message" in response.json()
        finally:
            app.dependency_overrides.clear()

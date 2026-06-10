"""
Tests for POST /user, GET /logs, and middleware enforcement.

Covers:
    - User registration happy path (201)
    - User ID format validation (422)
    - Logs endpoint authentication (401)
    - Logs endpoint response structure (200)
    - TenantContextMiddleware enforcement across all authenticated routes
    - ErrorHandlerMiddleware error envelope structure
    - Health endpoint exemption from auth
"""

from fastapi.testclient import TestClient


class TestUserRoute:

    def test_create_user_returns_201(self, client: TestClient) -> None:
        response = client.post("/user", json={"user_id": "alice"})
        assert response.status_code == 201

    def test_create_user_response_contains_user_id(
        self, client: TestClient
    ) -> None:
        response = client.post("/user", json={"user_id": "alice"})
        assert response.json()["user_id"] == "alice"

    def test_create_user_response_contains_created_at(
        self, client: TestClient
    ) -> None:
        response = client.post("/user", json={"user_id": "alice"})
        assert "created_at" in response.json()

    def test_create_user_response_contains_message(
        self, client: TestClient
    ) -> None:
        response = client.post("/user", json={"user_id": "alice"})
        assert response.status_code == 201
        assert "message" in response.json()

    def test_create_user_with_hyphens_returns_201(
        self, client: TestClient
    ) -> None:
        response = client.post("/user", json={"user_id": "alice-bob"})
        assert response.status_code == 201

    def test_create_user_with_underscores_returns_201(
        self, client: TestClient
    ) -> None:
        response = client.post("/user", json={"user_id": "alice_bob"})
        assert response.status_code == 201

    def test_create_user_no_auth_header_required(
        self, client: TestClient
    ) -> None:
        # /user is in the unauthenticated path list -- no X-User-Id needed
        # because user is being created (bootstrap problem)
        response = client.post("/user", json={"user_id": "alice"})
        assert response.status_code == 201

    def test_user_id_with_spaces_returns_422(self, client: TestClient) -> None:
        response = client.post("/user", json={"user_id": "alice bob"})
        assert response.status_code == 422

    def test_user_id_with_special_chars_returns_422(
        self, client: TestClient
    ) -> None:
        response = client.post("/user", json={"user_id": "alice@bob.com"})
        assert response.status_code == 422

    def test_empty_user_id_returns_422(self, client: TestClient) -> None:
        response = client.post("/user", json={"user_id": ""})
        assert response.status_code == 422

    def test_missing_user_id_field_returns_422(self, client: TestClient) -> None:
        response = client.post("/user", json={})
        assert response.status_code == 422

    def test_create_user_response_user_id_matches_input(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.post(
            "/user", json={"user_id": "my-tenant-123"}, headers=auth_headers
        )
        if response.status_code == 201:
            assert response.json()["user_id"] == "my-tenant-123"


class TestLogsRoute:

    def test_logs_requires_auth_header(self, client: TestClient) -> None:
        response = client.get("/logs")
        assert response.status_code == 401

    def test_logs_returns_200_with_auth(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.get("/logs", headers=auth_headers)
        assert response.status_code == 200

    def test_logs_response_contains_user_id(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.get("/logs", headers=auth_headers)
        assert "user_id" in response.json()

    def test_logs_response_contains_request_metrics(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.get("/logs", headers=auth_headers)
        assert "request_metrics" in response.json()

    def test_logs_response_contains_cache_statistics(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.get("/logs", headers=auth_headers)
        assert "cache_statistics" in response.json()

    def test_logs_response_contains_route_decisions(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.get("/logs", headers=auth_headers)
        assert "route_decisions" in response.json()

    def test_logs_response_contains_documents(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        response = client.get("/logs", headers=auth_headers)
        assert "documents" in response.json()


class TestTenantContextMiddleware:

    def test_query_route_requires_header(self, client: TestClient) -> None:
        response = client.post("/query", json={"query": "test"})
        assert response.status_code == 401

    def test_upload_route_requires_header(self, client: TestClient) -> None:
        response = client.post(
            "/upload-doc",
            data={"file_type": "pdf"},
            files={"file": ("r.pdf", b"%PDF-1", "application/pdf")},
        )
        assert response.status_code == 401

    def test_logs_route_requires_header(self, client: TestClient) -> None:
        response = client.get("/logs")
        assert response.status_code == 401

    def test_health_does_not_require_header(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_401_has_error_code_field(self, client: TestClient) -> None:
        response = client.post("/query", json={"query": "test"})
        body = response.json()
        assert "error_code" in body

    def test_401_has_message_field(self, client: TestClient) -> None:
        response = client.post("/query", json={"query": "test"})
        body = response.json()
        assert "message" in body

    def test_401_error_code_is_unauthorized(self, client: TestClient) -> None:
        response = client.get("/logs")
        assert response.json()["error_code"] == "UNAUTHORIZED"

    def test_valid_header_passes_through(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        # Any route with a valid header should not 401
        response = client.get("/logs", headers=auth_headers)
        assert response.status_code != 401


class TestErrorHandlerMiddleware:

    def test_error_envelope_has_error_code(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        from unittest.mock import MagicMock

        from app.config.dependencies import get_cached_query_service
        from app.models.exceptions import VectorStoreError
        from main import app

        mock_svc = MagicMock()
        mock_svc.query.side_effect = VectorStoreError("db down")
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert "error_code" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_error_envelope_has_message(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        from unittest.mock import MagicMock

        from app.config.dependencies import get_cached_query_service
        from app.models.exceptions import LLMTimeoutError
        from main import app

        mock_svc = MagicMock()
        mock_svc.query.side_effect = LLMTimeoutError("timeout")
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            assert "message" in response.json()
        finally:
            app.dependency_overrides.clear()

    def test_no_stack_trace_in_error_response(
        self, client: TestClient, auth_headers: dict
    ) -> None:
        from unittest.mock import MagicMock

        from app.config.dependencies import get_cached_query_service
        from app.models.exceptions import VectorStoreError
        from main import app

        mock_svc = MagicMock()
        mock_svc.query.side_effect = VectorStoreError("db down")
        app.dependency_overrides[get_cached_query_service] = lambda: mock_svc
        try:
            response = client.post(
                "/query", json={"query": "q"}, headers=auth_headers
            )
            body_str = str(response.json())
            assert "Traceback" not in body_str
            assert "traceback" not in body_str
        finally:
            app.dependency_overrides.clear()

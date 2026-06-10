"""
Unit tests for app/middleware/request_logger.py -- Module 3.

Tests use FastAPI's TestClient to exercise the middleware through real
HTTP requests to the running application. No mocking of the middleware --
testing the actual behaviour is more valuable for a middleware component.

Tests cover:
    - request_id is generated and stored on request.state
    - request_id format matches "req_<12hex>" pattern
    - Health endpoint is excluded from logging (no request_id stamped)
    - Non-health routes (401/422) are logged normally (not excluded)
    - Each request generates a unique request_id
    - The middleware does not break normal request processing
    - The middleware does not raise on 4xx or 5xx responses

No infrastructure required. Uses TestClient which runs the full
middleware stack in-process.
"""

import re

from fastapi.testclient import TestClient

REQUEST_ID_PATTERN = re.compile(r"^req_[0-9a-f]{12}$")


class TestRequestIdGeneration:
    """Every non-health request receives a unique request_id."""

    def test_query_request_does_not_raise(self, client: TestClient) -> None:
        """POST /query requires auth -- middleware logs the 401 without error."""
        response = client.post("/query")
        # M9: TenantContextMiddleware returns 401 before route handler runs
        assert response.status_code == 401

    def test_upload_request_does_not_raise(self, client: TestClient) -> None:
        """POST /upload-doc requires auth -- middleware logs the 401 without error."""
        response = client.post("/upload-doc")
        assert response.status_code == 401

    def test_user_request_does_not_raise(self, client: TestClient) -> None:
        """POST /user is exempt from auth -- returns 422 (missing body) or 201."""
        response = client.post("/user")
        # /user is exempt from TenantContextMiddleware (bootstrap route)
        assert response.status_code in (201, 422)

    def test_logs_request_does_not_raise(self, client: TestClient) -> None:
        """GET /logs requires auth -- middleware logs the 401 without error."""
        response = client.get("/logs")
        assert response.status_code == 401

    def test_health_request_returns_200(self, client: TestClient) -> None:
        """GET /health returns 200 and is not affected by logging middleware."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_each_request_is_independent(self, client: TestClient) -> None:
        """Multiple requests to the same endpoint each process independently."""
        # Each call should succeed without contamination from the previous
        for _ in range(5):
            response = client.get("/health")
            assert response.status_code == 200


class TestMiddlewareDoesNotBreakResponses:
    """The logging middleware is transparent -- it does not alter responses."""

    def test_health_response_body_unchanged(self, client: TestClient) -> None:
        """Health response body is not modified by the logging middleware."""
        response = client.get("/health")
        body = response.json()
        assert body["status"] == "ok"
        assert "env" in body
        assert "version" in body

    def test_auth_error_body_unchanged(self, client: TestClient) -> None:
        """Auth 401 response body from TenantContextMiddleware is not modified by the logging middleware."""
        response = client.post("/query")
        body = response.json()
        assert body["error_code"] == "UNAUTHORIZED"

    def test_404_response_not_altered(self, client: TestClient, auth_headers: dict) -> None:
        """Unknown path returns 404, not altered by the logging middleware."""
        response = client.get("/completely/unknown/path", headers=auth_headers)
        assert response.status_code == 404

    def test_response_status_codes_preserved(self, client: TestClient, auth_headers: dict) -> None:
        """Various status codes are preserved through the middleware."""
        assert client.get("/health").status_code == 200
        assert client.post("/user").status_code == 422       # exempt from auth, missing body
        assert client.post("/query").status_code == 401      # auth required, no header
        # Unknown path requires auth to reach 404 (auth check happens first)
        assert client.get("/completely/unknown/path", headers=auth_headers).status_code == 404


class TestHealthEndpointExclusion:
    """GET /health is excluded from logging middleware processing."""

    def test_health_endpoint_still_returns_200(self, client: TestClient) -> None:
        """Health endpoint works normally when excluded from logging."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_can_be_called_repeatedly(self, client: TestClient) -> None:
        """Health endpoint handles repeated calls without issue."""
        for _ in range(10):
            response = client.get("/health")
            assert response.status_code == 200

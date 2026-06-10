"""
Tests for GET /health.

The health endpoint is the only route fully implemented in Module 1.
Route-specific tests for POST /user, POST /upload-doc, POST /query,
and GET /logs live in their own test files (Module 9).

Test coverage:
    - Happy path: 200 with correct response schema
    - No authentication required
    - Correct field types in response
    - Non-existent paths return 404 (not 500)
    - All routes are registered and do not return 404 or 501
"""

from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Test suite for GET /health."""

    def test_health_returns_200(self, client: TestClient) -> None:
        """Health endpoint returns HTTP 200 OK."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_has_required_fields(self, client: TestClient) -> None:
        """Health response contains all required fields."""
        response = client.get("/health")
        body = response.json()

        assert "status" in body
        assert "env" in body
        assert "version" in body

    def test_health_response_fields_are_strings(self, client: TestClient) -> None:
        """All health response fields are strings."""
        response = client.get("/health")
        body = response.json()

        assert isinstance(body["status"], str)
        assert isinstance(body["env"], str)
        assert isinstance(body["version"], str)

    def test_health_status_is_ok(self, client: TestClient) -> None:
        """Health status field value is 'ok' for a running application."""
        response = client.get("/health")
        assert response.json()["status"] == "ok"

    def test_health_requires_no_authentication(self, client: TestClient) -> None:
        """Health endpoint is reachable without X-User-Id header."""
        response = client.get("/health")
        assert response.status_code != 401
        assert response.status_code != 403

    def test_nonexistent_path_returns_404(self, client: TestClient, auth_headers: dict) -> None:
        """Requests to unknown paths return 404, not 500."""
        response = client.get("/this-path-does-not-exist", headers=auth_headers)
        assert response.status_code == 404

    def test_all_routes_are_registered(self, client: TestClient) -> None:
        """
        All routes are registered and respond (not 404).

        After Module 9, routes return real responses (200/201/401/422)
        rather than 501 stubs. This test verifies registration only.
        Authentication details are tested in route-specific test files.
        """
        routes = [
            ("POST", "/user"),
            ("POST", "/upload-doc"),
            ("POST", "/query"),
            ("GET", "/logs"),
        ]
        for method, path in routes:
            if method == "POST":
                response = client.post(path)
            else:
                response = client.get(path)

            assert response.status_code != 404, (
                f"{method} {path} returned 404 -- route not registered."
            )
            assert response.status_code != 501, (
                f"{method} {path} returned 501 -- route is still a stub."
            )

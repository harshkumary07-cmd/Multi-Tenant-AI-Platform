"""
API tests -- full HTTP request lifecycle.

Use FastAPI TestClient (no real HTTP server required).
Exercise the full middleware stack (auth, logging, error handling).
Test every defined status code for every endpoint.
Include cross-tenant API attack scenarios.
CI gate: runs on every PR.
"""

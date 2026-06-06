"""
Test suite root package.

Structure mirrors app/ exactly:
    tests/unit/        -- fast, no infrastructure, all deps mocked
    tests/integration/ -- requires real ChromaDB + Redis via Docker
    tests/api/         -- full HTTP stack via FastAPI TestClient
    tests/fixtures/    -- committed binary test files (PDF, CSV)

Every test that touches user-scoped data uses the unique_user_id fixture.
No test hardcodes user_id values like "u1" or "test".
"""

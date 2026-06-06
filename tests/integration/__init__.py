"""
Integration tests -- require real ChromaDB and Redis.

Require: docker compose up chromadb redis
Test actual database behaviour -- no mocks for external systems.
Includes the critical tenant isolation tests (hard CI merge gate).
CI gate: runs on every PR.

CRITICAL: test_chroma_isolation.py must always pass.
No PR merges if any isolation test fails.
"""

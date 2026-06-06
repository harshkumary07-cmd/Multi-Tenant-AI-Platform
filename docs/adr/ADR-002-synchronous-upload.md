# ADR-002: Synchronous document upload processing

**Date:** 2024-01-15
**Status:** Accepted

---

## Context

Uploaded documents must be parsed, chunked, embedded, and stored in ChromaDB.
This pipeline takes 5-60 seconds. Design question: return immediately (async)
or hold the HTTP connection until complete (sync)?

## Decision

Process uploads **synchronously**. Return `201 Created` only when ChromaDB
confirms all chunks are written.

## Alternatives considered

| Option | Pro | Con | Decision |
|---|---|---|---|
| Async with Celery + Redis queue | Non-blocking | 4 extra infrastructure components | Rejected (Phase 1) |
| Async with FastAPI BackgroundTasks | Simpler than Celery | Tasks lost on container restart | Rejected |
| Synchronous | No extra infrastructure | Held HTTP connection (up to 60s) | **Accepted** |

## Consequences

**Becomes easier:**
- No message queue, worker processes, or job status polling
- Failure = 5xx response; nothing was stored

**Becomes harder:**
- Large files hold an HTTP connection for up to 60 seconds
- Mitigation: MAX_UPLOAD_SIZE_MB=50 and UPLOAD_TIMEOUT_SECONDS=120

**Phase 2 upgrade path:**
Async processing (Celery + Redis queue) is the documented Phase 2 upgrade.
The `document_service.ingest()` interface is designed to be upgrade-compatible.

## References

- Architecture Handbook, Section 4: Document Ingestion Pipeline
- Implementation Blueprint, Module 5: Document Ingestion

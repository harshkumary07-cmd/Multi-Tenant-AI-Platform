# API Reference

Base URL: `http://localhost:8000`

All endpoints except `GET /health` and `POST /user` require:
```
X-User-Id: <your_user_id>
Content-Type: application/json   (for JSON body endpoints)
```

## Error Response Envelope

All error responses use a consistent structure:
```json
{
  "error_code": "SCREAMING_SNAKE_CASE_EXCEPTION_NAME",
  "message": "Human-readable description of the error."
}
```

## Endpoints

### GET /health
No authentication. Used by Docker HEALTHCHECK and load balancers.

**Response 200:**
```json
{"status": "ok", "env": "development", "version": "0.1.0"}
```

---

### POST /user
Register a user identity. No `X-User-Id` header required (bootstrap route).

**Request:**
```json
{"user_id": "alice-tenant-01"}
```
Constraints: 1–64 chars, `[a-zA-Z0-9_-]` only.

**Response 201:**
```json
{
  "user_id": "alice-tenant-01",
  "created_at": "2024-06-01T12:00:00+00:00",
  "message": "User 'alice-tenant-01' registered successfully. Use X-User-Id: alice-tenant-01 header on all subsequent requests."
}
```

**Errors:** `422` invalid format.

---

### POST /upload-doc
Ingest a document. Synchronous: 201 means all chunks are stored and queryable.

**Request:** `multipart/form-data`
- `file` (required): PDF or CSV file binary
- `file_type` (required): `"pdf"` or `"csv"`

**Response 201:**
```json
{
  "document_id": "doc_a1b2c3d4",
  "user_id": "alice-tenant-01",
  "filename": "q3-report.pdf",
  "file_type": "pdf",
  "chunks_stored": 47,
  "status": "complete",
  "upload_timestamp": "2024-06-01T12:00:00+00:00"
}
```

**Errors:**
| Code | Condition |
|---|---|
| 400 | Invalid file type, corrupt file, empty document, CSV parse error |
| 413 | File exceeds `MAX_UPLOAD_SIZE_MB` (default 50MB) |
| 500 | Embedding model failure |
| 503 | ChromaDB unavailable |

---

### POST /query
Submit a natural language query.

**Request:**
```json
{
  "query": "What were the Q3 revenue figures?",
  "top_k": 5
}
```
- `query`: 1–2000 characters (required)
- `top_k`: 1–20 (optional, overrides `RETRIEVAL_TOP_K` setting)

**Response 200 (answer):**
```json
{
  "query": "What were the Q3 revenue figures?",
  "answer": "Q3 revenue was $2.4B, a 34% increase year-over-year. [Source: q3-report.pdf, chunk 3]",
  "sources": [
    {
      "doc_id": "doc_a1b2c3d4",
      "source": "q3-report.pdf",
      "chunk_count": 2,
      "top_score": 0.91
    }
  ],
  "route": "RETRIEVE",
  "chunks_retrieved": 5,
  "chunks_used": 2,
  "token_usage": {
    "prompt_tokens": 820,
    "completion_tokens": 64,
    "total_tokens": 884
  },
  "latency_ms": 1240,
  "no_result_reason": null,
  "cache_hit": false,
  "timestamp": "2024-06-01T12:00:00+00:00"
}
```

**Response 200 (no result):**
```json
{
  "answer": null,
  "sources": [],
  "no_result_reason": "No chunks met the confidence threshold of 0.35. Examined 5 candidate chunks.",
  "cache_hit": false,
  ...
}
```

**Errors:**
| Code | Condition |
|---|---|
| 422 | Empty query, top_k out of range |
| 502 | LLM provider returned an error |
| 503 | ChromaDB unavailable |
| 504 | LLM provider timed out |

---

### GET /logs
Retrieve operational metrics.

**Response 200:**
```json
{
  "user_id": "alice-tenant-01",
  "note": "Phase 1: metrics persistence not yet implemented.",
  "request_metrics": {"total_requests": 0, "avg_latency_ms": 0, "p95_latency_ms": 0},
  "cache_statistics": {"hit_rate_pct": 0.0, "cache_hits": 0, "cache_misses": 0},
  "route_decisions": {"direct_count": 0, "retrieve_count": 0},
  "documents": {"total_uploaded": 0, "total_chunks_stored": 0},
  "recent_events": [],
  "error_summary": {}
}
```

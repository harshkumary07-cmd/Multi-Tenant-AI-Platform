# Multi-Tenant AI Platform

> Production-grade RAG platform enabling multiple isolated users to upload
> documents and query them using natural language.

[![CI](https://github.com/yourusername/ai-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/ai-platform/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This platform allows multiple isolated users (tenants) to:

1. **Upload** PDF and CSV documents
2. **Query** those documents using natural language
3. **Receive** accurate, cited answers grounded in their own documents

The system uses **RAG** (Retrieval-Augmented Generation): documents are
embedded into a vector database and semantically relevant passages are
retrieved at query time. Those passages are passed to an LLM as context,
producing answers grounded in the user's actual files rather than general
training data.

Every user's data is completely isolated from every other user through
six enforced layers of tenant isolation.

---

## Features

- **Multi-tenant isolation** — each user's documents and cache entries are
  completely separated; cross-tenant data access is architecturally impossible
- **PDF and CSV ingestion** — automatic text extraction, chunking, and embedding
- **Semantic retrieval** — cosine similarity search over embedded document chunks
- **Smart routing** — RouterAgent decides DIRECT (general knowledge) vs RETRIEVE
  (document search) before any expensive work begins
- **Multi-LLM support** — OpenAI, Anthropic, or a local provider via a single env var
- **Redis caching** — query results cached with per-user invalidation on upload
- **Structured logging** — every request logs JSON with request_id, user_id, latency
- **Token cost tracking** — prompt and completion tokens on every LLM response

---

## Architecture

```
POST /query
    → TenantContextMiddleware    (validates X-User-Id header)
    → CachedQueryService         (Redis cache lookup)
        → RoutedQueryService
                → RouterAgent   (DIRECT vs RETRIEVE — deterministic, 5 rules)
                ┌─ DIRECT ──→ LLMProvider.generate()
                └─ RETRIEVE → QueryService
                                → embed_single()
                                → ChromaRepository.search_chunks(user_id=...)
                                → assemble_context()
                                → build_messages()
                                → LLMProvider.generate()
    ← QueryResult (answer, sources, route, token_usage, cache_hit, latency_ms)
```

**Tenant isolation layers:**

1. `X-User-Id` header validated on every request by `TenantContextMiddleware`
2. `user_id` injected into every service call via `Depends(get_current_user_id)`
3. ChromaDB queries always include `where={"user_id": {"$eq": user_id}}`
4. Cache keys prefixed `query:{user_id}:...` — no cross-tenant key collisions
5. Cache invalidation on upload scoped to `query:{user_id}:*`
6. All domain exception messages exclude other users' data

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| API framework | FastAPI | 0.111 |
| Vector database | ChromaDB | 0.5.0 |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | 3.0.0 |
| LLM (default) | OpenAI gpt-4o | via API |
| Cache | Redis | 7 |
| Text splitting | LangChain text splitters | 0.2.0 |
| PDF parsing | pdfplumber | 0.11.0 |
| CSV parsing | pandas | 2.2.0 |
| Validation | Pydantic v2 | 2.7.0 |
| Server | Uvicorn | 0.29.0 |

---

## Quick Start

### Option A — Docker (recommended)

**Prerequisites:** Docker Desktop or Docker Engine with Compose plugin.

```bash
# 1. Clone
git clone https://github.com/yourusername/ai-platform.git
cd ai-platform

# 2. Configure
cp .env.example .env
# Edit .env -- set LLM_API_KEY to your real key:
#   LLM_API_KEY=sk-...        (OpenAI)
#   LLM_API_KEY=sk-ant-...    (Anthropic, also set LLM_PROVIDER=anthropic)

# 3. Start all services
docker compose up

# 4. Verify
curl http://localhost:8000/health
```

### Option B — Local Python (development)

**Prerequisites:** Python 3.11+, ChromaDB running locally, Redis running locally.

```bash
# 1. Clone and enter
git clone https://github.com/yourusername/ai-platform.git
cd ai-platform

# 2. Virtual environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for testing

# 4. Configure
cp .env.example .env
# Edit .env -- set LLM_API_KEY at minimum

# 5. Start ChromaDB (separate terminal)
pip install chromadb
chroma run --host localhost --port 8001

# 6. Start Redis (separate terminal)
redis-server

# 7. Start the API
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# 8. Verify
curl http://localhost:8000/health
```

---

## First Query (end-to-end)

```bash
# 1. Register a user
curl -s -X POST http://localhost:8000/user \
     -H "Content-Type: application/json" \
     -d '{"user_id": "alice"}' | python3 -m json.tool

# 2. Upload a document
curl -s -X POST http://localhost:8000/upload-doc \
     -H "X-User-Id: alice" \
     -F "file=@/path/to/your-document.pdf" \
     -F "file_type=pdf" | python3 -m json.tool

# 3. Query it
curl -s -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -H "X-User-Id: alice" \
     -d '{"query": "What are the main findings in this document?"}' \
     | python3 -m json.tool
```

---

## Environment Variables

See [`.env.example`](.env.example) for the full annotated list. Key variables:

| Variable | Default | Required | Description |
|---|---|---|---|
| `LLM_API_KEY` | `changeme` | **Yes** | OpenAI or Anthropic API key |
| `LLM_PROVIDER` | `openai` | No | `openai` \| `anthropic` \| `local` |
| `LLM_MODEL_NAME` | `gpt-4o` | No | Model identifier for the chosen provider |
| `APP_ENV` | `development` | No | `development` \| `production` |
| `CHROMA_HOST` | `localhost` | No | ChromaDB hostname (Docker: `chromadb`) |
| `REDIS_HOST` | `localhost` | No | Redis hostname (Docker: `redis`) |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.35` | No | Minimum cosine similarity to include a chunk |

---

## API Reference

All authenticated endpoints require the `X-User-Id: <your_user_id>` header.

### `GET /health`

Health check. No authentication required.

**Response 200:**
```json
{"status": "ok", "env": "development", "version": "0.1.0"}
```

---

### `POST /user`

Register a user identity. No authentication required.

**Request body:**
```json
{"user_id": "alice"}
```

`user_id`: 1–64 characters, alphanumeric + hyphens + underscores.

**Response 201:**
```json
{
  "user_id": "alice",
  "created_at": "2024-06-01T12:00:00Z",
  "message": "User 'alice' registered successfully."
}
```

---

### `POST /upload-doc`

Upload and ingest a PDF or CSV document. Synchronous — returns 201 after all
chunks are confirmed stored.

**Request:** `multipart/form-data`
- `file`: the document file
- `file_type`: `"pdf"` or `"csv"`

**Response 201:**
```json
{
  "document_id": "doc_a1b2c3d4",
  "user_id": "alice",
  "filename": "report.pdf",
  "file_type": "pdf",
  "chunks_stored": 42,
  "status": "complete",
  "upload_timestamp": "2024-06-01T12:00:00Z"
}
```

**Error codes:** `400` bad file type / corrupt / empty, `413` too large, `503` ChromaDB unavailable.

---

### `POST /query`

Submit a natural language query for AI-powered answering.

**Request body:**
```json
{
  "query": "What were the Q3 revenue figures?",
  "top_k": 5
}
```

`top_k` is optional (default: `RETRIEVAL_TOP_K` setting, range 1–20).

**Response 200:**
```json
{
  "query": "What were the Q3 revenue figures?",
  "answer": "Q3 revenue was $2.4B, up 34% YoY. [Source: report.pdf, chunk 3]",
  "sources": [
    {"doc_id": "doc_a1b2c3d4", "source": "report.pdf", "chunk_count": 2, "top_score": 0.91}
  ],
  "route": "RETRIEVE",
  "chunks_retrieved": 5,
  "chunks_used": 2,
  "token_usage": {"prompt_tokens": 820, "completion_tokens": 64, "total_tokens": 884},
  "latency_ms": 1240,
  "no_result_reason": null,
  "cache_hit": false,
  "timestamp": "2024-06-01T12:00:00Z"
}
```

No-result response (`answer` is `null` when no chunks meet the confidence threshold):
```json
{"answer": null, "no_result_reason": "No chunks met the confidence threshold of 0.35.", ...}
```

**Error codes:** `401` missing header, `422` validation error, `502` LLM provider error, `503` ChromaDB unavailable, `504` LLM timeout.

---

### `GET /logs`

Retrieve operational metrics for the authenticated user.

**Response 200:**
```json
{
  "user_id": "alice",
  "note": "Phase 1: metrics persistence not yet implemented.",
  "request_metrics": {"total_requests": 0, "avg_latency_ms": 0},
  "cache_statistics": {"hit_rate_pct": 0.0, "cache_hits": 0, "cache_misses": 0},
  "route_decisions": {"direct_count": 0, "retrieve_count": 0},
  "documents": {"total_uploaded": 0, "total_chunks_stored": 0}
}
```

---

## Testing

```bash
# All tests
python -m pytest -v

# Unit tests only (no external services)
python -m pytest tests/unit/ -v

# Integration tests (requires ChromaDB + Redis running)
python -m pytest tests/integration/ tests/api/ -v

# With coverage report
python -m pytest --cov=app --cov-report=term-missing
```

**Test counts (M9 baseline):**
- Unit: 524 tests
- Integration: 13 tests
- API: 102 tests
- **Total: 639 tests**

---

## Docker

```bash
# Start all services (builds image on first run)
docker compose up

# Start in background
docker compose up -d

# Follow application logs
docker compose logs -f app

# Rebuild after code changes
docker compose build app && docker compose up app

# Infrastructure only (for local Python development)
docker compose up chromadb redis -d

# Stop services (data preserved in named volumes)
docker compose down

# Stop and delete all persisted data
docker compose down -v

# Backup ChromaDB and Redis
./scripts/backup.sh ./backups
```

**Port mapping:**

| Service | Host port | Container port |
|---|---|---|
| FastAPI app | 8000 | 8000 |
| ChromaDB | 8001 | 8000 |
| Redis | 6379 | 6379 |

---

## Project Structure

```
ai-platform/
├── app/
│   ├── agents/              # RouterAgent (M7)
│   ├── api/routes/          # FastAPI route handlers (M9)
│   ├── cache/               # Redis client + CacheService (M8)
│   ├── config/              # Settings, dependency injection (M1/M9)
│   ├── logging/             # Structured JSON logging (M3)
│   ├── middleware/          # TenantContext + ErrorHandler (M9), RequestLogger (M3)
│   ├── models/              # Domain dataclasses (no HTTP concerns)
│   ├── rag/                 # Context assembler, prompt builder, token utils (M6)
│   │   └── parsers/         # PDF + CSV parsers (M5)
│   ├── repositories/        # ChromaRepository (M4)
│   ├── schemas/             # Pydantic HTTP schemas (M5/M6/M9)
│   ├── services/            # Business logic orchestration (M5–M9)
│   └── vectorstore/         # ChromaDB client singleton (M4)
├── docs/
│   ├── adr/                 # Architecture Decision Records
│   ├── api/                 # API reference
│   └── architecture/        # Architecture overview
├── scripts/
│   ├── backup.sh            # ChromaDB + Redis backup
│   ├── healthcheck.sh       # Docker HEALTHCHECK
│   └── reingest.sh          # Re-embed after model change
├── tests/
│   ├── api/                 # Route handler tests
│   ├── fixtures/            # Binary test fixtures (PDF, CSV)
│   ├── integration/         # ChromaDB integration tests
│   └── unit/                # All unit tests (no infrastructure)
├── .env.example             # All environment variables documented
├── .github/workflows/ci.yml # CI pipeline
├── docker-compose.yml       # Full stack: app + chromadb + redis
├── Dockerfile               # Two-stage production image
├── main.py                  # FastAPI app factory + lifespan
├── pyproject.toml           # ruff, mypy, pytest configuration
├── requirements.txt         # Pinned production dependencies
└── requirements-dev.txt     # Pinned development/test dependencies
```

---

## Implementation Status

| Module | Description | Status |
|---|---|---|
| M1 | Project Scaffold | ✅ Complete |
| M2 | Configuration Validation + Exception Hierarchy | ✅ Complete |
| M3 | Structured JSON Logging Framework | ✅ Complete |
| M4 | ChromaDB Vector Store + Multi-Tenant Isolation | ✅ Complete |
| M5 | Document Ingestion Pipeline (PDF/CSV) | ✅ Complete |
| M6 | RAG Query Engine | ✅ Complete |
| M7 | Router Agent (DIRECT vs RETRIEVE) | ✅ Complete |
| M8 | Redis Cache Layer | ✅ Complete |
| M9 | API Layer (routes, middleware, DI) | ✅ Complete |
| M10 | Deployment Hardening (Docker, CI) | ✅ Complete |

---

## Future Roadmap

- **Phase 2 authentication:** JWT-based auth replacing `X-User-Id` header
- **Metrics persistence:** Populate `GET /logs` from a real metrics store
- **Document management:** `DELETE /upload-doc/{document_id}` endpoint
- **Async ingestion:** Background task queue for large document uploads
- **Ollama integration:** Local LLM provider using Ollama HTTP API
- **Tiktoken integration:** Exact token counting replacing character approximation

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT

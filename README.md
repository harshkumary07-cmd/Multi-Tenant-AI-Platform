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

- **Multi-tenant architecture** -- strict per-user isolation at six layers
- **PDF ingestion** -- pdfplumber extraction, cleaning, 512-token chunking
- **CSV ingestion** -- pandas parsing with column:value row serialisation
- **Semantic search** -- `all-MiniLM-L6-v2` (384-dimensional vectors, CPU-capable)
- **ChromaDB vector storage** -- single collection with metadata-filtered isolation
- **Router Agent** -- rule-based DIRECT vs RETRIEVE routing (deterministic, free)
- **Redis query cache** -- sub-100ms cache hits, normalised key strategy
- **Multi-LLM support** -- OpenAI, Anthropic, local (Ollama) via abstraction layer
- **Structured JSON logging** -- `request_id` propagation via Python `contextvars`
- **Per-request cost tracking** -- token usage and USD estimate per query
- **Docker Compose deployment** -- three-container stack on isolated network
- **CI/CD pipeline** -- lint -> unit -> integration -> isolation -> docker build

---

## Architecture

```
Client
  |
  v
FastAPI  (API + Middleware Layer)
  |-- TenantContextMiddleware   validates X-User-Id, injects user_id
  |-- RequestLoggerMiddleware   stamps request_id, measures latency
  |-- ErrorHandlerMiddleware    maps exceptions to structured responses
  |
  v
Services Layer
  |-- DocumentService    orchestrates PDF/CSV ingestion pipeline
  |-- QueryService       orchestrates cache -> router -> RAG -> response
  |-- RouterAgent        DIRECT vs RETRIEVE (rule-based, deterministic)
  |-- LLMService         provider abstraction (OpenAI / Anthropic / local)
  |
  v
Data Layer
  |-- ChromaDB   single 'documents' collection, tenant: where={"user_id": X}
  |-- Redis      query cache, key: query:{user_id}:{sha256}
  |-- LLM APIs   OpenAI / Anthropic / Ollama
```

See `docs/architecture/` for detailed flow diagrams (added per module).

---

## Tech Stack

| Component | Technology | Version | Purpose |
|---|---|---|---|
| Backend | FastAPI | 0.111.0 | Async HTTP API framework |
| Vector DB | ChromaDB | latest | Embedding storage and similarity search |
| Embeddings | sentence-transformers | 3.0.1 | `all-MiniLM-L6-v2` (384 dims) |
| Cache | Redis | 7 | Query result caching with TTL |
| LLM | OpenAI / Anthropic | latest | Answer generation |
| PDF parsing | pdfplumber | 0.11.0 | Layout-aware text extraction |
| CSV parsing | pandas | 2.2.2 | Tabular data processing |
| Validation | pydantic | 2.7.1 | Request/response schemas + settings |
| Linting | ruff | 0.4.4 | Style and import enforcement |
| Type checking | mypy | 1.10.0 | Static type verification |
| Testing | pytest | 8.2.0 | Unit, integration, and API tests |
| Containers | Docker Compose | v2 | Three-service orchestration |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker Desktop
- An OpenAI or Anthropic API key (optional for M1 -- required from M6 onwards)

### 1. Clone

```bash
git clone https://github.com/yourusername/ai-platform.git
cd ai-platform
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt -r requirements-dev.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Optionally set LLM_API_KEY in .env for Module 6+ features
```

### 5. Start the application

**Option A -- Full Docker stack:**
```bash
docker compose up
```

**Option B -- Local development (infrastructure in Docker):**
```bash
docker compose up chromadb redis -d
uvicorn main:app --reload
```

### 6. Verify

```bash
curl http://localhost:8000/health
# {"status": "ok", "env": "development", "version": "0.1.0"}
```

### 7. Browse the API

Open http://localhost:8000/docs for interactive documentation.

---

## Environment Variables

Copy `.env.example` to `.env` and configure as needed.

| Variable | Default | Required | Description |
|---|---|---|---|
| `APP_ENV` | `development` | No | `development` / `staging` / `production` |
| `LLM_API_KEY` | `changeme` | M6+ | OpenAI or Anthropic API key |
| `CHROMA_HOST` | `localhost` | No | `chromadb` in Docker Compose |
| `REDIS_HOST` | `localhost` | No | `redis` in Docker Compose |
| `CHUNK_SIZE_TOKENS` | `512` | No | Tokens per document chunk |
| `RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.35` | No | Minimum cosine similarity score |
| `REDIS_CACHE_TTL_SECONDS` | `1800` | No | Cache entry TTL (seconds) |
| `MAX_UPLOAD_SIZE_MB` | `50` | No | Maximum file upload size |

See `.env.example` for the complete list with descriptions.

---

## API Reference

| Method | Endpoint | Description | Module |
|---|---|---|---|
| `GET` | `/health` | Application health check | M1 |
| `POST` | `/user` | Register a new user (tenant) | M9 |
| `POST` | `/upload-doc` | Upload PDF or CSV document | M9 |
| `POST` | `/query` | Submit a natural language query | M9 |
| `GET` | `/logs` | Retrieve operational metrics | M9 |

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=app --cov-report=html

# Run by category
pytest tests/unit/           # fast, no infrastructure
pytest tests/integration/    # requires ChromaDB + Redis
pytest tests/api/            # full HTTP stack

# Run only tenant isolation tests (critical security tests)
pytest tests/integration/ -k "isolation" -v
```

---

## Project Structure

```
ai-platform/
├── main.py                 Application entrypoint (uvicorn main:app)
├── app/
│   ├── api/routes/         FastAPI route handlers
│   ├── services/           Business logic orchestration
│   ├── repositories/       All I/O with external systems
│   ├── rag/                RAG pipeline (parsing, chunking, retrieval)
│   ├── agents/             Router Agent (DIRECT vs RETRIEVE)
│   ├── middleware/         Cross-cutting HTTP concerns
│   ├── schemas/            Pydantic HTTP contracts
│   ├── models/             Internal domain objects
│   ├── config/             Settings and dependency injection
│   └── logging/            Structured JSON logger factory
├── tests/
│   ├── unit/               Fast tests, no infrastructure
│   ├── integration/        Real ChromaDB + Redis tests
│   ├── api/                HTTP endpoint tests
│   └── fixtures/           Test PDFs and CSVs
└── docs/adr/               Architecture Decision Records
```

---

## Docker

```bash
docker compose up                        # start all services
docker compose up chromadb redis -d      # infrastructure only
docker compose logs -f app               # follow app logs
docker compose down                      # stop (data preserved)
docker compose down -v                   # stop and DELETE all data
```

---

## Implementation Status

| Module | Description | Status |
|---|---|---|
| M1 | Project Scaffold | Complete |
| M2 | Configuration Layer | Pending |
| M3 | Logging Layer | Pending |
| M4 | Vector Database Layer | Pending |
| M5 | Document Ingestion | Pending |
| M6 | RAG Query Engine | Pending |
| M7 | Router Agent | Pending |
| M8 | Redis Cache | Pending |
| M9 | API Layer | Pending |
| M10 | Deployment and Hardening | Pending |

---

## Future Roadmap

- Phase 2 JWT authentication (replace `X-User-Id` header)
- Async upload processing (Celery + Redis queue)
- Hybrid search (BM25 + vector similarity)
- Cross-encoder re-ranking for retrieved chunks
- Streaming LLM responses via server-sent events
- Per-user rate limiting

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

---

## License

MIT License -- see [LICENSE](LICENSE) for details.

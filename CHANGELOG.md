# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

---

## [0.10.0] -- Module 10: Deployment Hardening

### Added
- `Dockerfile` -- two-stage production image; model baked in at build time
- `docker-compose.yml` -- full stack: app + ChromaDB + Redis with healthchecks
- `.dockerignore` -- excludes credentials, caches, test fixtures from build context
- `.github/workflows/ci.yml` -- four-job CI pipeline: lint, unit-tests, integration, docker-build
- `scripts/backup.sh` -- ChromaDB volume + Redis RDB backup with 7-day retention
- `docs/architecture/overview.md` -- architecture overview document
- `docs/api/reference.md` -- full API reference with request/response examples

### Modified
- `README.md` -- complete rewrite: Docker quick start, full API reference, architecture diagram
- `CHANGELOG.md` -- added all module entries

---

## [0.9.0] -- Module 9: API Layer

### Added
- `app/middleware/tenant_context.py` -- validates X-User-Id header; sets request.state.user_id
- `app/middleware/error_handler.py` -- maps PlatformError subclasses to HTTP status codes
- `tests/api/test_query_route.py` -- 34 tests for POST /query
- `tests/api/test_upload_route.py` -- 27 tests for POST /upload-doc
- `tests/api/test_user_and_misc_routes.py` -- 24 tests for POST /user, GET /logs, middleware

### Modified
- `app/api/routes/query.py` -- 501 stub replaced with CachedQueryService wiring
- `app/api/routes/upload.py` -- 501 stub replaced with DocumentService wiring
- `app/api/routes/user.py` -- 501 stub replaced with Phase 1 stateless registration
- `app/api/routes/logs.py` -- 501 stub replaced with structured 200 metrics skeleton
- `app/config/dependencies.py` -- all DI providers: get_current_user_id, get_cached_query_service, get_document_service, etc.
- `app/schemas/query_response.py` -- cache_hit: bool field added
- `main.py` -- TenantContextMiddleware and ErrorHandlerMiddleware registered
- `pyproject.toml` -- middleware modules added to mypy ignore_errors

---

## [0.8.0] -- Module 8: Redis Cache Layer

### Added
- `app/cache/redis_client.py` -- singleton connection pool; initialise_redis, close_redis_client, get_redis_client
- `app/cache/cache_service.py` -- build_query_cache_key (SHA-256), CacheService.get/set_result/invalidate_user_cache
- `app/services/cached_query_service.py` -- CachedQueryService wrapping RoutedQueryService
- `tests/unit/test_cache_layer.py` -- 58 unit tests across 9 test classes

### Modified
- `requirements.txt` -- redis==5.0.4 activated
- `pyproject.toml` -- app.cache.redis_client, app.cache.cache_service added to mypy ignore_errors
- `app/models/query_result.py` -- cache_hit: bool = False field added
- `app/services/document_service.py` -- optional cache_service param; invalidate_user_cache called post-upload
- `main.py` -- initialise_redis and close_redis_client wired into lifespan
- `tests/conftest.py` -- Redis initialisation patched in client fixture

---

## [0.7.0] -- Module 7: Router Agent

### Added
- `app/agents/router_agent.py` -- RouteDecision frozen dataclass; RouterAgent with 5 ordered rules
- `app/services/routed_query_service.py` -- RoutedQueryService: RouterAgent + QueryService coordinator
- `tests/unit/test_router_agent.py` -- 137 tests; 50-case parametrized fixture suite (ADR-003)

### Modified
- `app/rag/prompt_builder.py` -- DIRECT_SYSTEM_PROMPT and build_direct_messages() added
- `app/agents/__init__.py` -- docstring updated
- `app/services/__init__.py` -- docstring updated
- `tests/unit/test_rag_pipeline.py` -- TestBuildDirectMessages class added

---

## [0.6.0] -- Module 6: RAG Query Engine

### Added
- `app/models/query_result.py` -- QueryResult, SourceReference, TokenUsage frozen dataclasses
- `app/schemas/query_request.py` -- POST /query request pydantic schema
- `app/schemas/query_response.py` -- POST /query 200 response pydantic schema
- `app/rag/context_assembler.py` -- filter/rank/budget/deduplicate retrieved chunks
- `app/rag/prompt_builder.py` -- build [system, context, user] message list
- `app/rag/token_utils.py` -- single source of truth for token estimation
- `app/services/llm_service.py` -- LLMProvider ABC + Local/OpenAI/Anthropic + factory
- `app/services/query_service.py` -- RAG query pipeline orchestrator
- `tests/unit/test_rag_pipeline.py` -- 93 unit tests

---

## [0.5.0] -- Module 5: Document Ingestion Pipeline

### Added
- `app/rag/parsers/pdf_parser.py` -- pdfplumber extraction and cleaning
- `app/rag/parsers/csv_parser.py` -- pandas parsing with row serialisation
- `app/services/chunking_service.py` -- RecursiveCharacterTextSplitter wrapper
- `app/services/embedding_service.py` -- SentenceTransformer singleton
- `app/services/document_service.py` -- full ingestion pipeline orchestrator
- `app/models/document.py` -- DocumentRecord domain model
- `app/schemas/upload_request.py` -- POST /upload-doc form schema
- `app/schemas/upload_response.py` -- POST /upload-doc 201 response schema
- `tests/unit/test_ingestion_pipeline.py` -- 62 tests
- `tests/fixtures/sample.pdf`, `sample.csv`, `corrupt.pdf` -- binary test fixtures

---

## [0.4.0] -- Module 4: ChromaDB Vector Store

### Added
- `app/vectorstore/client.py` -- ChromaDB HttpClient singleton
- `app/vectorstore/tenant.py` -- chunk metadata constants and builders
- `app/models/chunk.py` -- Chunk and ChunkResult domain models
- `app/repositories/chroma_repository.py` -- ChromaRepository with user_id isolation
- `tests/integration/test_chroma_repository.py` -- 13 integration tests
- `tests/unit/test_chroma_repository_unit.py` -- 32 unit tests

---

## [0.3.0] -- Module 3: Structured Logging Framework

### Added
- `app/logging/context.py` -- LogContext frozen dataclass; contextvars request context
- `app/logging/formatters.py` -- JSONFormatter
- `app/logging/timing.py` -- start_timer, elapsed_ms, LatencyTracker
- `app/middleware/request_logger.py` -- RequestLoggerMiddleware
- `tests/unit/test_logging_context.py` -- 17 tests
- `tests/unit/test_json_formatter.py` -- 28 tests
- `tests/unit/test_timing.py` -- 15 tests
- `tests/unit/test_request_logger_middleware.py` -- 12 tests

---

## [0.2.0] -- Module 2: Configuration Validation + Exception Hierarchy

### Added
- `app/models/exceptions.py` -- PlatformError hierarchy (15 typed exceptions)
- `app/config/settings.py` -- validate_startup_config (9 rules), get_settings_summary
- `tests/unit/test_settings.py` -- 40 tests
- `tests/unit/test_exceptions.py` -- 56 tests
- `docs/adr/ADR-001-single-collection.md`
- `docs/adr/ADR-002-synchronous-upload.md`
- `docs/adr/ADR-003-rule-based-router.md`

---

## [0.1.0] -- Module 1: Project Scaffold

### Added
- Complete project folder structure following clean architecture principles
- `main.py` -- FastAPI app factory with lifespan hooks and router registration
- `app/config/settings.py` -- pydantic BaseSettings with type validation
- `app/logging/logger.py` -- structured logger factory
- `app/api/routes/health.py` -- GET /health (fully implemented)
- `app/api/routes/user.py`, `upload.py`, `query.py`, `logs.py` -- stubs
- `pyproject.toml` -- ruff, mypy, pytest configuration
- `requirements.txt`, `requirements-dev.txt` -- pinned dependencies
- `.env.example` -- all environment variables documented
- `.gitignore`, `CONTRIBUTING.md`
- `scripts/healthcheck.sh`, `scripts/reingest.sh`
- `tests/conftest.py`, `tests/api/test_health_endpoint.py`

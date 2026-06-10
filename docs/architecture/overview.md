# Architecture Overview

## Service Graph

```
Client
  │
  ▼
TenantContextMiddleware   ← validates X-User-Id, stamps request.state.user_id
  │
RequestLoggerMiddleware   ← stamps request_id, logs entry/exit
  │
ErrorHandlerMiddleware    ← maps PlatformError → HTTP status codes
  │
Route Handler             ← FastAPI async endpoint
  │
  ├─ POST /upload-doc ──→ DocumentService
  │                           ├── parse_pdf / parse_csv
  │                           ├── chunk_text (RecursiveCharacterTextSplitter)
  │                           ├── embed_chunks (SentenceTransformer)
  │                           ├── ChromaRepository.add_chunks(user_id=...)
  │                           └── CacheService.invalidate_user_cache(user_id)
  │
  └─ POST /query ────────→ CachedQueryService
                               ├── CacheService.get(user_id, query)  [HIT → return]
                               └── RoutedQueryService.query()
                                       ├── RouterAgent.decide()
                                       │       ├── Rule 1: no documents → DIRECT
                                       │       ├── Rule 2: filename signal → RETRIEVE
                                       │       ├── Rule 3: RETRIEVE keywords → RETRIEVE
                                       │       ├── Rule 4: DIRECT keywords → DIRECT
                                       │       └── Rule 5: ambiguous → RETRIEVE (default)
                                       ├── DIRECT path: build_direct_messages → LLM
                                       └── RETRIEVE path: QueryService.query()
                                               ├── embed_single(query)
                                               ├── ChromaRepository.search_chunks(user_id=...)
                                               ├── assemble_context (threshold + budget)
                                               ├── build_messages (system + context + query)
                                               └── LLMProvider.generate()
```

## Multi-Tenant Isolation

Six independent enforcement layers prevent any cross-tenant data access:

1. **Header validation** — `TenantContextMiddleware` returns 401 on missing/blank `X-User-Id`
2. **Request state injection** — `get_current_user_id()` dependency reads from `request.state`, never from request body
3. **Repository filter** — every `ChromaRepository` method requires `user_id` as a positional argument; all ChromaDB queries include `where={"user_id": {"$eq": user_id}}`
4. **Cache key namespace** — keys are `query:{user_id}:{hash}`, making cross-tenant collisions structurally impossible
5. **Cache invalidation scope** — `invalidate_user_cache(user_id)` scans `query:{user_id}:*` only
6. **No-result path** — when retrieval returns zero results for a user, the LLM is not called; it cannot draw on another user's data

## Key Design Decisions

See `docs/adr/` for the full Architecture Decision Records:

- **ADR-001**: Single ChromaDB collection with `user_id` metadata filter (vs per-user collections)
- **ADR-002**: Synchronous document upload (vs async queue) — 201 means immediately queryable
- **ADR-003**: Deterministic rule-based router (vs LLM-based routing) — zero latency, zero cost, zero hallucination risk on routing

## Token Estimation

All token arithmetic passes through `app/rag/token_utils.py`. The current implementation uses a 4-chars-per-token approximation. To upgrade to exact tokenisation (e.g. tiktoken for OpenAI, Anthropic tokeniser for Claude), replace `estimate_tokens()` in that module — zero changes to callers.

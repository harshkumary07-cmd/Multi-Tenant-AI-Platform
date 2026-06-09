"""
Business logic orchestration layer.

Services coordinate repositories, embedding models, and external APIs.
They work in domain models (app/models/), not HTTP schemas (app/schemas/).
Services never import from app.api.

Services added per module:
    M5: chunking_service.py        -- text -> list[Chunk]
    M5: embedding_service.py       -- list[Chunk] -> list[Chunk] with vectors; embed_single()
    M5: document_service.py        -- full ingestion pipeline orchestrator
    M6: llm_service.py             -- LLMProvider ABC + LocalProvider + OpenAI + Anthropic
    M6: query_service.py           -- RAG query pipeline orchestrator
    M7: routed_query_service.py    -- RoutedQueryService: RouterAgent + QueryService coordinator
    M8: cached_query_service.py    -- CachedQueryService: Redis caching wrapper
"""

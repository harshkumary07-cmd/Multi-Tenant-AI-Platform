"""
Business logic orchestration layer.

Services coordinate repositories, embedding models, and external APIs.
They work in domain models (app/models/), not HTTP schemas (app/schemas/).
Services never import from app.api.

Services added per module:
    M5: chunking_service.py   -- text -> list[Chunk]
    M5: embedding_service.py  -- list[Chunk] -> list[Chunk] with vectors
    M5: document_service.py   -- full ingestion pipeline orchestrator
    M6: query_service.py      -- RAG query orchestrator
    M6: llm_service.py        -- LLM provider abstraction
    M6: cost_tracker.py       -- per-request cost recording
"""

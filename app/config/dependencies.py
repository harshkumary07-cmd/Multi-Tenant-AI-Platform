"""
FastAPI dependency injection providers.

This module exposes all Depends() providers used across route handlers.
Centralising DI here means:
    1. Route handlers import from one location -- no circular imports.
    2. Tests override from one location -- clean test isolation.
    3. Future providers (ChromaDB client, Redis pool) are added here
       without modifying any existing import.

Usage in route handlers:
    from fastapi import Depends
    from app.config.dependencies import get_settings

    @router.get("/example")
    async def example(settings: Settings = Depends(get_settings)):
        ...

Usage in tests (dependency override):
    from app.config.dependencies import get_settings
    from app.config.settings import Settings

    test_settings = Settings(APP_ENV="development")
    app.dependency_overrides[get_settings] = lambda: test_settings
"""

# Re-export get_settings so route handlers have a single import location.
# Implementation lives in settings.py to avoid circular imports.
from app.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]

# ---------------------------------------------------------------------------
# Future dependency providers -- added in their respective modules
# ---------------------------------------------------------------------------
#
# Module 4 -- ChromaDB:
#   async def get_chroma_client() -> chromadb.HttpClient: ...
#   async def get_chroma_repository() -> ChromaRepository: ...
#
# Module 6 -- LLM:
#   async def get_llm_service() -> LLMService: ...
#
# Module 8 -- Redis:
#   async def get_redis_client() -> redis.Redis: ...
#   async def get_cache_service() -> CacheService: ...
#
# Module 9 -- Request state helpers:
#   async def get_current_user_id(request: Request) -> str: ...

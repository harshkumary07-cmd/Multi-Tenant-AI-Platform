"""
FastAPI dependency injection providers.

This module exposes all Depends() providers used across route handlers.
Centralising DI here means:
    1. Route handlers have a single import location -- no circular imports.
    2. Tests override from a single location -- clean test isolation.
    3. Future providers (ChromaDB client, Redis pool) are added here
       without modifying any existing imports.

Pattern -- route handler usage:
    from fastapi import Depends
    from app.config.dependencies import get_settings

    @router.post("/example")
    async def example(settings: Settings = Depends(get_settings)) -> ...:
        host = settings.CHROMA_HOST

Pattern -- test override:
    from app.config.dependencies import get_settings
    from app.config.settings import Settings

    def override_settings() -> Settings:
        return Settings(APP_ENV="development", LLM_API_KEY="test-key")

    app.dependency_overrides[get_settings] = override_settings

    # After the test:
    app.dependency_overrides.clear()
"""

# Re-export get_settings so route handlers have a single import location.
# Implementation lives in settings.py to avoid circular imports.
from app.config.settings import (
    Settings,
    get_settings,
    get_settings_summary,
    validate_startup_config,
)

__all__ = [
    "Settings",
    "get_settings",
    "get_settings_summary",
    "validate_startup_config",
]

# ---------------------------------------------------------------------------
# Future dependency providers -- added in their respective modules
# ---------------------------------------------------------------------------
#
# Module 4 -- ChromaDB:
#   async def get_chroma_client() -> chromadb.HttpClient: ...
#   async def get_chroma_repository(
#       client: chromadb.HttpClient = Depends(get_chroma_client),
#   ) -> ChromaRepository: ...
#
# Module 6 -- LLM:
#   async def get_llm_service(
#       settings: Settings = Depends(get_settings),
#   ) -> LLMService: ...
#
# Module 8 -- Redis:
#   async def get_redis_client() -> redis.Redis: ...
#   async def get_cache_service(
#       redis: redis.Redis = Depends(get_redis_client),
#   ) -> CacheService: ...
#
# Module 9 -- Request state helpers:
#   async def get_current_user_id(request: Request) -> str:
#       """Extract validated user_id from request.state (set by TenantContextMiddleware)."""
#       user_id: str = request.state.user_id
#       return user_id

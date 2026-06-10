"""
FastAPI dependency injection providers.

This module exposes all Depends() providers used across route handlers.
Centralising DI here means:
    1. Route handlers have a single import location -- no circular imports.
    2. Tests override from a single location -- clean test isolation.
    3. Future providers are added here without modifying existing imports.

Pattern -- route handler usage:
    from fastapi import Depends, Request
    from app.config.dependencies import get_current_user_id, get_cached_query_service

    @router.post("/query")
    async def submit_query(
        body: QueryRequest,
        user_id: str = Depends(get_current_user_id),
        service: CachedQueryService = Depends(get_cached_query_service),
    ) -> ...:
        result = service.query(user_id, body.query, body.top_k)

Pattern -- test override:
    from app.config.dependencies import get_cached_query_service
    from main import app

    app.dependency_overrides[get_cached_query_service] = lambda: mock_service
    # After test:
    app.dependency_overrides.clear()
"""

from fastapi import Request

from app.agents.router_agent import RouterAgent
from app.cache.cache_service import CacheService
from app.cache.redis_client import get_redis_client
from app.config.settings import Settings, get_settings
from app.repositories.chroma_repository import ChromaRepository
from app.services.cached_query_service import CachedQueryService
from app.services.document_service import DocumentService
from app.services.llm_service import LLMProvider, create_llm_provider
from app.services.query_service import QueryService
from app.services.routed_query_service import RoutedQueryService
from app.vectorstore.client import get_chroma_client, get_or_create_collection

__all__ = [
    "Settings",
    "get_settings",
    "get_current_user_id",
    "get_chroma_repository",
    "get_cache_service",
    "get_llm_provider",
    "get_router_agent",
    "get_query_service",
    "get_routed_query_service",
    "get_cached_query_service",
    "get_document_service",
]


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------


def get_current_user_id(request: Request) -> str:
    """
    Extract the validated user_id from request.state.

    TenantContextMiddleware sets request.state.user_id before any route
    handler runs. This dependency surfaces it with a clean signature.

    Returns:
        str: The tenant identifier for this request.
    """
    return request.state.user_id  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Infrastructure providers
# ---------------------------------------------------------------------------


def get_chroma_repository() -> ChromaRepository:
    """
    Return a ChromaRepository backed by the singleton ChromaDB client.

    The collection is the application-wide singleton initialised in main.py.
    Construction is cheap (ChromaRepository holds a reference, no I/O).
    """
    settings = get_settings()
    client = get_chroma_client()
    collection = get_or_create_collection(client, settings.CHROMA_COLLECTION_NAME)
    return ChromaRepository(collection)


def get_cache_service() -> CacheService:
    """Return a CacheService backed by the singleton Redis connection pool."""
    settings = get_settings()
    redis_client = get_redis_client()
    return CacheService(
        client=redis_client,
        ttl_seconds=settings.REDIS_CACHE_TTL_SECONDS,
        empty_result_ttl_seconds=settings.REDIS_EMPTY_RESULT_TTL_SECONDS,
    )


def get_llm_provider() -> LLMProvider:
    """Return the configured LLM provider instance."""
    settings = get_settings()
    return create_llm_provider(
        provider_name=settings.LLM_PROVIDER,
        model_name=settings.LLM_MODEL_NAME,
        api_key=settings.LLM_API_KEY.get_secret_value(),
        timeout_seconds=settings.LLM_TIMEOUT_SECONDS,
    )


# ---------------------------------------------------------------------------
# Service providers
# ---------------------------------------------------------------------------


def get_router_agent() -> RouterAgent:
    """Return a RouterAgent with the ChromaDB repository for doc-count checks."""
    return RouterAgent(repository=get_chroma_repository())


def get_query_service() -> QueryService:
    """Return a QueryService with all RAG pipeline dependencies."""
    return QueryService(
        repository=get_chroma_repository(),
        llm_provider=get_llm_provider(),
        settings=get_settings(),
    )


def get_routed_query_service() -> RoutedQueryService:
    """Return a RoutedQueryService with RouterAgent + QueryService."""
    settings = get_settings()
    repository = get_chroma_repository()
    llm_provider = get_llm_provider()
    router = RouterAgent(repository=repository)
    query_service = QueryService(
        repository=repository,
        llm_provider=llm_provider,
        settings=settings,
    )
    return RoutedQueryService(
        router=router,
        query_service=query_service,
        llm_provider=llm_provider,
        settings=settings,
    )


def get_cached_query_service() -> CachedQueryService:
    """
    Return the top-level query service used by the POST /query route.

    Wraps RoutedQueryService with Redis caching.
    Cache failures are silent (handled inside CacheService).
    """
    return CachedQueryService(
        routed_service=get_routed_query_service(),
        cache_service=get_cache_service(),
    )


def get_document_service() -> DocumentService:
    """
    Return a DocumentService with ChromaRepository and CacheService.

    Cache invalidation runs after successful upload (inside DocumentService).
    """
    return DocumentService(
        repository=get_chroma_repository(),
        settings=get_settings(),
        cache_service=get_cache_service(),
    )

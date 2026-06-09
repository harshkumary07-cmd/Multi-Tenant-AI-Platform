"""
Cached query service -- wraps RoutedQueryService with Redis-backed caching.

This is the top-level service that Module 9 (API layer) wires to the
POST /query route handler. The wrapping chain is:

    CachedQueryService
        └── RoutedQueryService
                ├── RouterAgent
                └── QueryService

Cache behaviour:
    1. Check cache: build_query_cache_key(user_id, query) -> Redis GET
    2. HIT:  return cached QueryResult (with cache_hit=True) immediately.
             No embedding, no ChromaDB query, no LLM call.
    3. MISS: delegate to RoutedQueryService.query()
    4. Write the result to cache (regardless of route or answer presence)
    5. Return the result

Cache failures:
    If Redis is unavailable, all cache operations silently fail (cache miss,
    no write). The request is served by RoutedQueryService without caching.
    No error is surfaced to the user.

RoutedQueryService is NOT modified:
    This class wraps it -- does not extend or inherit from it.
    RoutedQueryService.query() is called unchanged.

Module 9 extension note:
    The query route handler will construct CachedQueryService with real
    dependencies. For testing, inject mock CacheService and
    RoutedQueryService instances.
"""

from app.cache.cache_service import CacheService
from app.logging.logger import get_logger
from app.logging.timing import LatencyTracker
from app.models.query_result import QueryResult
from app.services.routed_query_service import RoutedQueryService

logger = get_logger(__name__)


class CachedQueryService:
    """
    Adds Redis caching to the RoutedQueryService pipeline.

    Args:
        routed_service: RoutedQueryService to delegate misses to.
        cache_service:  CacheService for get/set operations.
    """

    def __init__(
        self,
        routed_service: RoutedQueryService,
        cache_service: CacheService,
    ) -> None:
        self._routed = routed_service
        self._cache = cache_service

    def query(
        self,
        user_id: str,
        query_text: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """
        Serve a query from cache if available, otherwise run the full pipeline.

        Args:
            user_id:    Tenant identifier.
            query_text: The user's natural language query.
            top_k:      Optional top-k override passed to RoutedQueryService
                        on a cache miss. Ignored on cache hit.

        Returns:
            QueryResult: cache_hit=True if served from cache,
                         cache_hit=False if served by the pipeline.
        """
        tracker = LatencyTracker()

        # ------------------------------------------------------------------
        # Step 1: Cache lookup
        # ------------------------------------------------------------------
        cached = self._cache.get(user_id, query_text)
        tracker.checkpoint("cache_lookup")

        if cached is not None:
            logger.info(
                "query served from cache",
                extra={
                    "event": "QUERY_CACHE_HIT",
                    "user_id": user_id,
                    "route": cached.route,
                    "has_answer": cached.has_answer,
                    **tracker.to_log_fields(),
                },
            )
            return cached

        # ------------------------------------------------------------------
        # Step 2: Pipeline execution (cache miss)
        # ------------------------------------------------------------------
        result = self._routed.query(
            user_id=user_id,
            query_text=query_text,
            top_k=top_k,
        )
        tracker.checkpoint("pipeline")

        # ------------------------------------------------------------------
        # Step 3: Write to cache
        # ------------------------------------------------------------------
        self._cache.set_result(user_id, query_text, result)
        tracker.checkpoint("cache_write")

        logger.info(
            "query served from pipeline",
            extra={
                "event": "QUERY_CACHE_MISS",
                "user_id": user_id,
                "route": result.route,
                "has_answer": result.has_answer,
                "cache_hit": False,
                **tracker.to_log_fields(),
            },
        )
        return result

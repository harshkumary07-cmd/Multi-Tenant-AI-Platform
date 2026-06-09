"""
Cache service for query result caching.

Responsibilities:
    - Build deterministic, privacy-preserving cache keys
    - Serialise QueryResult to JSON for Redis storage
    - Deserialise cached JSON back to QueryResult
    - Invalidate all cache entries for a user on document upload
    - Treat every Redis failure as a cache miss (never propagate)

Cache key format:
    query:{user_id}:{sha256_hex[:16]}

    The query string is normalised (lowercased, whitespace collapsed) before
    hashing. This makes "what is revenue?" and "What is Revenue ?" hit the
    same cache entry.

    SHA-256 is truncated to 16 hex characters (64 bits). At realistic query
    volumes the collision probability is negligible (~1 in 18 quintillion).
    The hash makes keys fixed-length and prevents query content from
    appearing in Redis key listings (privacy preservation).

Serialisation:
    dataclasses.asdict() converts QueryResult (and nested SourceReference,
    TokenUsage) to a plain dict. json.dumps() serialises it to a string.
    The datetime timestamp is stored as an ISO 8601 string.
    Deserialisation reconstructs the nested dataclasses from the dict.

Cache invalidation:
    Uses Redis SCAN with pattern query:{user_id}:* to find and delete all
    entries for a user. SCAN is non-blocking (unlike KEYS) and safe for
    production Redis instances with large key counts.

Failure policy (critical):
    Every public method catches ALL exceptions and treats them as misses or
    silent no-ops. The cache is an optimisation. A Redis outage must never
    cause a user-facing error. Cache failures are logged at WARNING level
    with event=CACHE_ERROR for operator alerting.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

import redis

from app.logging.logger import get_logger
from app.models.query_result import QueryResult, SourceReference, TokenUsage

logger = get_logger(__name__)

# Key namespace prefix. All query cache entries share this prefix.
# Other future namespaces: "upload:", "session:", etc.
_KEY_PREFIX = "query"

# Number of SHA-256 hex characters to use in the cache key.
# 16 chars = 64 bits = negligible collision probability.
_HASH_LENGTH = 16


def build_query_cache_key(user_id: str, query_text: str) -> str:
    """
    Build a deterministic, privacy-preserving cache key for a query.

    Normalises the query (lowercase + whitespace collapse) before hashing
    so that trivially equivalent queries share the same cache entry.

    Args:
        user_id:    Tenant identifier. Scopes the key to one user.
        query_text: The raw query string from the user.

    Returns:
        str: Cache key in the format "query:{user_id}:{sha256_16}".

    Examples:
        build_query_cache_key("u1", "What is revenue?")
        -> "query:u1:7c7373fabfa90823"

        build_query_cache_key("u1", "what is revenue?")
        -> "query:u1:7c7373fabfa90823"  # same key -- normalised
    """
    normalised = " ".join(query_text.lower().split())
    digest = hashlib.sha256(normalised.encode()).hexdigest()[:_HASH_LENGTH]
    return f"{_KEY_PREFIX}:{user_id}:{digest}"


def _serialise_result(result: QueryResult) -> str:
    """
    Serialise a QueryResult to a JSON string for Redis storage.

    Converts the frozen dataclass (and all nested dataclasses) to a plain
    dict, then serialises to JSON. The datetime timestamp is stored as an
    ISO 8601 string with timezone information.

    Args:
        result: The QueryResult to serialise.

    Returns:
        str: JSON string representation of the result.
    """
    raw = dataclasses.asdict(result)

    def default(obj: Any) -> Any:
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

    return json.dumps(raw, default=default)


def _deserialise_result(data: str) -> QueryResult:
    """
    Deserialise a JSON string back to a QueryResult.

    Reconstructs the full nested dataclass structure from the JSON dict.
    Marks the result as a cache hit.

    Args:
        data: JSON string previously produced by _serialise_result().

    Returns:
        QueryResult: Reconstructed result with cache_hit=True.

    Raises:
        ValueError: If the JSON is malformed or missing required fields.
                    The caller catches this and treats it as a miss.
    """
    raw = json.loads(data)

    sources = [
        SourceReference(**s)
        for s in raw.get("sources", [])
    ]

    token_usage = TokenUsage(**raw["token_usage"])

    timestamp = datetime.fromisoformat(raw["timestamp"])
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)

    return QueryResult(
        query=raw["query"],
        user_id=raw["user_id"],
        answer=raw.get("answer"),
        sources=sources,
        route=raw["route"],
        chunks_retrieved=raw["chunks_retrieved"],
        chunks_used=raw["chunks_used"],
        token_usage=token_usage,
        latency_ms=raw["latency_ms"],
        no_result_reason=raw.get("no_result_reason"),
        timestamp=timestamp,
        cache_hit=True,
    )


class CacheService:
    """
    Redis-backed query result cache.

    All public methods catch Redis exceptions and treat them as misses
    or silent no-ops. The cache never causes user-facing errors.

    Args:
        client:                  redis.Redis instance from get_redis_client().
        ttl_seconds:             TTL for answer entries (settings.REDIS_CACHE_TTL_SECONDS).
        empty_result_ttl_seconds: TTL for no-result entries
                                  (settings.REDIS_EMPTY_RESULT_TTL_SECONDS).
    """

    def __init__(
        self,
        client: redis.Redis,
        ttl_seconds: int,
        empty_result_ttl_seconds: int,
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._empty_ttl = empty_result_ttl_seconds

    def get(self, user_id: str, query_text: str) -> QueryResult | None:
        """
        Look up a cached result for this user and query.

        Returns None on any error (connection failure, deserialization
        failure, key not found). Never raises.

        Args:
            user_id:    Tenant identifier.
            query_text: The raw query string.

        Returns:
            QueryResult with cache_hit=True, or None on miss/error.
        """
        key = build_query_cache_key(user_id, query_text)
        try:
            data = self._client.get(key)
            if data is None:
                logger.debug(
                    "cache miss",
                    extra={"event": "CACHE_MISS", "user_id": user_id, "key": key},
                )
                return None

            result = _deserialise_result(data)
            logger.info(
                "cache hit",
                extra={"event": "CACHE_HIT", "user_id": user_id, "key": key},
            )
            return result

        except Exception as exc:
            logger.warning(
                "cache get error -- treating as miss",
                extra={
                    "event": "CACHE_ERROR",
                    "operation": "get",
                    "user_id": user_id,
                    "key": key,
                    "error": str(exc),
                    "cache_available": False,
                },
            )
            return None

    def set_result(self, user_id: str, query_text: str, result: QueryResult) -> None:
        """
        Store a query result in the cache.

        Uses the shorter TTL for no-result entries so they expire quickly
        after new documents are uploaded.

        Fails silently on any error.

        Args:
            user_id:    Tenant identifier.
            query_text: The raw query string.
            result:     The QueryResult to cache.
        """
        key = build_query_cache_key(user_id, query_text)
        ttl = self._empty_ttl if result.is_no_result else self._ttl

        try:
            serialised = _serialise_result(result)
            self._client.setex(key, ttl, serialised)
            logger.debug(
                "cache set",
                extra={
                    "event": "CACHE_SET",
                    "user_id": user_id,
                    "key": key,
                    "ttl_seconds": ttl,
                    "is_no_result": result.is_no_result,
                },
            )
        except Exception as exc:
            logger.warning(
                "cache set error -- continuing without cache write",
                extra={
                    "event": "CACHE_ERROR",
                    "operation": "set",
                    "user_id": user_id,
                    "key": key,
                    "error": str(exc),
                    "cache_available": False,
                },
            )

    def invalidate_user_cache(self, user_id: str) -> int:
        """
        Delete all cached query results for a user.

        Called after a successful document upload so that subsequent
        queries use the freshly ingested content rather than stale results.

        Uses SCAN to find matching keys non-blocking. Safe for production
        Redis instances with large key counts.

        Args:
            user_id: Tenant identifier.

        Returns:
            int: Number of keys deleted. 0 on error or no matches.
        """
        pattern = f"{_KEY_PREFIX}:{user_id}:*"
        deleted = 0

        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    self._client.delete(*keys)
                    deleted += len(keys)
                if cursor == 0:
                    break

            logger.info(
                "cache invalidated",
                extra={
                    "event": "CACHE_INVALIDATED",
                    "user_id": user_id,
                    "keys_deleted": deleted,
                },
            )
            return deleted

        except Exception as exc:
            logger.warning(
                "cache invalidation error -- continuing",
                extra={
                    "event": "CACHE_ERROR",
                    "operation": "invalidate",
                    "user_id": user_id,
                    "pattern": pattern,
                    "error": str(exc),
                    "cache_available": False,
                },
            )
            return 0

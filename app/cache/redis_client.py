"""
Redis client singleton.

Provides a single connection pool for the lifetime of the application process.
The pool is created once in the lifespan startup hook and closed at shutdown.

Design:
    redis.ConnectionPool is used rather than redis.Redis directly. A pool
    manages a set of underlying TCP connections and hands them out on demand.
    redis.Redis is a thin wrapper that borrows a connection from the pool for
    each command and returns it immediately after. This is more efficient than
    creating a new TCP connection per command.

    The pool is stored as a module-level singleton and accessed via
    get_redis_client(). All callers receive the same Redis client wrapping
    the same pool -- no duplicate connections.

Sync client rationale:
    All service-layer calls in this codebase are synchronous (ChromaDB,
    embedding, LLM). The Redis client follows the same pattern. Redis
    operations complete in <5ms and do not hold the GIL during network I/O,
    making them safe to call from async route handlers without blocking the
    event loop for a meaningful duration. Upgrading to redis.asyncio later
    is a drop-in change to this module only.

Failure isolation:
    initialise_redis() raises VectorStoreError if the connection pool cannot
    be created. This is intentional -- a failure to initialise Redis at startup
    is a configuration error, not a cache miss. Once initialised, individual
    Redis commands that fail are caught and treated as misses in CacheService.
"""

from __future__ import annotations

import redis

from app.logging.logger import get_logger

logger = get_logger(__name__)

# Module-level singleton pool. Set by initialise_redis(), read by get_redis_client().
_pool: redis.ConnectionPool | None = None


def initialise_redis(host: str, port: int) -> redis.Redis:
    """
    Create the Redis connection pool singleton and verify connectivity.

    Called from main.py lifespan startup hook.

    Args:
        host: Redis server hostname (from settings.REDIS_HOST).
        port: Redis server port (from settings.REDIS_PORT).

    Returns:
        redis.Redis: Client wrapping the new connection pool.

    Raises:
        RuntimeError: If the pool cannot be created or the PING fails.
                      Startup is aborted -- Redis is a required service.
    """
    global _pool

    logger.info(
        "connecting to redis",
        extra={
            "event": "REDIS_CONNECTING",
            "host": host,
            "port": port,
        },
    )

    try:
        _pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            max_connections=20,
        )
        client = redis.Redis(connection_pool=_pool)
        client.ping()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to Redis at {host}:{port}: {exc}. "
            "Ensure Redis is running and REDIS_HOST / REDIS_PORT are correct."
        ) from exc

    logger.info(
        "redis connected",
        extra={
            "event": "REDIS_CONNECTED",
            "host": host,
            "port": port,
        },
    )
    return client


def get_redis_client() -> redis.Redis:
    """
    Return a Redis client backed by the singleton connection pool.

    Must be called after initialise_redis() has run.

    Raises:
        RuntimeError: If called before initialise_redis().
    """
    if _pool is None:
        raise RuntimeError(
            "Redis connection pool has not been initialised. "
            "Ensure initialise_redis() is called in the lifespan startup hook."
        )
    return redis.Redis(connection_pool=_pool)


def close_redis_client() -> None:
    """
    Disconnect all pooled connections and release resources.

    Called from main.py lifespan shutdown hook.
    Safe to call if the pool was never initialised (no-op).
    """
    global _pool
    if _pool is not None:
        try:
            _pool.disconnect()
        except Exception:
            pass
        _pool = None
        logger.info(
            "redis disconnected",
            extra={"event": "REDIS_DISCONNECTED"},
        )

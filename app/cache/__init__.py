"""
Redis client wrapper and cache key construction (implemented in M8).

    redis_client.py  -- singleton connection pool with bypass pattern
    cache_service.py -- key construction, get/set/invalidate operations

Cache failures are caught here and treated as cache misses.
They are never propagated as service-level errors.
"""

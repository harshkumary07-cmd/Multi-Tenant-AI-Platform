"""
Redis client and cache service -- implemented in Module 8.

    redis_client.py  -- singleton connection pool; initialise_redis(),
                        close_redis_client(), get_redis_client()
    cache_service.py -- build_query_cache_key(), CacheService with
                        get(), set_result(), invalidate_user_cache()

Cache failures are always silently treated as misses.
They are never propagated as service-level errors.
"""

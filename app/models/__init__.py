"""
Internal domain objects passed between services and repositories.

Domain models have no HTTP concerns and no pydantic validation overhead.
They are plain Python dataclasses.

Models added per module:
    M2: exceptions.py  -- PlatformError hierarchy (ConfigurationError, etc.)
    M4: chunk.py       -- Chunk (for ingestion) and ChunkResult (for retrieval)

Domain exceptions raised by services are defined in exceptions.py.
Pydantic HTTP contracts (request/response schemas) live in app/schemas/.
"""

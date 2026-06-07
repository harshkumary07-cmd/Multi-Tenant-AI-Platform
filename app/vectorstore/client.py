"""
ChromaDB client management.

Provides:
    get_chroma_client()         -- singleton client connected to ChromaDB
    get_or_create_collection()  -- idempotent collection initialisation
    check_chroma_health()       -- connectivity probe for startup validation
    initialise_chroma()         -- full startup sequence called from lifespan
    close_chroma_client()       -- graceful shutdown called from lifespan

Architecture:
    chromadb.HttpClient() is a factory function returning chromadb.api.ClientAPI.
    The singleton is typed as Optional[chromadb.api.ClientAPI] to avoid the
    runtime TypeError that occurs when using the | None union syntax with
    chromadb's function-based HttpClient.
"""

from __future__ import annotations

import chromadb
import chromadb.api

from app.config.settings import Settings
from app.logging.logger import get_logger
from app.models.exceptions import VectorStoreError
from app.vectorstore.tenant import COSINE_DISTANCE

logger = get_logger(__name__)

# chromadb.HttpClient() returns a chromadb.api.ClientAPI instance.
_client: chromadb.api.ClientAPI | None = None


def get_chroma_client() -> chromadb.api.ClientAPI:
    """
    Return the ChromaDB client singleton.

    Raises:
        VectorStoreError: If called before initialise_chroma().
    """
    if _client is None:
        raise VectorStoreError(
            "ChromaDB client has not been initialised. "
            "Ensure initialise_chroma() is called in the lifespan startup hook."
        )
    return _client


def check_chroma_health(client: chromadb.api.ClientAPI) -> bool:
    """
    Probe ChromaDB connectivity via the heartbeat endpoint.

    Raises:
        VectorStoreError: If the heartbeat raises any exception.
    """
    try:
        client.heartbeat()
        return True
    except Exception as exc:
        raise VectorStoreError(
            f"ChromaDB heartbeat failed: {exc}. "
            "Check that ChromaDB is running at the configured host/port."
        ) from exc


def get_or_create_collection(
    client: chromadb.api.ClientAPI,
    collection_name: str,
) -> chromadb.Collection:
    """
    Return the named collection, creating it with cosine distance if needed.

    Idempotent: safe to call on every startup.

    Raises:
        VectorStoreError: If the existing collection has a different distance
                          metric, or if ChromaDB raises any exception.
    """
    try:
        collection: chromadb.Collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": COSINE_DISTANCE},
        )

        configured_metric = (collection.metadata or {}).get("hnsw:space", "l2")
        if configured_metric != COSINE_DISTANCE:
            raise VectorStoreError(
                f"Collection '{collection_name}' exists with distance metric "
                f"'{configured_metric}' but the application requires '{COSINE_DISTANCE}'. "
                "Delete the collection and restart, or change CHROMA_COLLECTION_NAME."
            )

        logger.info(
            "chromadb collection ready",
            extra={
                "event": "CHROMA_COLLECTION_READY",
                "collection_name": collection_name,
                "distance_metric": COSINE_DISTANCE,
            },
        )
        return collection

    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to get or create collection '{collection_name}': {exc}"
        ) from exc


def initialise_chroma(settings: Settings) -> chromadb.Collection:
    """
    Initialise the ChromaDB client singleton and verify the collection.

    Called from main.py lifespan startup hook.

    Raises:
        VectorStoreError: If connection or collection initialisation fails.
    """
    global _client

    logger.info(
        "connecting to chromadb",
        extra={
            "event": "CHROMA_CONNECTING",
            "host": settings.CHROMA_HOST,
            "port": settings.CHROMA_PORT,
        },
    )

    try:
        _client = chromadb.HttpClient(
            host=settings.CHROMA_HOST,
            port=settings.CHROMA_PORT,
            settings=chromadb.config.Settings(anonymized_telemetry=False),
        )
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to create ChromaDB HttpClient "
            f"({settings.CHROMA_HOST}:{settings.CHROMA_PORT}): {exc}"
        ) from exc

    check_chroma_health(_client)

    logger.info(
        "chromadb connected",
        extra={
            "event": "CHROMA_CONNECTED",
            "host": settings.CHROMA_HOST,
            "port": settings.CHROMA_PORT,
        },
    )

    return get_or_create_collection(_client, settings.CHROMA_COLLECTION_NAME)


def close_chroma_client() -> None:
    """
    Close the ChromaDB client and release the connection pool.

    Called from main.py lifespan shutdown hook.
    Safe to call if the client was never initialised (no-op).
    """
    global _client
    if _client is not None:
        try:
            _client._client.close()
        except Exception:
            pass
        _client = None
        logger.info(
            "chromadb client closed",
            extra={"event": "CHROMA_DISCONNECTED"},
        )

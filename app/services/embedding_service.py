"""
Embedding service.

Wraps sentence-transformers SentenceTransformer to:
    1. Load the model once as a module-level singleton
    2. Encode chunks in batches, returning float lists
    3. Attach embedding vectors back to Chunk objects

Model: all-MiniLM-L6-v2
    - 80MB on disk
    - CPU-capable (no GPU required)
    - 384-dimensional output vectors
    - Strong semantic similarity performance

Model singleton:
    initialise_embedding_model() is called from the FastAPI lifespan hook
    in main.py before any request is served.

CRITICAL: The model loaded here must be identical to the model used
at query time (Module 6). Changing it requires re-embedding all stored
documents via scripts/reingest.sh.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.logging.logger import get_logger
from app.models.chunk import Chunk
from app.models.exceptions import EmbeddingFailedError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as STModel

logger = get_logger(__name__)

_model: STModel | None = None


def initialise_embedding_model(model_name: str) -> None:
    """
    Load the SentenceTransformer model and store as a singleton.

    Called from the FastAPI lifespan startup hook. Subsequent calls
    with the same model_name are no-ops.

    Args:
        model_name: HuggingFace model identifier from settings.

    Raises:
        EmbeddingFailedError: If the model cannot be loaded.
    """
    global _model

    if _model is not None:
        logger.debug(
            "embedding model already loaded -- skipping",
            extra={"model": model_name},
        )
        return

    logger.info(
        "loading embedding model",
        extra={"event": "EMBEDDING_MODEL_LOADING", "model": model_name},
    )

    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(model_name)

        # Warmup call to confirm the model is functional
        test_output = _model.encode(["warmup"], normalize_embeddings=True)
        dimension = len(test_output[0]) if hasattr(test_output[0], "__len__") else test_output.shape[1]

        logger.info(
            "embedding model loaded",
            extra={
                "event": "EMBEDDING_MODEL_LOADED",
                "model": model_name,
                "dimension": dimension,
            },
        )
    except Exception as exc:
        raise EmbeddingFailedError(
            f"Failed to load embedding model '{model_name}': {exc}. "
            "Check that sentence-transformers is installed and the model name is correct."
        ) from exc


def get_embedding_model() -> STModel:
    """
    Return the embedding model singleton.

    Raises:
        EmbeddingFailedError: If called before initialise_embedding_model().
    """
    if _model is None:
        raise EmbeddingFailedError(
            "Embedding model has not been initialised. "
            "Ensure initialise_embedding_model() is called in the lifespan startup hook."
        )
    return _model


def _to_float_list(vector: object) -> list[float]:
    """Convert a numpy array or list to a Python list of floats."""
    if hasattr(vector, "tolist"):
        return vector.tolist()  # type: ignore[union-attr]
    return list(float(x) for x in vector)  # type: ignore[arg-type]


def embed_chunks(chunks: list[Chunk], batch_size: int) -> list[Chunk]:
    """
    Generate embedding vectors for a list of chunks in batches.

    Attaches the resulting float vectors to each Chunk's embedding field.
    The input list is modified in place and also returned.

    Args:
        chunks:     Chunks with non-empty text fields. Modified in place.
        batch_size: Chunks per encode() call (from settings).

    Returns:
        list[Chunk]: Same list with embedding fields populated.

    Raises:
        EmbeddingFailedError: If encode() raises or model is not initialised.
        ValueError:           If chunks is empty.
    """
    if not chunks:
        raise ValueError("embed_chunks called with empty list.")

    model = get_embedding_model()
    total = len(chunks)
    batch_start = 0

    try:
        for batch_start in range(0, total, batch_size):
            batch = chunks[batch_start : batch_start + batch_size]
            texts = [c.text for c in batch]

            raw = model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            for i, chunk in enumerate(batch):
                chunk.embedding = _to_float_list(raw[i])

            logger.debug(
                "embedding batch complete",
                extra={
                    "batch_start": batch_start,
                    "batch_size": len(batch),
                    "total": total,
                },
            )

    except EmbeddingFailedError:
        raise
    except Exception as exc:
        raise EmbeddingFailedError(
            f"Embedding failed on batch starting at index {batch_start}: {exc}"
        ) from exc

    logger.info(
        "chunks embedded",
        extra={
            "event": "CHUNKS_EMBEDDED",
            "total_chunks": total,
            "batch_size": batch_size,
        },
    )

    return chunks


def embed_single(text: str) -> list[float]:
    """
    Generate an embedding vector for a single text string.

    Used at query time (Module 6). Must use the same model as embed_chunks().

    Args:
        text: The query string to embed.

    Returns:
        list[float]: A 384-dimensional embedding vector.

    Raises:
        EmbeddingFailedError: If the model raises or is not initialised.
    """
    model = get_embedding_model()
    try:
        raw = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return _to_float_list(raw[0])
    except Exception as exc:
        raise EmbeddingFailedError(f"Failed to embed query text: {exc}") from exc

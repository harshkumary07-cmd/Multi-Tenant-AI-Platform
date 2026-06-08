"""
Text chunking service.

Splits a cleaned text string into overlapping chunks using
RecursiveCharacterTextSplitter from langchain-text-splitters.

The splitter tries to break text on paragraph boundaries first (\n\n),
then sentence boundaries (\n), then clause boundaries (. ), then word
boundaries ( ), and finally character boundaries as a last resort. This
priority order preserves semantic coherence within chunks.

Chunk size is measured in characters, not tokens, for performance.
The configured CHUNK_SIZE_TOKENS value is treated as approximately
equal to characters at a 1:1 ratio for the all-MiniLM-L6-v2 model
(which uses subword tokenisation -- 512 tokens ≈ 400-600 characters
for typical English text). For production workloads with precise token
budgets, replace the character splitter with a tiktoken-based splitter
in Module 6.

Chunk minimum length:
    Chunks shorter than MIN_CHUNK_LENGTH characters after stripping
    are discarded. Short chunks (e.g. from table cells, headers, or
    repeated separator lines that survived cleaning) add noise to
    the vector store without providing meaningful retrieval signal.
"""

from app.logging.logger import get_logger
from app.models.chunk import Chunk
from app.models.exceptions import EmptyDocumentError
from app.vectorstore.tenant import build_chunk_id

logger = get_logger(__name__)

# Minimum chunk length in characters. Chunks shorter than this are dropped.
MIN_CHUNK_LENGTH = 20

# Separator priority order for RecursiveCharacterTextSplitter.
# The splitter tries each separator in order, preferring longer structural
# breaks over shorter ones.
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_text(
    text: str,
    doc_id: str,
    user_id: str,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """
    Split text into overlapping Chunk objects.

    Uses RecursiveCharacterTextSplitter to split the text, then wraps
    each text segment in a Chunk domain object with the appropriate
    metadata. Chunks shorter than MIN_CHUNK_LENGTH are discarded.

    Args:
        text:          Cleaned document text from the parser.
        doc_id:        Document identifier. Used to construct chunk_ids.
        user_id:       Tenant identifier. Stored on every chunk.
        source:        Original filename. Stored on every chunk.
        chunk_size:    Maximum characters per chunk (from settings).
        chunk_overlap: Character overlap between adjacent chunks (from settings).

    Returns:
        list[Chunk]: Non-empty list of Chunk objects without embeddings.
                     Embeddings are populated by EmbeddingService.

    Raises:
        EmptyDocumentError: If splitting produces zero usable chunks.
                            This can happen if the input text is very short
                            or consists entirely of short strings.
        ValueError:         If chunk_overlap >= chunk_size (caught by
                            validate_startup_config in M2 -- should not
                            reach here in production).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        length_function=len,
        is_separator_regex=False,
        add_start_index=False,
    )

    raw_chunks: list[str] = splitter.split_text(text)

    chunks: list[Chunk] = []
    discarded = 0

    for _, chunk_text_str in enumerate(raw_chunks):
        stripped = chunk_text_str.strip()
        if len(stripped) < MIN_CHUNK_LENGTH:
            discarded += 1
            continue

        chunks.append(
            Chunk(
                chunk_id=build_chunk_id(doc_id, len(chunks)),
                doc_id=doc_id,
                user_id=user_id,
                source=source,
                chunk_index=len(chunks),
                text=stripped,
            )
        )

    logger.debug(
        "text chunked",
        extra={
            "doc_id": doc_id,
            "user_id": user_id,
            "source": source,
            "raw_chunks": len(raw_chunks),
            "usable_chunks": len(chunks),
            "discarded_chunks": discarded,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
    )

    if not chunks:
        raise EmptyDocumentError(
            f"Chunking '{source}' produced zero usable chunks from "
            f"{len(text)} characters of text. "
            "The document may contain only very short text segments."
        )

    return chunks

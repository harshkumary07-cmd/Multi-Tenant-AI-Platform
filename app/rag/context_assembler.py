"""
Context assembler for the RAG pipeline.

Transforms a list of raw ChunkResult objects (from ChromaDB similarity search)
into a structured context string ready for prompt construction.

Responsibilities:
    1. Filter: discard chunks below the confidence threshold
    2. Sort:   order remaining chunks by score descending (most relevant first)
    3. Budget: enforce a character limit so the LLM context window is never exceeded
    4. Format: produce a labelled context string with chunk source headers
    5. Deduplicate: build a SourceReference list with one entry per document

Character budget vs token budget:
    Exact token counting requires tiktoken (listed as a future dependency).
    This module uses a character approximation: char_limit = token_budget * 4.
    For typical English text chunked at 512 tokens, 1 token ≈ 4 characters.
    This approximation is conservative -- it will never exceed the real token
    limit, though it may under-use the budget by 10-20%.

    When tiktoken is added, upgrade _char_budget() and token counting without
    changing the public interface of this module.

Context format (each included chunk):
    [Source: {filename}, chunk {index}]
    {chunk text}

    ---

This format gives the LLM clear source attribution for every passage,
enabling it to produce cited answers.

Raises:
    NoRelevantChunksError: If zero chunks pass the confidence threshold.
                           QueryService catches this and returns a
                           no-result response rather than calling the LLM.
"""

from dataclasses import dataclass

from app.logging.logger import get_logger
from app.models.chunk import ChunkResult
from app.models.exceptions import NoRelevantChunksError
from app.models.query_result import SourceReference

logger = get_logger(__name__)

# Approximate token-to-character ratio for English text.
# Used to convert a token budget into a character limit.
CHARS_PER_TOKEN: int = 4

# Default context window budget in tokens.
# 2000 tokens leaves headroom for the system prompt (~300 tokens)
# and the query itself (~50 tokens) within a 4096-token context window.
DEFAULT_CONTEXT_TOKEN_BUDGET: int = 2000

# Separator between chunks in the assembled context string.
CHUNK_SEPARATOR: str = "\n\n---\n\n"

# Header template for each chunk in the context string.
# Format: [Source: filename.pdf, chunk 3]
CHUNK_HEADER_TEMPLATE: str = "[Source: {source}, chunk {index}]"


@dataclass
class AssembledContext:
    """
    Result of context assembly for a single query.

    Attributes:
        context_text:   The formatted context string to include in the prompt.
        chunks_used:    ChunkResult objects that were included (passed threshold
                        and fit within the character budget).
        sources:        Deduplicated source reference list -- one entry per
                        unique document, with chunk count and top score.
        char_count:     Total characters in context_text.
    """

    context_text: str
    chunks_used: list[ChunkResult]
    sources: list[SourceReference]
    char_count: int

    @property
    def chunk_count(self) -> int:
        """Number of chunks included in the context."""
        return len(self.chunks_used)


def _char_budget(token_budget: int) -> int:
    """Convert a token budget to a character limit."""
    return token_budget * CHARS_PER_TOKEN


def _format_chunk(chunk: ChunkResult) -> str:
    """Format a single chunk as a labelled context block."""
    header = CHUNK_HEADER_TEMPLATE.format(
        source=chunk.source,
        index=chunk.chunk_index,
    )
    return f"{header}\n{chunk.text}"


def _build_sources(chunks: list[ChunkResult]) -> list[SourceReference]:
    """
    Build a deduplicated SourceReference list from included chunks.

    One SourceReference per unique doc_id. Aggregates chunk count
    and tracks the highest score across all chunks from each document.

    Args:
        chunks: Chunks that were included in the context.

    Returns:
        list[SourceReference]: Sorted by top_score descending.
    """
    doc_chunks: dict[str, list[ChunkResult]] = {}
    for chunk in chunks:
        doc_chunks.setdefault(chunk.doc_id, []).append(chunk)

    sources: list[SourceReference] = []
    for doc_id, doc_chunk_list in doc_chunks.items():
        sources.append(
            SourceReference(
                doc_id=doc_id,
                source=doc_chunk_list[0].source,
                chunk_count=len(doc_chunk_list),
                top_score=max(c.score for c in doc_chunk_list),
            )
        )

    return sorted(sources, key=lambda s: s.top_score, reverse=True)


def assemble_context(
    chunks: list[ChunkResult],
    threshold: float,
    token_budget: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> AssembledContext:
    """
    Filter, rank, budget, and format retrieved chunks into a context string.

    Pipeline:
        1. Filter chunks below the confidence threshold
        2. Sort passing chunks by score descending
        3. Greedily add chunks until the character budget is exhausted
        4. Format the selected chunks with source headers
        5. Build deduplicated source references

    Args:
        chunks:       Raw ChunkResult list from ChromaRepository.search_chunks().
        threshold:    Minimum cosine similarity score to include a chunk.
                      Chunks below this score are discarded.
        token_budget: Maximum tokens of context to include. Converted to a
                      character budget internally.

    Returns:
        AssembledContext: The formatted context, included chunks, and sources.

    Raises:
        NoRelevantChunksError: If zero chunks pass the confidence threshold.
                               This signals QueryService to return a no-result
                               response without calling the LLM.
    """
    # Step 1: Filter by confidence threshold
    passing = [c for c in chunks if c.is_above_threshold(threshold)]

    logger.debug(
        "context assembly filtering",
        extra={
            "chunks_in": len(chunks),
            "chunks_passing_threshold": len(passing),
            "threshold": threshold,
        },
    )

    if not passing:
        raise NoRelevantChunksError(
            f"No chunks met the confidence threshold of {threshold:.2f}. "
            f"Examined {len(chunks)} candidate chunks. "
            "The uploaded documents do not appear to contain information "
            "relevant to this query."
        )

    # Step 2: Sort by score descending
    ranked = sorted(passing, key=lambda c: c.score, reverse=True)

    # Step 3: Apply character budget
    char_limit = _char_budget(token_budget)
    selected: list[ChunkResult] = []
    total_chars = 0
    separator_chars = len(CHUNK_SEPARATOR)

    for chunk in ranked:
        formatted = _format_chunk(chunk)
        # Account for the separator that will be added between chunks
        needed = len(formatted) + (separator_chars if selected else 0)
        if total_chars + needed > char_limit:
            # Budget exhausted -- stop adding chunks
            break
        selected.append(chunk)
        total_chars += needed

    logger.debug(
        "context assembly budget",
        extra={
            "chunks_passing": len(passing),
            "chunks_selected": len(selected),
            "char_budget": char_limit,
            "chars_used": total_chars,
            "token_budget": token_budget,
        },
    )

    # Step 4: Format context string
    context_text = CHUNK_SEPARATOR.join(_format_chunk(c) for c in selected)

    # Step 5: Build source references
    sources = _build_sources(selected)

    logger.info(
        "context assembled",
        extra={
            "event": "CONTEXT_ASSEMBLED",
            "chunks_retrieved": len(chunks),
            "chunks_used": len(selected),
            "sources_count": len(sources),
            "char_count": len(context_text),
        },
    )

    return AssembledContext(
        context_text=context_text,
        chunks_used=selected,
        sources=sources,
        char_count=len(context_text),
    )

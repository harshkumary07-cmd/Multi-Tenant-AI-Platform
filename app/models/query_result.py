"""
Query result domain model.

QueryResult is the internal representation of a completed query pipeline run.
It travels from QueryService to the route handler (Module 9), which converts
it into the pydantic QueryResponse schema for the HTTP response.

It is a frozen dataclass -- immutable once produced by QueryService. This
prevents accidental mutation during routing decisions (Module 7) or response
serialisation (Module 9).
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass(frozen=True)
class SourceReference:
    """
    A single source document cited in a query result.

    Produced by the context assembler when deduplicating retrieved chunks
    by their source document. One SourceReference per unique doc_id.

    Attributes:
        doc_id:         Document identifier.
        source:         Original filename (e.g. "q3_report.pdf").
        chunk_count:    Number of chunks from this document included in context.
        top_score:      Highest cosine similarity score among included chunks.
    """

    doc_id: str
    source: str
    chunk_count: int
    top_score: float


@dataclass(frozen=True)
class TokenUsage:
    """
    Token consumption for a single LLM call.

    Attributes:
        prompt_tokens:     Tokens in the full prompt (system + context + query).
        completion_tokens: Tokens in the model's response.
        total_tokens:      prompt_tokens + completion_tokens.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    @classmethod
    def zero(cls) -> "TokenUsage":
        """Return a zero-usage instance for no-result responses."""
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    @classmethod
    def from_counts(cls, prompt: int, completion: int) -> "TokenUsage":
        """Construct from prompt and completion counts."""
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        )


@dataclass(frozen=True)
class QueryResult:
    """
    Result of a completed RAG query pipeline run.

    Produced by QueryService.query() and passed to the route handler
    (Module 9), which converts it to a QueryResponse pydantic schema.

    The result represents one of three outcomes:

    1. Successful retrieval and answer (answer is not None, sources is non-empty):
       route="RETRIEVE", chunks were found above threshold, LLM generated answer.

    2. No-result response (answer is None, no_result_reason is set):
       route="RETRIEVE", no chunks above threshold -- LLM was NOT called.
       The LLM never invents answers when documents don't contain the information.

    3. Direct LLM response (answer is not None, sources is empty):
       route="DIRECT" -- Router Agent (Module 7) bypassed retrieval entirely.
       Module 6 does not produce this outcome -- QueryService always retrieves.
       Module 7 will extend QueryService to support DIRECT routing.

    Attributes:
        query:            The original query string.
        user_id:          Tenant identifier.
        answer:           LLM-generated answer, or None for no-result responses.
        sources:          Deduplicated list of source documents cited.
        route:            "RETRIEVE" for all Module 6 responses.
        chunks_retrieved: Total chunks returned by ChromaDB before threshold filter.
        chunks_used:      Chunks that passed the threshold and were sent to LLM.
        token_usage:      Token consumption for this query.
        latency_ms:       Total wall-clock time from query entry to result ready.
        no_result_reason: Set when answer is None. Explains why no answer was given.
        timestamp:        UTC timestamp when the result was produced.
    """

    query: str
    user_id: str
    answer: str | None
    sources: list[SourceReference]
    route: Literal["RETRIEVE", "DIRECT"]
    chunks_retrieved: int
    chunks_used: int
    token_usage: TokenUsage
    latency_ms: int
    no_result_reason: str | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(tz=UTC)
    )

    @property
    def has_answer(self) -> bool:
        """Return True if the query produced an answer."""
        return self.answer is not None

    @property
    def is_no_result(self) -> bool:
        """Return True if no relevant chunks were found."""
        return self.answer is None and self.no_result_reason is not None

"""
Query response schema.

Pydantic models for the 200 response from POST /query.
Built from a QueryResult domain model in the route handler (Module 9).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class SourceReferenceSchema(BaseModel):
    """Source document cited in the query response."""

    doc_id: str = Field(description="Document identifier.")
    source: str = Field(description="Original filename.")
    chunk_count: int = Field(description="Chunks from this document used in context.")
    top_score: float = Field(description="Highest similarity score among used chunks.")


class TokenUsageSchema(BaseModel):
    """Token consumption for the LLM call."""

    prompt_tokens: int = Field(description="Tokens in the prompt.")
    completion_tokens: int = Field(description="Tokens in the completion.")
    total_tokens: int = Field(description="Total tokens consumed.")


class QueryResponse(BaseModel):
    """
    Response body for POST /query.

    Returned with HTTP 200 OK for both successful answers and
    no-result responses. A no-result response has answer=null and
    a non-null no_result_reason.

    Attributes:
        query:            The original query string.
        answer:           LLM-generated answer, or null for no-result responses.
        sources:          Source documents cited in the answer.
        route:            Routing decision -- "RETRIEVE" or "DIRECT".
        chunks_retrieved: Total chunks retrieved from ChromaDB.
        chunks_used:      Chunks that passed the confidence threshold.
        token_usage:      Token consumption breakdown.
        latency_ms:       Total query latency in milliseconds.
        no_result_reason: Explains why answer is null, when applicable.
        timestamp:        UTC timestamp of query completion.
    """

    query: str = Field(description="Original query string.")
    answer: str | None = Field(
        default=None,
        description="LLM-generated answer, or null if no relevant chunks found.",
    )
    sources: list[SourceReferenceSchema] = Field(
        default_factory=list,
        description="Source documents cited in the answer.",
    )
    route: Literal["RETRIEVE", "DIRECT"] = Field(
        description="Routing decision for this query.",
    )
    chunks_retrieved: int = Field(description="Chunks retrieved from ChromaDB.")
    chunks_used: int = Field(description="Chunks passed to the LLM context.")
    token_usage: TokenUsageSchema = Field(description="Token consumption.")
    latency_ms: int = Field(description="Total query latency in milliseconds.")
    no_result_reason: str | None = Field(
        default=None,
        description="Reason for null answer, when applicable.",
    )
    cache_hit: bool = Field(
        default=False,
        description="True if this result was served from Redis cache.",
    )
    timestamp: datetime = Field(description="UTC timestamp of query completion.")

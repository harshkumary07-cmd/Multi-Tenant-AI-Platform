"""
Query request schema.

Pydantic model for the POST /query request body.
Validated at the API boundary before QueryService receives the query.
"""

from typing import Annotated

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Request body for POST /query.

    The user_id is NOT part of this schema. It is always read from
    request.state.user_id (set by TenantContextMiddleware in Module 9),
    never from the request body. Including user_id in the body would
    allow a client to attempt to query another user's documents.

    Attributes:
        query:  The natural language question. 1-2000 characters.
        top_k:  Optional override for the number of chunks to retrieve.
                If not set, falls back to settings.RETRIEVAL_TOP_K.
                Range: 1-20.
    """

    query: Annotated[str, Field(min_length=1, max_length=2000)] = Field(
        description="Natural language query. 1-2000 characters.",
    )
    top_k: Annotated[int, Field(ge=1, le=20)] | None = Field(
        default=None,
        description=(
            "Optional: number of chunks to retrieve. "
            "Overrides settings.RETRIEVAL_TOP_K if set. Range: 1-20."
        ),
    )

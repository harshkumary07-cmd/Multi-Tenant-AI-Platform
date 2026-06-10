"""
User registration route.

POST /user -- register a new tenant identity (Phase 1: stateless).

Phase 1 design (current):
    - No user database. user_id is returned as-is after format validation.
    - 201 is returned for any valid format, idempotently.
    - X-User-Id header is NOT required on this route (user is being created).

Phase 2 design (future):
    - JWT issued on registration.
    - UserAlreadyExistsError raised on duplicate.
    - X-User-Id replaced by Authorization: Bearer <token>.

HTTP codes:
    201 Created              -- registration accepted
    422 Unprocessable Entity -- user_id fails format validation
"""

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator

from app.logging.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class UserCreateRequest(BaseModel):
    """
    Request body for POST /user.

    user_id: 1-64 characters, alphanumeric plus hyphens and underscores.
    """

    user_id: Annotated[str, Field(min_length=1, max_length=64)] = Field(
        description="Unique tenant identifier. 1-64 chars, alphanumeric + hyphens/underscores.",
    )

    @field_validator("user_id")
    @classmethod
    def validate_user_id_format(cls, value: str) -> str:
        import re
        if not re.match(r"^[a-zA-Z0-9_\-]+$", value):
            raise ValueError(
                "user_id must contain only alphanumeric characters, hyphens, "
                "and underscores."
            )
        return value


class UserCreateResponse(BaseModel):
    """Response body for POST /user."""

    user_id: str = Field(description="The registered tenant identifier.")
    created_at: datetime = Field(description="UTC timestamp of registration.")
    message: str = Field(description="Confirmation message.")


@router.post(
    "",
    response_model=UserCreateResponse,
    status_code=201,
    summary="Register a new user (tenant)",
    description=(
        "Phase 1: stateless registration. No auth header required. "
        "Returns 201 for any valid user_id format. "
        "Phase 2 will add JWT issuance and duplicate detection."
    ),
)
async def create_user(body: UserCreateRequest) -> UserCreateResponse:
    """
    Register a new tenant identity.

    Phase 1: validates the user_id format and returns it with a timestamp.
    No persistence. Idempotent -- calling twice with the same user_id
    produces the same result.
    """
    logger.info(
        "user registration",
        extra={
            "event": "USER_REGISTERED",
            "user_id": body.user_id,
        },
    )

    return UserCreateResponse(
        user_id=body.user_id,
        created_at=datetime.now(tz=UTC),
        message=(
            f"User '{body.user_id}' registered successfully. "
            "Use X-User-Id: {user_id} header on all subsequent requests."
        ).replace("{user_id}", body.user_id),
    )

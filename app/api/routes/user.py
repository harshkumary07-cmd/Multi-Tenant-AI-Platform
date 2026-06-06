"""
User registration route.

POST /user -- register a new tenant identity.

STATUS: STUB -- returns 501 Not Implemented.
Fully implemented in Module 9 (API Layer).

Planned behaviour (approved architecture):
    Request:  {"user_id": "u1"}
    Response: {"user_id": "u1", "created_at": "..."}
    Codes:    201 Created | 409 Conflict | 422 Validation Error

Design decisions (locked, from approved architecture):
    - Phase 1: user_id-only registration (no email, no password)
    - Phase 2: JWT authentication added in this file without changing downstream
    - No X-User-Id header required -- user is being created here
    - user_id format: 1-64 chars, alphanumeric + hyphens/underscores only
    - Duplicate user_id returns 409 (not 200)
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.post(
    "",
    summary="Register a new user (tenant)",
    description="Phase 1: user_id only. Phase 2: JWT. No auth header required.",
    status_code=501,
)
async def create_user() -> JSONResponse:
    """
    Register a new tenant identity.

    Not yet implemented. Returns 501 until Module 9.
    """
    return JSONResponse(
        status_code=501,
        content={
            "error_code": "NOT_IMPLEMENTED",
            "message": "POST /user will be implemented in Module 9.",
        },
    )

"""
Health check route.

GET /health -- returns application status, environment, and version.

This is the only route fully implemented in Module 1.
All other routes are stubs returning 501 until their modules are built.

Used by:
    - Docker HEALTHCHECK instruction
    - docker-compose depends_on condition: service_healthy
    - Load balancer / reverse proxy health probing
    - CI smoke tests

Design decisions:
    - No authentication required (must be reachable before user creation)
    - Health check does NOT verify ChromaDB or Redis -- process liveness only
    - Request logging suppressed for this endpoint in M3 (avoids poll noise)
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.config.settings import get_settings

router = APIRouter()
settings = get_settings()


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    env: str
    version: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Application health check",
    description=(
        "Returns application status. Used by Docker healthchecks and "
        "load balancer probes. No authentication required."
    ),
)
async def health_check() -> HealthResponse:
    """
    Return the current application health status.

    This endpoint confirms the FastAPI process is alive and able to serve
    requests. It does NOT check ChromaDB, Redis, or LLM provider health.
    Dependency health is tracked via the /logs endpoint metrics.

    Returns:
        HealthResponse: status, environment name, and application version.
    """
    return HealthResponse(
        status="ok",
        env=settings.APP_ENV,
        version="0.1.0",
    )

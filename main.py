"""
Multi-Tenant AI Platform -- Application Entrypoint.

Uvicorn entry point:
    uvicorn main:app --host 0.0.0.0 --port 8000

Responsibilities:
    - Create and configure the FastAPI application instance
    - Register all API routers
    - Define application lifespan (startup / shutdown hooks)
    - Register middleware (order matters -- see inline comments)

Architecture note:
    No business logic lives here. This file is a wiring layer only.
    All logic belongs in app/services/, app/repositories/, or app/agents/.

Why main.py is at project root (not app/main.py):
    uvicorn expects `main:app` by convention. Placing main.py inside app/
    would require `uvicorn app.main:app` -- non-standard and inconsistent
    with every FastAPI tutorial, Docker example, and deployment guide.
    The root file is a wiring layer only; all testable code lives in app/.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, logs, query, upload, user
from app.config.settings import get_settings

settings = get_settings()


# ---------------------------------------------------------------------------
# Lifespan -- startup and shutdown hooks
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Startup tasks (executed before first request is served):
        - M2: validate_startup_config() -- assert no placeholder values in prod
        - M3: configure structured JSON logger
        - M4: initialise ChromaDB client and verify collection exists
        - M5: load sentence-transformers embedding model singleton
        - M8: initialise Redis connection pool

    Shutdown tasks (executed after last request completes):
        - M4: close ChromaDB HTTP client
        - M8: close Redis connection pool

    Each task is added in its respective module. This scaffold
    provides the hook; modules populate it incrementally.
    """
    # ------------------------------------------------------------------ #
    # STARTUP                                                              #
    # ------------------------------------------------------------------ #
    # M2 -- startup config validation added here
    # M3 -- structured logging initialised here
    # M4 -- ChromaDB client and collection verified here
    # M5 -- embedding model loaded here
    # M8 -- Redis pool initialised here

    yield

    # ------------------------------------------------------------------ #
    # SHUTDOWN                                                             #
    # ------------------------------------------------------------------ #
    # M4 -- ChromaDB client closed here
    # M8 -- Redis pool closed here


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns a fully configured FastAPI instance with:
        - All routers registered
        - Middleware stack configured
        - OpenAPI metadata set

    Middleware registration order (FastAPI/Starlette executes in LIFO):
        The LAST middleware registered executes FIRST on incoming requests.

        Registration order  ->  Execution order on request:
          1. CORSMiddleware          (registered 1st) -> executes last
          2. ErrorHandlerMiddleware  (M9, registered 2nd) -> executes 3rd
          3. RequestLoggerMiddleware (M3, registered 3rd) -> executes 2nd
          4. TenantContextMiddleware (M9, registered last) -> executes FIRST

        This ensures TenantContextMiddleware stamps user_id into
        request.state before any route handler or logger fires.
    """
    application = FastAPI(
        title="Multi-Tenant AI Platform",
        description=(
            "Production-grade RAG platform supporting multi-tenant document "
            "upload, semantic search, and AI-powered query answering."
        ),
        version="0.1.0",
        docs_url="/docs" if settings.APP_ENV != "production" else None,
        redoc_url="/redoc" if settings.APP_ENV != "production" else None,
        lifespan=lifespan,
    )

    # ------------------------------------------------------------------
    # Middleware registration
    # Add in reverse execution order (last registered = first to execute).
    # M3 and M9 will add RequestLoggerMiddleware and TenantContextMiddleware.
    # ------------------------------------------------------------------
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tightened in production via env config
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # M9 -- ErrorHandlerMiddleware registered here
    # M3 -- RequestLoggerMiddleware registered here
    # M9 -- TenantContextMiddleware registered here (last = executes first)

    # ------------------------------------------------------------------
    # Router registration
    # ------------------------------------------------------------------
    application.include_router(health.router, tags=["health"])
    application.include_router(user.router, prefix="/user", tags=["users"])
    application.include_router(upload.router, prefix="/upload-doc", tags=["documents"])
    application.include_router(query.router, prefix="/query", tags=["query"])
    application.include_router(logs.router, prefix="/logs", tags=["observability"])

    return application


app = create_app()

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
from app.cache.redis_client import close_redis_client, initialise_redis
from app.config.settings import get_settings, get_settings_summary, validate_startup_config
from app.logging.logger import configure_logging, get_logger
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.middleware.request_logger import RequestLoggerMiddleware
from app.middleware.tenant_context import TenantContextMiddleware
from app.services.embedding_service import initialise_embedding_model
from app.vectorstore.client import close_chroma_client, initialise_chroma

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

    # M3 -- Configure structured JSON logging before anything else.
    # This ensures that the M2 config summary and all subsequent startup
    # log lines are emitted as structured JSON, not plain text.
    configure_logging(settings.LOG_LEVEL)
    _logger = get_logger(__name__)

    # M2 -- Semantic startup validation.
    # Raises ConfigurationError if any production safety rule is violated.
    # The process exits here before serving a single request.
    validate_startup_config(settings)

    # M3 -- Emit structured startup log with masked settings summary.
    summary = get_settings_summary(settings)
    _logger.info(
        "platform starting",
        extra={
            "event": "PLATFORM_STARTUP",
            "app_env": settings.APP_ENV,
            "log_level": settings.LOG_LEVEL,
            "llm_provider": settings.LLM_PROVIDER,
            "settings_summary": summary,
        },
    )

    # M4 -- Initialise ChromaDB: connect, verify collection, validate distance metric.
    # Raises VectorStoreError if ChromaDB is unreachable or misconfigured.
    # The process exits here before serving requests if the vector store is unavailable.
    chroma_collection = initialise_chroma(settings)
    _logger.info(
        "chromadb ready",
        extra={
            "event": "CHROMA_READY",
            "collection": settings.CHROMA_COLLECTION_NAME,
            "collection_count": chroma_collection.count(),
        },
    )

    # M5 -- Load embedding model singleton.
    # This triggers a ~80MB model download on first run if not cached.
    # Subsequent startups load from ~/.cache/huggingface/ in ~2-4 seconds.
    initialise_embedding_model(settings.EMBEDDING_MODEL_NAME)

    # M8 -- Initialise Redis connection pool and verify connectivity.
    initialise_redis(settings.REDIS_HOST, settings.REDIS_PORT)
    _logger.info(
        "redis ready",
        extra={
            "event": "REDIS_READY",
            "host": settings.REDIS_HOST,
            "port": settings.REDIS_PORT,
        },
    )

    yield

    # ------------------------------------------------------------------ #
    # SHUTDOWN                                                             #
    # ------------------------------------------------------------------ #

    # M3 -- Structured shutdown log line.
    _logger.info(
        "platform shutting down",
        extra={"event": "PLATFORM_SHUTDOWN"},
    )
    # M4 -- Close ChromaDB HTTP client and release connection pool.
    close_chroma_client()
    # M8 -- Close Redis connection pool.
    close_redis_client()


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
    # ------------------------------------------------------------------
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Tightened in production via env config
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # M9 -- ErrorHandlerMiddleware: maps domain exceptions to HTTP status codes
    application.add_middleware(ErrorHandlerMiddleware)
    # M3 -- RequestLoggerMiddleware: logs every request entry/exit
    application.add_middleware(RequestLoggerMiddleware)
    # M9 -- TenantContextMiddleware: validates X-User-Id, sets request.state.user_id
    # Registered last so it executes first on every incoming request.
    application.add_middleware(TenantContextMiddleware)

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

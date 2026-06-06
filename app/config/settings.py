"""
Application configuration.

This module is the SINGLE source of truth for all environment variables.
No other module reads os.environ directly.

Loading priority (highest to lowest):
    1. Actual environment variables (shell export or Docker env injection)
    2. .env file (local development only -- never present in production)
    3. Field default values (where defined)

Startup behaviour:
    Settings() is instantiated once via get_settings() at import time.
    pydantic validates every field type at instantiation.
    If a field has the wrong type, the process exits immediately with a
    clear ValidationError naming the offending field.

Secret handling:
    LLM_API_KEY is declared as SecretStr.
    str(settings.LLM_API_KEY)              -> "**********"
    settings.LLM_API_KEY.get_secret_value() -> the real key string
    The real value is only accessed inside llm_service.py (Module 6).
    This prevents accidental exposure in logs and error messages.

LLM_API_KEY default:
    The default is SecretStr("changeme") so the application starts cleanly
    on a fresh clone without any configuration. Module 6 adds a startup
    assertion that refuses to initialise the LLM service when the value is
    still "changeme" and APP_ENV is "production". This separates the concern
    of config loading (here) from the concern of config adequacy (M6).

Usage:
    from app.config.settings import get_settings

    settings = get_settings()
    host = settings.CHROMA_HOST
    key  = settings.LLM_API_KEY.get_secret_value()  # llm_service.py only
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All fields are typed and validated at startup.
    Fields with no default cause an immediate startup abort if missing.
    Currently all fields have defaults, so the app starts on a fresh clone.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",  # Unknown keys in .env are silently ignored
    )

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    APP_ENV: Literal["development", "staging", "production"] = Field(
        default="development",
        description=(
            "Runtime environment. Controls /docs exposure and log verbosity. "
            "Values: development | staging | production"
        ),
    )
    APP_HOST: str = Field(
        default="0.0.0.0",
        description="Host interface for uvicorn to bind to.",
    )
    APP_PORT: int = Field(
        default=8000,
        description="Port for uvicorn to listen on.",
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging verbosity. Use DEBUG in development, INFO in production.",
    )

    # ------------------------------------------------------------------
    # ChromaDB (client initialised in Module 4)
    # ------------------------------------------------------------------
    CHROMA_HOST: str = Field(
        default="localhost",
        description="ChromaDB server hostname. Use service name in Docker Compose.",
    )
    CHROMA_PORT: int = Field(
        default=8001,
        description="ChromaDB HTTP server port.",
    )
    CHROMA_COLLECTION_NAME: str = Field(
        default="documents",
        description=(
            "Single collection name for all tenants. "
            "Tenant isolation is enforced via metadata filtering, "
            "not separate collections. Changing this after ingestion "
            "requires a full re-migration of all vectors."
        ),
    )

    # ------------------------------------------------------------------
    # Redis (client initialised in Module 8)
    # ------------------------------------------------------------------
    REDIS_HOST: str = Field(
        default="localhost",
        description="Redis hostname. Use service name in Docker Compose.",
    )
    REDIS_PORT: int = Field(
        default=6379,
        description="Redis server port.",
    )
    REDIS_CACHE_TTL_SECONDS: int = Field(
        default=1800,
        description=(
            "TTL for cached query results in seconds. "
            "Default: 30 minutes. Balances freshness and LLM cost savings."
        ),
    )
    REDIS_EMPTY_RESULT_TTL_SECONDS: int = Field(
        default=300,
        description=(
            "TTL for no-result cache entries in seconds. "
            "Shorter than normal TTL so new document uploads surface quickly. "
            "Default: 5 minutes."
        ),
    )

    # ------------------------------------------------------------------
    # Embedding model (singleton loaded in Module 5)
    # ------------------------------------------------------------------
    EMBEDDING_MODEL_NAME: str = Field(
        default="all-MiniLM-L6-v2",
        description=(
            "sentence-transformers model name. "
            "Produces 384-dimensional vectors. CPU-capable. "
            "CRITICAL: changing this model after documents have been ingested "
            "requires full re-embedding. Run scripts/reingest.sh."
        ),
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=100,
        description=(
            "Chunks per model.encode() call. "
            "Larger batches are more efficient but use more RAM."
        ),
    )

    # ------------------------------------------------------------------
    # Chunking (used in Module 5)
    # ------------------------------------------------------------------
    CHUNK_SIZE_TOKENS: int = Field(
        default=512,
        description=(
            "Maximum tokens per document chunk. "
            "512 balances retrieval granularity and context richness. "
            "Consider 256 for highly structured tabular data."
        ),
    )
    CHUNK_OVERLAP_TOKENS: int = Field(
        default=50,
        description=(
            "Token overlap between adjacent chunks. "
            "Prevents information loss at chunk boundaries. "
            "Must be less than CHUNK_SIZE_TOKENS."
        ),
    )

    # ------------------------------------------------------------------
    # Retrieval (used in Module 6)
    # ------------------------------------------------------------------
    RETRIEVAL_TOP_K: int = Field(
        default=5,
        description=(
            "Number of candidate chunks to retrieve per query. "
            "Actual chunks used in context may be fewer after confidence filtering."
        ),
    )
    RETRIEVAL_CONFIDENCE_THRESHOLD: float = Field(
        default=0.35,
        description=(
            "Minimum cosine similarity score (0.0-1.0) for a chunk to be "
            "included in the LLM context. Below threshold = no-result response. "
            "Tune without code changes by updating this value and redeploying."
        ),
    )

    # ------------------------------------------------------------------
    # LLM (provider initialised in Module 6)
    # ------------------------------------------------------------------
    LLM_PROVIDER: Literal["openai", "anthropic", "local"] = Field(
        default="openai",
        description="Primary LLM provider. Failover chain: primary -> secondary -> local.",
    )
    LLM_MODEL_NAME: str = Field(
        default="gpt-4o",
        description="Model identifier passed to the LLM provider API.",
    )
    LLM_API_KEY: SecretStr = Field(
        default=SecretStr("changeme"),
        description=(
            "LLM provider API key. Stored as SecretStr -- masked in all logs. "
            "Default 'changeme' allows the app to start without configuration. "
            "Module 6 startup validation refuses to initialise the LLM service "
            "when this is 'changeme' and APP_ENV is 'production'. "
            "Access the real value only via .get_secret_value() in llm_service.py."
        ),
    )
    LLM_TIMEOUT_SECONDS: int = Field(
        default=30,
        description="Max seconds to wait for LLM API response before LLMTimeoutError.",
    )

    # ------------------------------------------------------------------
    # File upload (used in Module 5)
    # ------------------------------------------------------------------
    MAX_UPLOAD_SIZE_MB: int = Field(
        default=50,
        description=(
            "Maximum upload file size in megabytes. "
            "Enforced before reading bytes into memory. "
            "Prevents DoS via oversized file uploads."
        ),
    )
    UPLOAD_TIMEOUT_SECONDS: int = Field(
        default=120,
        description=(
            "Maximum seconds for synchronous upload pipeline. "
            "Covers: parse + chunk + embed + ChromaDB write. "
            "Files near MAX_UPLOAD_SIZE_MB may take 60-90s on CPU."
        ),
    )


@lru_cache
def get_settings() -> Settings:
    """
    Return the cached Settings singleton.

    Uses @lru_cache to ensure Settings() is instantiated exactly once per
    process. All subsequent calls return the same instance without re-reading
    environment variables or the .env file.

    Test isolation:
        Call get_settings.cache_clear() before tests that need fresh settings.
        Override via FastAPI dependency injection:
            app.dependency_overrides[get_settings] = lambda: test_settings

    Returns:
        Settings: The validated application configuration instance.

    Raises:
        pydantic.ValidationError: If any field fails type validation. The
            application exits before serving a single request.
    """
    return Settings()

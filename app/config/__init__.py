"""
Application configuration and dependency injection.

    settings.py     -- pydantic BaseSettings; single source of truth for env vars
    dependencies.py -- FastAPI Depends() providers for DI

No other file reads os.environ directly.
"""

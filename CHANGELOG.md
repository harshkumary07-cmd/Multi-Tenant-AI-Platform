# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)

## [Unreleased]

## [0.1.0] -- Module 1: Project Scaffold

### Added
- Complete project folder structure following clean architecture principles
- `main.py` -- FastAPI app factory with lifespan hooks and router registration
- `app/config/settings.py` -- pydantic BaseSettings with type validation
- `app/config/dependencies.py` -- FastAPI dependency injection providers
- `app/logging/logger.py` -- structured logger factory scaffold
- `app/api/routes/health.py` -- GET /health (200 OK, fully implemented)
- `app/api/routes/user.py` -- POST /user stub (501, implemented in M9)
- `app/api/routes/upload.py` -- POST /upload-doc stub (501, implemented in M9)
- `app/api/routes/query.py` -- POST /query stub (501, implemented in M9)
- `app/api/routes/logs.py` -- GET /logs stub (501, implemented in M9)
- `tests/conftest.py` -- shared fixtures including `unique_user_id` factory
- `tests/api/test_health_endpoint.py` -- 7 health endpoint tests, all passing
- `requirements.txt` -- pinned production dependencies (M1 subset)
- `requirements-dev.txt` -- pinned development and CI dependencies
- `pyproject.toml` -- ruff, mypy, and pytest configuration
- `.env.example` -- all 20 environment variables documented
- `.gitignore` -- Python, Docker, IDE, and secret file exclusions
- `docs/adr/ADR-001-single-collection.md`
- `docs/adr/ADR-002-synchronous-upload.md`
- `docs/adr/ADR-003-rule-based-router.md`
- `CHANGELOG.md`, `CONTRIBUTING.md`, `README.md`

# Contributing

## Development setup

1. Clone the repository
2. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt -r requirements-dev.txt`
4. Copy environment config: `cp .env.example .env`
5. Start infrastructure: `docker compose up chromadb redis -d`
6. Verify: `uvicorn main:app --reload` then `curl localhost:8000/health`

## Before every commit

```bash
ruff check .        # must pass -- zero violations
mypy app/           # must pass -- zero errors
pytest tests/unit/  # must pass -- under 30 seconds
```

## Before opening a PR

```bash
pytest tests/       # full suite (requires docker compose up chromadb redis -d)
```

## Commit message format

Conventional Commits: `type(scope): description`

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`

Examples:
```
feat(router): add filename-based RETRIEVE signal detection
fix(cache): use SCAN instead of KEYS for user cache invalidation
test(isolation): add cross-tenant API attack scenarios
docs(adr): document single-collection design decision
```

## Branch naming

- `feat/description` -- new feature
- `fix/description` -- bug fix
- `docs/description` -- documentation only
- `test/description` -- tests only

## Non-negotiable rules

1. No hardcoded `user_id` values in tests -- use the `unique_user_id` fixture
2. No `os.environ` reads outside `app/config/settings.py`
3. Every new env variable documented in `.env.example`
4. Tenant isolation tests must remain green (hard merge gate)
5. `ruff` and `mypy` must pass before any PR is opened

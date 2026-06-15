# =============================================================================
# Multi-Tenant AI Platform -- Production Dockerfile
# =============================================================================
#
# Two-stage build:
#   Stage 1 (builder): installs Python deps + compiles C extensions into /opt/venv
#   Stage 2 (runtime): copies only the venv + application code; no build tools
#
# The sentence-transformers model (all-MiniLM-L6-v2, ~80MB) is baked into the
# image at build time so startup does not require network access or a warm cache.
#
# Usage:
#   docker build -t ai-platform:latest .
#   docker run -p 8000:8000 --env-file .env ai-platform:latest
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# Build dependencies for packages with C extensions (chromadb, numpy, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Create and activate a virtual environment so Stage 2 can copy it cleanly.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install production dependencies first (layer-cached unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Pre-download the sentence-transformers model so the image is self-contained.
# This bakes the ~80MB model into the image and eliminates cold-start latency
# and Hugging Face network dependency at container startup.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model cached successfully')"

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Runtime system dependency: curl is used by the HEALTHCHECK script.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Copy the complete virtual environment from the builder stage.
# This excludes gcc and all build tools -- final image is ~300MB lighter.
COPY --from=builder /opt/venv /opt/venv

# Copy the Hugging Face model cache from builder.
# Located at /root/.cache/huggingface in the builder stage.
COPY --from=builder /root/.cache /root/.cache

# Always use the venv Python / pip in subsequent RUN and CMD instructions.
ENV PATH="/opt/venv/bin:$PATH"

# Prevent Python from writing .pyc files and buffer stdout/stderr.
# These are critical for log visibility in Docker.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create a non-root user and group to run the application.
# Running as root in a container is a security anti-pattern.
RUN groupadd --system appgroup && \
    useradd --system --gid appgroup --create-home appuser

RUN mkdir -p /home/appuser/.cache && \
    chown -R appuser:appgroup /home/appuser

ENV HOME=/home/appuser

WORKDIR /app

# Copy application source. Excluded by .dockerignore: .git, .env,
# __pycache__, tests/fixtures, uploads, *.pyc, cache directories.
COPY --chown=appuser:appgroup . .

# Create the uploads directory the application writes to.
RUN mkdir -p /app/uploads && chown appuser:appgroup /app/uploads

# Switch to non-root user for all subsequent operations.
USER appuser

# Expose the application port. Matches APP_PORT default and uvicorn binding.
EXPOSE 8000

# Health check using the pre-existing healthcheck script.
# Interval: every 30s. Timeout: 5s. Start period: 60s (model init time).
# Retries: 3 before marking unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD bash scripts/healthcheck.sh

# Production server: uvicorn with a single worker.
# For multi-worker deployments, override CMD with:
#   uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
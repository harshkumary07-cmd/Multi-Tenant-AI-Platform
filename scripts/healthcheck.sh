#!/usr/bin/env bash
# Docker HEALTHCHECK script.
# Returns 0 (healthy) or 1 (unhealthy).
set -euo pipefail
curl -f "http://localhost:${APP_PORT:-8000}/health" \
  --silent --max-time 5 --output /dev/null

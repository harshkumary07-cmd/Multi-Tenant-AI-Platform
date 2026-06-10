#!/usr/bin/env bash
# =============================================================================
# backup.sh -- Backup ChromaDB volume and Redis AOF to a local archive
# =============================================================================
#
# Usage:
#   ./scripts/backup.sh [BACKUP_DIR]
#
# Arguments:
#   BACKUP_DIR   Directory to write archives to. Default: ./backups
#
# Creates two timestamped archives:
#   chroma-backup-YYYYMMDD-HHMMSS.tar.gz  -- ChromaDB persistent volume
#   redis-backup-YYYYMMDD-HHMMSS.rdb      -- Redis RDB snapshot
#
# Requirements:
#   - docker compose services must be running
#   - docker CLI available on PATH
#
# Example cron (daily at 02:00):
#   0 2 * * * /app/scripts/backup.sh /mnt/backups >> /var/log/backup.log 2>&1
# =============================================================================

set -euo pipefail

BACKUP_DIR="${1:-./backups}"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
CHROMA_CONTAINER="ai-platform-chromadb"
REDIS_CONTAINER="ai-platform-redis"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Starting backup to: $BACKUP_DIR"

# Create backup directory if it doesn't exist
mkdir -p "$BACKUP_DIR"

# ---------------------------------------------------------------------------
# ChromaDB backup
# ---------------------------------------------------------------------------
CHROMA_ARCHIVE="$BACKUP_DIR/chroma-backup-${TIMESTAMP}.tar.gz"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Backing up ChromaDB..."

if docker ps --format '{{.Names}}' | grep -q "^${CHROMA_CONTAINER}$"; then
    docker exec "$CHROMA_CONTAINER" \
        tar czf - /chroma/chroma 2>/dev/null \
        > "$CHROMA_ARCHIVE"
    CHROMA_SIZE=$(du -sh "$CHROMA_ARCHIVE" | cut -f1)
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] ChromaDB backup complete: $CHROMA_ARCHIVE ($CHROMA_SIZE)"
else
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] WARNING: ChromaDB container '$CHROMA_CONTAINER' not running -- skipping"
fi

# ---------------------------------------------------------------------------
# Redis backup (BGSAVE + copy RDB)
# ---------------------------------------------------------------------------
REDIS_ARCHIVE="$BACKUP_DIR/redis-backup-${TIMESTAMP}.rdb"

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Backing up Redis..."

if docker ps --format '{{.Names}}' | grep -q "^${REDIS_CONTAINER}$"; then
    # Trigger a background save and wait for it to complete
    docker exec "$REDIS_CONTAINER" redis-cli BGSAVE > /dev/null
    sleep 2

    # Copy the RDB file out of the container
    docker cp "${REDIS_CONTAINER}:/data/dump.rdb" "$REDIS_ARCHIVE" 2>/dev/null || \
    docker cp "${REDIS_CONTAINER}:/data/appendonly.aof" \
        "$BACKUP_DIR/redis-aof-backup-${TIMESTAMP}.aof" 2>/dev/null || \
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] WARNING: Could not copy Redis data file"

    REDIS_SIZE=$(du -sh "$REDIS_ARCHIVE" 2>/dev/null | cut -f1 || echo "0")
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Redis backup complete: $REDIS_ARCHIVE ($REDIS_SIZE)"
else
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] WARNING: Redis container '$REDIS_CONTAINER' not running -- skipping"
fi

# ---------------------------------------------------------------------------
# Cleanup: remove backups older than 7 days
# ---------------------------------------------------------------------------
find "$BACKUP_DIR" -name "chroma-backup-*.tar.gz" -mtime +7 -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "redis-backup-*.rdb" -mtime +7 -delete 2>/dev/null || true
find "$BACKUP_DIR" -name "redis-aof-backup-*.aof" -mtime +7 -delete 2>/dev/null || true

echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Backup complete."

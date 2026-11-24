#!/usr/bin/env sh
# Snapshot Redis RDB into BACKUP_DIR (default: ./backups/redis).
# Usage:
#   REDIS_HOST=localhost REDIS_PORT=6379 ./scripts/backup_redis.sh
#   # or inside compose network:
#   REDIS_HOST=redis ./scripts/backup_redis.sh

set -eu

REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
BACKUP_DIR="${BACKUP_DIR:-./backups/redis}"
KEEP_DAYS="${KEEP_DAYS:-7}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TARGET="${BACKUP_DIR}/dump-${STAMP}.rdb"

mkdir -p "${BACKUP_DIR}"

# REDISCLI_AUTH is read by redis-cli and stays out of argv (unlike -a, which
# shows up in ps). Empty password must not set the variable.
if [ -n "${REDIS_PASSWORD}" ]; then
  REDISCLI_AUTH="${REDIS_PASSWORD}"
  export REDISCLI_AUTH
fi

echo "Triggering BGSAVE on ${REDIS_HOST}:${REDIS_PORT}..."
redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" BGSAVE >/dev/null

# Wait until lastsave advances (max ~60s).
BEFORE="$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" LASTSAVE)"
i=0
while [ "$i" -lt 60 ]; do
  sleep 1
  NOW="$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" LASTSAVE)"
  if [ "${NOW}" != "${BEFORE}" ]; then
    break
  fi
  i=$((i + 1))
done

# Prefer --rdb when available (Redis 6+ redis-cli).
if redis-cli --version >/dev/null 2>&1; then
  if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" --rdb "${TARGET}"; then
    echo "Saved ${TARGET}"
  else
    echo "redis-cli --rdb failed; trying docker volume copy fallback is required manually" >&2
    exit 1
  fi
fi

# Prune old backups.
find "${BACKUP_DIR}" -type f -name 'dump-*.rdb' -mtime "+${KEEP_DAYS}" -delete 2>/dev/null || true
echo "Backup complete: ${TARGET}"
ls -lh "${TARGET}"

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

AUTH_ARGS=""
if [ -n "${REDIS_PASSWORD}" ]; then
  AUTH_ARGS="-a ${REDIS_PASSWORD}"
fi

echo "Triggering BGSAVE on ${REDIS_HOST}:${REDIS_PORT}..."
# shellcheck disable=SC2086
redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ${AUTH_ARGS} BGSAVE >/dev/null

# Wait until lastsave advances (max ~60s).
BEFORE="$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ${AUTH_ARGS} LASTSAVE)"
i=0
while [ "$i" -lt 60 ]; do
  sleep 1
  NOW="$(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ${AUTH_ARGS} LASTSAVE)"
  if [ "${NOW}" != "${BEFORE}" ]; then
    break
  fi
  i=$((i + 1))
done

# Prefer --rdb when available (Redis 6+ redis-cli).
if redis-cli --version >/dev/null 2>&1; then
  # shellcheck disable=SC2086
  if redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" ${AUTH_ARGS} --rdb "${TARGET}"; then
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

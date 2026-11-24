#!/bin/sh
# Собирает REDIS_URL, когда compose задал REDIS_HOST.
# Пароль percent-encoding; пустой пароль — URL без userinfo.
set -e
if [ -n "${REDIS_HOST:-}" ]; then
  REDIS_URL="$(
    python -c 'import os; from app.infra.redis_url import compose_redis_url; print(compose_redis_url(os.environ.get("REDIS_HOST", "redis"), os.environ.get("REDIS_PASSWORD", "")))'
  )"
  export REDIS_URL
fi
exec "$@"

"""Per-user rate limits with Redis sliding window and in-memory fallback.

Картинки — отдельные token bucket ``image_generate`` и ``image_edit``.
Текст, медиа и HTTP-загрузка остаются скользящим окном.
"""

from __future__ import annotations

import asyncio
import math
import time
import uuid
from collections import defaultdict, deque
from typing import Any

import structlog

logger = structlog.get_logger("rate_limiter")

TEXT_LIMIT = 10
TEXT_WINDOW_SECONDS = 60.0
IMAGE_LIMIT = 5
IMAGE_WINDOW_SECONDS = 3600.0
MEDIA_LIMIT = 10
MEDIA_WINDOW_SECONDS = 3600.0
# HTTP загрузки нет в Telegram-лимите: страница и POST идут снаружи бота.
UPLOAD_READ_LIMIT = 60
UPLOAD_READ_WINDOW_SECONDS = 60.0
UPLOAD_WRITE_LIMIT = 12
UPLOAD_WRITE_WINDOW_SECONDS = 600.0

# Atomic sliding-window consume: prune → count → zadd (or deny).
_ALLOW_LUA = """
redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, ARGV[1])
local count = redis.call('ZCARD', KEYS[1])
if count >= tonumber(ARGV[2]) then
  return 0
end
redis.call('ZADD', KEYS[1], ARGV[3], ARGV[4])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[5]))
return 1
"""

# Token bucket. ARGV: capacity, refill_per_min, now, ttl seconds, consume (1/0).
# Hash fields: tokens, ts. Missing key starts full. Peek does not create a key.
_BUCKET_LUA = """
local capacity = tonumber(ARGV[1])
local refill_per_min = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local ttl = tonumber(ARGV[4])
local consume = tonumber(ARGV[5])
local rate = refill_per_min / 60

local raw_tokens = redis.call('HGET', KEYS[1], 'tokens')
local raw_ts = redis.call('HGET', KEYS[1], 'ts')
local tokens
if raw_tokens == false or raw_ts == false then
  tokens = capacity
else
  tokens = tonumber(raw_tokens)
  local ts = tonumber(raw_ts)
  if tokens == nil then
    tokens = capacity
  end
  local elapsed = 0
  if ts ~= nil then
    elapsed = now - ts
    if elapsed < 0 then
      elapsed = 0
    end
  end
  tokens = math.min(capacity, tokens + elapsed * rate)
end

local allow = 0
local retry_after = 0
if tokens >= 1 then
  allow = 1
  if consume == 1 then
    tokens = tokens - 1
  end
else
  local deficit = 1 - tokens
  if deficit < 0 then
    deficit = 0
  end
  if refill_per_min > 0 then
    retry_after = math.ceil(deficit * 60 / refill_per_min)
  else
    retry_after = 1
  end
end

if consume == 1 then
  redis.call(
    'HSET', KEYS[1],
    'tokens', string.format('%.8f', tokens),
    'ts', string.format('%.6f', now)
  )
  redis.call('EXPIRE', KEYS[1], ttl)
end
return {allow, retry_after}
"""

_IMAGE_BUCKET_OPS = frozenset({"image_generate", "image_edit"})


class RateLimiter:
    """Sliding-window limiter for LLM text and image generation.

    When a Redis client is attached, counters survive process restarts.
    Otherwise an in-memory deque is used (single-process only).
    """

    def __init__(self) -> None:
        self._redis: Any | None = None
        self._lock = asyncio.Lock()
        self._text_hits: dict[int, deque[float]] = defaultdict(deque)
        self._image_hits: dict[int, deque[float]] = defaultdict(deque)
        self._media_hits: dict[int, deque[float]] = defaultdict(deque)
        self._upload_hits: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(deque)
        )
        self._image_buckets: dict[tuple[int, str], tuple[float, float]] = {}

    def attach_redis(self, redis_client: Any | None) -> None:
        """Use Redis for durable counters when available."""
        self._redis = redis_client
        if redis_client is not None:
            logger.info("rate_limiter_using_redis")
        else:
            logger.warning("rate_limiter_using_memory")

    async def check_text_limit(self, user_id: int) -> bool:
        """Return True and consume a slot if under the text/LLM limit."""
        return await self._allow(
            user_id,
            operation="text",
            limit=TEXT_LIMIT,
            window=TEXT_WINDOW_SECONDS,
        )

    async def check_image_limit(self, user_id: int) -> bool:
        """Legacy sliding window. Image entry points use ``consume_image_bucket``."""
        return await self._allow(
            user_id,
            operation="image",
            limit=IMAGE_LIMIT,
            window=IMAGE_WINDOW_SECONDS,
        )

    async def consume_image_bucket(self, user_id: int, operation: str) -> bool:
        """Забирает один токен ``image_generate`` или ``image_edit``.

        В production сбой Redis — отказ, без перехода на память процесса.
        """
        capacity, refill, _ttl = self._bucket_params()
        now = time.time()
        if self._redis is not None:
            try:
                allowed, _retry = await self._bucket_redis(
                    user_id,
                    operation,
                    capacity=capacity,
                    refill_per_min=refill,
                    now=now,
                    consume=True,
                )
                return allowed
            except Exception as exc:
                from app.infra.access import is_production

                if is_production():
                    logger.error(
                        "rate_limiter_redis_failed_deny",
                        error=str(exc),
                        operation=operation,
                        user_id=user_id,
                    )
                    return False
                logger.warning(
                    "rate_limiter_redis_failed_fallback_memory",
                    error=str(exc),
                    operation=operation,
                    user_id=user_id,
                )
        allowed, _retry = await self._bucket_memory(
            user_id,
            operation,
            capacity=capacity,
            refill_per_min=refill,
            now=now,
            consume=True,
        )
        return allowed

    async def check_media_limit(self, user_id: int) -> bool:
        """Return True and consume a slot if under the STT/TTS/video limit."""
        return await self._allow(
            user_id,
            operation="media",
            limit=MEDIA_LIMIT,
            window=MEDIA_WINDOW_SECONDS,
        )

    async def check_upload_limit(self, identity: str, *, write: bool) -> bool:
        """Скользящее окно на IP HTTP-загрузки. ``write`` — POST файла или presign."""
        operation = "upload_write" if write else "upload_read"
        limit = UPLOAD_WRITE_LIMIT if write else UPLOAD_READ_LIMIT
        window = UPLOAD_WRITE_WINDOW_SECONDS if write else UPLOAD_READ_WINDOW_SECONDS
        return await self._allow_named(
            identity, operation=operation, limit=limit, window=window
        )

    async def get_upload_retry_after(self, identity: str, *, write: bool) -> float:
        """Секунды до освобождения слота HTTP-загрузки."""
        operation = "upload_write" if write else "upload_read"
        limit = UPLOAD_WRITE_LIMIT if write else UPLOAD_READ_LIMIT
        window = UPLOAD_WRITE_WINDOW_SECONDS if write else UPLOAD_READ_WINDOW_SECONDS
        now = time.time()
        key = self._named_key(identity, operation)
        if self._redis is not None:
            try:
                await self._redis.zremrangebyscore(key, 0, now - window)
                oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                count = await self._redis.zcard(key)
                if count < limit or not oldest:
                    return 0.0
                return max(0.0, float(oldest[0][1]) + window - now)
            except Exception as exc:
                logger.warning("rate_limiter_redis_retry_after_failed", error=str(exc))
        async with self._lock:
            self._sweep_expired(now)
            hits = self._upload_hits[operation][identity]
            self._prune(hits, now, window)
            if not hits:
                self._upload_hits[operation].pop(identity, None)
                return 0.0
            if len(hits) < limit:
                return 0.0
            return max(0.0, hits[0] + window - now)

    async def get_retry_after(self, user_id: int, operation: str) -> float:
        """Seconds until the oldest hit leaves the window (0 if capacity free)."""
        if operation in _IMAGE_BUCKET_OPS:
            return await self._image_bucket_retry_after(user_id, operation)
        if operation == "text":
            window, limit = TEXT_WINDOW_SECONDS, TEXT_LIMIT
        elif operation == "media":
            window, limit = MEDIA_WINDOW_SECONDS, MEDIA_LIMIT
        else:
            window, limit = IMAGE_WINDOW_SECONDS, IMAGE_LIMIT
        now = time.time()

        if self._redis is not None:
            key = self._key(user_id, operation)
            try:
                await self._redis.zremrangebyscore(key, 0, now - window)
                oldest = await self._redis.zrange(key, 0, 0, withscores=True)
                count = await self._redis.zcard(key)
                if count < limit or not oldest:
                    return 0.0
                return max(0.0, float(oldest[0][1]) + window - now)
            except Exception as exc:
                logger.warning("rate_limiter_redis_retry_after_failed", error=str(exc))

        async with self._lock:
            self._sweep_expired(now)
            hits = self._hits(user_id, operation)
            self._prune(hits, now, window)
            if not hits:
                self._drop_hits(user_id, operation)
                return 0.0
            if len(hits) < limit:
                return 0.0
            return max(0.0, hits[0] + window - now)

    async def _image_bucket_retry_after(self, user_id: int, operation: str) -> float:
        capacity, refill, _ttl = self._bucket_params()
        now = time.time()
        if self._redis is not None:
            try:
                allowed, retry = await self._bucket_redis(
                    user_id,
                    operation,
                    capacity=capacity,
                    refill_per_min=refill,
                    now=now,
                    consume=False,
                )
                return 0.0 if allowed else retry
            except Exception as exc:
                from app.infra.access import is_production

                logger.warning(
                    "rate_limiter_redis_retry_after_failed",
                    error=str(exc),
                    operation=operation,
                    user_id=user_id,
                )
                if is_production():
                    return self._bucket_one_token_wait(refill)
        allowed, retry = await self._bucket_memory(
            user_id,
            operation,
            capacity=capacity,
            refill_per_min=refill,
            now=now,
            consume=False,
        )
        return 0.0 if allowed else retry

    async def _bucket_redis(
        self,
        user_id: int,
        operation: str,
        *,
        capacity: float,
        refill_per_min: float,
        now: float,
        consume: bool,
    ) -> tuple[bool, float]:
        if operation not in _IMAGE_BUCKET_OPS:
            raise ValueError(f"unknown image bucket {operation}")
        assert self._redis is not None
        _capacity, _refill, ttl = self._bucket_params()
        raw = await self._redis.eval(
            _BUCKET_LUA,
            1,
            self._key(user_id, operation),
            capacity,
            refill_per_min,
            now,
            ttl,
            1 if consume else 0,
        )
        allowed = int(raw[0]) == 1
        retry = max(0.0, float(raw[1]))
        return allowed, retry

    async def _bucket_memory(
        self,
        user_id: int,
        operation: str,
        *,
        capacity: float,
        refill_per_min: float,
        now: float,
        consume: bool,
    ) -> tuple[bool, float]:
        if operation not in _IMAGE_BUCKET_OPS:
            raise ValueError(f"unknown image bucket {operation}")
        rate = refill_per_min / 60.0
        async with self._lock:
            stored = self._image_buckets.get((user_id, operation))
            if stored is None:
                tokens = capacity
            else:
                tokens, ts = stored
                elapsed = max(0.0, now - ts)
                tokens = min(capacity, tokens + elapsed * rate)
            allowed = tokens >= 1.0
            retry = 0.0
            if consume and allowed:
                tokens -= 1.0
            elif not allowed:
                retry = self._bucket_retry(1.0 - tokens, refill_per_min)
            if consume:
                self._image_buckets[(user_id, operation)] = (tokens, now)
            return allowed, retry

    @staticmethod
    def _bucket_params() -> tuple[float, float, int]:
        from app.infra.settings import get_settings

        settings = get_settings()
        capacity = float(settings.image_bucket_capacity)
        refill = float(settings.image_bucket_refill_per_min)
        rate = refill / 60.0
        ttl = math.ceil(capacity / rate) + 3600 if rate > 0 else 3600
        return capacity, refill, ttl

    @staticmethod
    def _bucket_retry(deficit: float, refill_per_min: float) -> float:
        if deficit <= 0:
            return 0.0
        if refill_per_min <= 0:
            return 1.0
        return float(math.ceil(deficit * 60.0 / refill_per_min))

    @classmethod
    def _bucket_one_token_wait(cls, refill_per_min: float) -> float:
        return cls._bucket_retry(1.0, refill_per_min)

    async def _allow(
        self, user_id: int, *, operation: str, limit: int, window: float
    ) -> bool:
        if self._redis is not None:
            try:
                return await self._allow_redis(user_id, operation, limit, window)
            except Exception as exc:
                from app.infra.access import is_production

                if is_production():
                    logger.error(
                        "rate_limiter_redis_failed_deny",
                        error=str(exc),
                        operation=operation,
                        user_id=user_id,
                    )
                    return False
                logger.warning(
                    "rate_limiter_redis_failed_fallback_memory",
                    error=str(exc),
                    operation=operation,
                    user_id=user_id,
                )
        return await self._allow_memory(user_id, operation, limit, window)

    async def _allow_named(
        self, identity: str, *, operation: str, limit: int, window: float
    ) -> bool:
        if self._redis is not None:
            try:
                return await self._allow_redis_key(
                    self._named_key(identity, operation), limit, window
                )
            except Exception as exc:
                from app.infra.access import is_production

                if is_production():
                    logger.error(
                        "rate_limiter_redis_failed_deny",
                        error=str(exc),
                        operation=operation,
                    )
                    return False
                logger.warning(
                    "rate_limiter_redis_failed_fallback_memory",
                    error=str(exc),
                    operation=operation,
                )
        now = time.time()
        async with self._lock:
            self._sweep_expired(now)
            hits = self._upload_hits[operation][identity]
            self._prune(hits, now, window)
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    async def _allow_redis_key(self, key: str, limit: int, window: float) -> bool:
        assert self._redis is not None
        now = time.time()
        member = f"{now}:{uuid.uuid4().hex}"
        allowed = await self._redis.eval(
            _ALLOW_LUA,
            1,
            key,
            now - window,
            limit,
            now,
            member,
            int(window) + 1,
        )
        return int(allowed) == 1

    async def _allow_redis(
        self, user_id: int, operation: str, limit: int, window: float
    ) -> bool:
        assert self._redis is not None
        key = self._key(user_id, operation)
        now = time.time()
        member = f"{now}:{uuid.uuid4().hex}"
        allowed = await self._redis.eval(
            _ALLOW_LUA,
            1,
            key,
            now - window,
            limit,
            now,
            member,
            int(window) + 1,
        )
        return int(allowed) == 1

    async def _allow_memory(
        self, user_id: int, operation: str, limit: int, window: float
    ) -> bool:
        now = time.time()
        async with self._lock:
            self._sweep_expired(now)
            hits = self._hits(user_id, operation)
            self._prune(hits, now, window)
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def _hits(self, user_id: int, operation: str) -> deque[float]:
        if operation == "text":
            return self._text_hits[user_id]
        if operation == "media":
            return self._media_hits[user_id]
        return self._image_hits[user_id]

    def _drop_hits(self, user_id: int, operation: str) -> None:
        if operation == "text":
            store = self._text_hits
        elif operation == "media":
            store = self._media_hits
        else:
            store = self._image_hits
        store.pop(user_id, None)

    def _sweep_expired(self, now: float) -> None:
        """Drop memory keys whose windows are fully expired."""
        for user_id, hits in list(self._text_hits.items()):
            self._prune(hits, now, TEXT_WINDOW_SECONDS)
            if not hits:
                self._text_hits.pop(user_id, None)
        for user_id, hits in list(self._image_hits.items()):
            self._prune(hits, now, IMAGE_WINDOW_SECONDS)
            if not hits:
                self._image_hits.pop(user_id, None)
        for user_id, hits in list(self._media_hits.items()):
            self._prune(hits, now, MEDIA_WINDOW_SECONDS)
            if not hits:
                self._media_hits.pop(user_id, None)
        windows = {
            "upload_read": UPLOAD_READ_WINDOW_SECONDS,
            "upload_write": UPLOAD_WRITE_WINDOW_SECONDS,
        }
        for operation, bucket in list(self._upload_hits.items()):
            window = windows.get(operation, UPLOAD_READ_WINDOW_SECONDS)
            for identity, hits in list(bucket.items()):
                self._prune(hits, now, window)
                if not hits:
                    bucket.pop(identity, None)

    @staticmethod
    def _prune(hits: deque[float], now: float, window: float) -> None:
        cutoff = now - window
        while hits and hits[0] < cutoff:
            hits.popleft()

    @staticmethod
    def _key(user_id: int, operation: str) -> str:
        return f"rl:{operation}:{user_id}"

    @staticmethod
    def _named_key(identity: str, operation: str) -> str:
        safe = identity.strip()[:128] or "unknown"
        return f"rl:{operation}:{safe}"


rate_limiter = RateLimiter()

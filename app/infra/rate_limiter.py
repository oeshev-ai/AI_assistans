"""Per-user rate limits with Redis sliding window and in-memory fallback."""

from __future__ import annotations

import asyncio
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
        """Return True and consume a slot if under the image-generation limit."""
        return await self._allow(
            user_id,
            operation="image",
            limit=IMAGE_LIMIT,
            window=IMAGE_WINDOW_SECONDS,
        )

    async def get_retry_after(self, user_id: int, operation: str) -> float:
        """Seconds until the oldest hit leaves the window (0 if capacity free)."""
        window = TEXT_WINDOW_SECONDS if operation == "text" else IMAGE_WINDOW_SECONDS
        limit = TEXT_LIMIT if operation == "text" else IMAGE_LIMIT
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
            hits = self._hits(user_id, operation)
            self._prune(hits, now, window)
            if len(hits) < limit or not hits:
                return 0.0
            return max(0.0, hits[0] + window - now)

    async def _allow(
        self, user_id: int, *, operation: str, limit: int, window: float
    ) -> bool:
        if self._redis is not None:
            try:
                return await self._allow_redis(user_id, operation, limit, window)
            except Exception as exc:
                logger.warning(
                    "rate_limiter_redis_failed_fallback_memory",
                    error=str(exc),
                    operation=operation,
                    user_id=user_id,
                )
        return await self._allow_memory(user_id, operation, limit, window)

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
            hits = self._hits(user_id, operation)
            self._prune(hits, now, window)
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def _hits(self, user_id: int, operation: str) -> deque[float]:
        if operation == "text":
            return self._text_hits[user_id]
        return self._image_hits[user_id]

    @staticmethod
    def _prune(hits: deque[float], now: float, window: float) -> None:
        cutoff = now - window
        while hits and hits[0] < cutoff:
            hits.popleft()

    @staticmethod
    def _key(user_id: int, operation: str) -> str:
        return f"rl:{operation}:{user_id}"


rate_limiter = RateLimiter()

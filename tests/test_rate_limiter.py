"""Unit tests for per-user sliding-window rate limits."""

from __future__ import annotations

import time

import pytest

from app.infra.rate_limiter import (
    IMAGE_LIMIT,
    MEDIA_LIMIT,
    TEXT_LIMIT,
    RateLimiter,
)


@pytest.mark.asyncio
async def test_text_limit_blocks_after_threshold() -> None:
    limiter = RateLimiter()
    user_id = 42
    for _ in range(TEXT_LIMIT):
        assert await limiter.check_text_limit(user_id) is True
    assert await limiter.check_text_limit(user_id) is False
    retry = await limiter.get_retry_after(user_id, "text")
    assert retry > 0


@pytest.mark.asyncio
async def test_image_limit_independent_of_text() -> None:
    limiter = RateLimiter()
    user_id = 7
    for _ in range(IMAGE_LIMIT):
        assert await limiter.check_image_limit(user_id) is True
    assert await limiter.check_image_limit(user_id) is False
    assert await limiter.check_text_limit(user_id) is True


@pytest.mark.asyncio
async def test_media_limit_independent_of_image() -> None:
    limiter = RateLimiter()
    user_id = 11
    for _ in range(MEDIA_LIMIT):
        assert await limiter.check_media_limit(user_id) is True
    assert await limiter.check_media_limit(user_id) is False
    assert await limiter.check_image_limit(user_id) is True
    retry = await limiter.get_retry_after(user_id, "media")
    assert retry > 0


@pytest.mark.asyncio
async def test_users_are_isolated() -> None:
    limiter = RateLimiter()
    for _ in range(TEXT_LIMIT):
        assert await limiter.check_text_limit(1) is True
    assert await limiter.check_text_limit(1) is False
    assert await limiter.check_text_limit(2) is True


@pytest.mark.asyncio
async def test_redis_backed_limit(fakeredis_client) -> None:
    limiter = RateLimiter()
    limiter.attach_redis(fakeredis_client)
    user_id = 99
    for _ in range(TEXT_LIMIT):
        assert await limiter.check_text_limit(user_id) is True
    assert await limiter.check_text_limit(user_id) is False


@pytest.fixture
async def fakeredis_client():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_window_expiry_frees_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = RateLimiter()
    user_id = 3
    now = {"t": 1_000_000.0}

    def fake_time() -> float:
        return now["t"]

    monkeypatch.setattr(time, "time", fake_time)
    for _ in range(TEXT_LIMIT):
        assert await limiter.check_text_limit(user_id) is True
    assert await limiter.check_text_limit(user_id) is False
    now["t"] += 61.0
    assert await limiter.check_text_limit(user_id) is True


@pytest.mark.asyncio
async def test_expired_memory_keys_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    limiter = RateLimiter()
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(time, "time", lambda: now["t"])
    for uid in range(15):
        assert await limiter.check_text_limit(uid) is True
    assert len(limiter._text_hits) == 15
    now["t"] += 120.0
    assert await limiter.check_text_limit(0) is True
    # Sweep drops fully expired peers; only the touched user remains.
    assert 0 in limiter._text_hits
    assert len(limiter._text_hits) == 1


def test_env_int_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infra.envparse import env_int

    monkeypatch.setenv("BAD_PORT", "xyz")
    assert env_int("BAD_PORT", 8082) == 8082


@pytest.mark.asyncio
async def test_redis_atomic_limit_under_concurrency(fakeredis_client) -> None:
    import asyncio

    limiter = RateLimiter()
    limiter.attach_redis(fakeredis_client)
    user_id = 77

    results = await asyncio.gather(
        *[limiter.check_text_limit(user_id) for _ in range(TEXT_LIMIT + 8)]
    )
    assert sum(1 for ok in results if ok) == TEXT_LIMIT
    assert sum(1 for ok in results if not ok) == 8

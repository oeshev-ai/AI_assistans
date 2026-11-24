"""Token bucket генерации и правки картинок."""

from __future__ import annotations

import math
import time

import pytest

from app.infra.rate_limiter import RateLimiter
from app.infra.settings import get_settings


@pytest.fixture
async def fakeredis_client():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()


def _expected_ttl() -> int:
    settings = get_settings()
    capacity = float(settings.image_bucket_capacity)
    refill = float(settings.image_bucket_refill_per_min)
    rate = refill / 60.0
    return math.ceil(capacity / rate) + 3600


@pytest.mark.asyncio
async def test_burst_then_retry_after_then_refill(
    fakeredis_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    limiter = RateLimiter()
    limiter.attach_redis(fakeredis_client)
    clock = {"t": 1_700_000_000.0}
    monkeypatch.setattr(time, "time", lambda: clock["t"])
    user_id = 15
    capacity = int(get_settings().image_bucket_capacity)

    for _ in range(capacity):
        assert await limiter.consume_image_bucket(user_id, "image_generate") is True
    assert await limiter.consume_image_bucket(user_id, "image_generate") is False
    assert await limiter.get_retry_after(user_id, "image_generate") == 120

    clock["t"] += 60
    assert await limiter.consume_image_bucket(user_id, "image_generate") is False
    assert await limiter.get_retry_after(user_id, "image_generate") == 60

    clock["t"] += 60
    assert await limiter.consume_image_bucket(user_id, "image_generate") is True
    assert await limiter.get_retry_after(user_id, "image_generate") == 120


@pytest.mark.asyncio
async def test_generate_and_edit_buckets_are_independent(
    fakeredis_client, monkeypatch: pytest.MonkeyPatch
) -> None:
    limiter = RateLimiter()
    limiter.attach_redis(fakeredis_client)
    monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
    user_id = 8
    capacity = int(get_settings().image_bucket_capacity)
    for _ in range(capacity):
        assert await limiter.consume_image_bucket(user_id, "image_generate") is True
    assert await limiter.consume_image_bucket(user_id, "image_generate") is False
    assert await limiter.consume_image_bucket(user_id, "image_edit") is True
    generate = await fakeredis_client.hget(f"rl:image_generate:{user_id}", "tokens")
    edit = await fakeredis_client.hget(f"rl:image_edit:{user_id}", "tokens")
    assert float(generate) == 0
    assert float(edit) == capacity - 1


@pytest.mark.asyncio
async def test_bucket_key_has_ttl_and_peek_does_not_create_it(fakeredis_client) -> None:
    limiter = RateLimiter()
    limiter.attach_redis(fakeredis_client)
    user_id = 3
    key = f"rl:image_generate:{user_id}"
    assert await limiter.get_retry_after(user_id, "image_generate") == 0
    assert await fakeredis_client.exists(key) == 0

    assert await limiter.consume_image_bucket(user_id, "image_generate") is True
    assert await fakeredis_client.type(key) == "hash"
    ttl = await fakeredis_client.ttl(key)
    assert ttl == _expected_ttl()
    assert await fakeredis_client.hexists(key, "tokens")
    assert await fakeredis_client.hexists(key, "ts")


@pytest.mark.asyncio
async def test_production_redis_outage_denies_without_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Down:
        async def eval(self, *_args: object, **_kwargs: object) -> None:
            raise ConnectionError("redis down")

    limiter = RateLimiter()
    limiter.attach_redis(_Down())
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert await limiter.consume_image_bucket(4, "image_generate") is False
    assert limiter._image_buckets == {}
    assert await limiter.get_retry_after(4, "image_generate") == 120
    assert limiter._image_buckets == {}


@pytest.mark.asyncio
async def test_development_redis_outage_falls_back_to_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Down:
        async def eval(self, *_args: object, **_kwargs: object) -> None:
            raise ConnectionError("redis down")

    limiter = RateLimiter()
    limiter.attach_redis(_Down())
    monkeypatch.setenv("ENVIRONMENT", "development")
    assert await limiter.consume_image_bucket(4, "image_generate") is True
    assert (4, "image_generate") in limiter._image_buckets

"""Отложенные фото: TTL, file_id без байтов, Redis-ключ."""

from __future__ import annotations

import asyncio

import pytest

from app.storage.pending_media import (
    InMemoryPendingMediaStorage,
    PendingMedia,
    PendingMediaItem,
    RedisPendingMediaStorage,
)


def _sample() -> PendingMedia:
    return PendingMedia(
        chat_id=7,
        assistant_id="asst",
        items=[PendingMediaItem(file_id="file-1", message_id=3, file_size=10)],
        caption="секретная подпись",
    )


@pytest.mark.asyncio
async def test_memory_put_get_clear() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    await store.put(_sample())
    loaded = await store.get(7, "asst")
    assert loaded is not None
    assert loaded.items[0].file_id == "file-1"
    assert not hasattr(loaded.items[0], "data")
    await store.clear(7, "asst")
    assert await store.get(7, "asst") is None
    await store.close()


@pytest.mark.asyncio
async def test_memory_expires() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=0.01)
    await store.put(_sample())
    await asyncio.sleep(0.03)
    assert await store.get(7, "asst") is None


@pytest.mark.asyncio
async def test_base_override_roundtrip() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    await store.put(_sample())
    updated = await store.set_base_override(7, "asst", 0)
    assert updated is not None
    assert updated.base_override == 0
    await store.mark_role_retry(7, "asst")
    again = await store.get(7, "asst")
    assert again is not None
    assert again.role_retry_used is True


@pytest.mark.asyncio
async def test_redis_key_has_file_id_only() -> None:
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisPendingMediaStorage(client, ttl_seconds=30, owns_client=False)
    await store.put(_sample())
    raw = await client.get("chat:7:asst:asst:pending_photo")
    assert isinstance(raw, str)
    assert "file-1" in raw
    assert "секретная подпись" in raw
    assert "base64" not in raw
    loaded = await store.get(7, "asst")
    assert loaded is not None
    assert loaded.items[0].file_id == "file-1"
    await client.aclose()

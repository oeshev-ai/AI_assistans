"""Отложенные фото: TTL, file_id без байтов, Redis-ключ с user_id."""

from __future__ import annotations

import asyncio

import pytest

from app.storage.pending_media import (
    InMemoryPendingMediaStorage,
    PendingMedia,
    PendingMediaItem,
    RedisPendingMediaStorage,
)


def _sample(*, user_id: int = 42) -> PendingMedia:
    return PendingMedia(
        chat_id=7,
        user_id=user_id,
        assistant_id="asst",
        items=[PendingMediaItem(file_id="file-1", message_id=3, file_size=10)],
        caption="секретная подпись",
    )


@pytest.mark.asyncio
async def test_memory_put_get_clear() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    await store.put(_sample())
    loaded = await store.get(7, "asst", 42)
    assert loaded is not None
    assert loaded.items[0].file_id == "file-1"
    assert loaded.user_id == 42
    assert not hasattr(loaded.items[0], "data")
    await store.clear(7, "asst", 42)
    assert await store.get(7, "asst", 42) is None
    await store.close()


@pytest.mark.asyncio
async def test_users_are_isolated() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    await store.put(_sample(user_id=10))
    await store.put(_sample(user_id=20))
    assert await store.get(7, "asst", 10) is not None
    assert await store.get(7, "asst", 20) is not None
    await store.clear(7, "asst", 10)
    assert await store.get(7, "asst", 10) is None
    assert await store.get(7, "asst", 20) is not None


@pytest.mark.asyncio
async def test_clear_user_removes_all_personas() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    for asst in ("a", "b"):
        media = _sample()
        media.assistant_id = asst
        await store.put(media)
    other = _sample(user_id=99)
    other.assistant_id = "a"
    await store.put(other)
    assert await store.clear_user(7, 42) is True
    assert await store.get(7, "a", 42) is None
    assert await store.get(7, "b", 42) is None
    assert await store.get(7, "a", 99) is not None
    assert await store.clear_user(7, 42) is False


@pytest.mark.asyncio
async def test_memory_expires() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=0.01)
    await store.put(_sample())
    await asyncio.sleep(0.03)
    assert await store.get(7, "asst", 42) is None


@pytest.mark.asyncio
async def test_bump_round_and_replace() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    await store.put(_sample())
    updated = await store.bump_round(7, "asst", 42, "Что сделать?")
    assert updated is not None
    assert updated.clarify_rounds == 1
    assert updated.last_question == "Что сделать?"
    replacement = _sample()
    replacement.items[0].file_id = "file-2"
    await store.put(replacement)
    loaded = await store.get(7, "asst", 42)
    assert loaded is not None
    assert loaded.items[0].file_id == "file-2"
    popped = await store.pop(7, "asst", 42)
    assert popped is not None
    assert await store.get(7, "asst", 42) is None


@pytest.mark.asyncio
async def test_base_override_roundtrip() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    await store.put(_sample())
    updated = await store.set_base_override(7, "asst", 42, 0)
    assert updated is not None
    assert updated.base_override == 0
    await store.mark_role_retry(7, "asst", 42)
    again = await store.get(7, "asst", 42)
    assert again is not None
    assert again.role_retry_used is True


@pytest.mark.asyncio
async def test_mutate_preserves_memory_deadline() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    await store.put(_sample())
    key = store._key(7, 42, "asst")
    _media, deadline = store._items[key]
    await store.set_base_override(7, "asst", 42, 1)
    _media2, deadline2 = store._items[key]
    assert deadline2 == pytest.approx(deadline)
    assert _media2.base_override == 1


@pytest.mark.asyncio
async def test_redis_mutate_keeps_ttl() -> None:
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisPendingMediaStorage(client, ttl_seconds=60, owns_client=False)
    await store.put(_sample())
    key = "chat:7:user:42:asst:asst:pending_photo"
    ttl_before = await client.ttl(key)
    await store.set_base_override(7, "asst", 42, 2)
    ttl_after = await client.ttl(key)
    loaded = await store.get(7, "asst", 42)
    assert loaded is not None
    assert loaded.base_override == 2
    assert ttl_after > 0
    # Must not reset back to a full fresh 60s window relative to before.
    assert ttl_after <= ttl_before
    await client.aclose()


@pytest.mark.asyncio
async def test_redis_key_has_file_id_only() -> None:
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisPendingMediaStorage(client, ttl_seconds=30, owns_client=False)
    await store.put(_sample())
    raw = await client.get("chat:7:user:42:asst:asst:pending_photo")
    assert isinstance(raw, str)
    assert "file-1" in raw
    assert "секретная подпись" in raw
    assert "base64" not in raw
    loaded = await store.get(7, "asst", 42)
    assert loaded is not None
    assert loaded.items[0].file_id == "file-1"
    await client.aclose()


@pytest.mark.asyncio
async def test_memory_pop_is_single_winner() -> None:
    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    await store.put(_sample())
    first = await store.pop(7, "asst", 42)
    second = await store.pop(7, "asst", 42)
    assert first is not None
    assert first.items[0].file_id == "file-1"
    assert second is None


@pytest.mark.asyncio
async def test_redis_pop_is_atomic() -> None:
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    store = RedisPendingMediaStorage(client, ttl_seconds=30, owns_client=False)
    await store.put(_sample())
    first = await store.pop(7, "asst", 42)
    second = await store.pop(7, "asst", 42)
    assert first is not None
    assert second is None
    assert await client.get("chat:7:user:42:asst:asst:pending_photo") is None
    await client.aclose()

"""Сборка альбома и debounce без Telegram и сети."""

from __future__ import annotations

import asyncio

import pytest

from app.bot.album import AlbumHub, IncomingPhoto, assemble_album


def _photo(message_id: int, caption: str = "", size: int = 10) -> IncomingPhoto:
    return IncomingPhoto(
        message_id=message_id,
        file_id=f"id-{message_id}",
        size=size,
        caption=caption,
    )


def test_sorts_by_message_id_and_keeps_first_caption() -> None:
    built = assemble_album([_photo(5, ""), _photo(2, "вторая"), _photo(9, "позже")])
    assert [item.message_id for item in built.photos] == [2, 5, 9]
    assert built.caption == "вторая"
    assert built.error is None


def test_trims_to_five() -> None:
    photos = [_photo(index) for index in range(6, 0, -1)]
    built = assemble_album(photos, max_items=5)
    assert len(built.photos) == 5
    assert [item.message_id for item in built.photos] == [1, 2, 3, 4, 5]
    assert built.notice == "Взял первые 5 фото"


def test_rejects_oversize_album() -> None:
    built = assemble_album(
        [_photo(1, size=11), _photo(2, size=11)],
        max_total_bytes=20,
    )
    assert built.error is not None
    assert built.photos == []


@pytest.mark.asyncio
async def test_text_during_debounce_becomes_instruction() -> None:
    hub = AlbumHub(debounce=5)
    ready: list[tuple[str, list[int]]] = []

    async def on_ready(built, _services) -> None:  # type: ignore[no-untyped-def]
        ready.append((built.caption, [item.message_id for item in built.photos]))

    await hub.add(1, 10, "group", _photo(2, "из альбома"), "svc", on_ready)
    await hub.add(1, 10, "group", _photo(1), "svc", on_ready)
    flushed = await hub.flush_text(1, 10, "замени фон", on_ready)
    assert flushed is True
    assert ready == [("замени фон", [1, 2])]
    await asyncio.sleep(0.05)
    assert len(ready) == 1


@pytest.mark.asyncio
async def test_single_photo_cancels_open_buffer() -> None:
    hub = AlbumHub(debounce=5)
    ready: list[str] = []

    async def on_ready(built, _services) -> None:  # type: ignore[no-untyped-def]
        ready.append(built.photos[0].file_id)

    await hub.add(1, 10, "group", _photo(1), "svc", on_ready)
    await hub.add(1, 10, None, _photo(8), "svc", on_ready)
    assert ready == ["id-8"]
    assert await hub.cancel_chat(1, 10) is False


@pytest.mark.asyncio
async def test_users_do_not_flush_each_others_albums() -> None:
    hub = AlbumHub(debounce=5)
    ready: list[tuple[int, str]] = []

    async def on_ready(built, services) -> None:  # type: ignore[no-untyped-def]
        ready.append((services, built.caption))

    await hub.add(1, 10, "g", _photo(1, "a"), 10, on_ready)
    await hub.add(1, 20, "g", _photo(2, "b"), 20, on_ready)
    assert hub.has_pending(1, 10) is True
    assert hub.has_pending(1, 20) is True
    flushed = await hub.flush_text(1, 10, "только для 10", on_ready)
    assert flushed is True
    assert ready == [(10, "только для 10")]
    assert hub.has_pending(1, 20) is True
    assert await hub.flush_text(1, 20, "для 20", on_ready) is True
    assert ready[-1] == (20, "для 20")


@pytest.mark.asyncio
async def test_redis_merges_photos_from_peer_hub(fakeredis_client) -> None:
    """Two hubs sharing Redis glue one media_group_id."""
    left = AlbumHub(debounce=5)
    right = AlbumHub(debounce=5)
    left.attach_redis(fakeredis_client)
    right.attach_redis(fakeredis_client)
    ready: list[list[int]] = []

    async def on_ready(built, _services) -> None:  # type: ignore[no-untyped-def]
        ready.append([item.message_id for item in built.photos])

    await left.add(1, 10, "g", _photo(1), "svc-a", on_ready)
    await right.add(1, 10, "g", _photo(2), "svc-b", on_ready)
    flushed = await left.flush_text(1, 10, "склей", on_ready)
    assert flushed is True
    assert ready == [[1, 2]]
    # Peer must not deliver leftovers after Redis claim.
    assert await right.flush_text(1, 10, "повтор", on_ready) is False
    assert ready == [[1, 2]]


@pytest.fixture
async def fakeredis_client():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()

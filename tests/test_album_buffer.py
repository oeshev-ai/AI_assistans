"""Сборка альбома и debounce без Telegram и сети."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

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


@pytest.mark.asyncio
async def test_redis_take_error_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With Redis attached, claim errors must not flush local (avoids multi-replica dupes)."""
    hub = AlbumHub(debounce=5)
    hub.attach_redis(object())  # non-None marks Redis mode
    ready: list[list[int]] = []

    async def on_ready(built, _services) -> None:  # type: ignore[no-untyped-def]
        ready.append([item.message_id for item in built.photos])

    async def boom(_key):  # type: ignore[no-untyped-def]
        return None, []

    async def noop_append(_key, _photo) -> None:  # type: ignore[no-untyped-def]
        return None

    monkeypatch.setattr(hub, "_redis_take", boom)
    monkeypatch.setattr(hub, "_redis_append", noop_append)

    await hub.add(1, 10, "g", _photo(3, "подпись"), "svc", on_ready)
    flushed = await hub.flush_text(1, 10, "из локального", on_ready)
    assert flushed is True
    assert ready == []


@pytest.mark.asyncio
async def test_append_failure_merges_local_on_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub = AlbumHub(debounce=5)
    hub.attach_redis(object())
    ready: list[list[int]] = []

    async def on_ready(built, _services) -> None:  # type: ignore[no-untyped-def]
        ready.append([item.message_id for item in built.photos])

    async def take_empty(_key):  # type: ignore[no-untyped-def]
        return True, []

    async def append_fail(key, _photo) -> None:  # type: ignore[no-untyped-def]
        hub._append_failed.add(key)

    monkeypatch.setattr(hub, "_redis_take", take_empty)
    monkeypatch.setattr(hub, "_redis_append", append_fail)

    await hub.add(1, 10, "g", _photo(4, "подпись"), "svc", on_ready)
    flushed = await hub.flush_text(1, 10, "merge", on_ready)
    assert flushed is True
    assert ready == [[4]]


@pytest.mark.asyncio
async def test_cancel_drops_open_album_when_pending_slot_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """«Отмена» гасит и слот, и альбом, который ещё собирается."""
    import app.bot.album as album_module
    from app.bot.handlers.chat import handle_text
    from app.storage.pending_media import (
        InMemoryPendingMediaStorage,
        PendingMedia,
        PendingMediaItem,
    )

    store = InMemoryPendingMediaStorage(ttl_seconds=30)
    await store.put(
        PendingMedia(
            chat_id=7,
            user_id=42,
            assistant_id="asst",
            items=[PendingMediaItem(file_id="kept", message_id=1)],
        )
    )
    hub = AlbumHub(debounce=30)
    ready: list[int] = []

    async def on_ready(built, _services) -> None:  # type: ignore[no-untyped-def]
        ready.append(len(built.photos))

    previous = album_module._hub
    album_module._hub = hub

    async def _profile(_chat_id: int, _storage: object) -> SimpleNamespace:
        return SimpleNamespace(id="asst")

    async def _reset(_state: object) -> None:
        return None

    monkeypatch.setattr("app.bot.runtime._active_profile", _profile)
    monkeypatch.setattr("app.bot.runtime._reset_flow", _reset)

    class _Message:
        text = "отмена"

        class chat:
            id = 7

        class from_user:
            id = 42

        def __init__(self) -> None:
            self.answers: list[str] = []

        async def answer(self, text: str) -> None:
            self.answers.append(text)

    message = _Message()
    try:
        await hub.add(7, 42, "group", _photo(3), "svc", on_ready)
        await handle_text(
            message,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            None,  # type: ignore[arg-type]
            store,
        )
        assert message.answers == ["Отменено."]
        assert hub.has_pending(7, 42) is False
        assert await store.get(7, "asst", 42) is None
        await asyncio.sleep(0.05)
        assert ready == []
    finally:
        album_module._hub = previous
        await store.close()


@pytest.fixture
async def fakeredis_client():
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    try:
        yield client
    finally:
        await client.aclose()

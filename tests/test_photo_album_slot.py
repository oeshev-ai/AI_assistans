"""Слот отложенного альбома не теряется и не затирает следующий."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.bot.photo_flow import _release_album, _restore_album
from app.storage.pending_media import (
    InMemoryPendingMediaStorage,
    PendingMedia,
    PendingMediaItem,
)


def _media(file_id: str, message_id: int = 1) -> PendingMedia:
    return PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id=file_id, message_id=message_id)],
    )


def _turn(store: InMemoryPendingMediaStorage) -> SimpleNamespace:
    return SimpleNamespace(pending=store)


@pytest.mark.asyncio
async def test_restore_puts_album_back_into_empty_slot() -> None:
    store = InMemoryPendingMediaStorage()
    await _restore_album(_turn(store), _media("old"))
    got = await store.get(1, "a", 2)
    assert got is not None
    assert got.items[0].file_id == "old"


@pytest.mark.asyncio
async def test_restore_and_release_leave_a_newer_album() -> None:
    store = InMemoryPendingMediaStorage()
    await store.put(_media("new", 9))
    await _restore_album(_turn(store), _media("old"))
    await _release_album(_turn(store), _media("old"), "result-file")
    got = await store.get(1, "a", 2)
    assert got is not None
    assert got.items[0].file_id == "new"


@pytest.mark.asyncio
async def test_release_keeps_result_when_slot_is_ours() -> None:
    store = InMemoryPendingMediaStorage()
    original = _media("old")
    await store.put(original)
    await _release_album(_turn(store), original, "result-file")
    got = await store.get(1, "a", 2)
    assert got is not None
    assert got.items[0].file_id == "result-file"

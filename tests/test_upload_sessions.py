"""Тесты сессий загрузки больших медиа."""

from __future__ import annotations

import pytest

from upload_sessions import (
    UploadRequestContext,
    UploadSessionStore,
    get_upload_request_context,
    object_key_for,
    reset_upload_request_context,
    set_upload_request_context,
)


@pytest.mark.asyncio
async def test_memory_session_create_and_auth() -> None:
    store = UploadSessionStore(redis=None)
    session = await store.create(chat_id=10, user_id=20, media_kind="video")
    got = await store.get_authorized(session.upload_id, session.token)
    assert got is not None
    assert got.chat_id == 10
    assert got.media_kind == "video"
    assert await store.get_authorized(session.upload_id, "wrong") is None


@pytest.mark.asyncio
async def test_object_key_prefix() -> None:
    store = UploadSessionStore(redis=None)
    session = await store.create(chat_id=1, user_id=2)
    key = object_key_for(session, "My Lecture (1).mp4")
    assert key.startswith(f"uploads/{session.upload_id}/")
    assert key.endswith(".mp4")
    assert " " not in key


def test_upload_request_context() -> None:
    token = set_upload_request_context(UploadRequestContext(chat_id=5, user_id=6))
    try:
        ctx = get_upload_request_context()
        assert ctx is not None
        assert ctx.chat_id == 5
        assert ctx.user_id == 6
    finally:
        reset_upload_request_context(token)
    assert get_upload_request_context() is None

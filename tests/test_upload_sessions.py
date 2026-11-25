"""Тесты сессий загрузки больших медиа."""

from __future__ import annotations

import json

import pytest

from app.storage.upload_sessions import (
    UploadRequestContext,
    UploadSession,
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


def test_old_session_json_without_verified_mime_loads() -> None:
    store_payload = {
        "upload_id": "abc",
        "token": "tok",
        "chat_id": 1,
        "user_id": 2,
        "purpose": "transcribe",
        "media_kind": "audio",
        "status": "uploaded",
        "created_at": 1.0,
        "expires_at": 9_999_999_999.0,
        "mime_type": "audio/mpeg",
    }
    session = UploadSession.from_json(json.dumps(store_payload))
    assert session.verified_mime is None
    assert session.mime_type == "audio/mpeg"


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


@pytest.mark.asyncio
async def test_claim_processing_is_single_winner() -> None:
    store = UploadSessionStore(redis=None)
    session = await store.create(chat_id=1, user_id=2)
    first = await store.claim_processing(
        session.upload_id,
        session.token,
        from_statuses={"pending"},
        updates={"filename": "a.mp3"},
    )
    second = await store.claim_processing(
        session.upload_id,
        session.token,
        from_statuses={"pending"},
    )
    assert first is not None
    assert first.status == "processing"
    assert first.filename == "a.mp3"
    assert second is None


@pytest.mark.asyncio
async def test_claim_processing_redis_single_winner() -> None:
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    client = fakeredis.FakeRedis(decode_responses=True)
    store = UploadSessionStore(redis=client)
    session = await store.create(chat_id=1, user_id=2)
    first = await store.claim_processing(
        session.upload_id, session.token, from_statuses={"pending"}
    )
    second = await store.claim_processing(
        session.upload_id, session.token, from_statuses={"pending"}
    )
    assert first is not None and first.status == "processing"
    assert second is None
    await client.aclose()

"""Тесты HTTP API загрузки в бакет."""

from __future__ import annotations

import asyncio
import io
import wave
from dataclasses import dataclass
from typing import Any

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from app.media.upload_server import _copy_until_limit, setup_upload_routes
from app.storage.upload_sessions import UploadSessionStore


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 800)
    return buf.getvalue()


@dataclass
class FakeS3Config:
    signed_url_expires_in: int = 1800
    max_upload_mb: int = 100


class FakeStorage:
    def __init__(self) -> None:
        self.config = FakeS3Config()
        self.objects: dict[str, bytes] = {}

    async def create_presigned_put(self, *, object_key: str, content_type: str) -> str:
        return f"https://bucket.storage.example/{object_key}?sig=1&ct={content_type}"

    async def upload_file(
        self, *, object_key: str, file_path: str, content_type: str
    ) -> None:
        from pathlib import Path

        self.objects[object_key] = Path(file_path).read_bytes()

    async def head_object(self, object_key: str) -> dict[str, Any]:
        data = self.objects.get(object_key)
        if data is None:
            from app.storage.s3_storage import S3NotFoundError

            raise S3NotFoundError("missing")
        return {"ContentLength": len(data), "ContentType": "audio/mpeg"}

    async def download_bytes(self, object_key: str) -> bytes:
        return self.objects[object_key]

    async def delete_object(self, object_key: str) -> None:
        self.objects.pop(object_key, None)


@pytest.mark.asyncio
async def test_same_origin_file_upload_flow() -> None:
    store = UploadSessionStore(redis=None)
    storage = FakeStorage()
    completed: list[str] = []

    async def on_complete(session: Any) -> None:
        completed.append(session.upload_id)

    session = await store.create(chat_id=42, user_id=7, purpose="transcribe")
    app = web.Application(client_max_size=10 * 1024 * 1024)
    setup_upload_routes(
        app,
        store=store,
        storage=storage,  # Protocol duck-typing
        on_complete=on_complete,
    )

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        # ?t= → Set-Cookie + redirect to clean URL (TestClient follows).
        page = await client.get(f"/upload/{session.upload_id}?t={session.token}")
        assert page.status == 200
        assert page.url.path == f"/upload/{session.upload_id}"
        assert "t=" not in str(page.url)
        text = await page.text()
        assert "/api/uploads/" in text
        assert "FormData" in text
        assert "TOKEN" not in text
        assert session.token not in text
        assert "upload_token" in client.session.cookie_jar.filter_cookies(
            client.make_url("/")
        )

        form = FormData()
        form.add_field(
            "file",
            _wav_bytes(),
            filename="lecture.wav",
            content_type="audio/wav",
        )
        # Auth via HttpOnly cookie only — no ?t= on API.
        resp = await client.post(
            f"/api/uploads/{session.upload_id}/file",
            data=form,
        )
        assert resp.status == 200, await resp.text()
        body = await resp.json()
        assert body["ok"] is True
        assert body["objectKey"] in storage.objects

        # Second POST must not double-process.
        form2 = FormData()
        form2.add_field(
            "file",
            _wav_bytes(),
            filename="lecture2.wav",
            content_type="audio/wav",
        )
        dup = await client.post(
            f"/api/uploads/{session.upload_id}/file",
            data=form2,
        )
        assert dup.status == 409

        for _ in range(50):
            if completed:
                break
            await asyncio.sleep(0.01)
        assert completed == [session.upload_id]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upload_releases_claim_on_bad_multipart() -> None:
    """Битый multipart не должен оставлять сессию в processing."""
    store = UploadSessionStore(redis=None)
    storage = FakeStorage()
    session = await store.create(chat_id=1, user_id=2, purpose="transcribe")
    app = web.Application(client_max_size=10 * 1024 * 1024)
    setup_upload_routes(
        app,
        store=store,
        storage=storage,
        on_complete=lambda _s: asyncio.sleep(0),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        await client.get(f"/upload/{session.upload_id}?t={session.token}")
        resp = await client.post(
            f"/api/uploads/{session.upload_id}/file",
            data=b"not-multipart",
            headers={"Content-Type": "text/plain"},
        )
        assert resp.status == 400
        refreshed = await store.get(session.upload_id)
        assert refreshed is not None
        assert refreshed.status == "pending"
        # Повтор с нормальным файлом должен пройти после release.
        form = FormData()
        form.add_field(
            "file",
            _wav_bytes(),
            filename="retry.wav",
            content_type="audio/wav",
        )
        ok = await client.post(f"/api/uploads/{session.upload_id}/file", data=form)
        assert ok.status == 200, await ok.text()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upload_rejects_mime_mismatch() -> None:
    store = UploadSessionStore(redis=None)
    storage = FakeStorage()
    session = await store.create(chat_id=1, user_id=2, purpose="transcribe")
    app = web.Application(client_max_size=10 * 1024 * 1024)
    setup_upload_routes(
        app,
        store=store,
        storage=storage,
        on_complete=lambda _s: asyncio.sleep(0),
    )
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        await client.get(f"/upload/{session.upload_id}?t={session.token}")
        form = FormData()
        form.add_field(
            "file",
            b"%PDF-1.7 disguised",
            filename="lecture.mp3",
            content_type="audio/mpeg",
        )
        resp = await client.post(
            f"/api/uploads/{session.upload_id}/file",
            data=form,
        )
        assert resp.status == 400
        body = await resp.json()
        assert "соответствует" in body["error"]
        assert storage.objects == {}
    finally:
        await client.close()


def test_client_key_uses_last_forwarded_hop(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from app.media.upload_server import _client_key

    req = SimpleNamespace(
        remote="10.0.0.2",
        headers={"X-Forwarded-For": "1.2.3.4, 203.0.113.9"},
    )
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    assert _client_key(req) == "10.0.0.2"  # type: ignore[arg-type]
    monkeypatch.setenv("TRUST_PROXY", "1")
    assert _client_key(req) == "203.0.113.9"  # type: ignore[arg-type]


def test_forwarded_proto_requires_trust_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    from app.media.upload_server import _request_is_https

    req = SimpleNamespace(secure=False, headers={"X-Forwarded-Proto": "https"})
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    assert _request_is_https(req) is False  # type: ignore[arg-type]
    monkeypatch.setenv("TRUST_PROXY", "1")
    assert _request_is_https(req) is True  # type: ignore[arg-type]


def test_legacy_presign_routes_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UPLOAD_LEGACY_PRESIGN", "0")
    store = UploadSessionStore(redis=None)
    app = web.Application()
    setup_upload_routes(
        app,
        store=store,
        storage=FakeStorage(),
        on_complete=lambda _s: asyncio.sleep(0),
    )
    paths = []
    for resource in app.router.resources():
        info = resource.get_info()
        formatter = info.get("formatter") or info.get("path") or ""
        paths.append(str(formatter))
    joined = " ".join(paths)
    assert "{upload_id}/file" in joined or "/file" in joined
    assert "presign" not in joined
    assert "complete" not in joined


@pytest.mark.asyncio
async def test_presign_ignores_boolean_and_negative_file_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("UPLOAD_LEGACY_PRESIGN", "1")
    store = UploadSessionStore(redis=None)
    session = await store.create(chat_id=1, user_id=2, purpose="transcribe")
    app = web.Application()
    setup_upload_routes(
        app,
        store=store,
        storage=FakeStorage(),
        on_complete=lambda _s: asyncio.sleep(0),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        for raw_size in (True, False, -1):
            fresh = await store.create(chat_id=1, user_id=2, purpose="transcribe")
            resp = await client.post(
                f"/api/uploads/{fresh.upload_id}/presign",
                json={
                    "fileName": "lecture.mp3",
                    "fileType": "audio/mpeg",
                    "fileSize": raw_size,
                },
                headers={"X-Upload-Token": fresh.token},
            )
            assert resp.status == 200, await resp.text()
            saved = await store.get(fresh.upload_id)
            assert saved is not None
            assert saved.file_size is None
        sized = await client.post(
            f"/api/uploads/{session.upload_id}/presign",
            json={
                "fileName": "lecture.mp3",
                "fileType": "audio/mpeg",
                "fileSize": 1200,
            },
            headers={"X-Upload-Token": session.token},
        )
        assert sized.status == 200, await sized.text()
        saved_size = await store.get(session.upload_id)
        assert saved_size is not None
        assert saved_size.file_size == 1200
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upload_errors_are_opaque() -> None:
    from app.storage.s3_storage import S3StorageError

    class BoomStorage(FakeStorage):
        async def upload_file(
            self, *, object_key: str, file_path: str, content_type: str
        ) -> None:
            raise S3StorageError("bucket secret-xyz denied at 10.0.0.5")

    store = UploadSessionStore(redis=None)
    session = await store.create(chat_id=1, user_id=2, purpose="transcribe")
    app = web.Application(client_max_size=10 * 1024 * 1024)
    setup_upload_routes(
        app,
        store=store,
        storage=BoomStorage(),
        on_complete=lambda _s: asyncio.sleep(0),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        page = await client.get(f"/upload/{session.upload_id}?t={session.token}")
        assert page.status == 200
        form = FormData()
        form.add_field(
            "file",
            _wav_bytes(),
            filename="a.wav",
            content_type="audio/wav",
        )
        resp = await client.post(f"/api/uploads/{session.upload_id}/file", data=form)
        body = await resp.json()
        assert resp.status == 500
        assert body["error"] == "Ошибка загрузки"
        assert "secret-xyz" not in str(body)
        assert "10.0.0.5" not in str(body)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_upload_post_is_rate_limited_before_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import AsyncMock

    from app.infra.rate_limiter import rate_limiter

    store = UploadSessionStore(redis=None)
    storage = FakeStorage()
    session = await store.create(chat_id=1, user_id=2, purpose="transcribe")
    app = web.Application(client_max_size=10 * 1024 * 1024)
    setup_upload_routes(
        app,
        store=store,
        storage=storage,
        on_complete=lambda _s: asyncio.sleep(0),
    )
    monkeypatch.setattr(
        rate_limiter, "check_upload_limit", AsyncMock(return_value=False)
    )
    monkeypatch.setattr(
        rate_limiter, "get_upload_retry_after", AsyncMock(return_value=42.0)
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
        form.add_field(
            "file",
            b"not-a-real-upload",
            filename="lecture.mp3",
            content_type="audio/mpeg",
        )
        resp = await client.post(
            f"/api/uploads/{session.upload_id}/file?t={session.token}",
            data=form,
        )
        assert resp.status == 429
        assert resp.headers["Retry-After"] == "42"
        body = await resp.json()
        assert "позже" in body["error"]
        assert storage.objects == {}
        saved = await store.get_authorized(session.upload_id, session.token)
        assert saved is not None
        assert saved.status == "pending"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_request_large_media_upload_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.media.media_tools import RequestLargeMediaUploadTool
    from app.storage.upload_sessions import (
        UploadRequestContext,
        UploadSessionStore,
        reset_upload_request_context,
        set_upload_request_context,
    )

    monkeypatch.setenv("UPLOAD_PUBLIC_BASE_URL", "https://bot.example.com")
    monkeypatch.setenv("YC_BUCKET_NAME", "bucket")
    monkeypatch.setenv("YC_ACCESS_KEY_ID", "key")
    monkeypatch.setenv("YC_SECRET_ACCESS_KEY", "secret")

    class FakeClient:
        upload_session_store = UploadSessionStore(redis=None)

    token = set_upload_request_context(UploadRequestContext(chat_id=1, user_id=2))
    try:
        tool = RequestLargeMediaUploadTool(FakeClient())  # type: ignore[arg-type]
        raw = await tool.execute(media_kind="audio")
    finally:
        reset_upload_request_context(token)

    import json

    data = json.loads(raw)
    assert data["status"] == "success"
    assert data["upload_url"].startswith("https://bot.example.com/upload/")
    assert "t=" in data["upload_url"]


class _ChunkField:
    def __init__(self, parts: list[bytes]) -> None:
        self._parts = list(parts)
        self.calls = 0

    async def read_chunk(self, size: int = 8192) -> bytes:
        self.calls += 1
        if not self._parts:
            return b""
        return self._parts.pop(0)


@pytest.mark.asyncio
async def test_copy_stops_after_the_chunk_that_crosses_the_limit() -> None:
    field = _ChunkField([b"a" * 100, b"b" * 100, b"c" * 100])
    dest = io.BytesIO()
    size, too_large = await _copy_until_limit(field, dest, 150)
    assert too_large is True
    assert size == 200
    assert field.calls == 2
    assert dest.getvalue() == b"a" * 100


@pytest.mark.asyncio
async def test_oversized_upload_closes_without_storing() -> None:
    store = UploadSessionStore(redis=None)
    storage = FakeStorage()
    storage.config.max_upload_mb = 1
    session = await store.create(chat_id=1, user_id=2, purpose="transcribe")
    app = web.Application(client_max_size=8 * 1024 * 1024)
    setup_upload_routes(
        app,
        store=store,
        storage=storage,
        on_complete=lambda _s: asyncio.sleep(0),
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        form = FormData()
        form.add_field(
            "file",
            b"\x00" * (2 * 1024 * 1024),
            filename="a.wav",
            content_type="audio/wav",
        )
        resp = await client.post(
            f"/api/uploads/{session.upload_id}/file",
            data=form,
            headers={"X-Upload-Token": session.token},
        )
        body = await resp.json()
        assert resp.status == 413
        assert resp.headers["Connection"].lower() == "close"
        assert body["error"] == "Файл больше 1 МБ"
        saved = await store.get(session.upload_id)
        assert saved is not None
        assert saved.status == "pending"
        assert storage.objects == {}
    finally:
        await client.close()

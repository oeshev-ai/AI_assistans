"""Тесты HTTP API загрузки в бакет."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer

from app.media.upload_server import setup_upload_routes
from app.storage.upload_sessions import UploadSessionStore


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
            from app.storage.s3_storage import S3StorageError

            raise S3StorageError("missing")
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
            b"ID3fake-mp3-bytes",
            filename="lecture.mp3",
            content_type="audio/mpeg",
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
            b"ID3fake-mp3-bytes-2",
            filename="lecture2.mp3",
            content_type="audio/mpeg",
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
            b"ID3bytes",
            filename="a.mp3",
            content_type="audio/mpeg",
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

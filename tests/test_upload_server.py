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
        page = await client.get(f"/upload/{session.upload_id}?t={session.token}")
        assert page.status == 200
        text = await page.text()
        assert "/api/uploads/" in text
        assert "FormData" in text

        form = FormData()
        form.add_field(
            "file",
            b"ID3fake-mp3-bytes",
            filename="lecture.mp3",
            content_type="audio/mpeg",
        )
        resp = await client.post(
            f"/api/uploads/{session.upload_id}/file?t={session.token}",
            data=form,
        )
        assert resp.status == 200, await resp.text()
        body = await resp.json()
        assert body["ok"] is True
        assert body["objectKey"] in storage.objects

        for _ in range(50):
            if completed:
                break
            await asyncio.sleep(0.01)
        assert completed == [session.upload_id]
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

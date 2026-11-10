"""Тесты HTTP API загрузки в бакет."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from upload_server import setup_upload_routes
from upload_sessions import UploadSessionStore


@dataclass
class FakeS3Config:
    signed_url_expires_in: int = 1800
    max_upload_mb: int = 100


class FakeStorage:
    def __init__(self) -> None:
        self.config = FakeS3Config()
        self.objects: dict[str, bytes] = {}

    async def create_presigned_put(self, *, object_key: str, content_type: str) -> str:
        return f"https://storage.example/{object_key}?sig=1&ct={content_type}"

    async def head_object(self, object_key: str) -> dict[str, Any]:
        data = self.objects.get(object_key)
        if data is None:
            from s3_storage import S3StorageError

            raise S3StorageError("missing")
        return {"ContentLength": len(data), "ContentType": "audio/mpeg"}

    async def download_bytes(self, object_key: str) -> bytes:
        return self.objects[object_key]

    async def delete_object(self, object_key: str) -> None:
        self.objects.pop(object_key, None)


@pytest.mark.asyncio
async def test_presign_and_complete_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    store = UploadSessionStore(redis=None)
    storage = FakeStorage()
    completed: list[str] = []

    async def on_complete(session: Any) -> None:
        completed.append(session.upload_id)

    session = await store.create(chat_id=42, user_id=7, purpose="transcribe")
    app = web.Application()
    setup_upload_routes(
        app, store=store, storage=storage, on_complete=on_complete  # type: ignore[arg-type]
    )
    app["upload_enabled"] = True

    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    try:
        page = await client.get(f"/upload/{session.upload_id}?t={session.token}")
        assert page.status == 200
        assert "Большой файл" in await page.text()

        bad = await client.post(
            f"/api/uploads/{session.upload_id}/presign?t=bad",
            json={"fileName": "a.mp3", "fileType": "audio/mpeg", "fileSize": 10},
        )
        assert bad.status == 404

        presign = await client.post(
            f"/api/uploads/{session.upload_id}/presign?t={session.token}",
            json={
                "fileName": "lecture.mp3",
                "fileType": "audio/mpeg",
                "fileSize": 1024,
            },
        )
        assert presign.status == 200
        payload = await presign.json()
        assert "signedUrl" in payload
        assert payload["objectKey"].startswith(f"uploads/{session.upload_id}/")

        storage.objects[payload["objectKey"]] = b"x" * 100
        complete = await client.post(
            f"/api/uploads/{session.upload_id}/complete?t={session.token}",
            json={
                "objectKey": payload["objectKey"],
                "fileName": "lecture.mp3",
                "fileType": "audio/mpeg",
                "fileSize": 100,
            },
        )
        assert complete.status == 200
        body = await complete.json()
        assert body["ok"] is True

        # background task
        for _ in range(50):
            if completed:
                break
            await asyncio_sleep()
        assert completed == [session.upload_id]
    finally:
        await client.close()


async def asyncio_sleep() -> None:
    import asyncio

    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_request_large_media_upload_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    from media_tools import RequestLargeMediaUploadTool
    from upload_sessions import (
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

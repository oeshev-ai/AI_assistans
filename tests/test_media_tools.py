"""Unit tests for media tools outbound transcript file."""

from __future__ import annotations

import json

import pytest

from app.media.media_tools import (
    MediaTurnScope,
    TranscribeMediaTool,
    UserMedia,
    set_user_media,
)


@pytest.mark.asyncio
async def test_transcribe_media_sends_transcript_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_prepare(data, *, filename, is_video):
        return [(b"chunk", "chunk.mp3")]

    monkeypatch.setattr("app.media.media_tools.prepare_audio_for_stt", fake_prepare)

    class FakeClient:
        async def transcribe_audio(self, audio_bytes, filename="x", language=None):
            return "hello world " * 20

    tool = TranscribeMediaTool(FakeClient())  # type: ignore[arg-type]
    token = set_user_media(
        [
            UserMedia(
                kind="audio",
                filename="long.mp3",
                mime_type="audio/mpeg",
                data=b"raw",
            )
        ]
    )
    try:
        with MediaTurnScope() as scope:
            raw = await tool.execute()
            payload = json.loads(raw)
            assert payload["status"] == "success"
            assert payload["chars"] > 0
            assert len(scope.bucket.items) == 1
            doc = scope.bucket.items[0]
            assert doc.kind == "document"
            assert doc.filename == "transcript.txt"
            assert doc.data is not None
            assert b"hello world" in doc.data
    finally:
        from app.media.media_tools import reset_user_media

        reset_user_media(token)

"""Tests for audio extraction helpers (ffmpeg mocked)."""

from __future__ import annotations

import pytest

from media_utils import MediaProcessingError, prepare_audio_for_stt, split_audio_for_stt


@pytest.mark.asyncio
async def test_split_short_audio_passthrough() -> None:
    raw = b"tiny-audio"
    parts = await split_audio_for_stt(raw, filename="a.mp3")
    assert parts == [(raw, "a.mp3")]


@pytest.mark.asyncio
async def test_prepare_video_calls_extract(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_extract(video_bytes: bytes, *, source_name: str = "video.mp4"):
        assert video_bytes == b"video"
        return b"audio-bytes", "audio.mp3"

    monkeypatch.setattr("media_utils.extract_audio_from_video", fake_extract)
    parts = await prepare_audio_for_stt(b"video", filename="clip.mp4", is_video=True)
    assert parts == [(b"audio-bytes", "audio.mp3")]


@pytest.mark.asyncio
async def test_empty_audio_raises() -> None:
    with pytest.raises(MediaProcessingError):
        await split_audio_for_stt(b"")

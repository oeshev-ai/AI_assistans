"""Tests for audio extraction helpers (ffmpeg mocked)."""

from __future__ import annotations

import asyncio

import pytest

from app.media import media_utils
from app.media.media_utils import (
    MediaProcessingError,
    _duration_from_ffmpeg_text,
    audio_duration_seconds,
    communicate_with_timeout,
    extract_audio_from_video,
    prepare_audio_for_stt,
    split_audio_for_stt,
)


class _HungProc:
    def __init__(self) -> None:
        self.killed = False
        self.returncode: int | None = None

    async def communicate(self) -> tuple[bytes, bytes]:
        await asyncio.Event().wait()
        return b"", b""

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        self.returncode = -9
        return -9


class _FailProc:
    def __init__(self) -> None:
        self.killed = False
        self.returncode = 1

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b"no stream"

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


def test_duration_parser_reads_ffmpeg_line() -> None:
    text = "Input #0, mov, from 'a.mp4':\n  Duration: 01:02:03.50, start: 0.000000\n"
    assert _duration_from_ffmpeg_text(text) == pytest.approx(3723.5)
    assert _duration_from_ffmpeg_text("no duration") is None


@pytest.mark.asyncio
async def test_audio_duration_falls_back_to_extracted_bitrate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_probe(data: bytes, *, filename: str) -> None:
        return None

    monkeypatch.setattr(media_utils, "probe_duration_seconds", no_probe)
    seconds = await audio_duration_seconds([(b"x" * 16000, "audio.mp3")])
    assert seconds == pytest.approx(2.0)


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

    monkeypatch.setattr("app.media.media_utils.extract_audio_from_video", fake_extract)
    parts = await prepare_audio_for_stt(b"video", filename="clip.mp4", is_video=True)
    assert parts == [(b"audio-bytes", "audio.mp3")]


@pytest.mark.asyncio
async def test_empty_audio_raises() -> None:
    with pytest.raises(MediaProcessingError):
        await split_audio_for_stt(b"")


@pytest.mark.asyncio
async def test_hung_ffmpeg_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[_HungProc] = []

    async def fake_exec(*_args: object, **_kwargs: object) -> _HungProc:
        proc = _HungProc()
        started.append(proc)
        return proc

    monkeypatch.setattr(media_utils, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(media_utils, "_FFMPEG_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(media_utils.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(MediaProcessingError, match="слишком много времени"):
        await extract_audio_from_video(b"video", source_name="clip.mp4")

    assert len(started) == 1
    assert started[0].killed


@pytest.mark.asyncio
async def test_hung_split_does_not_start_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[_HungProc] = []

    async def fake_exec(*_args: object, **_kwargs: object) -> _HungProc:
        proc = _HungProc()
        started.append(proc)
        return proc

    monkeypatch.setattr(media_utils, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(media_utils, "_FFMPEG_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(media_utils, "_MAX_CHUNK_BYTES", 1)
    monkeypatch.setattr(media_utils.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(MediaProcessingError, match="слишком много времени"):
        await split_audio_for_stt(b"long-audio", filename="a.mp3")

    assert len(started) == 1
    assert started[0].killed


@pytest.mark.asyncio
async def test_extract_failure_still_tries_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started: list[_FailProc] = []

    async def fake_exec(*_args: object, **_kwargs: object) -> _FailProc:
        proc = _FailProc()
        started.append(proc)
        return proc

    monkeypatch.setattr(media_utils, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(media_utils.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(MediaProcessingError, match="Не удалось извлечь аудио"):
        await extract_audio_from_video(b"video", source_name="clip.mp4")

    assert len(started) == 2
    assert not any(proc.killed for proc in started)


@pytest.mark.asyncio
async def test_cancel_kills_ffmpeg() -> None:
    proc = _HungProc()
    task = asyncio.create_task(communicate_with_timeout(proc, timeout=30))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert proc.killed

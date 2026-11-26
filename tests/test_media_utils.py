"""Tests for audio extraction helpers (ffmpeg mocked)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.media import media_utils
from app.media.media_utils import (
    MediaProcessingError,
    SttChunks,
    _duration_from_ffmpeg_text,
    audio_duration_seconds,
    close_stt_chunks,
    communicate_with_timeout,
    extract_audio_from_video,
    iter_stt_chunk_bytes,
    prepare_audio_file_for_stt,
    prepare_audio_for_stt,
    split_audio_file_for_stt,
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


class _OkProc:
    def __init__(self) -> None:
        self.returncode = 0

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""

    def kill(self) -> None:
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode


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
async def test_long_file_is_split_from_disk_not_loaded_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "big.mp3"
    src.write_bytes(b"0123456789")
    seen: dict[str, object] = {}

    async def fake_split(
        path: Path, *, source_bytes: int, chunk_seconds: int
    ) -> list[tuple[bytes, str]]:
        seen["path"] = path
        seen["source_bytes"] = source_bytes
        seen["chunk_seconds"] = chunk_seconds
        return [(b"aa", "chunk_001.mp3"), (b"bb", "chunk_002.mp3")]

    monkeypatch.setattr(media_utils, "_MAX_CHUNK_BYTES", 4)
    monkeypatch.setattr(media_utils, "_split_audio_path", fake_split)
    parts = await split_audio_file_for_stt(src, filename="big.mp3")
    assert seen["path"] == src
    assert seen["source_bytes"] == 10
    assert b"0123456789" not in {item for item, _name in parts}
    assert parts == [(b"aa", "chunk_001.mp3"), (b"bb", "chunk_002.mp3")]


@pytest.mark.asyncio
async def test_long_split_reads_one_chunk_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "big.mp3"
    src.write_bytes(b"0123456789")
    reads: list[int] = []

    async def fake_exec(*args: object, **_kwargs: object) -> _OkProc:
        pattern = next(
            item for item in args if isinstance(item, str) and "chunk_%03d.mp3" in item
        )
        folder = Path(pattern).parent
        (folder / "chunk_000.mp3").write_bytes(b"aaaa")
        (folder / "chunk_001.mp3").write_bytes(b"bbbb")
        return _OkProc()

    real_read = Path.read_bytes

    def counting_read(self: Path) -> bytes:
        reads.append(self.stat().st_size)
        return real_read(self)

    monkeypatch.setattr(media_utils, "_MAX_CHUNK_BYTES", 4)
    monkeypatch.setattr(media_utils, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(media_utils.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(Path, "read_bytes", counting_read)
    chunks = await split_audio_file_for_stt(src, filename="big.mp3")
    try:
        assert isinstance(chunks, SttChunks)
        assert reads == []
        assert len(chunks) == 2
        loaded: list[tuple[bytes, str]] = []
        async for item in iter_stt_chunk_bytes(chunks):
            loaded.append(item)
            assert len(reads) == len(loaded)
        assert loaded == [
            (b"aaaa", "chunk_001.mp3"),
            (b"bbbb", "chunk_002.mp3"),
        ]
    finally:
        close_stt_chunks(chunks)


@pytest.mark.asyncio
async def test_oversized_chunk_is_rejected_without_keeping_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "big.mp3"
    src.write_bytes(b"0123456789")
    seen: dict[str, Path] = {}

    async def fake_exec(*args: object, **_kwargs: object) -> _OkProc:
        pattern = next(
            item for item in args if isinstance(item, str) and "chunk_%03d.mp3" in item
        )
        folder = Path(pattern).parent
        seen["dir"] = folder
        (folder / "chunk_000.mp3").write_bytes(b"aaaaa")
        return _OkProc()

    monkeypatch.setattr(media_utils, "_MAX_CHUNK_BYTES", 4)
    monkeypatch.setattr(media_utils, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(media_utils.asyncio, "create_subprocess_exec", fake_exec)

    with pytest.raises(MediaProcessingError, match="слишком большой"):
        await split_audio_file_for_stt(src, filename="big.mp3")
    assert not seen["dir"].exists()


@pytest.mark.asyncio
async def test_chunk_duration_uses_file_size_not_a_full_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    folder = tempfile.TemporaryDirectory()
    path = Path(folder.name) / "chunk_000.mp3"
    path.write_bytes(b"x" * 16000)
    chunks = SttChunks(folder, [(path, "chunk_001.mp3")], source_bytes=16000)

    async def no_probe(_path: Path) -> None:
        return None

    reads: list[Path] = []
    real_read = Path.read_bytes

    def counting_read(self: Path) -> bytes:
        reads.append(self)
        return real_read(self)

    monkeypatch.setattr(media_utils, "probe_file_duration_seconds", no_probe)
    monkeypatch.setattr(Path, "read_bytes", counting_read)
    try:
        seconds = await audio_duration_seconds(chunks)
    finally:
        chunks.close()
    assert seconds == pytest.approx(2.0)
    assert reads == []


@pytest.mark.asyncio
async def test_prepare_video_file_extracts_without_reading_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "clip.mp4"
    src.write_bytes(b"video-bytes-not-loaded")
    extracted: list[Path] = []

    async def fake_extract(path: Path, dst: Path) -> None:
        extracted.append(path)
        dst.write_bytes(b"mp3")

    async def fake_split(
        path: Path, *, filename: str, chunk_seconds: int = 600
    ) -> list[tuple[bytes, str]]:
        assert path.read_bytes() == b"mp3"
        assert filename == "audio.mp3"
        return [(b"mp3", filename)]

    monkeypatch.setattr(media_utils, "_extract_audio_to", fake_extract)
    monkeypatch.setattr(media_utils, "split_audio_file_for_stt", fake_split)
    parts = await prepare_audio_file_for_stt(src, filename="clip.mp4", is_video=True)
    assert extracted == [src]
    assert parts == [(b"mp3", "audio.mp3")]


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

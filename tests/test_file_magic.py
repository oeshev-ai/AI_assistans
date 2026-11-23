"""Контейнер загрузки: ffmpeg, а не префикс сигнатуры."""

from __future__ import annotations

import io
import subprocess
import tempfile
import wave
from pathlib import Path

import pytest

from app.media.file_magic import container_mime, mime_compatible
from app.media.media_utils import MediaProcessingError, find_ffmpeg


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(8000)
        handle.writeframes(b"\x00\x00" * 1600)
    return buf.getvalue()


def _ffmpeg() -> str:
    try:
        return find_ffmpeg()
    except MediaProcessingError:
        pytest.skip("ffmpeg не установлен")
        return ""


def test_nonstandard_mp4_brand_stays_audio() -> None:
    """Major brand вне короткого списка не меняет семейство, если есть аудиодорожка."""
    mime = container_mime(
        "mov,mp4,m4a,3gp,3g2,mj2",
        has_audio=True,
        has_video=False,
        ext=".m4a",
    )
    assert mime == "audio/mp4"


@pytest.mark.asyncio
async def test_rejects_forged_id3() -> None:
    assert not await mime_compatible(
        filename="lecture.mp3",
        claimed_mime="audio/mpeg",
        source=b"ID3" + b"\x00" * 32 + b"%PDF-1.7 fake",
    )


@pytest.mark.asyncio
async def test_rejects_pdf_disguised_as_mp3() -> None:
    assert not await mime_compatible(
        filename="lecture.mp3",
        claimed_mime="audio/mpeg",
        source=b"%PDF-1.7 fake",
    )


@pytest.mark.asyncio
async def test_accepts_wav() -> None:
    _ffmpeg()
    assert await mime_compatible(
        filename="lecture.wav",
        claimed_mime="audio/wav",
        source=_wav_bytes(),
    )


@pytest.mark.asyncio
async def test_rejects_wav_claimed_as_mpeg() -> None:
    _ffmpeg()
    assert not await mime_compatible(
        filename="lecture.wav",
        claimed_mime="audio/mpeg",
        source=_wav_bytes(),
    )


def _m4a_with_custom_brand() -> bytes:
    """Настоящий M4A, у которого major brand заменён на неизвестный."""
    ffmpeg = _ffmpeg()
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "in.wav"
        m4a_path = Path(tmp) / "out.m4a"
        wav_path.write_bytes(_wav_bytes())
        encoded = subprocess.run(  # noqa: S603
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-i",
                str(wav_path),
                "-c:a",
                "aac",
                str(m4a_path),
            ],
            capture_output=True,
            check=False,
        )
        if encoded.returncode != 0 or not m4a_path.is_file():
            pytest.skip("ffmpeg не собрал m4a")
        raw = bytearray(m4a_path.read_bytes())
    brand_at = raw.find(b"ftyp")
    assert brand_at > 0
    raw[brand_at + 4 : brand_at + 8] = b"zzzz"
    return bytes(raw)


@pytest.mark.asyncio
async def test_accepts_m4a_with_unknown_major_brand() -> None:
    assert await mime_compatible(
        filename="voice.m4a",
        claimed_mime="audio/mp4",
        source=_m4a_with_custom_brand(),
    )

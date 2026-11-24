"""Проверка контейнера загрузки через ffmpeg.

Сигнатура в начале файла этого не делает. MP4 с незнакомым major brand
остаётся обычным ISO-BMFF, а префикс ID3 или ftyp без настоящего контейнера
ffmpeg не открывает.
"""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path

import structlog

from app.media.media_utils import (
    MediaProcessingError,
    communicate_with_timeout,
    find_ffmpeg,
)

logger = structlog.get_logger("file_magic")

_PROBE_TIMEOUT_SECONDS = 30.0

# Расширение → допустимые MIME того же семейства.
_EXT_FAMILIES: dict[str, frozenset[str]] = {
    ".mp3": frozenset({"audio/mpeg", "audio/mp3"}),
    ".m4a": frozenset({"audio/mp4", "audio/m4a", "audio/x-m4a"}),
    ".wav": frozenset({"audio/wav", "audio/x-wav"}),
    ".ogg": frozenset({"audio/ogg", "video/ogg"}),
    ".oga": frozenset({"audio/ogg"}),
    ".webm": frozenset({"audio/webm", "video/webm"}),
    ".mp4": frozenset({"video/mp4", "audio/mp4"}),
    ".mov": frozenset({"video/quicktime", "video/mp4"}),
    ".mkv": frozenset({"video/x-matroska", "video/webm"}),
}

_CLAIM_ALIASES: dict[str, str] = {
    "audio/mp3": "audio/mpeg",
    "audio/x-wav": "audio/wav",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
}

_ISO_FORMATS = frozenset({"mov", "mp4", "m4a", "3gp", "3g2", "mj2"})
_INPUT_RE = re.compile(r"Input #\d+,\s*(.+),\s+from\s+")
_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?:\s*(Audio|Video):")


def normalize_mime(value: str) -> str:
    lowered = value.strip().lower().split(";", 1)[0].strip()
    return _CLAIM_ALIASES.get(lowered, lowered)


def _parse_probe(stderr: str) -> tuple[str, bool, bool] | None:
    """Имя формата ffmpeg и наличие аудио/видео. ``None``, если контейнер не открылся."""
    if "Error opening input" in stderr:
        return None
    match = _INPUT_RE.search(stderr)
    if match is None:
        return None
    kinds = {item.group(1).lower() for item in _STREAM_RE.finditer(stderr)}
    has_audio = "audio" in kinds
    has_video = "video" in kinds
    if not has_audio and not has_video:
        return None
    return match.group(1).strip(), has_audio, has_video


def container_mime(
    format_name: str,
    *,
    has_audio: bool,
    has_video: bool,
    ext: str,
) -> str | None:
    """MIME по формату ffmpeg и дорожкам, с учётом расширения."""
    if not has_audio and not has_video:
        return None
    tokens = {part.strip().lower() for part in format_name.split(",") if part.strip()}
    if "mp3" in tokens:
        return "audio/mpeg" if has_audio else None
    if "wav" in tokens:
        return "audio/wav" if has_audio else None
    if "ogg" in tokens:
        if has_video and not has_audio:
            return "video/ogg"
        return "audio/ogg" if has_audio else None
    if tokens & {"matroska", "webm"}:
        if "webm" in tokens:
            return "video/webm" if has_video else "audio/webm" if has_audio else None
        return "video/x-matroska" if has_video or has_audio else None
    if tokens & _ISO_FORMATS:
        if ext == ".m4a":
            return "audio/mp4" if has_audio else None
        if ext == ".mov":
            return (
                "video/quicktime" if has_video else "audio/mp4" if has_audio else None
            )
        if has_video:
            return "video/mp4"
        if has_audio:
            return "audio/mp4"
    return None


def _spill_probe_file(data: bytes, suffix: str) -> Path:
    """Пишет байты во временный файл вне event loop."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        return Path(tmp.name)


def _claim_ok(claimed_mime: str, family: frozenset[str]) -> bool:
    if not claimed_mime or claimed_mime == "application/octet-stream":
        return True
    return normalize_mime(claimed_mime) in family


async def _probe(path: Path) -> tuple[str, bool, bool] | None:
    try:
        ffmpeg = find_ffmpeg()
    except MediaProcessingError:
        logger.error("mime_probe_ffmpeg_missing")
        return None
    proc = await asyncio.create_subprocess_exec(
        ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-i",
        str(path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _stdout, stderr = await communicate_with_timeout(
            proc, timeout=_PROBE_TIMEOUT_SECONDS
        )
    except TimeoutError:
        logger.warning("mime_probe_timeout")
        return None
    return _parse_probe((stderr or b"").decode("utf-8", errors="replace"))


async def mime_compatible(
    *,
    filename: str,
    claimed_mime: str,
    source: bytes | Path,
) -> bool:
    """True, если ffmpeg открыл контейнер того же семейства, что расширение и MIME.

    ``source`` — байты файла или путь к уже записанному временному файлу.
    Поддельный префикс без контейнера даёт False. Нестандартный major brand
    у настоящего MP4/M4A остаётся True.
    """
    ext = Path(filename).suffix.lower()
    family = _EXT_FAMILIES.get(ext)
    if family is None or not _claim_ok(claimed_mime, family):
        return False
    if isinstance(source, Path):
        if not source.is_file() or source.stat().st_size <= 0:
            return False
        probed = await _probe(source)
    else:
        if not source:
            return False
        tmp_path = await asyncio.to_thread(_spill_probe_file, source, ext)
        try:
            probed = await _probe(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)
    if probed is None:
        return False
    format_name, has_audio, has_video = probed
    sniffed = container_mime(
        format_name, has_audio=has_audio, has_video=has_video, ext=ext
    )
    if sniffed is None:
        return False
    return normalize_mime(sniffed) in family

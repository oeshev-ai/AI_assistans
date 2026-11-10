"""Извлечение и нарезка аудио для STT (ProxyAPI принимает только аудио)."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

import structlog

logger = structlog.get_logger("media_utils")

# Whisper-класс API обычно ограничивает размер ~25 МБ.
_MAX_CHUNK_BYTES = 24 * 1024 * 1024
_DEFAULT_CHUNK_SECONDS = 600


class MediaProcessingError(Exception):
    """Сбой ffmpeg или неподдерживаемый медиафайл."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


def find_ffmpeg() -> str:
    """Ищет ffmpeg в PATH или во встроенном бинарнике imageio-ffmpeg."""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise MediaProcessingError(
            "ffmpeg не найден. Установите ffmpeg или пакет imageio-ffmpeg."
        ) from exc


async def extract_audio_from_video(
    video_bytes: bytes,
    *,
    source_name: str = "video.mp4",
) -> tuple[bytes, str]:
    """Достаёт аудиодорожку из видео в MP3.

    Returns:
        ``(audio_bytes, filename)``.
    """
    if not video_bytes:
        raise MediaProcessingError("Пустой видеофайл.")
    ffmpeg = find_ffmpeg()
    suffix = Path(source_name).suffix or ".mp4"
    with tempfile.TemporaryDirectory(prefix="assist_media_") as tmp:
        src = Path(tmp) / f"input{suffix}"
        dst = Path(tmp) / "audio.mp3"
        src.write_bytes(video_bytes)
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-vn",
            "-acodec",
            "libmp3lame",
            "-q:a",
            "4",
            str(dst),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0 or not dst.exists():
            detail = (stderr or b"").decode("utf-8", errors="replace")[-500:]
            logger.error("ffmpeg_extract_failed", detail=detail)
            raise MediaProcessingError(
                "Не удалось извлечь аудио из видео. Проверьте формат файла."
            )
        audio = dst.read_bytes()
        if not audio:
            raise MediaProcessingError("В видео нет звуковой дорожки.")
        logger.info(
            "audio_extracted",
            video_bytes=len(video_bytes),
            audio_bytes=len(audio),
        )
        return audio, "audio.mp3"


async def split_audio_for_stt(
    audio_bytes: bytes,
    *,
    filename: str = "audio.mp3",
    chunk_seconds: int = _DEFAULT_CHUNK_SECONDS,
) -> list[tuple[bytes, str]]:
    """Режет длинное аудио на куски, пригодные для transcriptions API.

    Короткий файл возвращается как один элемент без перекодирования.
    """
    if not audio_bytes:
        raise MediaProcessingError("Пустой аудиофайл.")
    if len(audio_bytes) <= _MAX_CHUNK_BYTES:
        return [(audio_bytes, filename)]

    ffmpeg = find_ffmpeg()
    suffix = Path(filename).suffix or ".mp3"
    with tempfile.TemporaryDirectory(prefix="assist_stt_") as tmp:
        src = Path(tmp) / f"input{suffix}"
        src.write_bytes(audio_bytes)
        pattern = str(Path(tmp) / "chunk_%03d.mp3")
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-f",
            "segment",
            "-segment_time",
            str(max(60, chunk_seconds)),
            "-c",
            "copy",
            pattern,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        chunks = sorted(Path(tmp).glob("chunk_*.mp3"))
        if proc.returncode != 0 or not chunks:
            # copy может не сработать без ключевых кадров — перекодируем.
            detail = (stderr or b"").decode("utf-8", errors="replace")[-300:]
            logger.warning("ffmpeg_segment_copy_failed", detail=detail)
            proc = await asyncio.create_subprocess_exec(
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-f",
                "segment",
                "-segment_time",
                str(max(60, chunk_seconds)),
                "-acodec",
                "libmp3lame",
                "-q:a",
                "4",
                pattern,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            chunks = sorted(Path(tmp).glob("chunk_*.mp3"))
        if not chunks:
            raise MediaProcessingError("Не удалось нарезать длинное аудио для STT.")
        result: list[tuple[bytes, str]] = []
        for index, path in enumerate(chunks):
            data = path.read_bytes()
            if not data:
                continue
            # Если кусок всё ещё слишком большой — режем ещё раз пополам по времени.
            if len(data) > _MAX_CHUNK_BYTES:
                raise MediaProcessingError(
                    "Фрагмент аудио слишком большой для распознавания. "
                    "Попробуйте файл покороче или с меньшим битрейтом."
                )
            name = f"chunk_{index + 1:03d}.mp3"
            result.append((data, name))
        if not result:
            raise MediaProcessingError("После нарезки не осталось аудиоданных.")
        logger.info("audio_split", parts=len(result), source_bytes=len(audio_bytes))
        return result


async def prepare_audio_for_stt(
    data: bytes,
    *,
    filename: str,
    is_video: bool,
) -> list[tuple[bytes, str]]:
    """Готовит один или несколько аудиофрагментов для ProxyAPI STT."""
    if is_video:
        data, filename = await extract_audio_from_video(data, source_name=filename)
    return await split_audio_for_stt(data, filename=filename)

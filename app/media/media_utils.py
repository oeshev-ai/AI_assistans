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


_MIN_IMAGE_AREA = 655_360
_MAX_IMAGE_AREA = 8_294_400
_MAX_IMAGE_SIDE = 3840
_MAX_IMAGE_RATIO = 3.0


def image_dimensions(data: bytes) -> tuple[int, int]:
    """Ширина и высота первого кадра. Пустой или битый файл — ошибка."""
    from io import BytesIO

    from PIL import Image

    if not data:
        raise MediaProcessingError("Пустой файл изображения.")
    with Image.open(BytesIO(data)) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise MediaProcessingError("Не удалось прочитать размер изображения.")
    return width, height


def source_image_size(data: bytes) -> tuple[int, int]:
    """Размер после EXIF-поворота. Для поля ``size`` берём оригинал, не даунскейл."""
    from io import BytesIO

    from PIL import Image, ImageOps

    if not data:
        raise MediaProcessingError("Пустой файл изображения.")
    with Image.open(BytesIO(data)) as image:
        oriented = ImageOps.exif_transpose(image) or image
        try:
            width, height = oriented.size
        finally:
            if oriented is not image:
                oriented.close()
    if width <= 0 or height <= 0:
        raise MediaProcessingError("Не удалось прочитать размер изображения.")
    return width, height


def downscale_for_edit(data: bytes, mime: str, max_edge: int) -> tuple[bytes, str]:
    """Сжимает длинную сторону до ``max_edge``. ``0`` выключает даунскейл.

    Файл и так меньше порога возвращается байт-в-байт: без апскейла и перекодирования.
    """
    if max_edge <= 0:
        return data, mime
    from io import BytesIO

    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError):
        logger.warning("edit_downscale_skipped")
        return data, mime
    with image:
        oriented = ImageOps.exif_transpose(image) or image
        try:
            if max(oriented.size) <= max_edge:
                return data, mime
            scale = max_edge / max(oriented.size)
            resized = oriented.resize(
                (
                    max(1, round(oriented.width * scale)),
                    max(1, round(oriented.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )
            buffer = BytesIO()
            if resized.mode in {"RGBA", "LA", "P"}:
                resized.convert("RGBA").save(buffer, format="PNG", optimize=True)
                return buffer.getvalue(), "image/png"
            resized.convert("RGB").save(
                buffer, format="JPEG", quality=92, optimize=True
            )
            return buffer.getvalue(), "image/jpeg"
        finally:
            if oriented is not image:
                oriented.close()


def fit_image_size(width: int, height: int) -> str:
    """Ближайший валидный размер GPT Image.

    Стороны кратны 16, длинная не больше 3840, соотношение не больше 3:1,
    площадь от 655_360 до 8_294_400. Исходные пропорции сохраняются.
    """
    wide = max(int(width), 1)
    tall = max(int(height), 1)
    if wide / tall > _MAX_IMAGE_RATIO:
        wide = max(1, round(tall * _MAX_IMAGE_RATIO))
    elif tall / wide > _MAX_IMAGE_RATIO:
        tall = max(1, round(wide * _MAX_IMAGE_RATIO))
    area = wide * tall
    if area > _MAX_IMAGE_AREA:
        scale = (_MAX_IMAGE_AREA / area) ** 0.5
    elif area < _MIN_IMAGE_AREA:
        scale = (_MIN_IMAGE_AREA / area) ** 0.5
    else:
        scale = 1.0
    wide = max(1, round(wide * scale))
    tall = max(1, round(tall * scale))
    long_side = max(wide, tall)
    if long_side > _MAX_IMAGE_SIDE:
        side_scale = _MAX_IMAGE_SIDE / long_side
        wide = max(1, round(wide * side_scale))
        tall = max(1, round(tall * side_scale))
    wide = _align16(wide)
    tall = _align16(tall)
    wide, tall = _constrain_image_size(wide, tall)
    return f"{wide}x{tall}"


def _align16(value: int) -> int:
    aligned = round(value / 16) * 16
    return max(16, aligned)


def _align16_down(value: int) -> int:
    return max(16, (int(value) // 16) * 16)


def _constrain_image_size(width: int, height: int) -> tuple[int, int]:
    """После округления до 16 возвращает размер внутрь лимитов GPT Image."""
    wide, tall = width, height
    for _ in range(8):
        if wide / tall > _MAX_IMAGE_RATIO:
            wide = _align16(int(tall * _MAX_IMAGE_RATIO))
        elif tall / wide > _MAX_IMAGE_RATIO:
            tall = _align16(int(wide * _MAX_IMAGE_RATIO))
        if max(wide, tall) > _MAX_IMAGE_SIDE:
            side_scale = _MAX_IMAGE_SIDE / max(wide, tall)
            wide = _align16(int(wide * side_scale))
            tall = _align16(int(tall * side_scale))
        area = wide * tall
        if area > _MAX_IMAGE_AREA:
            scale = (_MAX_IMAGE_AREA / area) ** 0.5
            wide = _align16_down(int(wide * scale))
            tall = _align16_down(int(tall * scale))
            continue
        if area < _MIN_IMAGE_AREA:
            scale = (_MIN_IMAGE_AREA / area) ** 0.5
            wide = _align16(int(wide * scale))
            tall = _align16(int(tall * scale))
            continue
        return wide, tall
    return wide, tall


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
        # 16 kHz mono — оптимально для Whisper; явный map первой аудиодорожки.
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-map",
            "0:a:0?",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "libmp3lame",
            "-b:a",
            "64k",
            str(dst),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _stdout, stderr = await proc.communicate()
        if proc.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
            # Fallback без map — некоторые контейнеры криво маркируют дорожки.
            detail = (stderr or b"").decode("utf-8", errors="replace")[-500:]
            logger.warning("ffmpeg_extract_map_failed", detail=detail)
            proc = await asyncio.create_subprocess_exec(
                ffmpeg,
                "-y",
                "-i",
                str(src),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "libmp3lame",
                "-b:a",
                "64k",
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
        # ~64 kbit/s → меньше ~8 КБ на секунду; очень маленький файл при
        # большом видео часто значит «почти нет речи» или битая дорожка.
        if len(video_bytes) > 5 * 1024 * 1024 and len(audio) < 32 * 1024:
            logger.warning(
                "audio_extracted_suspiciously_small",
                video_bytes=len(video_bytes),
                audio_bytes=len(audio),
            )
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

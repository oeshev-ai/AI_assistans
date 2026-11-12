"""Утилиты Vision: скачивание вложений Telegram и подготовка мультимодального контента."""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from aiogram import Bot
from aiogram.types import Message

logger = logging.getLogger("Vision")

MAX_IMAGE_SIZE_MB = 20
MAX_PDF_PAGES = 3
IMAGE_DETAIL = "auto"
_DEFAULT_VISION_PROMPT = "Опиши, что изображено на картинке."


class VisionError(Exception):
    """Сбой обработки изображения или документа для пользователя."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


async def download_telegram_file(bot: Bot, file_id: str) -> bytes:
    """Скачивает файл Telegram по ``file_id`` в память."""
    buffer = io.BytesIO()
    try:
        await bot.download(file_id, destination=buffer)
    except Exception as exc:
        logger.exception("Не удалось скачать файл file_id=%s", file_id[:16])
        raise VisionError("Не удалось загрузить файл") from exc
    data = buffer.getvalue()
    if not data:
        raise VisionError("Файл пустой.")
    return data


def encode_image_base64(data: bytes) -> str:
    """Кодирует байты в base64 без префикса ``data:...``."""
    return base64.b64encode(data).decode("ascii")


def build_image_content(
    data: bytes, mime: str, detail: str = IMAGE_DETAIL
) -> dict[str, Any]:
    """Элемент ``image_url`` для OpenAI-совместимого multimodal content."""
    encoded = encode_image_base64(data)
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime};base64,{encoded}",
            "detail": detail,
        },
    }


def process_pdf(data: bytes, max_pages: int = MAX_PDF_PAGES) -> list[bytes]:
    """Конвертирует первые страницы PDF в PNG (dpi=150)."""
    try:
        import fitz  # type: ignore[import-untyped]
    except ImportError as exc:
        raise VisionError(
            "Обработка PDF недоступна: установите пакет PyMuPDF."
        ) from exc
    if max_pages < 1:
        raise VisionError("Число страниц PDF должно быть не меньше 1.")
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        logger.exception("Не удалось открыть PDF")
        raise VisionError("Не удалось прочитать PDF.") from exc
    try:
        page_count = document.page_count
        limit = min(max_pages, page_count)
        pages: list[bytes] = []
        for index in range(limit):
            page = document.load_page(index)
            pixmap = page.get_pixmap(dpi=150)
            pages.append(pixmap.tobytes("png"))
        logger.info("PDF: страниц=%s, отправлено=%s", page_count, len(pages))
        if not pages:
            raise VisionError("PDF не содержит страниц.")
        return pages
    finally:
        document.close()


def extract_image_from_message(message: Message) -> tuple[str, str, str] | str:
    """Определяет поддерживаемое вложение и возвращает ``(file_id, mime, name)``."""
    if message.photo:
        largest = message.photo[-1]
        return largest.file_id, "image/jpeg", "photo.jpg"

    document = message.document
    if document is None:
        return "Этот тип файлов пока не поддерживается"

    mime = (document.mime_type or "").strip().casefold()
    file_name = document.file_name or "document"
    if mime.startswith("image/"):
        return document.file_id, mime, file_name
    if mime == "application/pdf":
        return document.file_id, "application/pdf", file_name
    return "Этот тип файлов пока не поддерживается"


def build_multimodal_content(
    text: str, images: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Собирает text + image_url блоки. Пустой текст заменяется дефолтной просьбой."""
    content: list[dict[str, Any]] = []
    cleaned = text.strip()
    if cleaned:
        content.append({"type": "text", "text": cleaned})
    else:
        content.append({"type": "text", "text": _DEFAULT_VISION_PROMPT})
    content.extend(images)
    return content


def ensure_within_size_limit(data: bytes, *, label: str = "Файл") -> None:
    """Проверяет лимит ``MAX_IMAGE_SIZE_MB``."""
    limit = MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(data) > limit:
        raise VisionError(
            f"{label} слишком большой. Максимум — {MAX_IMAGE_SIZE_MB} МБ."
        )

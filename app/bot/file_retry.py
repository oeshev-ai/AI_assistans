"""Повторная отправка озвучки и расшифровки из своего бакета."""

from __future__ import annotations

import asyncio
import uuid

import structlog
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import BufferedInputFile

from app.storage.file_delivery import (
    FileDelivery,
    InMemoryFileDeliveryStore,
    KeptFile,
    RedisFileDeliveryStore,
    _safe_filename,
    delivery_clock,
    retry_delay_seconds,
)
from app.storage.s3_storage import S3Storage, S3StorageError

logger = structlog.get_logger("file_delivery")

AUDIO_KEEP_TEXT = (
    "Озвучка уже готова, но файл в чат ещё не дошёл. "
    "Бот будет повторять отправку, пока копия хранится (до 7 дней). "
    "Новая озвучка для этого не запускается."
)
TEXT_KEEP_TEXT = (
    "Расшифровка уже готова, но файл в чат ещё не дошёл. "
    "Бот будет повторять отправку, пока копия хранится (до 7 дней). "
    "Новое распознавание для этого не запускается."
)
FILE_EXPIRED_TEXT = "Отправить готовый файл не удалось: срок хранения копии закончился."
FILE_LATE_CAPTION = "Готовый файл. Отправка в чат прошла не сразу."

FileDeliveryStore = InMemoryFileDeliveryStore | RedisFileDeliveryStore

_store: FileDeliveryStore | None = None
_storage: S3Storage | None = None


def bind_file_delivery(
    store: FileDeliveryStore | None, storage: S3Storage | None = None
) -> None:
    global _store, _storage
    _store = store
    _storage = storage


def keep_text_for(kind: str) -> str:
    if kind == "audio":
        return AUDIO_KEEP_TEXT
    return TEXT_KEEP_TEXT


async def keep_paid_bytes(
    chat_id: int,
    user_id: int,
    raw: bytes,
    filename: str,
    *,
    kind: str,
    mime: str,
) -> bool:
    """Кладёт байты в бакет и в очередь. False — хранить некуда или запись не легла."""
    store = _store
    storage = _storage
    if store is None or storage is None or not raw or kind not in {"audio", "document"}:
        return False
    safe_name = _safe_filename(filename)
    object_key = f"delivery/{chat_id}/{user_id}/{uuid.uuid4().hex}/{safe_name}"
    content_type = mime.strip() or ("audio/ogg" if kind == "audio" else "text/plain")
    try:
        await storage.upload_bytes(
            object_key=object_key, data=raw, content_type=content_type
        )
    except S3StorageError:
        logger.warning("file_delivery_upload_failed", chat_id=chat_id, kind=kind)
        return False
    saved = await store.add(
        KeptFile(key=object_key, filename=safe_name, kind=kind, mime=content_type),
        chat_id,
        user_id,
        now=delivery_clock(),
    )
    if not saved:
        try:
            await storage.delete_object(object_key)
        except S3StorageError:
            logger.warning("file_delivery_orphan", chat_id=chat_id)
        return False
    logger.info("file_delivery_scheduled", chat_id=chat_id, kind=kind)
    return True


async def file_delivery_loop(
    bot: Bot,
    store: FileDeliveryStore,
    storage: S3Storage | None,
    *,
    stop: asyncio.Event | None = None,
    interval: float = 20.0,
) -> None:
    """Фоновые попытки, пока бот жив. Остановка — через ``stop`` или cancel."""
    stopper = stop or asyncio.Event()
    while not stopper.is_set():
        try:
            await deliver_due(bot, store, storage)
        except Exception:
            logger.exception("file_delivery_tick_failed")
        try:
            await asyncio.wait_for(stopper.wait(), timeout=interval)
        except TimeoutError:
            continue


async def deliver_due(
    bot: Bot,
    store: FileDeliveryStore,
    storage: S3Storage | None,
    *,
    now: float | None = None,
) -> None:
    moment = delivery_clock() if now is None else now
    for record in await store.due(moment):
        if record.expired(moment):
            await _drop_objects(storage, record)
            await _notify(bot, record.chat_id, FILE_EXPIRED_TEXT)
            await store.delete(record.chat_id, record.user_id)
            logger.info("file_delivery_expired", chat_id=record.chat_id)
            continue
        if storage is None:
            continue
        pending: list[KeptFile] = []
        for item in record.files:
            try:
                raw = await storage.download_bytes(item.key)
            except S3StorageError:
                pending.append(item)
                logger.warning(
                    "file_delivery_retry",
                    chat_id=record.chat_id,
                    kind=item.kind,
                    attempts=record.attempts + 1,
                )
                continue
            try:
                await _send_bytes(bot, record.chat_id, raw, item)
            except Exception:
                logger.exception("file_delivery_send_failed", chat_id=record.chat_id)
                pending.append(item)
                continue
            await _drop_key(storage, item.key)
            logger.info(
                "file_delivery_sent",
                chat_id=record.chat_id,
                kind=item.kind,
                bytes=len(raw),
            )
        if not pending:
            await store.delete(record.chat_id, record.user_id)
            continue
        record.files = pending
        record.attempts += 1
        record.not_before = moment + retry_delay_seconds(record.attempts)
        await store.save(record, now=moment)


async def _send_bytes(bot: Bot, chat_id: int, raw: bytes, item: KeptFile) -> None:
    payload = BufferedInputFile(raw, filename=item.filename)
    if item.kind == "audio":
        voice = item.filename.endswith((".ogg", ".opus")) or item.mime == "audio/ogg"
        if voice:
            try:
                await bot.send_voice(
                    chat_id,
                    payload,
                    caption=FILE_LATE_CAPTION,
                    request_timeout=120,
                )
                return
            except TelegramBadRequest:
                payload = BufferedInputFile(raw, filename=item.filename)
        await bot.send_audio(
            chat_id,
            payload,
            caption=FILE_LATE_CAPTION,
            request_timeout=120,
        )
        return
    await bot.send_document(
        chat_id,
        payload,
        caption=FILE_LATE_CAPTION,
        request_timeout=120,
    )


async def _drop_objects(storage: S3Storage | None, record: FileDelivery) -> None:
    if storage is None:
        return
    for item in record.files:
        await _drop_key(storage, item.key)


async def _drop_key(storage: S3Storage, object_key: str) -> None:
    try:
        await storage.delete_object(object_key)
    except S3StorageError:
        logger.warning("file_delivery_delete_failed")


async def _notify(bot: Bot, chat_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id, text)
    except Exception:
        logger.exception("file_delivery_notify_failed", chat_id=chat_id)

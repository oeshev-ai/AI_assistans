"""Повторная доставка уже оплаченного ролика с CDN Polza."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

import httpx
import structlog
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.types import BufferedInputFile

from app.llm.errors import AssistAIError
from app.llm.media_api import download_public_video
from app.storage.video_delivery import (
    InMemoryVideoDeliveryStore,
    RedisVideoDeliveryStore,
    VideoDelivery,
    delivery_clock,
    retry_delay_seconds,
)

logger = structlog.get_logger("video_delivery")

VIDEO_RETRY_TEXT = (
    "Ролик уже готов, но файл с сервера сейчас не скачался. "
    "Попытки скачать его будут повторяться, пока файл хранится у сервиса "
    "(до 7 дней). Новая генерация для этого не запускается."
)
VIDEO_KEEP_TEXT = (
    "Ролик уже оплачен, но файл в чат ещё не дошёл. "
    "Бот будет повторять попытки, пока результат хранится у сервиса "
    "(до 7 дней). Новая генерация для этого не запускается."
)
VIDEO_EXPIRED_TEXT = (
    "Скачать готовый ролик не удалось: срок хранения у сервиса закончился. "
    "Если попросить ещё раз, начнётся новая генерация."
)
VIDEO_LATE_CAPTION = "Готовый ролик. Файл с сервера дошёл не сразу."

Download = Callable[[str], Awaitable[bytes]]
Resume = Callable[[str, str], Awaitable[bytes]]
_STOP_CODES = frozenset({"FAILED", "NOT_FOUND", "BAD_REQUEST"})
VideoDeliveryStore = InMemoryVideoDeliveryStore | RedisVideoDeliveryStore

_store: VideoDeliveryStore | None = None
_DOWNLOAD_TIMEOUT = httpx.Timeout(120.0, connect=15.0)


def bind_video_delivery(store: VideoDeliveryStore | None) -> None:
    global _store
    _store = store


def current_video_delivery() -> VideoDeliveryStore | None:
    return _store


async def note_failed_video(
    chat_id: int, user_id: int, url: str, filename: str
) -> bool:
    """Кладёт ссылку в очередь повторов. False — повторять нечего или хранилище молчит."""
    store = _store
    if store is None:
        return False
    saved = await store.add(chat_id, user_id, url, filename, now=delivery_clock())
    if saved:
        logger.info(
            "video_delivery_scheduled",
            chat_id=chat_id,
            host=urlparse(url).hostname,
        )
    return saved


async def note_failed_video_job(
    chat_id: int, user_id: int, provider: str, job_id: str
) -> bool:
    """Кладёт id уже оплаченной видео-задачи. False — повторять нечего."""
    store = _store
    if store is None:
        return False
    saved = await store.add_job(
        chat_id, user_id, provider, job_id, now=delivery_clock()
    )
    if saved:
        logger.info(
            "video_delivery_job_scheduled",
            chat_id=chat_id,
            provider=provider,
        )
    return saved


async def video_failure_reply(chat_id: int, user_id: int, exc: AssistAIError) -> str:
    """Текст пользователю: повтор забора ролика или исходная ошибка."""
    provider = exc.pending_provider
    job_id = exc.pending_job_id
    if (
        exc.pending_kind in {None, "video"}
        and isinstance(provider, str)
        and isinstance(job_id, str)
        and await note_failed_video_job(chat_id, user_id, provider, job_id)
    ):
        return VIDEO_KEEP_TEXT
    pending = exc.pending_url
    if isinstance(pending, str) and pending:
        if await note_failed_video(chat_id, user_id, pending, "video.mp4"):
            return VIDEO_RETRY_TEXT
    return exc.message


def _stop_retry(exc: AssistAIError) -> bool:
    code = (exc.error_code or "").upper()
    return code in _STOP_CODES or code.startswith("GUARD_")


async def deliver_due(
    bot: Bot,
    store: VideoDeliveryStore,
    download: Download,
    *,
    resume: Resume | None = None,
    now: float | None = None,
) -> None:
    """Одна проходка: просроченные закрыть, подошедшие скачать и отправить."""
    moment = delivery_clock() if now is None else now
    for record in await store.due(moment):
        if record.expired(moment):
            await _notify(bot, record.chat_id, VIDEO_EXPIRED_TEXT)
            await store.delete(record.chat_id, record.user_id)
            logger.info("video_delivery_expired", chat_id=record.chat_id)
            continue
        await _deliver_record(bot, store, download, record, moment, resume)


async def video_delivery_loop(
    bot: Bot,
    store: VideoDeliveryStore,
    download: Download,
    *,
    resume: Resume | None = None,
    stop: asyncio.Event | None = None,
    interval: float = 20.0,
) -> None:
    """Фоновые попытки, пока бот жив. Остановка — через ``stop`` или cancel."""
    stopper = stop or asyncio.Event()
    while not stopper.is_set():
        try:
            await deliver_due(bot, store, download, resume=resume)
        except Exception:
            logger.exception("video_delivery_tick_failed")
        try:
            await asyncio.wait_for(stopper.wait(), timeout=interval)
        except TimeoutError:
            continue


async def download_video_for_retry(url: str) -> bytes:
    """Скачивание для фоновой очереди: зеркало CDN и закреплённый адрес."""
    return await download_public_video(url, timeout=_DOWNLOAD_TIMEOUT)


async def _deliver_record(
    bot: Bot,
    store: VideoDeliveryStore,
    download: Download,
    record: VideoDelivery,
    now: float,
    resume: Resume | None,
) -> None:
    pending_urls: list[str] = []
    pending_names: list[str] = []
    for url, filename in zip(record.urls, record.filenames, strict=True):
        try:
            raw = await download(url)
        except AssistAIError:
            pending_urls.append(url)
            pending_names.append(filename)
            logger.warning(
                "video_delivery_retry",
                chat_id=record.chat_id,
                host=urlparse(url).hostname,
                attempts=record.attempts + 1,
            )
            continue
        if not raw:
            pending_urls.append(url)
            pending_names.append(filename)
            continue
        try:
            await _send_bytes(bot, record.chat_id, raw, filename)
        except Exception:
            logger.exception("video_delivery_send_failed", chat_id=record.chat_id)
            pending_urls.append(url)
            pending_names.append(filename)
            continue
        logger.info("video_delivery_sent", chat_id=record.chat_id, bytes=len(raw))
    pending_jobs: list[dict[str, str]] = []
    for job in record.jobs:
        if resume is None:
            pending_jobs.append(job)
            continue
        try:
            raw = await resume(job["provider"], job["id"])
        except AssistAIError as exc:
            if _stop_retry(exc):
                await _notify(bot, record.chat_id, exc.message)
                logger.info(
                    "video_delivery_job_stopped",
                    chat_id=record.chat_id,
                    provider=job["provider"],
                    error_code=exc.error_code,
                )
                continue
            pending_jobs.append(job)
            logger.warning(
                "video_delivery_job_retry",
                chat_id=record.chat_id,
                provider=job["provider"],
                error_code=exc.error_code,
            )
            continue
        if not raw:
            pending_jobs.append(job)
            continue
        try:
            await _send_bytes(bot, record.chat_id, raw, "video.mp4")
        except Exception:
            logger.exception("video_delivery_send_failed", chat_id=record.chat_id)
            pending_jobs.append(job)
            continue
        logger.info("video_delivery_sent", chat_id=record.chat_id, bytes=len(raw))
    if not pending_urls and not pending_jobs:
        await store.delete(record.chat_id, record.user_id)
        return
    record.urls = pending_urls
    record.filenames = pending_names
    record.jobs = pending_jobs
    record.attempts += 1
    record.not_before = now + retry_delay_seconds(record.attempts)
    await store.save(record, now=now)


async def _send_bytes(bot: Bot, chat_id: int, raw: bytes, filename: str) -> None:
    payload = BufferedInputFile(raw, filename=filename)
    try:
        await bot.send_video(
            chat_id,
            payload,
            caption=VIDEO_LATE_CAPTION,
            request_timeout=180,
        )
    except (TelegramBadRequest, TelegramNetworkError):
        from app.bot.runtime import _flood_send

        await _flood_send(
            lambda: bot.send_document(
                chat_id,
                BufferedInputFile(raw, filename=filename),
                caption=VIDEO_LATE_CAPTION,
                request_timeout=180,
            )
        )


async def _notify(bot: Bot, chat_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id, text)
    except Exception:
        logger.exception("video_delivery_notify_failed", chat_id=chat_id)

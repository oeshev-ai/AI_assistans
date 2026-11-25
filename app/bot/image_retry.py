"""Повторная доставка уже оплаченной картинки с CDN Polza."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from urllib.parse import urlparse

import structlog
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import BufferedInputFile

from app.llm.errors import AssistAIError
from app.storage.image_delivery import (
    ImageDelivery,
    InMemoryImageDeliveryStore,
    RedisImageDeliveryStore,
    delivery_clock,
    retry_delay_seconds,
)

logger = structlog.get_logger("image_delivery")

IMAGE_RETRY_TEXT = (
    "Картинка уже готова, но файл с сервера сейчас не скачался. "
    "Попытки скачать её будут повторяться, пока файл хранится у сервиса "
    "(до 7 дней). Новая генерация для этого не запускается."
)
IMAGE_KEEP_TEXT = (
    "Картинка уже оплачена, но файл в чат ещё не дошёл. "
    "Бот будет повторять попытки, пока результат хранится у сервиса "
    "(до 7 дней). Новая генерация для этого не запускается."
)
IMAGE_EXPIRED_TEXT = (
    "Скачать готовую картинку не удалось: срок хранения у сервиса закончился. "
    "Если попросить ещё раз, начнётся новая генерация."
)
IMAGE_LATE_CAPTION = "Готовая картинка. Файл с сервера дошёл не сразу."

_PHOTO_DOCUMENT_MARKERS = (
    "IMAGE_PROCESS_FAILED",
    "PHOTO_INVALID_DIMENSIONS",
    "PHOTO_SAVE_FILE_INVALID",
    "PHOTO_INVALID",
    "FILE_PARTS_INVALID",
    "FILE IS TOO BIG",
    "WRONG TYPE",
    "CAN'T PARSE",
)


def telegram_photo_needs_document(exc: BaseException) -> bool:
    """Telegram не принял sendPhoto: файл ещё можно отдать документом."""
    text = str(exc).upper()
    return any(marker in text for marker in _PHOTO_DOCUMENT_MARKERS)


Download = Callable[[str], Awaitable[bytes]]
Resume = Callable[[str, str], Awaitable[bytes]]
_STOP_CODES = frozenset({"FAILED", "NOT_FOUND", "BAD_REQUEST"})
ImageDeliveryStore = InMemoryImageDeliveryStore | RedisImageDeliveryStore

_store: ImageDeliveryStore | None = None


def bind_image_delivery(store: ImageDeliveryStore | None) -> None:
    global _store
    _store = store


async def note_failed_image(
    chat_id: int, user_id: int, url: str, filename: str, *, as_file: bool = False
) -> bool:
    """Кладёт ссылку в очередь повторов. False — повторять нечего или хранилище молчит."""
    store = _store
    if store is None:
        return False
    saved = await store.add(
        chat_id, user_id, url, filename, now=delivery_clock(), as_file=as_file
    )
    if saved:
        logger.info(
            "image_delivery_scheduled",
            chat_id=chat_id,
            host=urlparse(url).hostname,
        )
    return saved


async def note_failed_image_job(
    chat_id: int, user_id: int, provider: str, job_id: str, *, as_file: bool = False
) -> bool:
    """Кладёт id уже оплаченной задачи. False — повторять нечего."""
    store = _store
    if store is None:
        return False
    saved = await store.add_job(
        chat_id,
        user_id,
        provider,
        job_id,
        now=delivery_clock(),
        as_file=as_file,
    )
    if saved:
        logger.info(
            "image_delivery_job_scheduled",
            chat_id=chat_id,
            provider=provider,
        )
    return saved


async def image_failure_reply(
    chat_id: int, user_id: int, exc: AssistAIError, *, as_file: bool = False
) -> str:
    """Текст пользователю: повтор забора файла или исходная ошибка."""
    provider = exc.pending_provider
    job_id = exc.pending_job_id
    if (
        exc.pending_kind in {None, "image"}
        and isinstance(provider, str)
        and isinstance(job_id, str)
        and await note_failed_image_job(
            chat_id, user_id, provider, job_id, as_file=as_file
        )
    ):
        return IMAGE_KEEP_TEXT
    pending = exc.pending_url
    if isinstance(pending, str) and pending:
        if await note_failed_image(
            chat_id, user_id, pending, "image.png", as_file=as_file
        ):
            return IMAGE_RETRY_TEXT
    return exc.message


def _stop_retry(exc: AssistAIError) -> bool:
    code = (exc.error_code or "").upper()
    return code in _STOP_CODES or code.startswith("GUARD_")


def _has_arq(record: ImageDelivery) -> bool:
    return any(job.get("provider") == "arq" for job in record.jobs)


def _only_arq(record: ImageDelivery) -> bool:
    return bool(record.jobs) and all(
        job.get("provider") == "arq" for job in record.jobs
    )


def _due_records(records: list[ImageDelivery], now: float) -> list[ImageDelivery]:
    return [
        record for record in records if record.expired(now) or record.not_before <= now
    ]


async def deliver_due(
    bot: Bot,
    store: ImageDeliveryStore,
    download: Download,
    *,
    resume: Resume | None = None,
    now: float | None = None,
    records: list[ImageDelivery] | None = None,
) -> None:
    """Одна проходка: просроченные закрыть, подошедшие скачать и отправить.

    Запись без ссылок, где все задачи — arq, пропускается. Её на каждом тике
    ведёт promote, а пустой проход здесь только сжигал бы ``attempts`` и
    отодвигал первую настоящую закачку с CDN к потолку в полчаса.
    ``records`` — уже прочитанный индекс. Иначе чтение делает ``store.due``.
    """
    moment = delivery_clock() if now is None else now
    found = (
        _due_records(records, moment)
        if records is not None
        else await store.due(moment)
    )
    for record in found:
        if record.expired(moment):
            await _notify(bot, record.chat_id, IMAGE_EXPIRED_TEXT)
            await store.delete(record.chat_id, record.user_id)
            logger.info("image_delivery_expired", chat_id=record.chat_id)
            continue
        if not record.urls and _only_arq(record):
            continue
        await _deliver_record(bot, store, download, record, moment, resume)


async def image_delivery_tick(
    bot: Bot,
    store: ImageDeliveryStore,
    download: Download,
    *,
    resume: Resume | None = None,
    now: float | None = None,
) -> None:
    """Индекс читается один раз: arq-задачи, затем подошедшие ссылки.

    Без arq-задач опрос не начинается. Выключенная очередь тоже не ходит
    в Redis за результатом: живые arq-задачи закрываются отказом.
    """
    moment = delivery_clock() if now is None else now
    records = await store.indexed()
    live = [
        record for record in records if not record.expired(moment) and _has_arq(record)
    ]
    if not live:
        await deliver_due(
            bot, store, download, resume=resume, now=moment, records=records
        )
        return
    from app.infra.job_queue import is_queue_enabled

    if is_queue_enabled():
        await promote_arq_image_jobs(bot, store, records=live, now=moment)
    else:
        await _refuse_arq_without_queue(bot, store, live, moment)
    await deliver_due(bot, store, download, resume=resume, now=moment, records=records)


async def _refuse_arq_without_queue(
    bot: Bot,
    store: ImageDeliveryStore,
    records: list[ImageDelivery],
    now: float,
) -> None:
    """Очередь выключена: arq-задачи записи больше не у кого спросить.

    Текст один на запись, даже если задач несколько: сообщение одно и то же.
    """
    from app.infra.job_queue import arq_queue_unavailable

    exc = arq_queue_unavailable()
    for record in records:
        kept = [job for job in record.jobs if job.get("provider") != "arq"]
        if len(kept) == len(record.jobs):
            continue
        await _notify(bot, record.chat_id, exc.message)
        logger.info(
            "image_delivery_arq_stopped",
            chat_id=record.chat_id,
            error_code=exc.error_code,
        )
        record.jobs = kept
        await store.save(record, now=now)


async def image_delivery_loop(
    bot: Bot,
    store: ImageDeliveryStore,
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
            await image_delivery_tick(bot, store, download, resume=resume)
        except Exception:
            logger.exception("image_delivery_tick_failed")
        try:
            await asyncio.wait_for(stopper.wait(), timeout=interval)
        except TimeoutError:
            continue


async def _deliver_record(
    bot: Bot,
    store: ImageDeliveryStore,
    download: Download,
    record: ImageDelivery,
    now: float,
    resume: Resume | None,
) -> None:
    pending_urls: list[str] = []
    pending_names: list[str] = []
    pending_flags: list[bool] = []
    flags = _align_delivery_flags(record)
    for url, filename, as_file in zip(
        record.urls, record.filenames, flags, strict=True
    ):
        try:
            raw = await download(url)
        except AssistAIError:
            pending_urls.append(url)
            pending_names.append(filename)
            pending_flags.append(as_file)
            logger.warning(
                "image_delivery_retry",
                chat_id=record.chat_id,
                host=urlparse(url).hostname,
                attempts=record.attempts + 1,
            )
            continue
        if not raw:
            pending_urls.append(url)
            pending_names.append(filename)
            pending_flags.append(as_file)
            continue
        try:
            await _send_bytes(bot, record.chat_id, raw, filename, as_file=as_file)
        except Exception:
            logger.exception("image_delivery_send_failed", chat_id=record.chat_id)
            pending_urls.append(url)
            pending_names.append(filename)
            pending_flags.append(as_file)
            continue
        logger.info("image_delivery_sent", chat_id=record.chat_id, bytes=len(raw))
    pending_jobs: list[dict[str, str]] = []
    for job in record.jobs:
        if job.get("provider") == "arq":
            pending_jobs.append(job)
            continue
        if resume is None:
            pending_jobs.append(job)
            continue
        try:
            raw = await resume(job["provider"], job["id"])
        except AssistAIError as exc:
            if _stop_retry(exc):
                await _notify(bot, record.chat_id, exc.message)
                logger.info(
                    "image_delivery_job_stopped",
                    chat_id=record.chat_id,
                    provider=job["provider"],
                    error_code=exc.error_code,
                )
                continue
            pending_jobs.append(job)
            logger.warning(
                "image_delivery_job_retry",
                chat_id=record.chat_id,
                provider=job["provider"],
                error_code=exc.error_code,
            )
            continue
        if not raw:
            pending_jobs.append(job)
            continue
        try:
            await _send_bytes(
                bot,
                record.chat_id,
                raw,
                "image.png",
                as_file=job.get("as_file") == "1",
            )
        except Exception:
            logger.exception("image_delivery_send_failed", chat_id=record.chat_id)
            pending_jobs.append(job)
            continue
        logger.info("image_delivery_sent", chat_id=record.chat_id, bytes=len(raw))
    if not pending_urls and not pending_jobs:
        await store.delete(record.chat_id, record.user_id)
        return
    record.urls = pending_urls
    record.filenames = pending_names
    record.as_files = pending_flags
    record.jobs = pending_jobs
    record.attempts += 1
    record.not_before = now + retry_delay_seconds(record.attempts)
    await store.save(record, now=now)


def _align_delivery_flags(record: ImageDelivery) -> list[bool]:
    flags = list(record.as_files)
    if len(flags) < len(record.urls):
        flags.extend([False] * (len(record.urls) - len(flags)))
    return flags[: len(record.urls)]


async def _send_bytes(
    bot: Bot, chat_id: int, raw: bytes, filename: str, *, as_file: bool = False
) -> None:
    from app.bot.photo_pacer import send_photo_paced
    from app.bot.runtime import _flood_send

    async def _document() -> None:
        await bot.send_document(
            chat_id,
            BufferedInputFile(raw, filename=filename),
            caption=IMAGE_LATE_CAPTION,
            request_timeout=120,
        )

    if as_file:
        await _flood_send(_document)
        return

    async def _send_photo() -> None:
        await bot.send_photo(
            chat_id,
            BufferedInputFile(raw, filename=filename),
            caption=IMAGE_LATE_CAPTION,
            request_timeout=120,
        )

    try:
        await send_photo_paced(chat_id, _send_photo)
    except TelegramBadRequest as exc:
        if not telegram_photo_needs_document(exc):
            raise
        await _flood_send(_document)


def _next_image_name(index: int) -> str:
    if index == 0:
        return "image.png"
    return f"image_{index + 1}.png"


def _payload_skip(job: dict[str, str]) -> int:
    raw = job.get("sent")
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    return 0


async def _hand_off_image_bytes(
    bot: Bot,
    record: ImageDelivery,
    payloads: list[tuple[bytes, str]],
    *,
    as_file: bool,
    skip: int = 0,
) -> int:
    """Сколько кадров с начала пачки уже ушли в чат или в бакет.

    Кадр, который не ушёл и не лёг в бакет, останавливает пачку. Хвост
    уйдёт следующим тиком: в задаче запоминается длина сданного префикса,
    уже отправленные кадры не повторяются. Бакет доставки необязателен,
    без него прогресс всё равно держится на самой задаче.
    """
    from app.bot.file_retry import keep_paid_bytes

    settled = 0
    index = len(record.filenames)
    for offset, (raw, mime) in enumerate(payloads):
        filename = _next_image_name(index)
        index += 1
        if offset < skip:
            settled += 1
            continue
        try:
            await _send_bytes(
                bot,
                record.chat_id,
                raw,
                filename,
                as_file=as_file,
            )
        except Exception:
            logger.exception("image_delivery_send_failed", chat_id=record.chat_id)
            saved = await keep_paid_bytes(
                record.chat_id,
                record.user_id,
                raw,
                filename,
                kind="document",
                mime=mime or "image/png",
            )
            if not saved:
                logger.warning(
                    "image_delivery_bytes_unkept",
                    chat_id=record.chat_id,
                    bytes=len(raw),
                )
                return settled
            settled += 1
            continue
        logger.info(
            "image_delivery_sent",
            chat_id=record.chat_id,
            bytes=len(raw),
        )
        settled += 1
    return settled


async def promote_arq_image_jobs(
    bot: Bot,
    store: ImageDeliveryStore,
    *,
    records: list[ImageDelivery] | None = None,
    now: float | None = None,
) -> None:
    """Забирает готовую картинку у arq, пока ключ результата ещё жив.

    URL остаётся в очереди скачивания. base64 декодируется и уходит в чат
    байтами. Обычный повтор ждёт минуту и больше. ``keep_result`` картинки
    короче этого окна, поэтому опрос идёт на каждом тике, не дожидаясь
    ``not_before``. ``records`` — уже прочитанный индекс, без второго обхода.
    """
    from app.infra.job_queue import poll_arq_image_urls

    moment = delivery_clock() if now is None else now
    loaded = records if records is not None else await store.active(moment)
    for record in loaded:
        if record.expired(moment) or not _has_arq(record):
            continue
        kept: list[dict[str, str]] = []
        added = False
        stopped = False
        progressed = False
        flags = _align_delivery_flags(record)
        for job in record.jobs:
            if job.get("provider") != "arq":
                kept.append(job)
                continue
            try:
                ready = await poll_arq_image_urls(job["id"])
            except AssistAIError as exc:
                stopped = True
                await _notify(bot, record.chat_id, exc.message)
                logger.info(
                    "image_delivery_arq_stopped",
                    chat_id=record.chat_id,
                    error_code=exc.error_code,
                )
                continue
            if ready is None:
                kept.append(job)
                continue
            as_file = job.get("as_file") == "1"
            for url in ready.urls:
                if url in record.urls:
                    continue
                flags.append(as_file)
                record.urls.append(url)
                record.filenames.append(_next_image_name(len(record.filenames)))
                added = True
            skip = _payload_skip(job)
            settled = await _hand_off_image_bytes(
                bot, record, ready.payloads, as_file=as_file, skip=skip
            )
            if settled < len(ready.payloads):
                if settled > skip:
                    job["sent"] = str(settled)
                    progressed = True
                kept.append(job)
        if kept == record.jobs and not added and not stopped and not progressed:
            continue
        record.jobs = kept
        record.as_files = flags
        if added:
            record.not_before = moment
        await store.save(record, now=moment)


async def _notify(bot: Bot, chat_id: int, text: str) -> None:
    try:
        await bot.send_message(chat_id, text)
    except Exception:
        logger.exception("image_delivery_notify_failed", chat_id=chat_id)

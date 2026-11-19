"""Сборка альбома: сортировка, лимиты и debounce без сети."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger("album")

AlbumKey = tuple[int, int, str]  # chat_id, user_id, media_group_id


def album_debounce_seconds() -> float:
    raw = os.getenv("ALBUM_DEBOUNCE_SECONDS", "1.2").strip()
    try:
        value = float(raw)
    except ValueError:
        return 1.2
    return value if value > 0 else 1.2


def max_album_items() -> int:
    raw = os.getenv("MAX_ALBUM_ITEMS", "5").strip()
    try:
        value = int(raw)
    except ValueError:
        return 5
    return value if value > 0 else 5


def max_album_total_bytes() -> int:
    raw = os.getenv("MAX_ALBUM_TOTAL_MB", "20").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 20
    if value < 1:
        value = 20
    return value * 1024 * 1024


@dataclass
class IncomingPhoto:
    """Метаданные фото до скачивания. Байты сюда не кладём."""

    message_id: int
    file_id: str
    size: int = 0
    caption: str = ""
    filename: str = "photo.jpg"
    mime: str = "image/jpeg"


@dataclass
class AlbumBuild:
    """Результат сборки. ``error`` — отказ целиком, модель не вызывается."""

    photos: list[IncomingPhoto]
    caption: str
    notice: str | None
    error: str | None


def assemble_album(
    photos: list[IncomingPhoto],
    *,
    max_items: int | None = None,
    max_total_bytes: int | None = None,
    caption_override: str | None = None,
) -> AlbumBuild:
    """Сортирует по message_id, обрезает до лимита, берёт первую непустую подпись."""
    limit = max_album_items() if max_items is None else max_items
    budget = max_album_total_bytes() if max_total_bytes is None else max_total_bytes
    ordered = sorted(photos, key=lambda item: item.message_id)
    notice: str | None = None
    if len(ordered) > limit:
        ordered = ordered[:limit]
        notice = f"Взял первые {limit} фото"
    total = sum(max(item.size, 0) for item in ordered)
    if ordered and total > budget:
        mb = max(budget // (1024 * 1024), 1)
        return AlbumBuild(
            photos=[],
            caption="",
            notice=None,
            error=f"Альбом больше {mb} МБ. Отправьте меньше фото.",
        )
    caption = (caption_override or "").strip()
    if not caption:
        for item in ordered:
            text = item.caption.strip()
            if text:
                caption = text
                break
    return AlbumBuild(photos=ordered, caption=caption, notice=notice, error=None)


ReadyCallback = Callable[[AlbumBuild, Any], Awaitable[None]]


class AlbumHub:
    """Копит media_group_id и сбрасывает пачку после паузы.

    Одиночное фото обрабатывается сразу и отменяет открытый буфер
    того же пользователя в чате. Текст во время паузы сбрасывает
    буфер пользователя и становится инструкцией.

    Ключ включает ``user_id``, чтобы в группах чужой текст/фото не
    трогали чужой альбом.

    Буфер process-local: при нескольких репликах бота альбом может
    разъехаться по процессам. Держите один экземпляр ``bot`` (без
    горизонтального масштабирования polling) или sticky routing.
    """

    def __init__(self, debounce: float | None = None) -> None:
        self.debounce = album_debounce_seconds() if debounce is None else debounce
        self._buf: dict[AlbumKey, list[IncomingPhoto]] = {}
        self._tasks: dict[AlbumKey, asyncio.Task[None]] = {}
        self._latest: dict[AlbumKey, Any] = {}
        self._lock = asyncio.Lock()

    def _keys_for(self, chat_id: int, user_id: int | None = None) -> list[AlbumKey]:
        return [
            key
            for key in self._buf
            if key[0] == chat_id and (user_id is None or key[1] == user_id)
        ]

    def has_pending(self, chat_id: int, user_id: int) -> bool:
        """True if this user has an open album debounce buffer in the chat."""
        return any(key[0] == chat_id and key[1] == user_id for key in self._buf)

    def _cancel_user_unlocked(self, chat_id: int, user_id: int) -> bool:
        """Сброс буфера пользователя. Вызывать только под ``self._lock``."""
        keys = self._keys_for(chat_id, user_id)
        if not keys:
            return False
        for key in keys:
            self._buf.pop(key, None)
            self._latest.pop(key, None)
            task = self._tasks.pop(key, None)
            if task is not None and not task.done():
                task.cancel()
        return True

    async def cancel_chat(self, chat_id: int, user_id: int | None = None) -> bool:
        """Сбрасывает открытый буфер, не вызывая обработчик."""
        async with self._lock:
            if user_id is None:
                keys = self._keys_for(chat_id)
                if not keys:
                    return False
                for key in keys:
                    self._buf.pop(key, None)
                    self._latest.pop(key, None)
                    task = self._tasks.pop(key, None)
                    if task is not None and not task.done():
                        task.cancel()
                return True
            return self._cancel_user_unlocked(chat_id, user_id)

    async def add(
        self,
        chat_id: int,
        user_id: int,
        group_id: str | None,
        photo: IncomingPhoto,
        services: Any,
        on_ready: ReadyCallback,
    ) -> None:
        if group_id is None:
            async with self._lock:
                self._cancel_user_unlocked(chat_id, user_id)
            built = assemble_album([photo])
            await on_ready(built, services)
            return
        async with self._lock:
            for key in self._keys_for(chat_id, user_id):
                if key[2] == group_id:
                    continue
                self._buf.pop(key, None)
                self._latest.pop(key, None)
                task = self._tasks.pop(key, None)
                if task is not None and not task.done():
                    task.cancel()
            key = (chat_id, user_id, group_id)
            self._buf.setdefault(key, []).append(photo)
            self._latest[key] = services
            previous = self._tasks.get(key)
            if previous is not None and not previous.done():
                previous.cancel()
            self._tasks[key] = asyncio.create_task(self._fire(key, on_ready))

    async def flush_text(
        self,
        chat_id: int,
        user_id: int,
        text: str,
        on_ready: ReadyCallback,
    ) -> bool:
        """Если буфер пользователя открыт, закрывает его и отдаёт ``text`` как инструкцию."""
        async with self._lock:
            keys = self._keys_for(chat_id, user_id)
            if not keys:
                return False
            batches: list[tuple[list[IncomingPhoto], Any]] = []
            for key in keys:
                task = self._tasks.pop(key, None)
                if task is not None and not task.done():
                    task.cancel()
                photos = self._buf.pop(key, [])
                services = self._latest.pop(key, None)
                batches.append((photos, services))
        delivered = False
        for photos, services in batches:
            if not photos or services is None:
                continue
            built = assemble_album(photos, caption_override=text)
            await on_ready(built, services)
            delivered = True
        return delivered

    async def _fire(self, key: AlbumKey, on_ready: ReadyCallback) -> None:
        try:
            await asyncio.sleep(self.debounce)
        except asyncio.CancelledError:
            return
        async with self._lock:
            photos = self._buf.pop(key, None)
            services = self._latest.pop(key, None)
            self._tasks.pop(key, None)
        if not photos or services is None:
            return
        built = assemble_album(photos)
        logger.info(
            "album_flushed",
            chat_id=key[0],
            user_id=key[1],
            count=len(built.photos),
            trimmed=built.notice is not None,
        )
        try:
            await on_ready(built, services)
        except Exception:
            logger.exception("album_flush_failed", chat_id=key[0], user_id=key[1])


_hub: AlbumHub | None = None


def get_album_hub() -> AlbumHub:
    """Общий буфер процесса. Тесты создают свой ``AlbumHub``."""
    global _hub
    if _hub is None:
        _hub = AlbumHub()
    return _hub

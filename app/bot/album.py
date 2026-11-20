"""Сборка альбома: сортировка, лимиты и debounce.

С метаданными фото (без байтов). При ``attach_redis`` буфер дублируется в
Redis, чтобы несколько реплик могли склеить один ``media_group_id``.
Debounce и ``services`` остаются process-local; flush берёт lock в Redis.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import Any

import structlog

logger = structlog.get_logger("album")

AlbumKey = tuple[int, int, str]  # chat_id, user_id, media_group_id
_REDIS_TTL_SEC = 60
_LOCK_TTL_SEC = 5


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

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> IncomingPhoto:
        data = json.loads(raw)
        return cls(
            message_id=int(data["message_id"]),
            file_id=str(data["file_id"]),
            size=int(data.get("size") or 0),
            caption=str(data.get("caption") or ""),
            filename=str(data.get("filename") or "photo.jpg"),
            mime=str(data.get("mime") or "image/jpeg"),
        )


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

    Без Redis буфер process-local. С ``attach_redis`` метаданные фото
    пишутся в Redis, flush под lock — реплики могут склеить один альбом.
    """

    def __init__(self, debounce: float | None = None) -> None:
        self.debounce = album_debounce_seconds() if debounce is None else debounce
        self._buf: dict[AlbumKey, list[IncomingPhoto]] = {}
        self._tasks: dict[AlbumKey, asyncio.Task[None]] = {}
        self._latest: dict[AlbumKey, Any] = {}
        self._lock = asyncio.Lock()
        self._redis: Any | None = None

    def attach_redis(self, redis_client: Any | None) -> None:
        """Optional Redis for cross-replica album metadata."""
        self._redis = redis_client
        if redis_client is not None:
            logger.info("album_hub_using_redis")
        else:
            logger.warning(
                "album_hub_memory_only",
                hint="Multi-replica polling can split albums without Redis",
            )

    @staticmethod
    def _redis_key(key: AlbumKey) -> str:
        return f"album:buf:{key[0]}:{key[1]}:{key[2]}"

    @staticmethod
    def _lock_key(key: AlbumKey) -> str:
        return f"album:lock:{key[0]}:{key[1]}:{key[2]}"

    def _keys_for(self, chat_id: int, user_id: int | None = None) -> list[AlbumKey]:
        return [
            key
            for key in self._buf
            if key[0] == chat_id and (user_id is None or key[1] == user_id)
        ]

    def has_pending(self, chat_id: int, user_id: int) -> bool:
        """True if this user has an open album debounce buffer in the chat."""
        return any(key[0] == chat_id and key[1] == user_id for key in self._buf)

    async def _redis_append(self, key: AlbumKey, photo: IncomingPhoto) -> None:
        if self._redis is None:
            return
        rkey = self._redis_key(key)
        try:
            pipe = self._redis.pipeline()
            pipe.rpush(rkey, photo.to_json())
            pipe.expire(rkey, _REDIS_TTL_SEC)
            await pipe.execute()
        except Exception as exc:
            logger.warning("album_redis_append_failed", error=str(exc))

    async def _redis_take(
        self, key: AlbumKey
    ) -> tuple[bool | None, list[IncomingPhoto]]:
        """Atomically claim album photos.

        Returns ``(won, photos)``.
        ``won=False`` — peer holds the lock (drop local to avoid double delivery).
        ``won=None`` — Redis error (caller keeps local photos).
        When Redis is healthy, the list is the source of truth.
        """
        if self._redis is None:
            return True, []
        rkey = self._redis_key(key)
        lock = self._lock_key(key)
        try:
            got_lock = await self._redis.set(lock, "1", nx=True, ex=_LOCK_TTL_SEC)
            if not got_lock:
                return False, []
            try:
                raw_items = await self._redis.lrange(rkey, 0, -1)
                await self._redis.delete(rkey)
            finally:
                await self._redis.delete(lock)
        except Exception as exc:
            logger.warning("album_redis_take_failed", error=str(exc))
            return None, []
        photos: list[IncomingPhoto] = []
        for raw in raw_items or []:
            text = (
                raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
            )
            try:
                photos.append(IncomingPhoto.from_json(text))
            except (TypeError, ValueError, KeyError, json.JSONDecodeError):
                logger.warning("album_redis_photo_invalid")
        return True, photos

    async def _claim_photos(
        self, key: AlbumKey, local: list[IncomingPhoto]
    ) -> list[IncomingPhoto] | None:
        """Photos to flush, or ``None`` if a peer already claimed the album."""
        won, remote = await self._redis_take(key)
        if won is None:
            logger.warning(
                "album_redis_claim_fallback_local",
                local_count=len(local),
                chat_id=key[0],
                user_id=key[1],
            )
            return local
        if not won:
            return None
        if self._redis is None:
            return local
        # Redis SoT: empty after a won lock ⇒ peer already flushed.
        return remote

    async def _redis_clear(self, key: AlbumKey) -> None:
        if self._redis is None:
            return
        try:
            await self._redis.delete(self._redis_key(key), self._lock_key(key))
        except Exception as exc:
            logger.warning("album_redis_clear_failed", error=str(exc))

    def _cancel_user_unlocked(self, chat_id: int, user_id: int) -> list[AlbumKey]:
        """Сброс локального буфера. Вызывать только под ``self._lock``."""
        keys = self._keys_for(chat_id, user_id)
        for key in keys:
            self._buf.pop(key, None)
            self._latest.pop(key, None)
            task = self._tasks.pop(key, None)
            if task is not None and not task.done():
                task.cancel()
        return keys

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
            else:
                keys = self._cancel_user_unlocked(chat_id, user_id)
                if not keys:
                    return False
        for key in keys:
            await self._redis_clear(key)
        return True

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
            cleared: list[AlbumKey] = []
            async with self._lock:
                cleared = self._cancel_user_unlocked(chat_id, user_id)
            for key in cleared:
                await self._redis_clear(key)
            built = assemble_album([photo])
            await on_ready(built, services)
            return
        stale: list[AlbumKey] = []
        async with self._lock:
            for key in self._keys_for(chat_id, user_id):
                if key[2] == group_id:
                    continue
                self._buf.pop(key, None)
                self._latest.pop(key, None)
                task = self._tasks.pop(key, None)
                if task is not None and not task.done():
                    task.cancel()
                stale.append(key)
            key = (chat_id, user_id, group_id)
            self._buf.setdefault(key, []).append(photo)
            self._latest[key] = services
            previous = self._tasks.get(key)
            if previous is not None and not previous.done():
                previous.cancel()
            self._tasks[key] = asyncio.create_task(self._fire(key, on_ready))
        for stale_key in stale:
            await self._redis_clear(stale_key)
        await self._redis_append(key, photo)

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
            batches: list[tuple[AlbumKey, list[IncomingPhoto], Any]] = []
            for key in keys:
                task = self._tasks.pop(key, None)
                if task is not None and not task.done():
                    task.cancel()
                photos = self._buf.pop(key, [])
                services = self._latest.pop(key, None)
                batches.append((key, photos, services))
        delivered = False
        for key, photos, services in batches:
            claimed = await self._claim_photos(key, photos)
            if claimed is None or not claimed or services is None:
                continue
            built = assemble_album(claimed, caption_override=text)
            await on_ready(built, services)
            delivered = True
        return delivered

    async def _fire(self, key: AlbumKey, on_ready: ReadyCallback) -> None:
        try:
            await asyncio.sleep(self.debounce)
        except asyncio.CancelledError:
            return
        async with self._lock:
            photos = self._buf.pop(key, None) or []
            services = self._latest.pop(key, None)
            self._tasks.pop(key, None)
        claimed = await self._claim_photos(key, photos)
        if claimed is None or not claimed or services is None:
            return
        built = assemble_album(claimed)
        logger.info(
            "album_flushed",
            chat_id=key[0],
            user_id=key[1],
            count=len(built.photos),
            trimmed=built.notice is not None,
            redis=self._redis is not None,
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

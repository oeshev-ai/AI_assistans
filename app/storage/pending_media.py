"""Отложенные фото: только file_id, без байтов.

Ключ ``chat:{chat_id}:asst:{assistant_id}:pending_photo``. TTL по умолчанию 900 с.
Пустой ``REDIS_URL`` или недоступный Redis — память процесса с тем же TTL.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = structlog.get_logger("pending_media")

_KEY = "chat:{chat_id}:asst:{assistant_id}:pending_photo"


def pending_ttl_seconds() -> int:
    """TTL отложенных фото из ``PHOTO_PENDING_TTL_SECONDS``."""
    raw = os.getenv("PHOTO_PENDING_TTL_SECONDS", "900").strip()
    try:
        value = int(raw)
    except ValueError:
        return 900
    return value if value > 0 else 900


class PendingMediaError(Exception):
    """Сбой хранилища отложенных фото."""

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)


@dataclass
class PendingMediaItem:
    """Одно фото альбома. Байты не хранятся."""

    file_id: str
    message_id: int
    file_size: int = 0
    filename: str = "photo.jpg"
    mime_type: str = "image/jpeg"


@dataclass
class PendingMedia:
    """Альбом, который ждёт действие пользователя или ход модели."""

    chat_id: int
    assistant_id: str
    items: list[PendingMediaItem]
    caption: str = ""
    base_override: int | None = None
    role_retry_used: bool = False
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        payload = {
            "chat_id": self.chat_id,
            "assistant_id": self.assistant_id,
            "caption": self.caption,
            "base_override": self.base_override,
            "role_retry_used": self.role_retry_used,
            "created_at": self.created_at,
            "items": [
                {
                    "file_id": item.file_id,
                    "message_id": item.message_id,
                    "file_size": item.file_size,
                    "filename": item.filename,
                    "mime_type": item.mime_type,
                }
                for item in self.items
            ],
        }
        return json.dumps(payload, ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> PendingMedia:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise PendingMediaError("Повреждённые отложенные фото.")
        items_raw = data.get("items")
        items: list[PendingMediaItem] = []
        if isinstance(items_raw, list):
            for item in items_raw:
                if not isinstance(item, dict):
                    continue
                file_id = item.get("file_id")
                if not isinstance(file_id, str) or not file_id:
                    continue
                items.append(
                    PendingMediaItem(
                        file_id=file_id,
                        message_id=int(item.get("message_id") or 0),
                        file_size=int(item.get("file_size") or 0),
                        filename=str(item.get("filename") or "photo.jpg"),
                        mime_type=str(item.get("mime_type") or "image/jpeg"),
                    )
                )
        override = data.get("base_override")
        return cls(
            chat_id=int(data.get("chat_id") or 0),
            assistant_id=str(data.get("assistant_id") or ""),
            items=items,
            caption=str(data.get("caption") or ""),
            base_override=int(override) if isinstance(override, int) else None,
            role_retry_used=bool(data.get("role_retry_used")),
            created_at=float(data.get("created_at") or time.time()),
        )


class PendingMediaStorage(ABC):
    """Контракт хранилища отложенных фото."""

    @abstractmethod
    async def put(self, media: PendingMedia) -> None:
        """Сохраняет альбом и обновляет TTL."""

    @abstractmethod
    async def get(self, chat_id: int, assistant_id: str) -> PendingMedia | None:
        """Возвращает альбом или None, если его нет или TTL истёк."""

    @abstractmethod
    async def clear(self, chat_id: int, assistant_id: str) -> None:
        """Удаляет альбом."""

    async def pop(self, chat_id: int, assistant_id: str) -> PendingMedia | None:
        """Читает и удаляет альбом."""
        media = await self.get(chat_id, assistant_id)
        await self.clear(chat_id, assistant_id)
        return media

    async def set_base_override(
        self, chat_id: int, assistant_id: str, base_override: int | None
    ) -> PendingMedia | None:
        """Меняет выбранную основу, не продлевая смысл сверх текущего TTL."""
        media = await self.get(chat_id, assistant_id)
        if media is None:
            return None
        media.base_override = base_override
        await self.put(media)
        return media

    async def mark_role_retry(self, chat_id: int, assistant_id: str) -> None:
        """Отмечает, что кнопки выбора основы уже показаны один раз."""
        media = await self.get(chat_id, assistant_id)
        if media is None:
            return
        media.role_retry_used = True
        await self.put(media)

    @abstractmethod
    async def close(self) -> None:
        """Закрывает соединение, если хранилище им владеет."""


class InMemoryPendingMediaStorage(PendingMediaStorage):
    """Память процесса с тем же TTL, что и Redis."""

    def __init__(self, ttl_seconds: float | None = None) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else pending_ttl_seconds()
        self._items: dict[tuple[int, str], tuple[PendingMedia, float]] = {}
        self._closed = False

    def _key(self, chat_id: int, assistant_id: str) -> tuple[int, str]:
        return chat_id, assistant_id

    async def put(self, media: PendingMedia) -> None:
        if self._closed:
            raise PendingMediaError("Хранилище отложенных фото закрыто.")
        deadline = time.monotonic() + self._ttl
        self._items[self._key(media.chat_id, media.assistant_id)] = (media, deadline)

    async def get(self, chat_id: int, assistant_id: str) -> PendingMedia | None:
        key = self._key(chat_id, assistant_id)
        found = self._items.get(key)
        if found is None:
            return None
        media, deadline = found
        if time.monotonic() > deadline:
            self._items.pop(key, None)
            logger.info(
                "pending_expired",
                chat_id=chat_id,
                assistant_id=assistant_id,
            )
            return None
        return media

    async def clear(self, chat_id: int, assistant_id: str) -> None:
        self._items.pop(self._key(chat_id, assistant_id), None)

    async def close(self) -> None:
        self._closed = True
        self._items.clear()


class RedisPendingMediaStorage(PendingMediaStorage):
    """Redis-реализация. Клиент можно передать снаружи (тесты)."""

    def __init__(
        self,
        redis: Redis,
        *,
        ttl_seconds: float | None = None,
        owns_client: bool = True,
    ) -> None:
        self._redis = redis
        self._ttl = ttl_seconds if ttl_seconds is not None else pending_ttl_seconds()
        self._owns_client = owns_client
        self._closed = False

    def _redis_key(self, chat_id: int, assistant_id: str) -> str:
        return _KEY.format(chat_id=chat_id, assistant_id=assistant_id)

    async def ping(self) -> None:
        try:
            await self._redis.ping()
        except RedisError as exc:
            raise PendingMediaError("Redis недоступен.") from exc

    async def put(self, media: PendingMedia) -> None:
        if self._closed:
            raise PendingMediaError("Хранилище отложенных фото закрыто.")
        try:
            await self._redis.set(
                self._redis_key(media.chat_id, media.assistant_id),
                media.to_json(),
                ex=max(int(self._ttl), 1),
            )
        except RedisError as exc:
            raise PendingMediaError("Не удалось сохранить отложенные фото.") from exc

    async def get(self, chat_id: int, assistant_id: str) -> PendingMedia | None:
        try:
            raw = await self._redis.get(self._redis_key(chat_id, assistant_id))
        except RedisError as exc:
            raise PendingMediaError("Не удалось прочитать отложенные фото.") from exc
        if not raw:
            return None
        text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
        try:
            return PendingMedia.from_json(text)
        except (json.JSONDecodeError, PendingMediaError) as exc:
            raise PendingMediaError("Повреждённые отложенные фото.") from exc

    async def clear(self, chat_id: int, assistant_id: str) -> None:
        try:
            await self._redis.delete(self._redis_key(chat_id, assistant_id))
        except RedisError as exc:
            raise PendingMediaError("Не удалось удалить отложенные фото.") from exc

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            try:
                await self._redis.aclose()
            except RedisError:
                logger.exception("pending_media_close_failed")


async def create_pending_media_storage(
    redis_url: str | None = None,
) -> PendingMediaStorage:
    """Redis, если URL задан и сервер отвечает, иначе память процесса."""
    url = (redis_url if redis_url is not None else os.getenv("REDIS_URL", "")).strip()
    if not url:
        logger.info("pending_media_memory", reason="empty_redis_url")
        return InMemoryPendingMediaStorage()
    client: Redis = Redis.from_url(url, decode_responses=True)
    storage = RedisPendingMediaStorage(client, owns_client=True)
    try:
        await storage.ping()
    except PendingMediaError as exc:
        logger.warning("pending_media_memory", reason=exc.message)
        await storage.close()
        return InMemoryPendingMediaStorage()
    logger.info("pending_media_redis", ttl=storage._ttl)
    return storage

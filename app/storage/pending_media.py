"""Отложенные фото: только file_id, без байтов.

Ключ ``chat:{chat_id}:user:{user_id}:asst:{assistant_id}:pending_photo``.
TTL по умолчанию 900 с. Пустой ``REDIS_URL`` или недоступный Redis —
память процесса (в production при заданном Redis — fail-closed).
"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError, WatchError

logger = structlog.get_logger("pending_media")

_KEY = "chat:{chat_id}:user:{user_id}:asst:{assistant_id}:pending_photo"


def pending_ttl_seconds() -> int:
    """TTL отложенных фото из ``PHOTO_PENDING_TTL_SECONDS``."""
    from app.infra.settings import get_settings

    return get_settings().photo_pending_ttl_seconds


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
    user_id: int
    assistant_id: str
    items: list[PendingMediaItem]
    caption: str = ""
    topic_id: str = ""
    media_group_id: str | None = None
    clarify_rounds: int = 0
    last_question: str | None = None
    base_override: int | None = None
    role_retry_used: bool = False
    created_at: float = field(default_factory=time.time)

    def to_json(self) -> str:
        payload = {
            "chat_id": self.chat_id,
            "user_id": self.user_id,
            "assistant_id": self.assistant_id,
            "caption": self.caption,
            "topic_id": self.topic_id,
            "media_group_id": self.media_group_id,
            "clarify_rounds": self.clarify_rounds,
            "last_question": self.last_question,
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
        group = data.get("media_group_id")
        question = data.get("last_question")
        return cls(
            chat_id=int(data.get("chat_id") or 0),
            user_id=int(data.get("user_id") or 0),
            assistant_id=str(data.get("assistant_id") or ""),
            items=items,
            caption=str(data.get("caption") or ""),
            topic_id=str(data.get("topic_id") or ""),
            media_group_id=group if isinstance(group, str) and group else None,
            clarify_rounds=int(data.get("clarify_rounds") or 0),
            last_question=question if isinstance(question, str) and question else None,
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
    async def get(
        self, chat_id: int, assistant_id: str, user_id: int
    ) -> PendingMedia | None:
        """Возвращает альбом или None, если его нет или TTL истёк."""

    @abstractmethod
    async def clear(self, chat_id: int, assistant_id: str, user_id: int) -> None:
        """Удаляет альбом."""

    @abstractmethod
    async def clear_user(self, chat_id: int, user_id: int) -> bool:
        """Снимает pending у всех персон пользователя в чате. True, если что-то было."""

    async def pop(
        self, chat_id: int, assistant_id: str, user_id: int
    ) -> PendingMedia | None:
        """Atomically read-and-delete альбом (override in Redis/memory)."""
        media = await self.get(chat_id, assistant_id, user_id)
        if media is not None:
            await self.clear(chat_id, assistant_id, user_id)
        return media

    async def set_base_override(
        self,
        chat_id: int,
        assistant_id: str,
        user_id: int,
        base_override: int | None,
    ) -> PendingMedia | None:
        """Меняет выбранную основу, не продлевая TTL."""

        def _apply(media: PendingMedia) -> None:
            media.base_override = base_override

        return await self._mutate(chat_id, assistant_id, user_id, _apply)

    async def bump_round(
        self,
        chat_id: int,
        assistant_id: str,
        user_id: int,
        question: str | None = None,
    ) -> PendingMedia | None:
        """Увеличивает счётчик уточнений и запоминает последний вопрос."""

        def _apply(media: PendingMedia) -> None:
            media.clarify_rounds += 1
            media.last_question = question

        updated = await self._mutate(chat_id, assistant_id, user_id, _apply)
        if updated is not None:
            logger.info(
                "clarify_round",
                chat_id=chat_id,
                user_id=user_id,
                assistant_id=assistant_id,
                round=updated.clarify_rounds,
            )
        return updated

    async def mark_role_retry(
        self, chat_id: int, assistant_id: str, user_id: int
    ) -> None:
        """Отмечает, что кнопки выбора основы уже показаны один раз."""

        def _apply(media: PendingMedia) -> None:
            media.role_retry_used = True

        await self._mutate(chat_id, assistant_id, user_id, _apply)

    @abstractmethod
    async def _mutate(
        self,
        chat_id: int,
        assistant_id: str,
        user_id: int,
        mutator: Callable[[PendingMedia], None],
    ) -> PendingMedia | None:
        """Apply ``mutator`` in place without resetting TTL."""

    @abstractmethod
    async def close(self) -> None:
        """Закрывает соединение, если хранилище им владеет."""


class InMemoryPendingMediaStorage(PendingMediaStorage):
    """Память процесса с тем же TTL, что и Redis."""

    def __init__(self, ttl_seconds: float | None = None) -> None:
        self._ttl = ttl_seconds if ttl_seconds is not None else pending_ttl_seconds()
        self._items: dict[tuple[int, int, str], tuple[PendingMedia, float]] = {}
        self._closed = False
        self._lock = asyncio.Lock()

    def _key(
        self, chat_id: int, user_id: int, assistant_id: str
    ) -> tuple[int, int, str]:
        return chat_id, user_id, assistant_id

    async def put(self, media: PendingMedia) -> None:
        if self._closed:
            raise PendingMediaError("Хранилище отложенных фото закрыто.")
        deadline = time.monotonic() + self._ttl
        async with self._lock:
            self._items[self._key(media.chat_id, media.user_id, media.assistant_id)] = (
                media,
                deadline,
            )

    async def get(
        self, chat_id: int, assistant_id: str, user_id: int
    ) -> PendingMedia | None:
        key = self._key(chat_id, user_id, assistant_id)
        async with self._lock:
            found = self._items.get(key)
            if found is None:
                return None
            media, deadline = found
            if time.monotonic() > deadline:
                self._items.pop(key, None)
                logger.info(
                    "pending_expired",
                    chat_id=chat_id,
                    user_id=user_id,
                    assistant_id=assistant_id,
                )
                return None
            return media

    async def clear(self, chat_id: int, assistant_id: str, user_id: int) -> None:
        async with self._lock:
            self._items.pop(self._key(chat_id, user_id, assistant_id), None)

    async def clear_user(self, chat_id: int, user_id: int) -> bool:
        async with self._lock:
            keys = [
                key
                for key in self._items
                if key[0] == chat_id and key[1] == user_id
            ]
            for key in keys:
                self._items.pop(key, None)
            return bool(keys)

    async def pop(
        self, chat_id: int, assistant_id: str, user_id: int
    ) -> PendingMedia | None:
        key = self._key(chat_id, user_id, assistant_id)
        async with self._lock:
            found = self._items.pop(key, None)
            if found is None:
                return None
            media, deadline = found
            if time.monotonic() > deadline:
                logger.info(
                    "pending_expired",
                    chat_id=chat_id,
                    user_id=user_id,
                    assistant_id=assistant_id,
                )
                return None
            return media

    async def _mutate(
        self,
        chat_id: int,
        assistant_id: str,
        user_id: int,
        mutator: Callable[[PendingMedia], None],
    ) -> PendingMedia | None:
        key = self._key(chat_id, user_id, assistant_id)
        async with self._lock:
            found = self._items.get(key)
            if found is None:
                return None
            media, deadline = found
            if time.monotonic() > deadline:
                self._items.pop(key, None)
                logger.info(
                    "pending_expired",
                    chat_id=chat_id,
                    user_id=user_id,
                    assistant_id=assistant_id,
                )
                return None
            mutator(media)
            self._items[key] = (media, deadline)
            return media

    async def close(self) -> None:
        self._closed = True
        async with self._lock:
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

    def _redis_key(self, chat_id: int, user_id: int, assistant_id: str) -> str:
        return _KEY.format(chat_id=chat_id, user_id=user_id, assistant_id=assistant_id)

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
                self._redis_key(media.chat_id, media.user_id, media.assistant_id),
                media.to_json(),
                ex=max(int(self._ttl), 1),
            )
        except RedisError as exc:
            raise PendingMediaError("Не удалось сохранить отложенные фото.") from exc

    async def get(
        self, chat_id: int, assistant_id: str, user_id: int
    ) -> PendingMedia | None:
        try:
            raw = await self._redis.get(self._redis_key(chat_id, user_id, assistant_id))
        except RedisError as exc:
            raise PendingMediaError("Не удалось прочитать отложенные фото.") from exc
        if not raw:
            return None
        text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
        try:
            return PendingMedia.from_json(text)
        except (json.JSONDecodeError, PendingMediaError) as exc:
            raise PendingMediaError("Повреждённые отложенные фото.") from exc

    async def clear(self, chat_id: int, assistant_id: str, user_id: int) -> None:
        try:
            await self._redis.delete(self._redis_key(chat_id, user_id, assistant_id))
        except RedisError as exc:
            raise PendingMediaError("Не удалось удалить отложенные фото.") from exc

    async def clear_user(self, chat_id: int, user_id: int) -> bool:
        pattern = f"chat:{chat_id}:user:{user_id}:asst:*:pending_photo"
        cleared = False
        try:
            async for key in self._redis.scan_iter(match=pattern, count=50):
                await self._redis.delete(key)
                cleared = True
        except RedisError as exc:
            raise PendingMediaError("Не удалось удалить отложенные фото.") from exc
        return cleared

    async def pop(
        self, chat_id: int, assistant_id: str, user_id: int
    ) -> PendingMedia | None:
        key = self._redis_key(chat_id, user_id, assistant_id)
        try:
            # GETDEL is atomic (Redis 6.2+); fakeredis and redis:7 support it.
            raw = await self._redis.getdel(key)
        except RedisError as exc:
            raise PendingMediaError("Не удалось извлечь отложенные фото.") from exc
        if not raw:
            return None
        text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
        try:
            return PendingMedia.from_json(text)
        except (json.JSONDecodeError, PendingMediaError) as exc:
            raise PendingMediaError("Повреждённые отложенные фото.") from exc

    async def _mutate(
        self,
        chat_id: int,
        assistant_id: str,
        user_id: int,
        mutator: Callable[[PendingMedia], None],
    ) -> PendingMedia | None:
        if self._closed:
            raise PendingMediaError("Хранилище отложенных фото закрыто.")
        key = self._redis_key(chat_id, user_id, assistant_id)
        for _ in range(5):
            try:
                async with self._redis.pipeline(True) as pipe:
                    await pipe.watch(key)
                    raw = await pipe.get(key)
                    if not raw:
                        await pipe.reset()
                        return None
                    text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
                    try:
                        media = PendingMedia.from_json(text)
                    except (json.JSONDecodeError, PendingMediaError) as exc:
                        await pipe.reset()
                        raise PendingMediaError(
                            "Повреждённые отложенные фото."
                        ) from exc
                    mutator(media)
                    pipe.multi()
                    pipe.set(key, media.to_json(), keepttl=True)
                    await pipe.execute()
                    return media
            except WatchError:
                continue
            except RedisError as exc:
                raise PendingMediaError("Не удалось обновить отложенные фото.") from exc
        raise PendingMediaError("Не удалось обновить отложенные фото (конфликт).")

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
    """Redis, если URL задан и сервер отвечает, иначе память процесса.

    В production при заданном ``REDIS_URL`` недоступный Redis — fatal
    (как FSM), без тихой деградации в память процесса.
    """
    from app.infra.access import is_production
    from app.infra.settings import get_settings

    url = (redis_url if redis_url is not None else get_settings().redis_url).strip()
    if not url:
        if is_production():
            logger.error("pending_media_redis_required_in_production")
            raise SystemExit(1)
        logger.info("pending_media_memory", reason="empty_redis_url")
        return InMemoryPendingMediaStorage()
    client: Redis = Redis.from_url(url, decode_responses=True)
    storage = RedisPendingMediaStorage(client, owns_client=True)
    try:
        await storage.ping()
    except PendingMediaError as exc:
        await storage.close()
        if is_production():
            logger.error("pending_media_redis_failed_fatal", reason=exc.message)
            raise SystemExit(1) from exc
        logger.warning("pending_media_memory", reason=exc.message)
        return InMemoryPendingMediaStorage()
    logger.info("pending_media_redis", ttl=storage._ttl)
    return storage

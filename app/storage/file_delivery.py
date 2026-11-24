"""Недоставленные озвучка и расшифровка: ключ объекта в бакете.

Провайдер не оставляет ссылку на эти файлы. Бот сам кладёт байты в Object
Storage и 7 дней повторяет отправку. В лог ключ не пишется.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.storage.image_delivery import retry_delay_seconds
from app.storage.redis_await import await_redis

logger = structlog.get_logger("file_delivery")

FILE_DELIVERY_TTL_SECONDS = 7 * 24 * 3600
_INDEX_KEY = "file_delivery:index"
_RECORD_KEY = "file_delivery:{chat_id}:{user_id}"
_KINDS = frozenset({"audio", "document"})


def _safe_filename(name: str) -> str:
    base = name.replace("\\", "/").split("/")[-1].strip() or "file.bin"
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    return cleaned[:180] or "file.bin"


def _safe_key(key: str) -> str | None:
    if (
        isinstance(key, str)
        and key.startswith("delivery/")
        and ".." not in key
        and len(key) <= 300
    ):
        return key
    return None


@dataclass
class KeptFile:
    key: str
    filename: str
    kind: str
    mime: str

    def to_dict(self) -> dict[str, str]:
        return {
            "key": self.key,
            "filename": self.filename,
            "kind": self.kind,
            "mime": self.mime,
        }


def _files_from(payload: object) -> list[KeptFile]:
    if not isinstance(payload, list):
        return []
    files: list[KeptFile] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        key = _safe_key(str(item.get("key") or ""))
        kind = item.get("kind")
        if key is None or kind not in _KINDS:
            continue
        filename = _safe_filename(str(item.get("filename") or "file.bin"))
        mime = str(item.get("mime") or "application/octet-stream")[:120]
        files.append(KeptFile(key=key, filename=filename, kind=kind, mime=mime))
    return files


@dataclass
class FileDelivery:
    chat_id: int
    user_id: int
    files: list[KeptFile]
    created_at: float
    attempts: int = 0
    not_before: float = 0.0

    def expired(self, now: float) -> bool:
        return now >= self.created_at + FILE_DELIVERY_TTL_SECONDS

    def to_json(self) -> str:
        return json.dumps(
            {
                "chat_id": self.chat_id,
                "user_id": self.user_id,
                "files": [item.to_dict() for item in self.files],
                "created_at": self.created_at,
                "attempts": self.attempts,
                "not_before": self.not_before,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> FileDelivery | None:
        try:
            payload = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        files = _files_from(payload.get("files"))
        try:
            return cls(
                chat_id=int(payload["chat_id"]),
                user_id=int(payload["user_id"]),
                files=files,
                created_at=float(payload["created_at"]),
                attempts=int(payload.get("attempts") or 0),
                not_before=float(payload.get("not_before") or 0),
            )
        except (KeyError, TypeError, ValueError):
            return None


def _remaining_ttl(created_at: float, now: float) -> int:
    left = int(created_at + FILE_DELIVERY_TTL_SECONDS - now)
    return max(left, 1)


@dataclass
class InMemoryFileDeliveryStore:
    _items: dict[tuple[int, int], FileDelivery] = field(default_factory=dict)

    async def add(
        self, record_file: KeptFile, chat_id: int, user_id: int, *, now: float
    ) -> bool:
        key = (chat_id, user_id)
        current = self._items.get(key)
        if current is None or current.expired(now):
            self._items[key] = FileDelivery(
                chat_id=chat_id,
                user_id=user_id,
                files=[record_file],
                created_at=now,
                not_before=now + retry_delay_seconds(0),
            )
            return True
        if any(item.key == record_file.key for item in current.files):
            return True
        current.files.append(record_file)
        return True

    async def due(self, now: float) -> list[FileDelivery]:
        return [
            record
            for record in self._items.values()
            if record.expired(now) or record.not_before <= now
        ]

    async def save(self, record: FileDelivery, *, now: float) -> None:
        if record.expired(now) or not record.files:
            self._items.pop((record.chat_id, record.user_id), None)
            return
        self._items[(record.chat_id, record.user_id)] = record

    async def delete(self, chat_id: int, user_id: int) -> None:
        self._items.pop((chat_id, user_id), None)


class RedisFileDeliveryStore:
    """Те же записи в Redis. Клиент общий с ботом, здесь его не закрываем."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def _key(self, chat_id: int, user_id: int) -> str:
        return _RECORD_KEY.format(chat_id=chat_id, user_id=user_id)

    async def add(
        self, record_file: KeptFile, chat_id: int, user_id: int, *, now: float
    ) -> bool:
        key = self._key(chat_id, user_id)
        try:
            current = await self._load(key)
            if current is None or current.expired(now):
                record = FileDelivery(
                    chat_id=chat_id,
                    user_id=user_id,
                    files=[record_file],
                    created_at=now,
                    not_before=now + retry_delay_seconds(0),
                )
            elif any(item.key == record_file.key for item in current.files):
                return True
            else:
                current.files.append(record_file)
                record = current
            await self._redis.set(
                key, record.to_json(), ex=_remaining_ttl(record.created_at, now)
            )
            await await_redis(self._redis.sadd(_INDEX_KEY, key))
            await await_redis(self._redis.expire(_INDEX_KEY, FILE_DELIVERY_TTL_SECONDS))
            return True
        except RedisError:
            logger.warning("file_delivery_store_failed", chat_id=chat_id)
            return False

    async def due(self, now: float) -> list[FileDelivery]:
        try:
            keys = await await_redis(self._redis.smembers(_INDEX_KEY))
        except RedisError:
            logger.warning("file_delivery_index_failed")
            return []
        found: list[FileDelivery] = []
        for key in keys:
            text_key = key if isinstance(key, str) else bytes(key).decode("utf-8")
            try:
                record = await self._load(text_key)
            except RedisError:
                logger.warning("file_delivery_load_failed")
                continue
            if record is None:
                try:
                    await await_redis(self._redis.srem(_INDEX_KEY, text_key))
                except RedisError as exc:
                    logger.debug(
                        "file_delivery_index_prune_failed",
                        key=text_key,
                        error=str(exc),
                    )
                continue
            if record.expired(now) or record.not_before <= now:
                found.append(record)
        return found

    async def save(self, record: FileDelivery, *, now: float) -> None:
        key = self._key(record.chat_id, record.user_id)
        try:
            if record.expired(now) or not record.files:
                await self.delete(record.chat_id, record.user_id)
                return
            await self._redis.set(
                key, record.to_json(), ex=_remaining_ttl(record.created_at, now)
            )
            await await_redis(self._redis.sadd(_INDEX_KEY, key))
        except RedisError:
            logger.warning("file_delivery_store_failed", chat_id=record.chat_id)

    async def delete(self, chat_id: int, user_id: int) -> None:
        key = self._key(chat_id, user_id)
        try:
            await self._redis.delete(key)
            await await_redis(self._redis.srem(_INDEX_KEY, key))
        except RedisError:
            logger.warning("file_delivery_delete_failed", chat_id=chat_id)

    async def _load(self, key: str) -> FileDelivery | None:
        raw = await self._redis.get(key)
        if not raw:
            return None
        text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
        return FileDelivery.from_json(text)


def open_file_delivery_store(
    redis: Redis | None,
) -> InMemoryFileDeliveryStore | RedisFileDeliveryStore:
    if redis is None:
        logger.info("file_delivery_memory")
        return InMemoryFileDeliveryStore()
    return RedisFileDeliveryStore(redis)


def delivery_clock() -> float:
    return time.time()

"""Недоставленные картинки: ссылка или id уже оплаченной задачи.

Файл на CDN Polza живёт 7 дней. Пока срок не вышел, бот хранит
ссылку или id задачи и повторяет скачивание. В лог ссылка не пишется.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import structlog
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.llm.media_api import paid_job_id, public_https_image_url
from app.storage.redis_await import await_redis

logger = structlog.get_logger("image_delivery")

IMAGE_DELIVERY_TTL_SECONDS = 7 * 24 * 3600
_INDEX_KEY = "image_delivery:index"
_RECORD_KEY = "image_delivery:{chat_id}:{user_id}"


def retry_delay_seconds(attempts: int) -> int:
    """Пауза до следующей попытки: минута, затем дольше, не больше получаса."""
    shift = min(max(attempts, 0), 6)
    return min(60 * (2**shift), 1800)


def _safe_job(provider: str, job_id: str) -> dict[str, str] | None:
    if provider not in {"polza", "proxyapi"} or not paid_job_id(job_id):
        return None
    return {"provider": provider, "id": job_id}


def _jobs_from(payload: object) -> list[dict[str, str]]:
    if not isinstance(payload, list):
        return []
    jobs: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider")
        job_id = item.get("id")
        if not isinstance(provider, str) or not isinstance(job_id, str):
            continue
        job = _safe_job(provider, job_id)
        if job is not None and job not in jobs:
            jobs.append(job)
    return jobs


def _safe_filename(name: str) -> str:
    cleaned = name.strip()
    if cleaned == "image.png" or (
        cleaned.startswith("image_")
        and cleaned.endswith(".png")
        and cleaned[6:-4].isdigit()
    ):
        return cleaned
    return "image.png"


@dataclass
class ImageDelivery:
    """Один чат: несколько ещё не дошедших файлов одной или разных генераций."""

    chat_id: int
    user_id: int
    urls: list[str]
    filenames: list[str]
    created_at: float
    attempts: int = 0
    not_before: float = 0.0
    jobs: list[dict[str, str]] = field(default_factory=list)

    def expired(self, now: float) -> bool:
        return now >= self.created_at + IMAGE_DELIVERY_TTL_SECONDS

    def to_json(self) -> str:
        return json.dumps(
            {
                "chat_id": self.chat_id,
                "user_id": self.user_id,
                "urls": self.urls,
                "filenames": self.filenames,
                "created_at": self.created_at,
                "attempts": self.attempts,
                "not_before": self.not_before,
                "jobs": self.jobs,
            },
            ensure_ascii=False,
        )

    @classmethod
    def from_json(cls, raw: str) -> ImageDelivery | None:
        try:
            payload = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(payload, dict):
            return None
        urls = payload.get("urls")
        names = payload.get("filenames")
        if not isinstance(urls, list) or not isinstance(names, list):
            return None
        clean_urls = [item for item in urls if isinstance(item, str) and item]
        clean_names = [item for item in names if isinstance(item, str) and item]
        jobs = _jobs_from(payload.get("jobs"))
        if clean_urls and len(clean_urls) != len(clean_names):
            return None
        if not clean_urls and not jobs:
            return None
        try:
            return cls(
                chat_id=int(payload["chat_id"]),
                user_id=int(payload["user_id"]),
                urls=clean_urls,
                filenames=clean_names[: len(clean_urls)],
                created_at=float(payload["created_at"]),
                attempts=int(payload.get("attempts") or 0),
                not_before=float(payload.get("not_before") or 0),
                jobs=jobs,
            )
        except (KeyError, TypeError, ValueError):
            return None


def _remaining_ttl(created_at: float, now: float) -> int:
    left = int(created_at + IMAGE_DELIVERY_TTL_SECONDS - now)
    return max(left, 1)


@dataclass
class InMemoryImageDeliveryStore:
    """Память процесса, когда Redis нет (не production)."""

    _items: dict[tuple[int, int], ImageDelivery] = field(default_factory=dict)

    async def add(
        self, chat_id: int, user_id: int, url: str, filename: str, *, now: float
    ) -> bool:
        if not public_https_image_url(url):
            return False
        key = (chat_id, user_id)
        current = self._items.get(key)
        name = _safe_filename(filename)
        if current is None or current.expired(now):
            self._items[key] = ImageDelivery(
                chat_id=chat_id,
                user_id=user_id,
                urls=[url],
                filenames=[name],
                created_at=now,
                not_before=now + retry_delay_seconds(0),
            )
            return True
        if url in current.urls:
            return True
        current.urls.append(url)
        current.filenames.append(name)
        return True

    async def due(self, now: float) -> list[ImageDelivery]:
        found: list[ImageDelivery] = []
        for record in self._items.values():
            if record.expired(now) or record.not_before <= now:
                found.append(record)
        return found

    async def save(self, record: ImageDelivery, *, now: float) -> None:
        if record.expired(now) or (not record.urls and not record.jobs):
            self._items.pop((record.chat_id, record.user_id), None)
            return
        self._items[(record.chat_id, record.user_id)] = record

    async def add_job(
        self, chat_id: int, user_id: int, provider: str, job_id: str, *, now: float
    ) -> bool:
        job = _safe_job(provider, job_id)
        if job is None:
            return False
        key = (chat_id, user_id)
        current = self._items.get(key)
        if current is None or current.expired(now):
            self._items[key] = ImageDelivery(
                chat_id=chat_id,
                user_id=user_id,
                urls=[],
                filenames=[],
                created_at=now,
                jobs=[job],
                not_before=now + retry_delay_seconds(0),
            )
            return True
        if job in current.jobs:
            return True
        current.jobs.append(job)
        return True

    async def delete(self, chat_id: int, user_id: int) -> None:
        self._items.pop((chat_id, user_id), None)


class RedisImageDeliveryStore:
    """Те же записи в Redis. Клиент общий с ботом, здесь его не закрываем."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    def _key(self, chat_id: int, user_id: int) -> str:
        return _RECORD_KEY.format(chat_id=chat_id, user_id=user_id)

    async def add(
        self, chat_id: int, user_id: int, url: str, filename: str, *, now: float
    ) -> bool:
        if not public_https_image_url(url):
            return False
        key = self._key(chat_id, user_id)
        try:
            current = await self._load(key)
            name = _safe_filename(filename)
            if current is None or current.expired(now):
                record = ImageDelivery(
                    chat_id=chat_id,
                    user_id=user_id,
                    urls=[url],
                    filenames=[name],
                    created_at=now,
                    not_before=now + retry_delay_seconds(0),
                )
            elif url in current.urls:
                return True
            else:
                current.urls.append(url)
                current.filenames.append(name)
                record = current
            await self._redis.set(
                key, record.to_json(), ex=_remaining_ttl(record.created_at, now)
            )
            await await_redis(self._redis.sadd(_INDEX_KEY, key))
            return True
        except RedisError:
            logger.warning("image_delivery_store_failed", chat_id=chat_id)
            return False

    async def due(self, now: float) -> list[ImageDelivery]:
        try:
            keys = await await_redis(self._redis.smembers(_INDEX_KEY))
        except RedisError:
            logger.warning("image_delivery_index_failed")
            return []
        found: list[ImageDelivery] = []
        for key in keys:
            text_key = key if isinstance(key, str) else bytes(key).decode("utf-8")
            try:
                record = await self._load(text_key)
            except RedisError:
                logger.warning("image_delivery_load_failed")
                continue
            if record is None:
                try:
                    await await_redis(self._redis.srem(_INDEX_KEY, text_key))
                except RedisError as exc:
                    logger.debug(
                        "image_delivery_index_prune_failed",
                        key=text_key,
                        error=str(exc),
                    )
                continue
            if record.expired(now) or record.not_before <= now:
                found.append(record)
        return found

    async def save(self, record: ImageDelivery, *, now: float) -> None:
        key = self._key(record.chat_id, record.user_id)
        try:
            if record.expired(now) or (not record.urls and not record.jobs):
                await self.delete(record.chat_id, record.user_id)
                return
            await self._redis.set(
                key, record.to_json(), ex=_remaining_ttl(record.created_at, now)
            )
            await await_redis(self._redis.sadd(_INDEX_KEY, key))
        except RedisError:
            logger.warning("image_delivery_store_failed", chat_id=record.chat_id)

    async def add_job(
        self, chat_id: int, user_id: int, provider: str, job_id: str, *, now: float
    ) -> bool:
        job = _safe_job(provider, job_id)
        if job is None:
            return False
        key = self._key(chat_id, user_id)
        try:
            current = await self._load(key)
            if current is None or current.expired(now):
                record = ImageDelivery(
                    chat_id=chat_id,
                    user_id=user_id,
                    urls=[],
                    filenames=[],
                    created_at=now,
                    jobs=[job],
                    not_before=now + retry_delay_seconds(0),
                )
            elif job in current.jobs:
                return True
            else:
                current.jobs.append(job)
                record = current
            await self._redis.set(
                key, record.to_json(), ex=_remaining_ttl(record.created_at, now)
            )
            await await_redis(self._redis.sadd(_INDEX_KEY, key))
            return True
        except RedisError:
            logger.warning("image_delivery_store_failed", chat_id=chat_id)
            return False

    async def delete(self, chat_id: int, user_id: int) -> None:
        key = self._key(chat_id, user_id)
        try:
            await self._redis.delete(key)
            await await_redis(self._redis.srem(_INDEX_KEY, key))
        except RedisError:
            logger.warning("image_delivery_delete_failed", chat_id=chat_id)

    async def _load(self, key: str) -> ImageDelivery | None:
        raw = await self._redis.get(key)
        if not raw:
            return None
        text = raw if isinstance(raw, str) else bytes(raw).decode("utf-8")
        return ImageDelivery.from_json(text)


def open_image_delivery_store(
    redis: Redis | None,
) -> InMemoryImageDeliveryStore | RedisImageDeliveryStore:
    if redis is None:
        logger.info("image_delivery_memory")
        return InMemoryImageDeliveryStore()
    return RedisImageDeliveryStore(redis)


def delivery_clock() -> float:
    return time.time()

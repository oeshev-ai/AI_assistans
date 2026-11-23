"""Повторная доставка картинки с CDN Polza без новой генерации."""

from __future__ import annotations

import pytest

from app.bot.image_retry import (
    IMAGE_EXPIRED_TEXT,
    IMAGE_LATE_CAPTION,
    IMAGE_RETRY_TEXT,
    bind_image_delivery,
    deliver_due,
    telegram_photo_needs_document,
)
from app.bot.runtime import _send_generated_image
from app.llm.errors import AssistAIError
from app.storage.image_delivery import (
    IMAGE_DELIVERY_TTL_SECONDS,
    InMemoryImageDeliveryStore,
    RedisImageDeliveryStore,
    retry_delay_seconds,
)


class _Chat:
    id = 5


class _User:
    id = 9


class _Message:
    chat = _Chat()
    from_user = _User()


class _Bot:
    def __init__(self) -> None:
        self.photos: list[tuple[int, str | None]] = []
        self.texts: list[str] = []

    async def send_photo(
        self,
        chat_id: int,
        photo: object,
        caption: str | None = None,
        request_timeout: int | None = None,
    ) -> None:
        self.photos.append((chat_id, caption))

    async def send_message(self, chat_id: int, text: str) -> None:
        self.texts.append(text)

    async def send_document(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("document")


class _Redis:
    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}
        self.ttl: dict[str, int | None] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.kv[key] = value
        self.ttl[key] = ex

    async def get(self, key: str) -> str | None:
        return self.kv.get(key)

    async def delete(self, key: str) -> None:
        self.kv.pop(key, None)

    async def sadd(self, key: str, member: str) -> None:
        self.sets.setdefault(key, set()).add(member)

    async def srem(self, key: str, member: str) -> None:
        self.sets.get(key, set()).discard(member)

    async def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))


def test_retry_copy_promises_repeats_without_a_new_generation() -> None:
    for text in (IMAGE_RETRY_TEXT, IMAGE_EXPIRED_TEXT, IMAGE_LATE_CAPTION):
        assert "://" not in text
        assert "{" not in text
    assert "будут повторяться" in IMAGE_RETRY_TEXT
    assert "не запускается" in IMAGE_RETRY_TEXT
    assert "7 дней" in IMAGE_RETRY_TEXT
    assert IMAGE_DELIVERY_TTL_SECONDS == 7 * 24 * 3600
    assert retry_delay_seconds(0) == 60
    assert retry_delay_seconds(10) == 1800


@pytest.mark.asyncio
async def test_public_https_is_kept_and_private_hosts_are_not() -> None:
    store = InMemoryImageDeliveryStore()
    assert await store.add(1, 2, "https://s3.polza.ai/a.png", "image.png", now=10)
    assert await store.add(1, 2, "https://cdn.example/a.png", "image.png", now=10)
    assert (
        await store.add(1, 2, "http://s3.polza.ai/a.png", "image.png", now=10) is False
    )
    assert (
        await store.add(1, 2, "https://127.0.0.1/a.png", "image.png", now=10) is False
    )
    record = (await store.due(70))[0]
    assert record.urls == [
        "https://s3.polza.ai/a.png",
        "https://cdn.example/a.png",
    ]
    assert record.created_at == 10
    assert record.not_before == 70


@pytest.mark.asyncio
async def test_send_failure_schedules_the_same_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryImageDeliveryStore()
    bind_image_delivery(store)

    async def _boom(_url: str) -> bytes:
        raise AssistAIError("Не удалось скачать изображение с CDN провайдера.")

    monkeypatch.setattr("app.bot.runtime._download_image_url", _boom)
    try:
        with pytest.raises(AssistAIError) as info:
            await _send_generated_image(
                _Message(),  # type: ignore[arg-type]
                "https://s3.polza.ai/paid.png",
                0,
            )
    finally:
        bind_image_delivery(None)
    assert info.value.message == IMAGE_RETRY_TEXT
    waiting = await store.due(10**12)
    assert waiting[0].chat_id == 5
    assert waiting[0].urls == ["https://s3.polza.ai/paid.png"]
    assert waiting[0].filenames == ["image.png"]


class _FileBot(_Bot):
    def __init__(self) -> None:
        super().__init__()
        self.documents: list[tuple[int, str | None]] = []

    async def send_document(
        self,
        chat_id: int,
        document: object,
        caption: str | None = None,
        request_timeout: int | None = None,
    ) -> None:
        self.documents.append((chat_id, caption))


@pytest.mark.asyncio
async def test_due_download_sends_a_saved_file_choice() -> None:
    store = InMemoryImageDeliveryStore()
    await store.add(
        5,
        9,
        "https://s3-alt.polza.ai/b.png",
        "image.png",
        now=0,
        as_file=True,
    )
    bot = _FileBot()

    async def _ok(_url: str) -> bytes:
        return b"png"

    await deliver_due(bot, store, _ok, now=60)  # type: ignore[arg-type]
    assert bot.photos == []
    assert bot.documents == [(5, IMAGE_LATE_CAPTION)]


@pytest.mark.asyncio
async def test_due_download_sends_photo_and_drops_the_record() -> None:
    store = InMemoryImageDeliveryStore()
    await store.add(5, 9, "https://s3-alt.polza.ai/b.png", "image_2.png", now=0)
    bot = _Bot()

    async def _ok(_url: str) -> bytes:
        return b"png"

    await deliver_due(bot, store, _ok, now=60)  # type: ignore[arg-type]
    assert bot.photos == [(5, IMAGE_LATE_CAPTION)]
    assert await store.due(10**12) == []


@pytest.mark.asyncio
async def test_expired_delivery_tells_the_user_and_stops() -> None:
    store = InMemoryImageDeliveryStore()
    await store.add(5, 9, "https://s3.polza.ai/old.png", "image.png", now=0)
    bot = _Bot()

    async def _unused(_url: str) -> bytes:
        raise AssertionError("download")

    await deliver_due(
        bot,  # type: ignore[arg-type]
        store,
        _unused,
        now=IMAGE_DELIVERY_TTL_SECONDS,
    )
    assert bot.texts == [IMAGE_EXPIRED_TEXT]
    assert bot.photos == []
    assert await store.due(IMAGE_DELIVERY_TTL_SECONDS + 1) == []


@pytest.mark.asyncio
async def test_redis_roundtrip_keeps_ttl_from_the_first_save() -> None:
    redis = _Redis()
    store = RedisImageDeliveryStore(redis)  # type: ignore[arg-type]
    assert await store.add(3, 4, "https://s3.polza.ai/a.png", "image.png", now=100)
    assert await store.add(3, 4, "https://s3.polza.ai/b.png", "image_2.png", now=200)
    record = (await store.due(100 + 60))[0]
    assert record.created_at == 100
    assert record.urls == [
        "https://s3.polza.ai/a.png",
        "https://s3.polza.ai/b.png",
    ]
    assert redis.ttl["image_delivery:3:4"] == IMAGE_DELIVERY_TTL_SECONDS - 100
    await store.delete(3, 4)
    assert await store.due(10**12) == []


def test_photo_size_and_type_errors_fall_back_to_document() -> None:
    assert telegram_photo_needs_document(RuntimeError("IMAGE_PROCESS_FAILED"))
    assert telegram_photo_needs_document(RuntimeError("PHOTO_INVALID_DIMENSIONS"))
    assert telegram_photo_needs_document(RuntimeError("file is too big"))
    assert not telegram_photo_needs_document(
        RuntimeError("bot was blocked by the user")
    )

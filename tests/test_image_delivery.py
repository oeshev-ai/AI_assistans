"""Повторная доставка картинки с CDN Polza без новой генерации."""

from __future__ import annotations

import json

import pytest
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import SendMessage
from redis.exceptions import WatchError

from app.bot.image_retry import (
    IMAGE_EXPIRED_TEXT,
    IMAGE_KEEP_TEXT,
    IMAGE_LATE_CAPTION,
    IMAGE_RETRY_TEXT,
    bind_image_delivery,
    deliver_due,
    image_failure_reply,
    promote_arq_image_jobs,
    telegram_photo_needs_document,
)
from app.bot.runtime import _send_generated_image
from app.llm.errors import AssistAIError
from app.storage.image_delivery import (
    IMAGE_DELIVERY_TTL_SECONDS,
    InMemoryImageDeliveryStore,
    RedisImageDeliveryStore,
    delivery_clock,
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

    async def expire(self, key: str, seconds: int) -> None:
        self.ttl[key] = seconds

    def pipeline(self, transaction: bool = True) -> _Pipe:
        return _Pipe(self)


class _Pipe:
    def __init__(self, redis: _Redis) -> None:
        self._redis = redis
        self._watched: str | None = None
        self._snapshot: str | None = None
        self._ops: list[tuple[object, ...]] = []

    async def __aenter__(self) -> _Pipe:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def watch(self, key: str) -> None:
        self._watched = key
        self._snapshot = self._redis.kv.get(key)

    async def unwatch(self) -> None:
        self._watched = None

    async def get(self, key: str) -> str | None:
        return self._redis.kv.get(key)

    def multi(self) -> None:
        self._ops = []

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._ops.append(("set", key, value, ex))

    def sadd(self, key: str, member: str) -> None:
        self._ops.append(("sadd", key, member))

    def expire(self, key: str, seconds: int) -> None:
        self._ops.append(("expire", key, seconds))

    async def execute(self) -> None:
        if (
            self._watched is not None
            and self._redis.kv.get(self._watched) != self._snapshot
        ):
            raise WatchError()
        for op in self._ops:
            kind = op[0]
            if kind == "set":
                _, key, value, ex = op
                assert isinstance(key, str)
                assert isinstance(value, str)
                assert isinstance(ex, int)
                await self._redis.set(key, value, ex=ex)
            elif kind == "sadd":
                _, key, member = op
                assert isinstance(key, str)
                assert isinstance(member, str)
                await self._redis.sadd(key, member)
            elif kind == "expire":
                _, key, seconds = op
                assert isinstance(key, str)
                assert isinstance(seconds, int)
                await self._redis.expire(key, seconds)
        self._ops = []


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
    assert redis.ttl["image_delivery:index"] == IMAGE_DELIVERY_TTL_SECONDS
    await store.delete(3, 4)
    assert await store.due(10**12) == []


@pytest.mark.asyncio
async def test_add_retries_when_the_record_changes_under_watch() -> None:
    redis = _Redis()
    store = RedisImageDeliveryStore(redis)  # type: ignore[arg-type]
    await store.add(3, 4, "https://s3.polza.ai/a.png", "image.png", now=100)
    real_pipeline = redis.pipeline
    raced = False

    def pipeline(transaction: bool = True) -> _Pipe:
        pipe = real_pipeline(transaction)
        real_get = pipe.get

        async def get(key: str) -> str | None:
            nonlocal raced
            value = await real_get(key)
            if raced or not value or "b.png" in value:
                return value
            raced = True
            payload = json.loads(value)
            payload["urls"].append("https://s3.polza.ai/c.png")
            payload["filenames"].append("image_3.png")
            payload["as_files"].append(False)
            redis.kv[key] = json.dumps(payload, ensure_ascii=False)
            return value

        pipe.get = get  # type: ignore[method-assign]
        return pipe

    redis.pipeline = pipeline  # type: ignore[method-assign]
    assert await store.add(3, 4, "https://s3.polza.ai/b.png", "image_2.png", now=120)
    record = (await store.due(160))[0]
    assert record.urls == [
        "https://s3.polza.ai/a.png",
        "https://s3.polza.ai/c.png",
        "https://s3.polza.ai/b.png",
    ]


def _network_error() -> TelegramNetworkError:
    return TelegramNetworkError(
        method=SendMessage(chat_id=1, text="x"),
        message="HTTP Client says - ServerDisconnectedError: Server disconnected",
    )


class _PhotoMessage(_Message):
    def __init__(self, *, fail_photos: int) -> None:
        self.fail_photos = fail_photos
        self.photos = 0
        self.documents = 0
        self.sent = object()

    async def answer_photo(self, **_kwargs: object) -> object:
        self.photos += 1
        if self.photos <= self.fail_photos:
            raise _network_error()
        return self.sent

    async def answer_document(self, **_kwargs: object) -> object:
        self.documents += 1
        raise _network_error()


@pytest.mark.asyncio
async def test_telegram_disconnect_retries_photo_then_keeps_the_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обрыв sendPhoto не молчит: файл ставится в повторную доставку."""
    store = InMemoryImageDeliveryStore()
    bind_image_delivery(store)
    message = _PhotoMessage(fail_photos=10)

    async def _png(_url: str) -> bytes:
        return b"png"

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.bot.runtime._download_image_url", _png)
    monkeypatch.setattr("app.bot.runtime.asyncio.sleep", _no_sleep)
    try:
        with pytest.raises(AssistAIError) as info:
            await _send_generated_image(
                message,  # type: ignore[arg-type]
                "https://s3-alt.polza.ai/paid.png",
                0,
            )
    finally:
        bind_image_delivery(None)
    assert info.value.message == IMAGE_KEEP_TEXT
    assert message.photos == 4
    assert message.documents == 1
    waiting = await store.due(10**12)
    assert waiting[0].urls == ["https://s3-alt.polza.ai/paid.png"]


@pytest.mark.asyncio
async def test_telegram_disconnect_photo_succeeds_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = _PhotoMessage(fail_photos=2)

    async def _png(_url: str) -> bytes:
        return b"png"

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.bot.runtime._download_image_url", _png)
    monkeypatch.setattr("app.bot.runtime.asyncio.sleep", _no_sleep)
    sent = await _send_generated_image(
        message,  # type: ignore[arg-type]
        "https://s3-alt.polza.ai/paid.png",
        0,
    )
    assert sent is message.sent
    assert message.photos == 3
    assert message.documents == 0


@pytest.mark.asyncio
async def test_running_arq_image_is_scheduled_for_delivery() -> None:
    store = InMemoryImageDeliveryStore()
    bind_image_delivery(store)
    exc = AssistAIError(
        "Задача ещё выполняется. Повтор может списать деньги ещё раз.",
        error_code="QUEUE_JOB_RUNNING",
        job_committed=True,
        pending_job_id="abc123",
        pending_provider="arq",
        pending_kind="image",
    )
    try:
        text = await image_failure_reply(7, 8, exc)
    finally:
        bind_image_delivery(None)
    assert text == IMAGE_KEEP_TEXT
    waiting = await store.active(delivery_clock())
    assert waiting[0].jobs == [{"provider": "arq", "id": "abc123"}]
    assert await store.due(delivery_clock()) == []


@pytest.mark.asyncio
async def test_ready_arq_job_becomes_a_cdn_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryImageDeliveryStore()
    now = delivery_clock()
    assert await store.add_job(1, 2, "arq", "jobhex", now=now)

    async def _urls(job_id: str) -> list[str]:
        assert job_id == "jobhex"
        return ["https://s3.polza.ai/ready.png"]

    monkeypatch.setattr("app.infra.job_queue.poll_arq_image_urls", _urls)
    await promote_arq_image_jobs(_Bot(), store)  # type: ignore[arg-type]
    record = (await store.active(now))[0]
    assert record.urls == ["https://s3.polza.ai/ready.png"]
    assert record.filenames == ["image.png"]
    assert record.jobs == []
    assert record.not_before <= delivery_clock()


@pytest.mark.asyncio
async def test_arq_job_stays_while_the_worker_is_still_running(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryImageDeliveryStore()
    now = delivery_clock()
    await store.add_job(1, 2, "arq", "jobhex", now=now)

    async def _later(_job_id: str) -> None:
        return None

    monkeypatch.setattr("app.infra.job_queue.poll_arq_image_urls", _later)
    await promote_arq_image_jobs(_Bot(), store)  # type: ignore[arg-type]
    record = (await store.active(now))[0]
    assert record.jobs == [{"provider": "arq", "id": "jobhex"}]
    assert record.urls == []


@pytest.mark.asyncio
async def test_lost_arq_result_stops_and_tells_the_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryImageDeliveryStore()
    now = delivery_clock()
    await store.add_job(1, 2, "arq", "jobhex", now=now)
    bot = _Bot()

    async def _lost(_job_id: str) -> list[str]:
        raise AssistAIError(
            "Результат не сохранился в очереди. Повтор может списать деньги ещё раз.",
            error_code="QUEUE_RESULT_LOST",
            job_committed=True,
        )

    monkeypatch.setattr("app.infra.job_queue.poll_arq_image_urls", _lost)
    await promote_arq_image_jobs(bot, store)  # type: ignore[arg-type]
    assert await store.active(now) == []
    assert bot.texts == [
        "Результат не сохранился в очереди. Повтор может списать деньги ещё раз."
    ]


@pytest.mark.asyncio
async def test_delivery_pass_does_not_resume_an_arq_job() -> None:
    store = InMemoryImageDeliveryStore()
    now = delivery_clock()
    await store.add_job(1, 2, "arq", "jobhex", now=now)
    record = (await store.active(now))[0]
    record.not_before = now
    await store.save(record, now=now)

    async def _resume(_provider: str, _job_id: str) -> bytes:
        raise AssertionError("arq job is polled by the watcher")

    async def _download(_url: str) -> bytes:
        raise AssertionError("no url yet")

    await deliver_due(
        _Bot(),  # type: ignore[arg-type]
        store,
        _download,
        resume=_resume,
        now=now + 1,
    )
    kept = (await store.active(now))[0]
    assert kept.jobs == [{"provider": "arq", "id": "jobhex"}]


def test_photo_size_and_type_errors_fall_back_to_document() -> None:
    assert telegram_photo_needs_document(RuntimeError("IMAGE_PROCESS_FAILED"))
    assert telegram_photo_needs_document(RuntimeError("PHOTO_INVALID_DIMENSIONS"))
    assert telegram_photo_needs_document(RuntimeError("file is too big"))
    assert not telegram_photo_needs_document(
        RuntimeError("bot was blocked by the user")
    )

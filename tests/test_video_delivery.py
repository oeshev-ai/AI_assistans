"""Повторная доставка ролика с CDN Polza без новой генерации."""

from __future__ import annotations

import pytest
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import SendMessage

from app.bot.runtime import _send_outbound_media
from app.bot.video_retry import (
    VIDEO_EXPIRED_TEXT,
    VIDEO_KEEP_TEXT,
    VIDEO_LATE_CAPTION,
    VIDEO_RETRY_TEXT,
    bind_video_delivery,
    deliver_due,
    video_failure_reply,
)
from app.llm.errors import AssistAIError
from app.llm.media_api import video_user_error
from app.media.media_tools import OutboundMedia
from app.storage.video_delivery import InMemoryVideoDeliveryStore


class _Bot:
    def __init__(self) -> None:
        self.videos: list[tuple[int, str | None]] = []
        self.texts: list[str] = []

    async def send_video(
        self,
        chat_id: int,
        video: object,
        caption: str | None = None,
        request_timeout: int | None = None,
    ) -> None:
        self.videos.append((chat_id, caption))

    async def send_message(self, chat_id: int, text: str) -> None:
        self.texts.append(text)

    async def send_document(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("document")


@pytest.fixture
def store() -> InMemoryVideoDeliveryStore:
    bind_video_delivery(InMemoryVideoDeliveryStore())
    yield current()
    bind_video_delivery(None)


def current() -> InMemoryVideoDeliveryStore:
    from app.bot.video_retry import current_video_delivery

    found = current_video_delivery()
    assert isinstance(found, InMemoryVideoDeliveryStore)
    return found


@pytest.mark.asyncio
async def test_cdn_failure_is_queued_and_the_user_hears_about_retries(
    store: InMemoryVideoDeliveryStore,
) -> None:
    exc = AssistAIError(
        video_user_error("download"),
        error_code="NETWORK_ERROR",
        job_committed=True,
        pending_url="https://s3.polza.ai/media/clip.mp4",
    )
    text = await video_failure_reply(5, 9, exc)
    assert text == VIDEO_RETRY_TEXT
    assert "не придёт" not in text
    saved = store._items[(5, 9)]
    assert saved.urls == ["https://s3.polza.ai/media/clip.mp4"]


@pytest.mark.asyncio
async def test_paid_job_is_queued_and_then_sent(
    store: InMemoryVideoDeliveryStore,
) -> None:
    exc = AssistAIError(
        video_user_error("poll"),
        error_code="POLL_TIMEOUT",
        job_committed=True,
        pending_provider="proxyapi",
        pending_job_id="job_paid",
        pending_kind="video",
    )
    text = await video_failure_reply(5, 9, exc)
    assert text == VIDEO_KEEP_TEXT
    assert store._items[(5, 9)].jobs == [{"provider": "proxyapi", "id": "job_paid"}]
    store._items[(5, 9)].not_before = 0
    bot = _Bot()

    async def resume(provider: str, job_id: str) -> bytes:
        assert provider == "proxyapi"
        assert job_id == "job_paid"
        return b"mp4"

    async def unused(url: str) -> bytes:
        raise AssertionError(url)

    await deliver_due(bot, store, unused, resume=resume, now=10)
    assert bot.videos == [(5, VIDEO_LATE_CAPTION)]
    assert store._items == {}


@pytest.mark.asyncio
async def test_other_host_is_not_queued(store: InMemoryVideoDeliveryStore) -> None:
    exc = AssistAIError(
        video_user_error("download"),
        error_code="NETWORK_ERROR",
        job_committed=True,
        pending_url="https://cdn.example/out.mp4",
    )
    text = await video_failure_reply(5, 9, exc)
    assert text == video_user_error("download")
    assert store._items == {}


@pytest.mark.asyncio
async def test_due_video_is_sent_and_dropped(
    store: InMemoryVideoDeliveryStore,
) -> None:
    await store.add(5, 9, "https://s3.polza.ai/a.mp4", "video.mp4", now=0)
    store._items[(5, 9)].not_before = 0
    bot = _Bot()

    async def download(url: str) -> bytes:
        assert url == "https://s3.polza.ai/a.mp4"
        return b"mp4"

    await deliver_due(bot, store, download, now=120)
    assert bot.videos == [(5, VIDEO_LATE_CAPTION)]
    assert store._items == {}


@pytest.mark.asyncio
async def test_expired_video_tells_the_user_once(
    store: InMemoryVideoDeliveryStore,
) -> None:
    await store.add(5, 9, "https://s3.polza.ai/a.mp4", "video.mp4", now=0)
    bot = _Bot()
    await deliver_due(bot, store, download_unused, now=8 * 24 * 3600)
    assert bot.texts == [VIDEO_EXPIRED_TEXT]
    assert bot.videos == []
    assert store._items == {}


async def download_unused(url: str) -> bytes:
    raise AssertionError(url)


def _network_error() -> TelegramNetworkError:
    return TelegramNetworkError(
        method=SendMessage(chat_id=1, text="x"),
        message="HTTP Client says - ServerDisconnectedError: Server disconnected",
    )


class _Chat:
    id = 5


class _User:
    id = 9


class _VideoMessage:
    chat = _Chat()
    from_user = _User()

    def __init__(self, *, fail_videos: int) -> None:
        self.fail_videos = fail_videos
        self.videos = 0
        self.documents = 0
        self.texts: list[str] = []

    async def answer_video(self, **_kwargs: object) -> object:
        self.videos += 1
        if self.videos <= self.fail_videos:
            raise _network_error()
        return object()

    async def answer_document(self, **_kwargs: object) -> object:
        self.documents += 1
        raise _network_error()

    async def answer(self, text: str, **_kwargs: object) -> None:
        self.texts.append(text)


@pytest.mark.asyncio
async def test_telegram_disconnect_keeps_the_video_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Обрыв sendVideo сохраняет ссылку CDN для повторной отправки."""
    store = InMemoryVideoDeliveryStore()
    bind_video_delivery(store)
    message = _VideoMessage(fail_videos=10)

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.bot.runtime.asyncio.sleep", _no_sleep)
    try:
        await _send_outbound_media(
            message,  # type: ignore[arg-type]
            [
                OutboundMedia(
                    kind="video",
                    filename="video.mp4",
                    mime_type="video/mp4",
                    data=b"mp4",
                    url="https://s3.polza.ai/paid.mp4",
                )
            ],
        )
    finally:
        bind_video_delivery(None)
    assert message.texts == [VIDEO_KEEP_TEXT]
    assert message.videos == 4
    assert message.documents == 1
    waiting = await store.due(10**12)
    assert waiting[0].urls == ["https://s3.polza.ai/paid.mp4"]
    assert waiting[0].filenames == ["video.mp4"]


@pytest.mark.asyncio
async def test_telegram_disconnect_video_succeeds_on_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = _VideoMessage(fail_videos=2)

    async def _no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("app.bot.runtime.asyncio.sleep", _no_sleep)
    await _send_outbound_media(
        message,  # type: ignore[arg-type]
        [
            OutboundMedia(
                kind="video",
                filename="video.mp4",
                mime_type="video/mp4",
                data=b"mp4",
                url="https://s3.polza.ai/paid.mp4",
            )
        ],
    )
    assert message.texts == []
    assert message.videos == 3
    assert message.documents == 0

"""Повтор озвучки и расшифровки из своего бакета, без новой генерации."""

from __future__ import annotations

import pytest
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import SendMessage

from app.bot.file_retry import (
    AUDIO_KEEP_TEXT,
    FILE_EXPIRED_TEXT,
    TEXT_KEEP_TEXT,
    bind_file_delivery,
    deliver_due,
    keep_paid_bytes,
)
from app.bot.runtime import _send_outbound_media
from app.media.media_tools import OutboundMedia
from app.storage.file_delivery import InMemoryFileDeliveryStore


class _Storage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []

    async def upload_bytes(
        self, *, object_key: str, data: bytes, content_type: str
    ) -> None:
        self.objects[object_key] = data

    async def download_bytes(
        self, object_key: str, *, max_bytes: int | None = None
    ) -> bytes:
        return self.objects[object_key]

    async def delete_object(self, object_key: str) -> None:
        self.deleted.append(object_key)
        self.objects.pop(object_key, None)


class _Bot:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[str] = []
        self.texts: list[str] = []

    async def send_voice(self, chat_id: int, voice: object, **_kwargs: object) -> None:
        if self.fail:
            raise TelegramNetworkError(
                method=SendMessage(chat_id=chat_id, text="x"),
                message="Server disconnected",
            )
        self.sent.append("voice")

    async def send_audio(self, chat_id: int, audio: object, **_kwargs: object) -> None:
        if self.fail:
            raise TelegramNetworkError(
                method=SendMessage(chat_id=chat_id, text="x"),
                message="Server disconnected",
            )
        self.sent.append("audio")

    async def send_document(
        self, chat_id: int, document: object, **_kwargs: object
    ) -> None:
        if self.fail:
            raise TelegramNetworkError(
                method=SendMessage(chat_id=chat_id, text="x"),
                message="Server disconnected",
            )
        self.sent.append(getattr(document, "filename", ""))

    async def send_message(self, chat_id: int, text: str) -> None:
        self.texts.append(text)


class _Chat:
    id = 5


class _User:
    id = 9


class _Message:
    chat = _Chat()
    from_user = _User()

    def __init__(self) -> None:
        self.texts: list[str] = []

    async def answer_document(self, **_kwargs: object) -> None:
        raise TelegramNetworkError(
            method=SendMessage(chat_id=1, text="x"),
            message="Server disconnected",
        )

    async def answer(self, text: str, **_kwargs: object) -> None:
        self.texts.append(text)


@pytest.mark.asyncio
async def test_kept_audio_is_sent_and_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryFileDeliveryStore()
    storage = _Storage()
    bind_file_delivery(store, storage)  # type: ignore[arg-type]
    monkeypatch.setattr("app.bot.file_retry.delivery_clock", lambda: 1000.0)
    try:
        assert await keep_paid_bytes(
            5, 9, b"ogg", "answer.ogg", kind="audio", mime="audio/ogg"
        )
        bot = _Bot()
        await deliver_due(bot, store, storage, now=1060.0)  # type: ignore[arg-type]
    finally:
        bind_file_delivery(None)
    assert bot.sent == ["voice"]
    assert storage.objects == {}
    assert storage.deleted
    assert store._items == {}


@pytest.mark.asyncio
async def test_send_failure_keeps_the_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryFileDeliveryStore()
    storage = _Storage()
    bind_file_delivery(store, storage)  # type: ignore[arg-type]
    monkeypatch.setattr("app.bot.file_retry.delivery_clock", lambda: 1000.0)
    try:
        assert await keep_paid_bytes(
            5, 9, b"hello", "transcript.txt", kind="document", mime="text/plain"
        )
        bot = _Bot(fail=True)
        await deliver_due(bot, store, storage, now=1060.0)  # type: ignore[arg-type]
    finally:
        bind_file_delivery(None)
    assert storage.objects
    assert storage.deleted == []
    record = store._items[(5, 9)]
    assert record.files[0].filename == "transcript.txt"
    assert record.attempts == 1
    assert record.not_before == 1060.0 + 120


@pytest.mark.asyncio
async def test_expired_copy_is_deleted(monkeypatch: pytest.MonkeyPatch) -> None:
    store = InMemoryFileDeliveryStore()
    storage = _Storage()
    bind_file_delivery(store, storage)  # type: ignore[arg-type]
    monkeypatch.setattr("app.bot.file_retry.delivery_clock", lambda: 1000.0)
    try:
        assert await keep_paid_bytes(
            5, 9, b"ogg", "answer.ogg", kind="audio", mime="audio/ogg"
        )
        bot = _Bot()
        await deliver_due(bot, store, storage, now=1000.0 + 8 * 24 * 3600)  # type: ignore[arg-type]
    finally:
        bind_file_delivery(None)
    assert bot.texts == [FILE_EXPIRED_TEXT]
    assert bot.sent == []
    assert storage.objects == {}
    assert store._items == {}


@pytest.mark.asyncio
async def test_outbound_transcript_is_kept_when_telegram_drops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryFileDeliveryStore()
    storage = _Storage()
    bind_file_delivery(store, storage)  # type: ignore[arg-type]
    monkeypatch.setattr("app.bot.file_retry.delivery_clock", lambda: 1000.0)
    message = _Message()
    try:
        await _send_outbound_media(
            message,  # type: ignore[arg-type]
            [
                OutboundMedia(
                    kind="document",
                    filename="transcript.txt",
                    mime_type="text/plain",
                    data="речь".encode(),
                )
            ],
        )
    finally:
        bind_file_delivery(None)
    assert message.texts == [TEXT_KEEP_TEXT]
    assert list(storage.objects.values()) == ["речь".encode()]
    waiting = await store.due(10**12)
    assert waiting[0].files[0].kind == "document"


def test_keep_texts_name_the_file() -> None:
    assert "Озвучка" in AUDIO_KEEP_TEXT
    assert "7 дней" in AUDIO_KEEP_TEXT
    assert "Расшифровка" in TEXT_KEEP_TEXT

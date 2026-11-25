"""Статус «Рисую…» не должен зависать после сетевого сбоя Telegram."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from aiogram.exceptions import TelegramNetworkError
from aiogram.methods import SendMessage

from app.bot.photo_router import PhotoDecision
from app.llm.errors import AssistAIError
from app.media.media_tools import UserMedia
from app.storage.pending_media import (
    InMemoryPendingMediaStorage,
    PendingMedia,
    PendingMediaItem,
)


def _network_error() -> TelegramNetworkError:
    return TelegramNetworkError(
        method=SendMessage(chat_id=1, text="x"),
        message="disconnected",
    )


@pytest.mark.asyncio
async def test_delete_bot_message_swallows_network_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    class _Msg:
        async def delete(self) -> None:
            raise _network_error()

    await runtime._delete_bot_message(_Msg())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_generate_settles_when_status_delete_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.photo_flow as flow
    import app.infra.job_queue as job_queue

    pending = InMemoryPendingMediaStorage()
    media = PendingMedia(
        chat_id=1,
        user_id=2,
        assistant_id="a",
        items=[PendingMediaItem(file_id="source-file", message_id=1)],
        caption="нарисуй такое же",
    )
    await pending.put(media)

    class _Status:
        message_id = 99
        edits: list[str] = []

        async def edit_text(self, text: str) -> None:
            self.edits.append(text)

        async def delete(self) -> None:
            raise _network_error()

    status = _Status()
    answers: list[str] = []

    class _Message:
        chat = SimpleNamespace(id=1)
        from_user = SimpleNamespace(id=2)

        async def answer(self, text: str, **_kwargs: Any) -> SimpleNamespace:
            if text == "Рисую изображение…":
                return status
            answers.append(text)
            raise _network_error()

    async def edited(*_args: Any, **_kwargs: Any) -> list[str]:
        raise AssistAIError("сеть", error_code="NETWORK_ERROR")

    async def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    def sync_noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def allow(*_args: object, **_kwargs: object) -> bool:
        return True

    async def failure_text(*_args: Any, **_kwargs: Any) -> str:
        return "Сетевая ошибка при генерации."

    monkeypatch.setattr(job_queue, "edit_images", edited)
    monkeypatch.setattr(flow.rate_limiter, "consume_image_bucket", allow)
    monkeypatch.setattr(flow.cost_tracker, "release_reservation", noop)
    monkeypatch.setattr(flow.R, "_reset_flow", noop)
    monkeypatch.setattr(flow.R, "_append_chrome", noop)
    monkeypatch.setattr(flow.R, "_KEEP_OUTGOING", SimpleNamespace(set=lambda *_: None))
    monkeypatch.setattr(flow.R, "_release_outgoing", sync_noop)
    monkeypatch.setattr(flow.M, "_image_model_of", lambda *_a, **_k: "gen")
    monkeypatch.setattr(flow.M, "_stored_image_size", lambda *_a, **_k: "1K")
    monkeypatch.setattr(flow.M, "_image_as_file", lambda *_a, **_k: False)

    import app.bot.image_retry as image_retry

    monkeypatch.setattr(image_retry, "image_failure_reply", failure_text)

    turn = SimpleNamespace(
        message=_Message(),
        assistant=SimpleNamespace(
            config=SimpleNamespace(default_image_edit_model="edit-model")
        ),
        topic_storage=None,
        assistant_storage=None,
        state=SimpleNamespace(),
        pending=pending,
    )
    await flow._generate(
        turn,
        media,
        profile=SimpleNamespace(),
        loaded=[
            UserMedia(
                kind="image",
                filename="photo.jpg",
                mime_type="image/jpeg",
                data=b"jpeg",
            )
        ],
        decision=PhotoDecision(
            intent="generate",
            base_index=0,
            instruction_en="draw the same",
            instruction_ru="нарисуй такое же",
            confidence=1.0,
        ),
        topic_id="t",
    )
    stored = await pending.get(1, "a", 2)
    assert stored is not None
    assert stored.items[0].file_id == "source-file"
    assert status.edits == ["Сетевая ошибка при генерации."]

"""Голос уважает wizard; /cancel чистит pending у всех персон."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.bot.handlers.chat import handle_voice
from app.bot.handlers.commands import cmd_cancel
from app.storage.pending_media import (
    InMemoryPendingMediaStorage,
    PendingMedia,
    PendingMediaItem,
)


class _State:
    def __init__(self, current: str | None) -> None:
        self._current = current
        self.data: dict[str, Any] = {}

    async def get_state(self) -> str | None:
        return self._current

    async def set_state(self, value: str | None) -> None:
        self._current = value

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def set_data(self, data: dict[str, Any]) -> None:
        self.data = dict(data)

    async def update_data(self, **kwargs: Any) -> None:
        self.data.update(kwargs)


@pytest.mark.asyncio
async def test_voice_rejected_while_wizard_open() -> None:
    answers: list[str] = []

    class _Message:
        voice = SimpleNamespace(duration=3, file_size=100)
        chat = SimpleNamespace(id=1)
        from_user = SimpleNamespace(id=2)

        async def answer(self, text: str, **_kwargs: Any) -> Any:
            answers.append(text)
            return SimpleNamespace(message_id=9)

    await handle_voice(
        _Message(),  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        topic_storage=SimpleNamespace(),
        assistant_storage=SimpleNamespace(),
        state=_State("NewAssistantSG:name"),  # type: ignore[arg-type]
    )
    assert answers
    assert "Отмена" in answers[0]


@pytest.mark.asyncio
async def test_long_voice_transcript_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.handlers.chat as chat
    import app.bot.runtime as runtime
    from app.infra.security import MAX_MESSAGE_LENGTH
    from app.llm.media_api import TranscriptionResult

    edits: list[str] = []
    processed: list[str] = []

    async def edit_stream(_status: Any, text: str, _previous: str) -> str:
        edits.append(text)
        return text

    async def process(*_args: Any, **kwargs: Any) -> None:
        processed.append(kwargs["user_text"])

    monkeypatch.setattr(
        chat.rate_limiter, "check_text_limit", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(chat.cost_tracker, "check_budget", AsyncMock())
    monkeypatch.setattr(chat.cost_tracker, "record", AsyncMock())
    monkeypatch.setattr(chat.cost_tracker, "release_reservation", AsyncMock())
    monkeypatch.setattr(
        chat.job_queue,
        "transcribe_audio",
        AsyncMock(
            return_value=TranscriptionResult(text="я" * (MAX_MESSAGE_LENGTH + 1))
        ),
    )
    monkeypatch.setattr(
        runtime, "_download_voice", AsyncMock(return_value=(b"ogg", "voice.ogg"))
    )
    monkeypatch.setattr(runtime, "_active_profile", AsyncMock())
    monkeypatch.setattr(runtime, "_edit_stream", edit_stream)
    monkeypatch.setattr(runtime, "_append_chrome", AsyncMock())
    monkeypatch.setattr(runtime, "_process_user_message", process)

    class _Message:
        voice = SimpleNamespace(duration=60, file_size=1000)
        chat = SimpleNamespace(id=1)
        from_user = SimpleNamespace(id=2)

        async def answer(self, _text: str, **_kwargs: Any) -> Any:
            return SimpleNamespace(message_id=3)

    await handle_voice(
        _Message(),  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        topic_storage=SimpleNamespace(),
        assistant_storage=SimpleNamespace(),
        state=_State(None),  # type: ignore[arg-type]
    )
    assert processed == []
    assert any("4000" in text for text in edits)
    chat.cost_tracker.record.assert_awaited()


@pytest.mark.asyncio
async def test_cancel_clears_pending_for_all_personas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.handlers.commands as commands
    import app.bot.runtime as runtime

    pending = InMemoryPendingMediaStorage(ttl_seconds=60)
    for asst in ("a", "b"):
        await pending.put(
            PendingMedia(
                chat_id=5,
                user_id=9,
                assistant_id=asst,
                items=[PendingMediaItem(file_id=f"f-{asst}", message_id=1)],
            )
        )

    class _Hub:
        async def cancel_chat(self, *_args: Any, **_kwargs: Any) -> bool:
            return False

    monkeypatch.setattr(commands, "get_album_hub", lambda: _Hub())
    monkeypatch.setattr(runtime, "_menu_for", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(runtime, "_apply_reply_menu", AsyncMock())
    monkeypatch.setattr(runtime, "_reset_flow", AsyncMock())

    answers: list[str] = []

    class _Message:
        chat = SimpleNamespace(id=5)
        from_user = SimpleNamespace(id=9)

        async def answer(self, text: str, **_kwargs: Any) -> None:
            answers.append(text)

    await cmd_cancel(
        _Message(),  # type: ignore[arg-type]
        state=_State(None),  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=SimpleNamespace(),
        topic_storage=SimpleNamespace(),
        pending_media=pending,
    )
    assert await pending.get(5, "a", 9) is None
    assert await pending.get(5, "b", 9) is None
    assert answers == ["Отменено."]


@pytest.mark.asyncio
async def test_cancel_words_without_slot_do_not_reach_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime
    from app.bot.handlers.chat import handle_text

    processed: list[str] = []

    async def process(*_args: Any, **kwargs: Any) -> None:
        processed.append(kwargs["user_text"])

    monkeypatch.setattr(runtime, "_process_user_message", process)

    class _Message:
        chat = SimpleNamespace(id=909)
        from_user = SimpleNamespace(id=11)

        def __init__(self, text: str) -> None:
            self.text = text
            self.answers: list[str] = []

        async def answer(self, text: str, **_kwargs: Any) -> None:
            self.answers.append(text)

    for word in ("отмена", "Стоп", "Cancel"):
        message = _Message(word)
        await handle_text(
            message,  # type: ignore[arg-type]
            assistant=SimpleNamespace(),
            topic_storage=SimpleNamespace(),
            assistant_storage=SimpleNamespace(),
            state=_State(None),  # type: ignore[arg-type]
            pending_media=None,
        )
        assert message.answers == ["Нечего отменять."]
    assert processed == []

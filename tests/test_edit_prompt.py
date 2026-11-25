"""Правка промпта обрезает края и не принимает текст длиннее сообщения."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.bot.handlers.assistants import sg_edit_prompt
from app.bot.states import EditAssistantSG
from app.infra.security import MAX_MESSAGE_LENGTH
from app.storage.assistant_storage import (
    InMemoryAssistantStorage,
    default_assistant_profile,
)


class _State:
    def __init__(self, profile_id: str) -> None:
        self.current: str | None = EditAssistantSG.prompt.state
        self.data: dict[str, Any] = {"id": profile_id, "name": "Тест"}

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)


class _Message:
    def __init__(self, text: str, chat_id: int = 4) -> None:
        self.text = text
        self.chat = SimpleNamespace(id=chat_id)
        self.answers: list[str] = []

    async def answer(self, text: str, **_kwargs: Any) -> None:
        self.answers.append(text)


@pytest.mark.asyncio
async def test_edit_prompt_strips_and_rejects_over_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    storage = InMemoryAssistantStorage()
    chat_id = 4
    profile = default_assistant_profile().model_copy(
        update={
            "id": "edit1",
            "name": "Тест",
            "system_prompt": "Старый промпт персоны.",
        }
    )
    await storage.save_assistant(chat_id, profile)
    monkeypatch.setattr(runtime, "_reset_flow", AsyncMock())
    monkeypatch.setattr(runtime, "_open_persona_settings", AsyncMock())

    padded = _Message("\n  Ты отвечаешь коротко и по делу.  \n", chat_id)
    await sg_edit_prompt(
        padded,  # type: ignore[arg-type]
        state=_State(profile.id),  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=storage,
    )
    saved = next(
        item for item in await storage.list_assistants(chat_id) if item.id == profile.id
    )
    assert saved.system_prompt == "Ты отвечаешь коротко и по делу."

    too_long = _Message("я" * (MAX_MESSAGE_LENGTH + 1), chat_id)
    state = _State(profile.id)
    await sg_edit_prompt(
        too_long,  # type: ignore[arg-type]
        state=state,  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=storage,
    )
    kept = next(
        item for item in await storage.list_assistants(chat_id) if item.id == profile.id
    )
    assert kept.system_prompt == "Ты отвечаешь коротко и по делу."
    assert too_long.answers
    assert str(MAX_MESSAGE_LENGTH) in too_long.answers[0]
    assert state.current == EditAssistantSG.prompt.state

    blank = _Message(" \n\n          ", chat_id)
    await sg_edit_prompt(
        blank,  # type: ignore[arg-type]
        state=_State(profile.id),  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=storage,
    )
    still = next(
        item for item in await storage.list_assistants(chat_id) if item.id == profile.id
    )
    assert still.system_prompt == "Ты отвечаешь коротко и по делу."
    assert blank.answers == ["Промпт минимум 10 символов. /cancel — отмена."]

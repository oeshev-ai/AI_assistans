"""Повтор «Создать» в каталоге не заводит вторую персону «Имя (2)»."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.bot.handlers.presets import on_preset_apply
from app.bot.states import NewAssistantSG
from app.storage.assistant_presets import get_preset
from app.storage.assistant_storage import (
    DEFAULT_ASSISTANT_ID,
    AssistantStorageError,
    InMemoryAssistantStorage,
)


class _State:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.current: str | None = None

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def set_data(self, data: dict[str, Any]) -> None:
        self.data = dict(data)

    async def update_data(self, **kwargs: Any) -> dict[str, Any]:
        self.data.update(kwargs)
        return dict(self.data)

    async def get_state(self) -> str | None:
        return self.current

    async def set_state(self, value: str | None) -> None:
        self.current = value

    async def clear(self) -> None:
        self.data = {}
        self.current = None


def _assistant() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(
            default_text_model="openai/gpt-6-luna",
            default_image_model="openai/gpt-image-2.6-flare",
            default_tts_model="openai/gpt-4o-mini-tts",
        )
    )


@pytest.mark.asyncio
async def test_apply_refuses_while_wizard_is_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    preset = get_preset("helper")
    assert preset is not None
    storage = InMemoryAssistantStorage()
    chat_id = 9
    state = _State()
    state.current = NewAssistantSG.prompt.state
    state.data = {
        "flow_name": "Черновик",
        "flow_prompt": "Ты внимательный собеседник и помогаешь по делу.",
    }
    edit = AsyncMock()
    monkeypatch.setattr(runtime, "_query_message", lambda query: query.message)
    monkeypatch.setattr(runtime, "_edit_inline", edit)
    monkeypatch.setattr(runtime, "_reset_flow", AsyncMock())

    message = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=23,
        answer=AsyncMock(),
    )
    query = SimpleNamespace(
        data=f"preset:apply:{preset.id}",
        message=message,
        answer=AsyncMock(),
    )
    await on_preset_apply(
        query,  # type: ignore[arg-type]
        assistant=_assistant(),  # type: ignore[arg-type]
        assistant_storage=storage,
        topic_storage=SimpleNamespace(),
        state=state,  # type: ignore[arg-type]
    )
    assert _custom(await storage.list_assistants(chat_id)) == []
    assert state.current == NewAssistantSG.prompt.state
    assert state.data["flow_name"] == "Черновик"
    query.answer.assert_awaited_with(
        "Сначала завершите текущий шаг или нажмите /cancel.",
        show_alert=True,
    )
    edit.assert_not_awaited()
    runtime._reset_flow.assert_not_awaited()


def _custom(profiles: list[Any]) -> list[Any]:
    return [item for item in profiles if item.id != DEFAULT_ASSISTANT_ID]


@pytest.mark.asyncio
async def test_double_apply_keeps_one_persona(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    preset = get_preset("helper")
    assert preset is not None
    storage = InMemoryAssistantStorage()
    chat_id = 6
    state = _State()
    edits: list[Any] = []

    async def edit(_message: Any, _text: str, markup: Any) -> None:
        edits.append(markup)

    monkeypatch.setattr(runtime, "_query_message", lambda query: query.message)
    monkeypatch.setattr(runtime, "_edit_inline", edit)
    monkeypatch.setattr(runtime, "_menu_for", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(runtime, "_close_screen", AsyncMock())
    monkeypatch.setattr(runtime, "_apply_reply_menu", AsyncMock())

    message = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=21,
        answer=AsyncMock(),
    )
    query = SimpleNamespace(
        data=f"preset:apply:{preset.id}",
        message=message,
        answer=AsyncMock(),
    )
    await on_preset_apply(
        query,  # type: ignore[arg-type]
        assistant=_assistant(),  # type: ignore[arg-type]
        assistant_storage=storage,
        topic_storage=SimpleNamespace(),
        state=state,  # type: ignore[arg-type]
    )
    await on_preset_apply(
        query,  # type: ignore[arg-type]
        assistant=_assistant(),  # type: ignore[arg-type]
        assistant_storage=storage,
        topic_storage=SimpleNamespace(),
        state=state,  # type: ignore[arg-type]
    )

    saved = _custom(await storage.list_assistants(chat_id))
    assert [item.name for item in saved] == [preset.name]
    assert state.data["preset_apply"]["assistant_id"] == saved[0].id
    buttons = [
        button.callback_data for row in edits[0].inline_keyboard for button in row
    ]
    assert all(not item.startswith("preset:apply:") for item in buttons)
    assert message.answer.await_count == 1


@pytest.mark.asyncio
async def test_apply_retries_same_id_when_activation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    preset = get_preset("helper")
    assert preset is not None
    storage = InMemoryAssistantStorage()
    chat_id = 8
    state = _State()
    real_set = storage.set_active_assistant
    attempts = {"n": 0}

    async def flaky_set(saved_chat: int, profile_id: str) -> None:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise AssistantStorageError("Профили ассистентов временно недоступны.")
        await real_set(saved_chat, profile_id)

    monkeypatch.setattr(storage, "set_active_assistant", flaky_set)
    monkeypatch.setattr(runtime, "_query_message", lambda query: query.message)
    monkeypatch.setattr(runtime, "_edit_inline", AsyncMock())
    monkeypatch.setattr(runtime, "_menu_for", AsyncMock(return_value=SimpleNamespace()))
    monkeypatch.setattr(runtime, "_close_screen", AsyncMock())
    monkeypatch.setattr(runtime, "_apply_reply_menu", AsyncMock())

    message = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=22,
        answer=AsyncMock(),
    )
    query = SimpleNamespace(
        data=f"preset:apply:{preset.id}",
        message=message,
        answer=AsyncMock(),
    )
    await on_preset_apply(
        query,  # type: ignore[arg-type]
        assistant=_assistant(),  # type: ignore[arg-type]
        assistant_storage=storage,
        topic_storage=SimpleNamespace(),
        state=state,  # type: ignore[arg-type]
    )
    saved = _custom(await storage.list_assistants(chat_id))
    assert [item.id for item in saved] == [state.data["flow_id"]]
    assert saved[0].name == preset.name
    query.answer.assert_awaited_with(
        "Профили ассистентов временно недоступны.", show_alert=True
    )

    await on_preset_apply(
        query,  # type: ignore[arg-type]
        assistant=_assistant(),  # type: ignore[arg-type]
        assistant_storage=storage,
        topic_storage=SimpleNamespace(),
        state=state,  # type: ignore[arg-type]
    )
    saved = _custom(await storage.list_assistants(chat_id))
    assert [item.id for item in saved] == [state.data["preset_apply"]["assistant_id"]]
    assert saved[0].name == preset.name
    active = await storage.get_active_assistant(chat_id)
    assert active is not None
    assert active.id == saved[0].id

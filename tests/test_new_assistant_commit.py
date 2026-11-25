"""Повтор «Готово» не создаёт вторую персону, если активация упала после save."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.bot.handlers.assistants import on_asst_new_commit
from app.bot.states import NewAssistantSG
from app.storage.assistant_storage import (
    DEFAULT_ASSISTANT_ID,
    AssistantStorageError,
    InMemoryAssistantStorage,
)


class _State:
    def __init__(self) -> None:
        self.current: str | None = NewAssistantSG.models.state
        self.data: dict[str, Any] = {
            "flow_name": "Алиса",
            "flow_prompt": "Ты внимательный собеседник.",
            "flow_icon": "🤖",
        }

    async def get_state(self) -> str | None:
        return self.current

    async def set_state(self, value: str | None) -> None:
        self.current = value

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def update_data(self, **kwargs: Any) -> dict[str, Any]:
        self.data.update(kwargs)
        return dict(self.data)


def _custom_ids(profiles: list[Any]) -> list[str]:
    return [item.id for item in profiles if item.id != DEFAULT_ASSISTANT_ID]


@pytest.mark.asyncio
async def test_commit_retries_same_id_when_activation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    storage = InMemoryAssistantStorage()
    chat_id = 7
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
    monkeypatch.setattr(runtime, "_reset_flow", AsyncMock())
    monkeypatch.setattr(runtime, "_cleanup_wizard_ui", AsyncMock())
    monkeypatch.setattr(runtime, "_open_settings", AsyncMock())

    message = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=10,
        answer=AsyncMock(),
    )
    query = SimpleNamespace(message=message, answer=AsyncMock())

    await on_asst_new_commit(
        query,  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=storage,
        topic_storage=SimpleNamespace(),
        state=state,  # type: ignore[arg-type]
    )
    saved = _custom_ids(await storage.list_assistants(chat_id))
    assert saved == [state.data["flow_id"]]
    assert state.current == NewAssistantSG.models.state
    active = await storage.get_active_assistant(chat_id)
    assert active is not None
    assert active.id == DEFAULT_ASSISTANT_ID
    query.answer.assert_awaited_with(
        "Профили ассистентов временно недоступны.", show_alert=True
    )
    runtime._reset_flow.assert_not_awaited()

    await on_asst_new_commit(
        query,  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=storage,
        topic_storage=SimpleNamespace(),
        state=state,  # type: ignore[arg-type]
    )
    assert _custom_ids(await storage.list_assistants(chat_id)) == saved
    active = await storage.get_active_assistant(chat_id)
    assert active is not None
    assert active.id == saved[0]
    runtime._reset_flow.assert_awaited()

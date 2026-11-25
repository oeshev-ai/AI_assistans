"""Иконка пишется персоне, для которой открыли пикер, а не текущей активной."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.bot.handlers.assistants import on_asst_icon_menu, on_set_persona_icon
from app.storage.assistant_storage import (
    InMemoryAssistantStorage,
    default_assistant_profile,
)
from app.storage.persona_icons import PERSONA_ICONS


class _State:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {}

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def update_data(
        self, data: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        if data:
            kwargs.update(data)
        self.data.update(kwargs)
        return dict(self.data)


def _profile(profile_id: str, name: str) -> Any:
    return default_assistant_profile().model_copy(
        update={"id": profile_id, "name": name, "icon": "🤖"}
    )


@pytest.mark.asyncio
async def test_icon_click_uses_persona_from_picker_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    storage = InMemoryAssistantStorage()
    chat_id = 8
    alpha = _profile("alpha", "Альфа")
    beta = _profile("beta", "Бета")
    await storage.save_assistant(chat_id, alpha)
    await storage.save_assistant(chat_id, beta)
    await storage.set_active_assistant(chat_id, alpha.id)
    state = _State()
    message = SimpleNamespace(
        chat=SimpleNamespace(id=chat_id),
        message_id=10,
        answer=AsyncMock(return_value=SimpleNamespace(message_id=11)),
    )
    query = SimpleNamespace(data="asst_icon_menu", message=message, answer=AsyncMock())
    monkeypatch.setattr(runtime, "_query_message", lambda item: item.message)
    monkeypatch.setattr(runtime, "_return_to_settings", AsyncMock())

    await on_asst_icon_menu(
        query,  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=storage,
        state=state,  # type: ignore[arg-type]
    )
    assert state.data["id"] == alpha.id

    await storage.set_active_assistant(chat_id, beta.id)
    query.data = "set_icon:1"
    await on_set_persona_icon(
        query,  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=storage,
        state=state,  # type: ignore[arg-type]
    )

    saved = {item.id: item for item in await storage.list_assistants(chat_id)}
    assert saved[alpha.id].icon == PERSONA_ICONS[1]
    assert saved[beta.id].icon == "🤖"
    active = await storage.get_active_assistant(chat_id)
    assert active is not None
    assert active.id == beta.id
    runtime._return_to_settings.assert_awaited()
    returned = runtime._return_to_settings.await_args.args[2]
    assert returned.id == alpha.id
    assert returned.icon == PERSONA_ICONS[1]


@pytest.mark.asyncio
async def test_icon_click_without_stored_id_does_not_touch_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    storage = InMemoryAssistantStorage()
    chat_id = 9
    alpha = _profile("alpha", "Альфа")
    await storage.save_assistant(chat_id, alpha)
    await storage.set_active_assistant(chat_id, alpha.id)
    message = SimpleNamespace(chat=SimpleNamespace(id=chat_id), message_id=3)
    query = SimpleNamespace(data="set_icon:1", message=message, answer=AsyncMock())
    monkeypatch.setattr(runtime, "_query_message", lambda item: item.message)
    monkeypatch.setattr(runtime, "_return_to_settings", AsyncMock())

    await on_set_persona_icon(
        query,  # type: ignore[arg-type]
        assistant=SimpleNamespace(),
        assistant_storage=storage,
        state=_State(),  # type: ignore[arg-type]
    )
    saved = next(
        item for item in await storage.list_assistants(chat_id) if item.id == alpha.id
    )
    assert saved.icon == "🤖"
    query.answer.assert_awaited_with("Ассистент не найден", show_alert=True)
    runtime._return_to_settings.assert_not_awaited()

"""Уборка служебных реплик не переставляет меню, если якорь остаётся."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.bot.constants import _CHROME, _MENU_ANCHOR, _SCREEN_SETTINGS
from app.bot.runtime import _sweep_chrome
from app.storage.assistant_storage import InMemoryAssistantStorage
from app.storage.topic_history import InMemoryTopicStorage


class _State:
    def __init__(self, data: dict[str, Any]) -> None:
        self.data = dict(data)

    async def get_data(self) -> dict[str, Any]:
        return dict(self.data)

    async def update_data(
        self, data: dict[str, Any] | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        if data:
            kwargs.update(data)
        self.data.update(kwargs)
        return dict(self.data)

    async def set_data(self, data: dict[str, Any]) -> None:
        self.data = dict(data)


class _Bot:
    def __init__(self) -> None:
        self.deleted: list[int] = []

    async def delete_message(self, *, chat_id: int, message_id: int) -> None:
        self.deleted.append(message_id)


def _message(bot: _Bot) -> SimpleNamespace:
    return SimpleNamespace(chat=SimpleNamespace(id=3), bot=bot)


@pytest.mark.asyncio
async def test_sweep_does_not_move_menu_when_anchor_stays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    moved: list[str] = []

    async def menu(*_args: object, **_kwargs: object) -> str:
        moved.append("menu")
        return "menu"

    async def apply(*_args: object, **_kwargs: object) -> None:
        moved.append("apply")

    monkeypatch.setattr(runtime, "_menu_for", menu)
    monkeypatch.setattr(runtime, "_apply_reply_menu", apply)
    bot = _Bot()
    state = _State({_CHROME: [11, 12], _MENU_ANCHOR: 40, _SCREEN_SETTINGS: 11})
    await _sweep_chrome(
        _message(bot),  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        {
            "assistant_storage": InMemoryAssistantStorage(),
            "topic_storage": InMemoryTopicStorage(),
        },
    )
    assert moved == []
    assert bot.deleted == [11, 12]
    assert state.data[_MENU_ANCHOR] == 40
    assert state.data[_CHROME] == []
    assert _SCREEN_SETTINGS not in state.data


@pytest.mark.asyncio
async def test_sweep_moves_menu_when_anchor_is_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.bot.runtime as runtime

    calls: list[int] = []

    async def menu(*_args: object, **_kwargs: object) -> str:
        return "menu"

    async def apply(_message: object, _menu: object, fsm: _State) -> None:
        calls.append(1)
        await fsm.update_data({_MENU_ANCHOR: 90})

    monkeypatch.setattr(runtime, "_menu_for", menu)
    monkeypatch.setattr(runtime, "_apply_reply_menu", apply)
    bot = _Bot()
    state = _State({_CHROME: [11, 40], _MENU_ANCHOR: 40})
    await _sweep_chrome(
        _message(bot),  # type: ignore[arg-type]
        state,  # type: ignore[arg-type]
        {
            "assistant_storage": InMemoryAssistantStorage(),
            "topic_storage": InMemoryTopicStorage(),
        },
    )
    assert calls == [1]
    assert 90 not in bot.deleted
    assert 11 in bot.deleted
    assert state.data[_MENU_ANCHOR] == 90
    assert state.data[_CHROME] == []

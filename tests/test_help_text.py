"""Короткое приветствие и отдельная справка /help."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.bot.constants import HELP_TEXT, WELCOME
from app.bot.handlers.commands import cmd_help, cmd_start
from app.bot.handlers.register import _register_handlers


def test_welcome_and_help_are_different_texts() -> None:
    assert WELCOME != HELP_TEXT
    assert len(WELCOME) < 700
    assert len(HELP_TEXT) < 4096
    assert "/cancel" in HELP_TEXT
    assert "/help" in WELCOME
    assert "5/час" not in HELP_TEXT
    assert "до 15 за раз" in HELP_TEXT
    assert "одновременно до 3" in HELP_TEXT


def test_help_command_is_not_wired_to_start() -> None:
    import inspect

    from aiogram import Dispatcher

    source = inspect.getsource(_register_handlers)
    assert 'register(cmd_help, Command("help"))' in source
    assert 'register(cmd_start, Command("help"))' not in source
    assert "CommandStart()" in source
    dispatcher = Dispatcher()
    _register_handlers(dispatcher)


@pytest.mark.asyncio
async def test_cmd_help_answers_help_without_new_topic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: dict[str, Any] = {}

    async def menu(*args: object, **kwargs: object) -> str:
        sent["menu_args"] = args
        sent["ensure_topic"] = kwargs.get("ensure_topic", False)
        return "menu"

    class _Message:
        chat = SimpleNamespace(id=4)

        async def answer(self, text: str, reply_markup: object = None) -> None:
            sent["text"] = text
            sent["markup"] = reply_markup

    monkeypatch.setattr("app.bot.handlers.commands.R._menu_for", menu)
    await cmd_help(
        _Message(),  # type: ignore[arg-type]
        assistant_storage=None,  # type: ignore[arg-type]
        topic_storage=None,  # type: ignore[arg-type]
    )
    assert sent["text"] == HELP_TEXT
    assert sent["markup"] == "menu"
    assert sent["ensure_topic"] is False
    assert sent["menu_args"][0] == 4


@pytest.mark.asyncio
async def test_cmd_start_answers_welcome(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: dict[str, Any] = {}

    async def menu(*_args: object, **kwargs: object) -> str:
        sent["ensure_topic"] = kwargs.get("ensure_topic", False)
        return "menu"

    async def reset(_state: object) -> None:
        sent["reset"] = True

    class _Message:
        chat = SimpleNamespace(id=4)

        async def answer(self, text: str, reply_markup: object = None) -> None:
            sent["text"] = text

    monkeypatch.setattr("app.bot.handlers.commands.R._menu_for", menu)
    monkeypatch.setattr("app.bot.handlers.commands.R._reset_flow", reset)
    await cmd_start(
        _Message(),  # type: ignore[arg-type]
        state=None,  # type: ignore[arg-type]
        assistant_storage=None,  # type: ignore[arg-type]
        topic_storage=None,  # type: ignore[arg-type]
    )
    assert sent["text"] == WELCOME
    assert sent["text"] != HELP_TEXT
    assert sent["ensure_topic"] is True
    assert sent["reset"] is True

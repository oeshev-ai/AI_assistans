"""Ход подтверждения: без фото и без tools, сбой прячется за шаблоном."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.bot.photo_flow import confirm_reply
from app.llm.assist_ai import AssistAIError


class _Persona:
    def __init__(self, fail: bool) -> None:
        self.fail = fail
        self.messages: list[dict[str, Any]] = []
        self.kwargs: dict[str, Any] = {}

    async def chat_completion(
        self, messages: list[dict[str, Any]], model: str | None = None, **kwargs: Any
    ) -> str:
        self.messages = messages
        self.kwargs = kwargs
        if self.fail:
            raise AssistAIError("router down")
        return "Готово, поправил."


@pytest.mark.asyncio
async def test_confirm_has_no_images_or_tools() -> None:
    persona = _Persona(fail=False)
    profile = SimpleNamespace(system_prompt="Ты краткий.")
    text = await confirm_reply(
        persona,
        profile,
        "Правка выполнена.",
        text_model="openai/gpt-6-sol",
        fallback="Готово. Основа — фото 1.",
    )
    assert text == "Готово, поправил."
    assert persona.kwargs["use_tools"] is False
    assert persona.kwargs["max_tokens"] == 300
    for message in persona.messages:
        assert isinstance(message["content"], str)


@pytest.mark.asyncio
async def test_confirm_failure_returns_template() -> None:
    persona = _Persona(fail=True)
    profile = SimpleNamespace(system_prompt="Ты краткий.")
    text = await confirm_reply(
        persona,
        profile,
        "Правка выполнена.",
        text_model="openai/gpt-6-sol",
        fallback="Готово. Основа — фото 1.",
    )
    assert text == "Готово. Основа — фото 1."

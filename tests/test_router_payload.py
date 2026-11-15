"""Запрос роутера: без tools, с температурой 0 и номерами фото."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.bot.photo_router import decide


class _Router:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.config = SimpleNamespace(
            photo_router_model="google/gemini-3.5-flash-lite",
            router_confidence=0.55,
            router_max_tokens=600,
        )

    async def chat_completion(
        self, messages: list[dict[str, Any]], model: str | None = None, **kwargs: Any
    ) -> str:
        self.calls.append({"messages": messages, "model": model, "kwargs": kwargs})
        return '{"intent":"analyze","confidence":0.9,"reference_indexes":[]}'


@pytest.mark.asyncio
async def test_router_request_has_schema_and_numbered_photos() -> None:
    router = _Router()
    await decide(
        router,  # type: ignore[arg-type]
        [
            {"data": b"one", "mime": "image/png"},
            {"data": b"two", "mime": "image/jpeg"},
        ],
        "что это",
    )
    call = router.calls[0]
    kwargs = call["kwargs"]
    assert kwargs["use_tools"] is False
    assert "tool_choice" not in kwargs
    assert kwargs["temperature"] == 0
    assert kwargs["response_format"]["type"] == "json_schema"
    user = call["messages"][-1]["content"]
    texts = [part["text"] for part in user if part.get("type") == "text"]
    assert "Фото 1" in texts
    assert "Фото 2" in texts
    assert "Доступно 2 фото" in texts[0]

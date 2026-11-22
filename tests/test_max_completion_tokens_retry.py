"""Retry chat when provider rejects max_tokens in favor of max_completion_tokens."""

from __future__ import annotations

from typing import Any

import pytest

from app.llm.chat_runtime import (
    _needs_max_completion_tokens_retry,
    _swap_max_tokens_to_completion,
)
from app.llm.errors import AssistAIError


def test_needs_retry_matches_proxyapi_wording() -> None:
    exc = AssistAIError(
        "Unsupported parameter: 'max_tokens' is not supported with this model. "
        "Use 'max_completion_tokens' instead.",
        status_code=400,
    )
    assert _needs_max_completion_tokens_retry(exc) is True


def test_needs_retry_ignores_other_400() -> None:
    exc = AssistAIError("temperature is not supported", status_code=400)
    assert _needs_max_completion_tokens_retry(exc) is False


def test_swap_max_tokens_to_completion() -> None:
    payload: dict[str, Any] = {"model": "x", "max_tokens": 300}
    assert _swap_max_tokens_to_completion(payload) == 300
    assert "max_tokens" not in payload
    assert payload["max_completion_tokens"] == 300


@pytest.mark.asyncio
async def test_chat_completion_retries_with_max_completion_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.llm.assist_ai import AssistAIClient
    from app.llm.config import AssistAIConfig

    cfg = AssistAIConfig(
        proxyapi_key="test-key",
        base_url="https://example.test/v1",
        default_text_model="openai/gpt-6-sol",
    )
    client = AssistAIClient(cfg)
    calls: list[dict[str, Any]] = []

    async def fake_post(
        _path: str, payload: dict[str, Any], **_kwargs: Any
    ) -> dict[str, Any]:
        calls.append(dict(payload))
        if "max_tokens" in payload:
            raise AssistAIError(
                "Unsupported parameter: 'max_tokens' is not supported with this model. "
                "Use 'max_completion_tokens' instead.",
                status_code=400,
            )
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    async def fake_usage(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr(client, "_post_json", fake_post)
    monkeypatch.setattr(client, "_record_chat_usage", fake_usage)

    text = await client.chat_completion(
        [{"role": "user", "content": "hi"}],
        model="openai/gpt-6-sol",
        use_tools=False,
        max_tokens=300,
    )
    assert text == "ok"
    assert len(calls) == 2
    assert "max_tokens" in calls[0]
    assert "max_completion_tokens" in calls[1]
    assert calls[1]["max_completion_tokens"] == 300
    await client.close()

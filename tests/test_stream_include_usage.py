"""stream_chat запрашивает usage и повторяет запрос, если флаг отвергнут."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import structlog
from respx import MockRouter

from app.llm.assist_ai import AssistAIClient
from app.llm.chat_runtime import (
    ChatDelta,
    _adapt_rejected_stream_payload,
    _needs_stream_options_retry,
)
from app.llm.errors import AssistAIError

CHAT_URL = "https://api.proxyapi.ru/v1/chat/completions"
_STREAM_OPTIONS_400 = "Unrecognized request argument supplied: stream_options"
_MAX_TOKENS_400 = (
    "Unsupported parameter: 'max_tokens' is not supported with this model. "
    "Use 'max_completion_tokens' instead."
)


def _sse(events: list[dict[str, Any]]) -> str:
    lines = ["data: " + json.dumps(event, ensure_ascii=False) for event in events]
    lines.append("data: [DONE]")
    return "\n".join(lines) + "\n"


def _ok_stream() -> str:
    return _sse(
        [
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ]
            },
            {
                "choices": [],
                "model": "openai/gpt-4o",
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        ]
    )


def _http_400(message: str) -> httpx.Response:
    return httpx.Response(400, json={"error": {"message": message}})


def test_needs_retry_matches_unrecognized_stream_options() -> None:
    exc = AssistAIError(
        f"ProxyAPI вернул HTTP 400: {_STREAM_OPTIONS_400}",
        status_code=400,
    )
    assert _needs_stream_options_retry(exc) is True


def test_needs_retry_matches_include_usage_wording() -> None:
    exc = AssistAIError(
        "Unknown parameter: 'stream_options.include_usage'.",
        status_code=400,
    )
    assert _needs_stream_options_retry(exc) is True


def test_needs_retry_ignores_other_400_and_server_errors() -> None:
    other = AssistAIError("temperature is not supported", status_code=400)
    assert _needs_stream_options_retry(other) is False
    server = AssistAIError(
        "stream_options failed upstream",
        status_code=500,
    )
    assert _needs_stream_options_retry(server) is False


def test_adapt_drops_stream_options_and_can_remap_max_tokens() -> None:
    payload: dict[str, Any] = {
        "stream_options": {"include_usage": True},
        "max_tokens": 300,
    }
    dropped = _adapt_rejected_stream_payload(
        AssistAIError(_STREAM_OPTIONS_400, status_code=400),
        payload,
        allow_drop_stream_options=True,
        allow_remap_max_tokens=True,
    )
    assert dropped == (True, False)
    assert "stream_options" not in payload
    assert payload["max_tokens"] == 300

    both = dict(payload)
    both["stream_options"] = {"include_usage": True}
    adapted = _adapt_rejected_stream_payload(
        AssistAIError(
            f"{_MAX_TOKENS_400} Unrecognized stream_options.",
            status_code=400,
        ),
        both,
        allow_drop_stream_options=True,
        allow_remap_max_tokens=True,
    )
    assert adapted == (True, True)
    assert "stream_options" not in both
    assert both["max_completion_tokens"] == 300

    untouched = {"stream_options": {"include_usage": True}, "max_tokens": 20}
    assert (
        _adapt_rejected_stream_payload(
            AssistAIError("temperature is not supported", status_code=400),
            untouched,
            allow_drop_stream_options=True,
            allow_remap_max_tokens=True,
        )
        is None
    )
    assert untouched["stream_options"] == {"include_usage": True}


@pytest.mark.asyncio
async def test_stream_chat_retries_without_stream_options(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            _http_400(_STREAM_OPTIONS_400),
            httpx.Response(200, text=_ok_stream()),
        ]
    )
    with structlog.testing.capture_logs() as logs:
        deltas = [
            delta
            async for delta in assist_client.stream_chat(
                [{"role": "user", "content": "hi"}],
                use_tools=False,
            )
        ]
    assert "".join(delta.text for delta in deltas) == "ok"
    assert route.call_count == 2
    first = json.loads(route.calls[0].request.content)
    second = json.loads(route.calls[1].request.content)
    assert first["stream"] is True
    assert first["stream_options"] == {"include_usage": True}
    assert second["stream"] is True
    assert "stream_options" not in second
    spends = [item for item in logs if item.get("event") == "llm_spend_recorded"]
    assert spends[-1]["source"] == "usage"


@pytest.mark.asyncio
async def test_stream_chat_retries_stream_options_then_max_tokens(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        side_effect=[
            _http_400(_STREAM_OPTIONS_400),
            _http_400(_MAX_TOKENS_400),
            httpx.Response(200, text=_ok_stream()),
        ]
    )
    deltas = [
        delta
        async for delta in assist_client.stream_chat(
            [{"role": "user", "content": "hi"}],
            use_tools=False,
            max_tokens=300,
        )
    ]
    assert "".join(delta.text for delta in deltas) == "ok"
    assert route.call_count == 3
    bodies = [json.loads(call.request.content) for call in route.calls]
    assert bodies[0]["stream_options"] == {"include_usage": True}
    assert bodies[0]["max_tokens"] == 300
    assert "stream_options" not in bodies[1]
    assert bodies[1]["max_tokens"] == 300
    assert "stream_options" not in bodies[2]
    assert bodies[2]["max_completion_tokens"] == 300
    assert "max_tokens" not in bodies[2]


@pytest.mark.asyncio
async def test_stream_chat_does_not_retry_unrelated_400(
    assist_client: AssistAIClient,
    respx_mock: MockRouter,
) -> None:
    route = respx_mock.post(CHAT_URL).mock(
        return_value=_http_400("temperature is not supported")
    )
    with pytest.raises(AssistAIError, match="temperature"):
        async for _delta in assist_client.stream_chat(
            [{"role": "user", "content": "hi"}],
            use_tools=False,
        ):
            pass
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_stream_chat_does_not_retry_after_first_chunk(
    assist_client: AssistAIClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_iter(
        payload: dict[str, Any],
        *,
        provider: Any = None,
    ) -> AsyncIterator[dict[str, Any]]:
        del provider
        calls.append(dict(payload))
        yield {
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "hi"},
                    "finish_reason": None,
                }
            ]
        }
        raise AssistAIError(
            f"ProxyAPI вернул HTTP 400: {_STREAM_OPTIONS_400}",
            status_code=400,
        )

    monkeypatch.setattr(assist_client, "_iter_chat_chunks", fake_iter)
    deltas: list[ChatDelta] = []
    with pytest.raises(AssistAIError, match="stream_options"):
        async for delta in assist_client.stream_chat(
            [{"role": "user", "content": "hi"}],
            use_tools=False,
        ):
            deltas.append(delta)
    assert [delta.text for delta in deltas] == ["hi"]
    assert len(calls) == 1
    assert calls[0]["stream_options"] == {"include_usage": True}

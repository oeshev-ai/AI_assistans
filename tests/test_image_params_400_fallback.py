"""400 на quality/complexity: ровно один повтор без этих полей."""

from __future__ import annotations

import json

import httpx
import pytest
from respx import MockRouter

from app.llm.assist_ai import AssistAIClient


@pytest.mark.asyncio
async def test_generate_retries_once_without_rejected_params(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        side_effect=[
            httpx.Response(
                400,
                json={"error": {"message": "unknown parameter: quality"}},
            ),
            httpx.Response(200, json={"data": [{"url": "https://cdn.example/ok.png"}]}),
        ]
    )
    images = await assist_client.generate_image(
        "постер",
        model="openai/gpt-image-2.6-flare",
        size="1024x1024",
    )
    assert images == ["https://cdn.example/ok.png"]
    assert len(route.calls) == 2
    first = json.loads(route.calls[0].request.content)
    second = json.loads(route.calls[1].request.content)
    assert first["quality"] == "high"
    assert "complexity" not in first
    assert "quality" not in second
    assert "complexity" not in second


@pytest.mark.asyncio
async def test_unrelated_400_is_not_retried(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    from app.llm.assist_ai import AssistAIError

    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(
            400, json={"error": {"message": "content_policy_violation"}}
        )
    )
    with pytest.raises(AssistAIError):
        await assist_client.generate_image(
            "постер",
            model="openai/gpt-image-2.6-flare",
            size="1024x1024",
        )
    assert len(route.calls) == 1

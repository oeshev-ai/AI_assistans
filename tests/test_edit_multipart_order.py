"""POST /images/edits: основа первой частью multipart, без JSON Content-Type."""

from __future__ import annotations

import base64

import httpx
import pytest
from respx import MockRouter

from app.llm.assist_ai import AssistAIClient, AssistAIError


@pytest.mark.asyncio
async def test_base_file_is_first(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/edits").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"b64_json": base64.b64encode(b"ok").decode("ascii")}]},
        )
    )
    images = await assist_client.edit_images(
        [
            (b"base-bytes", "base.png", "image/png"),
            (b"ref-bytes", "ref.png", "image/png"),
        ],
        "поменяй фон",
        model="openai/gpt-image-2.6-flare",
    )
    assert images
    request = route.calls[0].request
    assert request.url.path.endswith("images/edits")
    content_type = request.headers["content-type"]
    assert content_type.startswith("multipart/form-data")
    assert "application/json" not in content_type
    body = request.content.decode("latin1")
    assert body.index('filename="base.png"') < body.index('filename="ref.png"')


@pytest.mark.asyncio
async def test_content_policy_text_is_passed_through(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    respx_mock.post("https://api.proxyapi.ru/v1/images/edits").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "content_policy_violation: rejected"}},
        )
    )
    with pytest.raises(AssistAIError, match="content_policy_violation: rejected"):
        await assist_client.edit_images(
            [(b"base-bytes", "base.png", "image/png")],
            "правка",
            model="openai/gpt-image-2.6-flare",
        )

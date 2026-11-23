"""generation_id остаётся в scope и не попадает в ответ модели."""

from __future__ import annotations

import base64

import httpx
import pytest
from respx import MockRouter

from app.llm.assist_ai import AssistAIClient
from app.llm.image_ops import ImageModelScope
from app.media.media_tools import (
    EditImageTool,
    UserMedia,
    reset_user_media,
    set_user_media,
)


@pytest.mark.asyncio
async def test_edit_tool_keeps_generation_id_out_of_model_json(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    encoded = base64.b64encode(b"ok").decode("ascii")
    respx_mock.post("https://api.proxyapi.ru/v1/images/edits").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"b64_json": encoded, "generation_id": "gen_1"}]},
        )
    )
    tool = EditImageTool(assist_client)
    token = set_user_media([UserMedia("image", "p.jpg", "image/jpeg", b"not-an-image")])
    try:
        with ImageModelScope(None) as scope:
            result = await tool.execute(instruction="замени фон")
    finally:
        reset_user_media(token)
    assert "generation_id" not in result
    assert scope.images[0].get("generation_id") == "gen_1"
    assert "value" not in scope.images[0]

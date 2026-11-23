"""Референсы generate_video: каталог хода и тела Polza / ProxyAPI."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.llm.errors import AssistAIError
from app.llm.image_ops import GenerateImageTool, ImageModelScope
from app.llm.media_api import (
    VideoReference,
    VideoResult,
    checked_video_references,
    proxy_input_references,
)
from app.media.media_tools import (
    GenerateVideoTool,
    UserMedia,
    reset_user_media,
    set_user_media,
)
from app.storage.assistant_storage import DEFAULT_SYSTEM_PROMPT


class _ImageClient:
    def __init__(self) -> None:
        self.config = SimpleNamespace(default_image_model="openai/gpt-image-2.6-flare")

    async def generate_image(
        self,
        prompt: str,
        model: str | None = None,
        size: str = "1024x1024",
        background: str | None = None,
    ) -> list[str]:
        return ["https://cdn.example/ref.png"]


class _VideoClient:
    def __init__(self) -> None:
        self.called = False
        self.references: list[VideoReference] | None = None

    async def generate_video(
        self,
        prompt: str,
        *,
        model: str | None = None,
        duration: int | None = None,
        resolution: str | None = None,
        references: list[VideoReference] | None = None,
    ) -> VideoResult:
        self.called = True
        self.references = references
        return VideoResult(data=b"mp4", filename="video.mp4")


def test_system_prompt_tells_persona_to_pass_reference_indexes() -> None:
    assert "reference_indexes" in DEFAULT_SYSTEM_PROMPT
    assert "только текст" in DEFAULT_SYSTEM_PROMPT


def test_proxy_input_references_url_and_bytes() -> None:
    payload = proxy_input_references(
        [
            VideoReference(url="https://cdn.example/girl.png"),
            VideoReference(data=b"png", mime_type="image/png"),
        ]
    )
    assert payload[0] == {
        "type": "image_url",
        "image_url": {"url": "https://cdn.example/girl.png"},
    }
    assert payload[1]["image_url"]["url"] == "data:image/png;base64,cG5n"


def test_reference_limit_matches_seedance() -> None:
    nine = [VideoReference(data=b"x") for _ in range(9)]
    assert len(checked_video_references(nine)) == 9
    with pytest.raises(AssistAIError) as info:
        checked_video_references([*nine, VideoReference(data=b"y")])
    assert info.value.error_code == "BAD_REQUEST"


def test_checked_references_reject_private_url() -> None:
    with pytest.raises(AssistAIError) as info:
        checked_video_references([VideoReference(url="https://127.0.0.1/a.png")])
    assert info.value.error_code == "BAD_REQUEST"
    with pytest.raises(AssistAIError):
        checked_video_references([VideoReference(url="http://cdn.example/a.png")])


@pytest.mark.asyncio
async def test_generate_image_numbers_references_after_user_photo() -> None:
    token = set_user_media(
        [
            UserMedia(
                kind="image",
                filename="photo.jpg",
                mime_type="image/jpeg",
                data=b"jpg",
            )
        ]
    )
    try:
        tool = GenerateImageTool(_ImageClient())  # type: ignore[arg-type]
        with ImageModelScope(None):
            payload = json.loads(await tool.execute(prompt="герой"))
    finally:
        reset_user_media(token)
    assert payload["reference_indexes"] == [1]


@pytest.mark.asyncio
async def test_generate_video_tool_passes_user_and_generated_refs() -> None:
    token = set_user_media(
        [
            UserMedia(
                kind="image",
                filename="photo.jpg",
                mime_type="image/jpeg",
                data=b"jpg",
            )
        ]
    )
    client = _VideoClient()
    try:
        image_tool = GenerateImageTool(_ImageClient())  # type: ignore[arg-type]
        video_tool = GenerateVideoTool(client)  # type: ignore[arg-type]
        with ImageModelScope(None):
            await image_tool.execute(prompt="герой")
            payload = json.loads(
                await video_tool.execute(
                    prompt="герой с первой картинки идёт по второй",
                    reference_indexes=[1, 0],
                )
            )
    finally:
        reset_user_media(token)
    assert payload["status"] == "success"
    assert payload["references"] == 2
    assert client.references is not None
    assert client.references[0].url == "https://cdn.example/ref.png"
    assert client.references[1].data == b"jpg"


@pytest.mark.asyncio
async def test_generate_video_tool_rejects_unknown_index_before_create() -> None:
    client = _VideoClient()
    tool = GenerateVideoTool(client)  # type: ignore[arg-type]
    payload = json.loads(await tool.execute(prompt="сцена", reference_indexes=[0]))
    assert "error" in payload
    assert client.called is False

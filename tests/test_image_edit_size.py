"""Пропорции правки: fit_image_size и режим source."""

from __future__ import annotations

import io

import httpx
import pytest
from respx import MockRouter

from app.llm.assist_ai import AssistAIClient
from app.media.media_utils import fit_image_size


def _limits(width: int, height: int) -> None:
    assert width % 16 == 0
    assert height % 16 == 0
    assert max(width, height) <= 3840
    assert max(width / height, height / width) <= 3.0 + 1e-6
    assert 655_360 <= width * height <= 8_294_400


@pytest.mark.parametrize(
    ("width", "height"),
    [(4000, 3000), (1024, 1024), (3000, 1000), (100, 100)],
)
def test_fit_image_size_stays_inside_gpt_limits(width: int, height: int) -> None:
    fitted = fit_image_size(width, height)
    out_w, out_h = (int(part) for part in fitted.split("x"))
    _limits(out_w, out_h)
    source_ratio = min(3.0, max(1 / 3, width / height))
    assert abs((out_w / out_h) - source_ratio) / source_ratio <= 0.02


@pytest.mark.asyncio
async def test_source_mode_keeps_three_to_two(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1536, 1024), (20, 40, 60)).save(buffer, format="PNG")
    png = buffer.getvalue()
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/edits").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/edited.png"}]}
        )
    )
    assist_client.config = assist_client.config.model_copy(
        update={"image_edit_size_mode": "source"}
    )
    await assist_client.edit_images(
        [(png, "base.png", "image/png")],
        "замени фон",
        model="openai/gpt-image-2.6-flare",
    )
    body = route.calls[0].request.content.decode("latin1")
    assert "1536x1024" in body
    assert "1024x1024" not in body


@pytest.mark.asyncio
async def test_fixed_mode_keeps_square(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (1536, 1024), (1, 2, 3)).save(buffer, format="PNG")
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/edits").mock(
        return_value=httpx.Response(
            200, json={"data": [{"url": "https://cdn.example/edited.png"}]}
        )
    )
    assist_client.config = assist_client.config.model_copy(
        update={"image_edit_size_mode": "fixed"}
    )
    await assist_client.edit_images(
        [(buffer.getvalue(), "base.png", "image/png")],
        "правка",
        model="openai/gpt-image-2.6-flare",
    )
    body = route.calls[0].request.content.decode("latin1")
    assert "1024x1024" in body

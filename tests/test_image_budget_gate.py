"""Budget preflight for generate/edit lives in AssistAIClient (no double reserve)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from io import BytesIO

import httpx
import pytest
from PIL import Image
from respx import MockRouter

from app.infra.cost_tracker import BudgetExceededError, cost_tracker, set_billing_user
from app.llm.assist_ai import AssistAIClient


def _png(size: tuple[int, int] = (64, 48)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
async def _reset_budget_state() -> AsyncIterator[None]:
    cost_tracker.daily_budget = 0.0
    cost_tracker.user_daily_budget = 0.0
    cost_tracker._memory.clear()
    await cost_tracker.release_reservation()
    set_billing_user(None)
    yield
    await cost_tracker.release_reservation()
    cost_tracker.daily_budget = 0.0
    cost_tracker.user_daily_budget = 0.0
    cost_tracker._memory.clear()
    set_billing_user(None)


@pytest.mark.asyncio
async def test_generate_blocks_before_http_when_budget_exceeded(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(200, json={"data": [{"url": "https://cdn/x.png"}]})
    )
    set_billing_user(42)
    cost_tracker.user_daily_budget = 0.001
    await cost_tracker.record(kind="image", amount_usd=0.001, user_id=42)

    with pytest.raises(BudgetExceededError):
        await assist_client.generate_image(
            "постер",
            model="openai/gpt-image-2.6-flare",
            size="1024x1024",
        )
    assert len(route.calls) == 0


@pytest.mark.asyncio
async def test_edit_blocks_before_http_when_budget_exceeded(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    route = respx_mock.post("https://api.proxyapi.ru/v1/images/edits").mock(
        return_value=httpx.Response(200, json={"data": [{"url": "https://cdn/e.png"}]})
    )
    set_billing_user(7)
    cost_tracker.user_daily_budget = 0.001
    await cost_tracker.record(kind="image", amount_usd=0.001, user_id=7)

    with pytest.raises(BudgetExceededError):
        await assist_client.edit_images(
            [(_png(), "base.png", "image/png")],
            "убери фон",
            model="openai/gpt-image-2.6-flare",
        )
    assert len(route.calls) == 0


@pytest.mark.asyncio
async def test_generate_single_reservation_near_budget_edge(
    assist_client: AssistAIClient, respx_mock: MockRouter
) -> None:
    """Outer+inner check_budget used to reserve ~2× and falsely reject near the cap."""
    respx_mock.post("https://api.proxyapi.ru/v1/images/generations").mock(
        return_value=httpx.Response(200, json={"data": [{"url": "https://cdn/ok.png"}]})
    )
    set_billing_user(9)
    estimate = cost_tracker.estimate_image_cost(
        1,
        model="openai/gpt-image-2.6-flare",
        quality=assist_client.config.image_quality,
        size="1024x1024",
        provider=assist_client._endpoint("image"),
        rub_per_usd=assist_client.config.rub_per_usd,
    )
    cost_tracker.user_daily_budget = estimate * 1.5
    images = await assist_client.generate_image(
        "постер",
        model="openai/gpt-image-2.6-flare",
        size="1024x1024",
    )
    assert images == ["https://cdn/ok.png"]

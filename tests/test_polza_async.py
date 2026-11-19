"""G3/G4: async-режим Polza — pending → polling → completed, исчерпание ожидания."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import httpx
import pytest
import respx

from app.llm import polza_media
from app.llm.assist_ai import AssistAIClient, AssistAIError
from app.llm.errors import fallback_reason
from app.llm.polza_media import generate
from tests.polza_helpers import make_polza_provider

_BASE = "https://polza.ai/api"
_MEDIA_URL = f"{_BASE}/v1/media"
_MODEL = "openai/gpt-image-2.6-flare"
_PROXY_IMAGES = "https://api.proxyapi.ru/v1/images/generations"


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Подменяет паузы: тесты не ждут, а паузы записываются."""
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(polza_media, "_sleep", fake_sleep)
    monkeypatch.setattr("app.llm.assist_ai._sleep", fake_sleep)
    return recorded


@pytest.mark.asyncio
async def test_pending_then_polling_then_completed(
    polza_fixture: Callable[[str], dict[str, Any]], sleeps: list[float]
) -> None:
    pending = polza_fixture("async_pending")
    done = polza_fixture("generate_completed")
    media_id = pending["id"]
    processing = {**pending, "status": "processing"}
    with respx.mock(assert_all_called=False) as router:
        router.post(_MEDIA_URL).mock(return_value=httpx.Response(200, json=pending))
        poll = router.get(f"{_BASE}/v1/media/{media_id}").mock(
            side_effect=[
                httpx.Response(200, json=processing),
                httpx.Response(200, json=processing),
                httpx.Response(200, json=done),
            ]
        )
        v2 = router.get(f"{_BASE}/v2/media/{media_id}").mock(
            return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})
        )
        async with httpx.AsyncClient() as client:
            result = await generate(
                client,
                make_polza_provider(),
                model=_MODEL,
                prompt="x",
                poll_interval=4.0,
                total_wait=240.0,
            )
    assert len(poll.calls) == 3
    assert not v2.called
    assert sleeps == [4.0, 4.0, 4.0]
    assert result.images[0]["value"].startswith("https://s3.polza.ai/")
    assert result.cost_rub == pytest.approx(2.37166294)


@pytest.mark.asyncio
async def test_poll_exhaustion_raises_poll_timeout(
    polza_fixture: Callable[[str], dict[str, Any]], sleeps: list[float]
) -> None:
    pending = polza_fixture("async_pending")
    with respx.mock(assert_all_called=False) as router:
        router.post(_MEDIA_URL).mock(return_value=httpx.Response(200, json=pending))
        poll = router.get(f"{_BASE}/v1/media/{pending['id']}").mock(
            return_value=httpx.Response(200, json=pending)
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(AssistAIError) as info:
                await generate(
                    client,
                    make_polza_provider(),
                    model=_MODEL,
                    prompt="x",
                    poll_interval=4.0,
                    total_wait=8.0,
                )
    assert len(poll.calls) == 2
    assert info.value.error_code == "POLL_TIMEOUT"
    assert fallback_reason(info.value) == "poll_timeout"


@pytest.mark.asyncio
async def test_failed_status_maps_error(
    polza_fixture: Callable[[str], dict[str, Any]], sleeps: list[float]
) -> None:
    pending = polza_fixture("async_pending")
    failed = {
        "id": pending["id"],
        "status": "failed",
        "error": {"code": "BAD_REQUEST", "message": "internal detail xyz"},
    }
    with respx.mock(assert_all_called=False) as router:
        router.post(_MEDIA_URL).mock(return_value=httpx.Response(200, json=pending))
        router.get(f"{_BASE}/v1/media/{pending['id']}").mock(
            return_value=httpx.Response(200, json=failed)
        )
        async with httpx.AsyncClient() as client:
            with pytest.raises(AssistAIError) as info:
                await generate(client, make_polza_provider(), model=_MODEL, prompt="x")
    assert info.value.error_code == "BAD_REQUEST"
    assert "xyz" not in info.value.message
    assert fallback_reason(info.value) is None


@pytest.mark.asyncio
async def test_transient_poll_5xx_is_tolerated(
    polza_fixture: Callable[[str], dict[str, Any]], sleeps: list[float]
) -> None:
    pending = polza_fixture("async_pending")
    done = polza_fixture("edit_completed")
    with respx.mock(assert_all_called=False) as router:
        router.post(_MEDIA_URL).mock(return_value=httpx.Response(200, json=pending))
        router.get(f"{_BASE}/v1/media/{pending['id']}").mock(
            side_effect=[
                httpx.Response(503, json={"error": {"code": "SERVICE_UNAVAILABLE"}}),
                httpx.Response(200, json=done),
            ]
        )
        async with httpx.AsyncClient() as client:
            result = await generate(
                client, make_polza_provider(), model=_MODEL, prompt="x"
            )
    assert result.images


@pytest.mark.asyncio
async def test_client_poll_exhaustion_falls_back_to_proxyapi(
    multi_client_factory: Callable[..., AssistAIClient],
    polza_fixture: Callable[[str], dict[str, Any]],
    sleeps: list[float],
) -> None:
    pending = polza_fixture("async_pending")
    client = multi_client_factory(polza_total_wait=8.0, polza_poll_interval=4.0)
    with respx.mock(assert_all_called=False) as router:
        router.post(_MEDIA_URL).mock(return_value=httpx.Response(200, json=pending))
        router.get(f"{_BASE}/v1/media/{pending['id']}").mock(
            return_value=httpx.Response(200, json=pending)
        )
        proxy = router.post(_PROXY_IMAGES).mock(
            return_value=httpx.Response(
                200, json={"data": [{"url": "https://cdn.example/fallback.png"}]}
            )
        )
        images = await client.generate_image("постер", model=_MODEL, size="1024x1024")
    assert images == ["https://cdn.example/fallback.png"]
    assert len(proxy.calls) == 1


@pytest.mark.asyncio
async def test_client_async_then_completed_records_cost_rub(
    multi_client_factory: Callable[..., AssistAIClient],
    polza_fixture: Callable[[str], dict[str, Any]],
    sleeps: list[float],
) -> None:
    pending = polza_fixture("async_pending")
    done = polza_fixture("generate_completed")
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        router.post(_MEDIA_URL).mock(return_value=httpx.Response(200, json=pending))
        router.get(f"{_BASE}/v1/media/{pending['id']}").mock(
            return_value=httpx.Response(200, json=done)
        )
        images = await client.generate_image("постер", model=_MODEL, size="1536x1024")
    assert images[0].startswith("https://s3.polza.ai/")

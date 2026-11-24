"""OpenWeather retries 429/5xx/timeouts after they become OpenWeatherAPIError."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest

from app.llm.openweather import OpenWeatherAPIError, OpenWeatherClient, OWMConfig


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.fixture
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)


def _response(status: int, body: dict | None = None) -> httpx.Response:
    request = httpx.Request("GET", "https://api.openweathermap.org/data/2.5/weather")
    kwargs: dict = {"status_code": status, "request": request}
    if body is not None:
        kwargs["json"] = body
    return httpx.Response(**kwargs)


async def _client_with(
    get: Callable[..., Awaitable[httpx.Response]],
) -> OpenWeatherClient:
    client = OpenWeatherClient(OWMConfig(api_key="test-key"))
    client._client.get = get  # type: ignore[method-assign]
    return client


@pytest.mark.asyncio
async def test_retries_503_then_returns_body(fast_retry: None) -> None:
    calls = 0

    async def get(url: str, params: dict | None = None) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls < 3:
            return _response(503)
        return _response(200, {"ok": True})

    client = await _client_with(get)
    try:
        assert await client._make_request_raw("https://example.test", {}) == {
            "ok": True
        }
    finally:
        await client.close()
    assert calls == 3


@pytest.mark.asyncio
async def test_does_not_retry_client_error(fast_retry: None) -> None:
    calls = 0

    async def get(url: str, params: dict | None = None) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(404)

    client = await _client_with(get)
    try:
        with pytest.raises(OpenWeatherAPIError) as info:
            await client._make_request_raw("https://example.test", {})
    finally:
        await client.close()
    assert info.value.status_code == 404
    assert calls == 1


@pytest.mark.asyncio
async def test_retries_timeout_then_raises(fast_retry: None) -> None:
    calls = 0

    async def get(url: str, params: dict | None = None) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("timed out")

    client = await _client_with(get)
    try:
        with pytest.raises(OpenWeatherAPIError, match="время ожидания"):
            await client._make_request_raw("https://example.test", {})
    finally:
        await client.close()
    assert calls == 3


@pytest.mark.asyncio
async def test_retries_network_error(fast_retry: None) -> None:
    calls = 0

    async def get(url: str, params: dict | None = None) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("down")

    client = await _client_with(get)
    try:
        with pytest.raises(OpenWeatherAPIError, match="Ошибка сети"):
            await client._make_request_raw("https://example.test", {})
    finally:
        await client.close()
    assert calls == 3

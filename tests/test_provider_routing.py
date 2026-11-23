"""G2/G4: маршруты, per-provider breaker, однократный fallback, 4xx без fallback."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
import respx
import structlog
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from prometheus_client import REGISTRY

from app.infra.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitState,
    get_breaker,
    llm_breaker,
)
from app.infra.health_server import health_check
from app.llm.assist_ai import AssistAIClient, AssistAIConfig, AssistAIError
from app.llm.errors import fallback_reason, is_rate_limited

_MODEL = "openai/gpt-image-2.6-flare"
_POLZA_MEDIA = "https://polza.ai/api/v1/media"
_PROXY_BASE = "https://api.proxyapi.ru/v1"
_PROXY_IMAGES = f"{_PROXY_BASE}/images/generations"
_PROXY_EDITS = f"{_PROXY_BASE}/images/edits"
_PROXY_OK = {"data": [{"url": "https://cdn.example/proxy.png"}]}
_POLZA_OK = {
    "id": "gen_1",
    "status": "completed",
    "data": [{"url": "https://s3.polza.ai/ok.png"}],
    "usage": {"cost_rub": 6.29},
}


def _fallback_count(source: str, target: str, reason: str) -> float:
    value = REGISTRY.get_sample_value(
        "provider_fallback_total", {"from": source, "to": target, "reason": reason}
    )
    return value or 0.0


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr("app.llm.image_ops._sleep", fake_sleep)
    monkeypatch.setattr("app.llm.transport._sleep", fake_sleep)
    monkeypatch.setattr("app.llm.polza_media._sleep", fake_sleep)
    return recorded


# --- G2: конфиг и выбор эндпоинтов ---


def test_proxyapi_only_client_routes_everything_to_proxyapi(
    assist_client: AssistAIClient,
) -> None:
    """AssistAIConfig без polza в providers → seed на proxyapi."""
    for operation in ("text", "chat", "image", "images_edit", "audio", "video"):
        provider = assist_client._endpoint(operation)
        assert provider.id == "proxyapi"
        assert provider.base_url == _PROXY_BASE
    assert assist_client.provider_ids() == ["proxyapi"]
    assert assist_client._client_for(assist_client._endpoint("image")) is (
        assist_client._http
    )


def test_from_env_reads_routes_and_waits(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("PROXYAPI_API_KEY", "px-key")
    monkeypatch.setenv("PROXYAPI_BASE_URL", "https://proxy.example/v9/")
    monkeypatch.setenv("POLZA_API_KEY", "pz-key")
    monkeypatch.setenv("TEXT_PROVIDER", "polza")
    monkeypatch.setenv("TEXT_PROVIDER_FALLBACK", "proxyapi")
    monkeypatch.setenv("IMAGE_PROVIDER", "polza")
    monkeypatch.setenv("IMAGE_PROVIDER_FALLBACK", "proxyapi")
    monkeypatch.setenv("RUB_PER_USD", "300")
    monkeypatch.setenv("POLZA_TOTAL_WAIT", "120")
    monkeypatch.setenv("POLZA_POLL_INTERVAL", "2")
    config = AssistAIConfig.from_env()
    assert config.base_url == "https://proxy.example/v9"
    assert config.routes["image"] == "polza"
    assert config.routes["text"] == "polza"
    assert config.image_provider_fallback == "proxyapi"
    assert config.text_provider_fallback == "proxyapi"
    assert config.rub_per_usd == 300.0
    assert config.polza_total_wait == 120.0
    assert config.polza_poll_interval == 2.0
    assert set(config.providers) == {"proxyapi", "polza"}


def test_from_env_defaults_polza_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("PROXYAPI_API_KEY", "px-key")
    monkeypatch.setenv("POLZA_API_KEY", "pz-key")
    for name in (
        "TEXT_PROVIDER",
        "TEXT_PROVIDER_FALLBACK",
        "IMAGE_PROVIDER",
        "IMAGE_PROVIDER_FALLBACK",
        "AUDIO_PROVIDER",
        "AUDIO_PROVIDER_FALLBACK",
        "VIDEO_PROVIDER",
        "VIDEO_PROVIDER_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    config = AssistAIConfig.from_env()
    assert config.base_url == _PROXY_BASE
    assert set(config.providers) == {"proxyapi", "polza"}
    assert set(config.routes.values()) == {"polza"}
    assert config.image_provider_fallback == "proxyapi"
    assert config.text_provider_fallback == "proxyapi"
    assert config.audio_provider_fallback == "proxyapi"
    assert config.video_provider_fallback == "proxyapi"
    assert config.rub_per_usd == 304.0
    assert config.polza_total_wait == 240.0
    assert config.polza_poll_interval == 4.0


def test_missing_polza_key_fails_when_routes_default_to_polza(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("PROXYAPI_API_KEY", "px-key")
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    for name in (
        "TEXT_PROVIDER",
        "IMAGE_PROVIDER",
        "AUDIO_PROVIDER",
        "VIDEO_PROVIDER",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="POLZA_API_KEY"):
        AssistAIConfig.from_env()


def test_unknown_image_provider_fails_at_startup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("app.llm.config.load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("PROXYAPI_API_KEY", "px-key")
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    monkeypatch.setenv("TEXT_PROVIDER", "proxyapi")
    monkeypatch.setenv("AUDIO_PROVIDER", "proxyapi")
    monkeypatch.setenv("VIDEO_PROVIDER", "proxyapi")
    monkeypatch.setenv("IMAGE_PROVIDER", "polza")
    with pytest.raises(ValueError, match=r"POLZA_API_KEY|polza"):
        AssistAIConfig.from_env()


@pytest.mark.asyncio
async def test_proxyapi_only_generate_hits_proxyapi(
    assist_client: AssistAIClient,
) -> None:
    with respx.mock(assert_all_called=False) as router:
        proxy = router.post(_PROXY_IMAGES).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        polza = router.post(_POLZA_MEDIA).mock(return_value=httpx.Response(200))
        images = await assist_client.generate_image(
            "постер", model=_MODEL, size="1024x1024"
        )
    assert images == ["https://cdn.example/proxy.png"]
    assert len(proxy.calls) == 1
    assert proxy.calls[0].request.headers["authorization"] == "Bearer test-proxy-key"
    assert not polza.called


@pytest.mark.asyncio
async def test_proxyapi_only_edit_hits_proxyapi(
    assist_client: AssistAIClient,
) -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (1, 2, 3)).save(buffer, format="PNG")
    with respx.mock(assert_all_called=False) as router:
        edits = router.post(_PROXY_EDITS).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        await assist_client.edit_image(buffer.getvalue(), "светлее", model=_MODEL)
    assert len(edits.calls) == 1


@pytest.mark.asyncio
async def test_image_provider_polza_uses_polza_base_text_stays_proxyapi(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    assert client._endpoint("image").id == "polza"
    assert client._endpoint("text").id == "proxyapi"
    assert client._endpoint("audio").id == "proxyapi"
    assert client._endpoint("video").id == "proxyapi"
    with respx.mock(assert_all_called=False) as router:
        media = router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(200, json=_POLZA_OK)
        )
        chat = router.post(f"{_PROXY_BASE}/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        proxy_images = router.post(_PROXY_IMAGES).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        images = await client.generate_image("постер", model=_MODEL, size="1024x1024")
        await client._post_json("chat/completions", {"model": "m", "messages": []})
    assert images == ["https://s3.polza.ai/ok.png"]
    assert media.calls[0].request.headers["authorization"] == "Bearer test-polza-key"
    assert len(chat.calls) == 1
    assert chat.calls[0].request.headers["authorization"] == "Bearer test-proxy-key"
    assert not proxy_images.called


@pytest.mark.asyncio
async def test_extra_clients_are_closed(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    polza_http = client._client_for(client._endpoint("image"))
    assert polza_http is not client._http
    await client.close()
    assert polza_http.is_closed
    assert client._http.is_closed


@pytest.mark.asyncio
async def test_text_route_to_polza_uses_chat_base_url(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(routes={"text": "polza", "image": "polza"})
    assert client._endpoint("text").id == "polza"
    assert client._endpoint("text").chat_base_url == "https://polza.ai/api/v1"
    with respx.mock(assert_all_called=False) as router:
        chat = router.post("https://polza.ai/api/v1/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        media = router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(200, json=_POLZA_OK)
        )
        wrong = router.post("https://polza.ai/api/chat/completions").mock(
            return_value=httpx.Response(200, json={"choices": []})
        )
        await client._post_json("chat/completions", {"model": "m", "messages": []})
        images = await client.generate_image("постер", model=_MODEL, size="1024x1024")
    assert len(chat.calls) == 1
    assert chat.calls[0].request.headers["authorization"] == "Bearer test-polza-key"
    assert not wrong.called
    assert images == ["https://s3.polza.ai/ok.png"]
    assert media.called


# --- G4: breaker ---


def test_llm_breaker_alias_and_isolation() -> None:
    assert get_breaker("proxyapi") is llm_breaker
    assert get_breaker("polza") is get_breaker("polza")
    assert get_breaker("polza") is not llm_breaker
    assert isinstance(llm_breaker, AsyncCircuitBreaker)


@pytest.mark.asyncio
async def test_open_polza_breaker_does_not_block_proxyapi() -> None:
    polza = get_breaker("polza")
    for _ in range(polza.failure_threshold):
        await polza._on_failure(RuntimeError("down"))
    assert polza.state == CircuitState.OPEN
    assert llm_breaker.state == CircuitState.CLOSED
    await llm_breaker.ensure_allows_call()


# --- G4: fallback ---

_CHAT_OK = {
    "choices": [
        {
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1},
}
_POLZA_CHAT = "https://polza.ai/api/v1/chat/completions"
_PROXY_CHAT = f"{_PROXY_BASE}/chat/completions"


@pytest.mark.asyncio
async def test_text_unavailable_falls_back_to_proxyapi(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={"text": "polza", "image": "polza"},
        text_fallback="proxyapi",
    )
    before = _fallback_count("polza", "proxyapi", "http_5xx")
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_CHAT).mock(return_value=httpx.Response(503))
        proxy = router.post(_PROXY_CHAT).mock(
            return_value=httpx.Response(200, json=_CHAT_OK)
        )
        with structlog.testing.capture_logs() as logs:
            text = await client.chat_completion(
                [{"role": "user", "content": "hi"}],
                use_tools=False,
            )
    assert text == "ok"
    assert len(polza.calls) == 1
    assert len(proxy.calls) == 1
    assert _fallback_count("polza", "proxyapi", "http_5xx") == before + 1
    event = next(item for item in logs if item["event"] == "provider_fallback")
    assert (event["from"], event["to"], event["reason"]) == (
        "polza",
        "proxyapi",
        "http_5xx",
    )


@pytest.mark.asyncio
async def test_text_client_error_does_not_fall_back(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={"text": "polza", "image": "polza"},
        text_fallback="proxyapi",
    )
    with respx.mock(assert_all_called=False) as router:
        polza = router.post(_POLZA_CHAT).mock(
            return_value=httpx.Response(400, json={"error": {"code": "BAD_REQUEST"}})
        )
        proxy = router.post(_PROXY_CHAT).mock(
            return_value=httpx.Response(200, json=_CHAT_OK)
        )
        with pytest.raises(AssistAIError) as info:
            await client.chat_completion(
                [{"role": "user", "content": "hi"}],
                use_tools=False,
            )
    assert len(polza.calls) == 1
    assert not proxy.called
    assert info.value.status_code == 400


@pytest.mark.asyncio
async def test_text_no_fallback_configured_raises(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(
        routes={"text": "polza", "image": "polza"},
        text_fallback="",
    )
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_CHAT).mock(return_value=httpx.Response(503))
        proxy = router.post(_PROXY_CHAT).mock(
            return_value=httpx.Response(200, json=_CHAT_OK)
        )
        with pytest.raises(AssistAIError):
            await client.chat_completion(
                [{"role": "user", "content": "hi"}],
                use_tools=False,
            )
    assert not proxy.called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("polza_response", "reason"),
    [
        (
            httpx.Response(503, json={"error": {"code": "SERVICE_UNAVAILABLE"}}),
            "service_unavailable",
        ),
        (httpx.Response(502, json={"error": {"code": "BAD_GATEWAY"}}), "bad_gateway"),
        (
            httpx.Response(408, json={"error": {"code": "REQUEST_TIMEOUT"}}),
            "request_timeout",
        ),
        (httpx.Response(500, text="boom"), "internal_error"),
    ],
)
async def test_unavailable_falls_back_exactly_once(
    multi_client_factory: Callable[..., AssistAIClient],
    polza_response: httpx.Response,
    reason: str,
) -> None:
    client = multi_client_factory()
    before = _fallback_count("polza", "proxyapi", reason)
    with respx.mock(assert_all_called=False) as router:
        media = router.post(_POLZA_MEDIA).mock(return_value=polza_response)
        proxy = router.post(_PROXY_IMAGES).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        with structlog.testing.capture_logs() as logs:
            images = await client.generate_image(
                "постер", model=_MODEL, size="1024x1024"
            )
    assert images == ["https://cdn.example/proxy.png"]
    assert len(media.calls) == 1
    assert len(proxy.calls) == 1
    assert _fallback_count("polza", "proxyapi", reason) == before + 1
    event = next(item for item in logs if item["event"] == "provider_fallback")
    assert (event["from"], event["to"], event["reason"]) == (
        "polza",
        "proxyapi",
        reason,
    )


@pytest.mark.asyncio
async def test_timeout_falls_back(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_MEDIA).mock(side_effect=httpx.ReadTimeout("slow"))
        proxy = router.post(_PROXY_IMAGES).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        images = await client.generate_image("постер", model=_MODEL, size="1024x1024")
    assert images == ["https://cdn.example/proxy.png"]
    assert len(proxy.calls) == 1


@pytest.mark.asyncio
async def test_open_circuit_falls_back_without_polza_call(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    polza = get_breaker("polza")
    for _ in range(polza.failure_threshold):
        await polza._on_failure(RuntimeError("down"))
    with respx.mock(assert_all_called=False) as router:
        media = router.post(_POLZA_MEDIA).mock(return_value=httpx.Response(200))
        proxy = router.post(_PROXY_IMAGES).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        images = await client.generate_image("постер", model=_MODEL, size="1024x1024")
    assert images == ["https://cdn.example/proxy.png"]
    assert not media.called
    assert len(proxy.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "BAD_REQUEST"),
        (401, "UNAUTHORIZED"),
        (403, "FORBIDDEN"),
        (403, "guard_blocked"),
        (403, "guard_no_route"),
        (404, "NOT_FOUND"),
        (413, "PAYLOAD_TOO_LARGE"),
    ],
)
async def test_client_errors_do_not_fall_back_or_retry(
    multi_client_factory: Callable[..., AssistAIClient], status: int, code: str
) -> None:
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        media = router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(status, json={"error": {"code": code}})
        )
        proxy = router.post(_PROXY_IMAGES).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        with pytest.raises(AssistAIError) as info:
            await client.generate_image("постер", model=_MODEL, size="1024x1024")
    assert len(media.calls) == 1
    assert not proxy.called
    assert info.value.status_code == status
    # 4xx не считаются недоступностью провайдера.
    assert get_breaker("polza").state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_fallback_happens_only_once(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        media = router.post(_POLZA_MEDIA).mock(return_value=httpx.Response(503))
        proxy = router.post(_PROXY_IMAGES).mock(return_value=httpx.Response(503))
        with pytest.raises(AssistAIError):
            await client.generate_image("постер", model=_MODEL, size="1024x1024")
    assert len(media.calls) == 1
    assert len(proxy.calls) == 1


@pytest.mark.asyncio
async def test_no_fallback_configured_raises(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(fallback="")
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_MEDIA).mock(return_value=httpx.Response(503))
        proxy = router.post(_PROXY_IMAGES).mock(return_value=httpx.Response(200))
        with pytest.raises(AssistAIError):
            await client.generate_image("постер", model=_MODEL, size="1024x1024")
    assert not proxy.called


@pytest.mark.asyncio
async def test_edit_falls_back_to_proxyapi_multipart(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (9, 9, 9)).save(buffer, format="PNG")
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_MEDIA).mock(return_value=httpx.Response(503))
        edits = router.post(_PROXY_EDITS).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        images = await client.edit_image(buffer.getvalue(), "светлее", model=_MODEL)
    assert images == ["https://cdn.example/proxy.png"]
    assert len(edits.calls) == 1
    assert b'name="image"' in edits.calls[0].request.content


# --- G4: 429 ---


@pytest.mark.asyncio
async def test_429_respects_retry_after(
    multi_client_factory: Callable[..., AssistAIClient], sleeps: list[float]
) -> None:
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        media = router.post(_POLZA_MEDIA).mock(
            side_effect=[
                httpx.Response(
                    429,
                    headers={"Retry-After": "7"},
                    json={"error": {"code": "TOO_MANY_REQUESTS"}},
                ),
                httpx.Response(200, json=_POLZA_OK),
            ]
        )
        proxy = router.post(_PROXY_IMAGES).mock(return_value=httpx.Response(200))
        images = await client.generate_image("постер", model=_MODEL, size="1024x1024")
    assert images == ["https://s3.polza.ai/ok.png"]
    assert sleeps == [7.0]
    assert len(media.calls) == 2
    assert not proxy.called


@pytest.mark.asyncio
async def test_429_without_retry_after_backs_off_then_falls_back(
    multi_client_factory: Callable[..., AssistAIClient], sleeps: list[float]
) -> None:
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        media = router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(
                429, json={"error": {"code": "TOO_MANY_REQUESTS"}}
            )
        )
        proxy = router.post(_PROXY_IMAGES).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        images = await client.generate_image("постер", model=_MODEL, size="1024x1024")
    assert images == ["https://cdn.example/proxy.png"]
    assert len(media.calls) == 3  # первый запрос + максимум два повтора
    assert sleeps == [1.0, 2.0]
    assert len(proxy.calls) == 1


# --- G4: 402 ---


@pytest.mark.asyncio
async def test_insufficient_balance_falls_back_and_alerts(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        router.post(_POLZA_MEDIA).mock(
            return_value=httpx.Response(
                402, json={"error": {"code": "INSUFFICIENT_BALANCE"}}
            )
        )
        proxy = router.post(_PROXY_IMAGES).mock(
            return_value=httpx.Response(200, json=_PROXY_OK)
        )
        with structlog.testing.capture_logs() as logs:
            images = await client.generate_image(
                "постер", model=_MODEL, size="1024x1024"
            )
    assert images == ["https://cdn.example/proxy.png"]
    assert len(proxy.calls) == 1
    alerts = [item for item in logs if item["event"] == "provider_balance_exhausted"]
    assert alerts
    assert alerts[0]["log_level"] == "error"
    assert alerts[0]["provider"] == "polza"


# --- классификация ошибок ---


def test_fallback_reason_classification() -> None:
    assert fallback_reason(AssistAIError("x", status_code=503)) == "http_5xx"
    assert fallback_reason(AssistAIError("x", error_code="TIMEOUT")) == "timeout"
    assert fallback_reason(AssistAIError("x", status_code=400)) is None
    assert fallback_reason(AssistAIError("x", status_code=413)) is None
    assert fallback_reason(AssistAIError("x", error_code="guard_blocked")) is None
    assert fallback_reason(AssistAIError("plain")) is None
    assert (
        fallback_reason(
            AssistAIError("x", error_code="POLL_TIMEOUT", job_committed=True)
        )
        is None
    )
    assert is_rate_limited(AssistAIError("x", status_code=429))
    assert not is_rate_limited(AssistAIError("x", status_code=503))


# --- test_connection и health ---


@pytest.mark.asyncio
async def test_test_connection_checks_all_providers(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        proxy = router.get(f"{_PROXY_BASE}/").mock(return_value=httpx.Response(404))
        polza = router.get("https://polza.ai/api/v1/models").mock(
            return_value=httpx.Response(200, json={})
        )
        await client.test_connection()
    assert proxy.called
    assert polza.called


@pytest.mark.asyncio
async def test_test_connection_reports_failed_provider(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory()
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{_PROXY_BASE}/").mock(return_value=httpx.Response(200))
        router.get("https://polza.ai/api/v1/models").mock(
            return_value=httpx.Response(503)
        )
        with pytest.raises(AssistAIError, match="polza"):
            await client.test_connection()


@pytest.mark.asyncio
async def test_health_reports_per_provider_status() -> None:
    async def ok() -> None:
        return None

    async def fail() -> None:
        raise ConnectionError("10.0.0.7 refused secret-token")

    app = web.Application()

    async def handler(request: web.Request) -> web.Response:
        return await health_check(request, None, ok, {"polza": fail})

    app.router.add_get("/health", handler)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        resp = await client.get("/health")
        body = await resp.json()
        assert resp.status == 503
        assert body["components"]["proxy_api"] == "ok"
        assert body["components"]["providers"] == {"proxyapi": "ok", "polza": "error"}
        raw = json.dumps(body)
        assert "10.0.0.7" not in raw
        assert "secret-token" not in raw
    finally:
        await client.close()

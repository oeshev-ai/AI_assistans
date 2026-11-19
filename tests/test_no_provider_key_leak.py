"""Ключи провайдеров не попадают в repr, логи, тексты ошибок и health."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest
import respx
import structlog

from app.llm.assist_ai import AssistAIClient, AssistAIConfig, AssistAIError
from app.llm.providers import ProviderRegistry
from tests.polza_helpers import make_polza_provider

_MODEL = "openai/gpt-image-2.6-flare"
_POLZA_KEY = "test-polza-key"
_PROXY_KEY = "test-proxy-key"


def test_repr_hides_keys() -> None:
    provider = make_polza_provider(api_key="sk-polza-SECRET-123")
    assert "sk-polza-SECRET-123" not in repr(provider)
    assert "sk-polza-SECRET-123" not in str(provider)
    config = AssistAIConfig(
        proxyapi_key="sk-proxy-SECRET-456",
        providers={"polza": provider},
        routes={"image": "polza"},
        image_provider_fallback="proxyapi",
    )
    text = repr(config) + str(config)
    assert "sk-proxy-SECRET-456" not in text
    assert "sk-polza-SECRET-123" not in text
    registry = ProviderRegistry(
        {"polza": provider}, {op: "polza" for op in ("text", "image", "audio", "video")}
    )
    assert "sk-polza-SECRET-123" not in repr(registry.providers)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 403, 503])
async def test_logs_and_errors_do_not_contain_keys(
    multi_client_factory: Callable[..., AssistAIClient], status: int
) -> None:
    client = multi_client_factory(fallback="")
    with respx.mock(assert_all_called=False) as router:
        router.post("https://polza.ai/api/v1/media").mock(
            return_value=httpx.Response(
                status,
                json={
                    "error": {"code": "BAD_REQUEST", "message": "boom"},
                    "trace_id": "trace-1",
                },
            )
        )
        with structlog.testing.capture_logs() as logs:
            with pytest.raises(AssistAIError) as info:
                await client.generate_image("постер", model=_MODEL, size="1024x1024")
    blob = repr(logs) + info.value.message + repr(info.value)
    assert _POLZA_KEY not in blob
    assert _PROXY_KEY not in blob
    assert "Bearer" not in blob


@pytest.mark.asyncio
async def test_proxyapi_error_text_has_no_key(
    assist_client: AssistAIClient,
) -> None:
    with respx.mock(assert_all_called=False) as router:
        router.post("https://api.proxyapi.ru/v1/images/generations").mock(
            return_value=httpx.Response(500, text="oops")
        )
        with structlog.testing.capture_logs() as logs:
            with pytest.raises(AssistAIError) as info:
                await assist_client.generate_image(
                    "постер", model=_MODEL, size="1024x1024"
                )
    blob = repr(logs) + info.value.message
    assert _PROXY_KEY not in blob


@pytest.mark.asyncio
async def test_polza_user_message_hides_internals(
    multi_client_factory: Callable[..., AssistAIClient],
) -> None:
    client = multi_client_factory(fallback="")
    with respx.mock(assert_all_called=False) as router:
        router.post("https://polza.ai/api/v1/media").mock(
            return_value=httpx.Response(
                502,
                json={
                    "error": {
                        "code": "BAD_GATEWAY",
                        "message": "upstream openai 10.1.2.3 failed",
                        "metadata": {"raw": "secret-raw", "provider_name": "OpenAI"},
                    },
                    "trace_id": "trace-xyz",
                },
            )
        )
        with pytest.raises(AssistAIError) as info:
            await client.generate_image("постер", model=_MODEL, size="1024x1024")
    for internal in ("10.1.2.3", "secret-raw", "trace-xyz", "OpenAI", "BAD_GATEWAY"):
        assert internal not in info.value.message

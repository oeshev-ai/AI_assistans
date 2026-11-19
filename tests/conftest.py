"""Общие фикстуры тестов AssistAIClient.

Ключи тестовые. Файл ``.env`` не читается, внешние API не вызываются.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from app.infra import circuit_breaker
from app.infra.circuit_breaker import CircuitState
from app.llm.assist_ai import AssistAIClient, AssistAIConfig
from tests.polza_helpers import make_polza_provider

_POLZA_FIXTURES = Path(__file__).parent / "fixtures" / "polza"
_PROVIDER_ENV = (
    "TEXT_PROVIDER",
    "IMAGE_PROVIDER",
    "IMAGE_PROVIDER_FALLBACK",
    "AUDIO_PROVIDER",
    "VIDEO_PROVIDER",
    "PROXYAPI_BASE_URL",
    "POLZA_BASE_URL",
    "POLZA_API_KEY",
    "POLZA_COST_MULTIPLIER",
    "POLZA_UPSTREAM_ONLY",
    "POLZA_ALLOW_UPSTREAM_FALLBACKS",
    "POLZA_MAX_PRICE_IMAGE_RUB",
    "POLZA_TOTAL_WAIT",
    "POLZA_POLL_INTERVAL",
    "RUB_PER_USD",
)


@pytest.fixture(autouse=True)
def _isolate_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Локальный ``.env`` разработчика не должен менять маршруты в тестах."""
    for name in _PROVIDER_ENV:
        monkeypatch.delenv(name, raising=False)
    # Production fail-closed paths (Redis eval, budgets) must not inherit .env.
    monkeypatch.setenv("ENVIRONMENT", "development")


@pytest.fixture(autouse=True)
def _reset_provider_breakers() -> None:
    """Breaker-ы кэшируются по id: сбрасываем состояние между тестами."""
    for breaker in circuit_breaker._provider_breakers.values():
        breaker._state = CircuitState.CLOSED
        breaker._failure_count = 0
        breaker._opened_at = 0.0


@pytest_asyncio.fixture
async def assist_client() -> AsyncIterator[AssistAIClient]:
    """Клиент ProxyAPI с фейковыми ключами и лимитом в 5 вызовов инструментов."""
    client = AssistAIClient(
        AssistAIConfig(
            proxyapi_key="test-proxy-key",
            openweather_api_key="",
            exchangerate_api_key="test-fx-key",
            base_url="https://api.proxyapi.ru/v1",
            default_text_model="openai/gpt-4o",
            tts_provider="proxyapi",
            timeout=5.0,
            max_tool_iterations=5,
        )
    )
    try:
        yield client
    finally:
        await client.close()


@pytest.fixture
def polza_fixture() -> Callable[[str], dict[str, Any]]:
    """Загрузчик JSON-фикстур G0 из ``tests/fixtures/polza``."""

    def load(name: str) -> dict[str, Any]:
        raw = (_POLZA_FIXTURES / f"{name}.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert isinstance(data, dict)
        return data

    return load


@pytest_asyncio.fixture
async def multi_client_factory() -> AsyncIterator[Callable[..., AssistAIClient]]:
    """Фабрика клиентов с Polza как основным image-провайдером."""
    created: list[AssistAIClient] = []

    def factory(
        *,
        image_provider: str = "polza",
        fallback: str = "proxyapi",
        provider_overrides: dict[str, Any] | None = None,
        **config_overrides: Any,
    ) -> AssistAIClient:
        providers = {"polza": make_polza_provider(**(provider_overrides or {}))}
        fields: dict[str, Any] = {
            "proxyapi_key": "test-proxy-key",
            "exchangerate_api_key": "",
            "base_url": "https://api.proxyapi.ru/v1",
            "default_text_model": "openai/gpt-4o",
            "tts_provider": "proxyapi",
            "timeout": 5.0,
            "providers": providers,
            "routes": {"image": image_provider},
            "image_provider_fallback": fallback,
            "polza_poll_interval": 4.0,
            "polza_total_wait": 240.0,
        }
        fields.update(config_overrides)
        client = AssistAIClient(AssistAIConfig(**fields))
        created.append(client)
        return client

    try:
        yield factory
    finally:
        for client in created:
            await client.close()

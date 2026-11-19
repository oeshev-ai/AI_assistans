"""G1: реестр провайдеров и resolve_route."""

from __future__ import annotations

import pytest

from app.llm.providers import ProviderRegistry, load_providers, load_routes


def test_load_providers_proxyapi_only_without_polza(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "proxy-key")
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    monkeypatch.delenv("POLZA_AI_API_KEY", raising=False)
    providers = load_providers()
    assert set(providers) == {"proxyapi"}
    assert providers["proxyapi"].kind == "openai"
    assert "images" in providers["proxyapi"].capabilities


def test_resolve_route_image_default_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "proxy-key")
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    for name in (
        "TEXT_PROVIDER",
        "IMAGE_PROVIDER",
        "AUDIO_PROVIDER",
        "VIDEO_PROVIDER",
        "IMAGE_PROVIDER_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    registry = ProviderRegistry.from_env()
    primary, fallback = registry.resolve_route("image")
    assert primary.id == "proxyapi"
    assert fallback is None
    assert load_routes()["image"] == "proxyapi"


def test_unknown_route_provider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "proxy-key")
    monkeypatch.delenv("POLZA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Неизвестный провайдер"):
        ProviderRegistry(
            load_providers(),
            {
                "text": "proxyapi",
                "image": "missing",
                "audio": "proxyapi",
                "video": "proxyapi",
            },
        )


def test_polza_registers_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "proxy-key")
    monkeypatch.setenv("POLZA_API_KEY", "polza-key")
    monkeypatch.setenv("IMAGE_PROVIDER", "polza")
    monkeypatch.setenv("IMAGE_PROVIDER_FALLBACK", "proxyapi")
    registry = ProviderRegistry.from_env()
    primary, fallback = registry.resolve_route("image")
    assert primary.id == "polza"
    assert primary.cost_source == "cost_rub"
    assert primary.kind == "polza_media"
    assert fallback is not None
    assert fallback.id == "proxyapi"
    assert "polza-key" not in repr(primary)

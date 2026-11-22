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


def test_resolve_route_defaults_polza_with_proxyapi_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "proxy-key")
    monkeypatch.setenv("POLZA_API_KEY", "polza-key")
    for name in (
        "TEXT_PROVIDER",
        "TEXT_PROVIDER_FALLBACK",
        "IMAGE_PROVIDER",
        "AUDIO_PROVIDER",
        "AUDIO_PROVIDER_FALLBACK",
        "VIDEO_PROVIDER",
        "VIDEO_PROVIDER_FALLBACK",
        "IMAGE_PROVIDER_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    registry = ProviderRegistry.from_env()
    primary, fallback = registry.resolve_route("image")
    assert primary.id == "polza"
    assert fallback is not None
    assert fallback.id == "proxyapi"
    assert load_routes()["image"] == "polza"
    text, text_fb = registry.resolve_route("text")
    assert text.id == "polza"
    assert text_fb is not None
    assert text_fb.id == "proxyapi"


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
    assert "chat" in primary.capabilities
    assert "audio" in primary.capabilities
    assert "video" in primary.capabilities
    assert primary.chat_base_url == "https://polza.ai/api/v1"
    assert fallback is not None
    assert fallback.id == "proxyapi"
    assert "polza-key" not in repr(primary)


def test_text_provider_polza_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "proxy-key")
    monkeypatch.setenv("POLZA_API_KEY", "polza-key")
    monkeypatch.setenv("TEXT_PROVIDER", "polza")
    monkeypatch.setenv("TEXT_PROVIDER_FALLBACK", "proxyapi")
    monkeypatch.setenv("IMAGE_PROVIDER", "polza")
    monkeypatch.setenv("IMAGE_PROVIDER_FALLBACK", "proxyapi")
    registry = ProviderRegistry.from_env()
    text, text_fb = registry.resolve_route("text")
    assert text.id == "polza"
    assert text_fb is not None
    assert text_fb.id == "proxyapi"
    assert text.chat_base_url == "https://polza.ai/api/v1"


def test_audio_provider_polza_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "proxy-key")
    monkeypatch.setenv("POLZA_API_KEY", "polza-key")
    monkeypatch.setenv("AUDIO_PROVIDER", "polza")
    monkeypatch.setenv("AUDIO_PROVIDER_FALLBACK", "proxyapi")
    registry = ProviderRegistry.from_env()
    audio, audio_fb = registry.resolve_route("audio")
    assert audio.id == "polza"
    assert "audio" in audio.capabilities
    assert audio_fb is not None
    assert audio_fb.id == "proxyapi"


def test_video_provider_polza_resolves(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROXYAPI_API_KEY", "proxy-key")
    monkeypatch.setenv("POLZA_API_KEY", "polza-key")
    monkeypatch.setenv("VIDEO_PROVIDER", "polza")
    monkeypatch.setenv("VIDEO_PROVIDER_FALLBACK", "proxyapi")
    registry = ProviderRegistry.from_env()
    video, video_fb = registry.resolve_route("video")
    assert video.id == "polza"
    assert "video" in video.capabilities
    assert video_fb is not None
    assert video_fb.id == "proxyapi"

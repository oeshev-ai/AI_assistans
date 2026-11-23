"""Снятые image-id не возвращаются в каталог и дефолты."""

from __future__ import annotations

import pytest

from app.llm.assist_ai import AssistAIConfig
from app.llm.models_config import AVAILABLE_IMAGE_MODELS, AVAILABLE_TEXT_MODELS
from app.storage.assistant_presets import ASSISTANT_PRESETS

_DEPRECATED = {
    "gpt-image-1-mini",
    "gpt-image-1.5",
    "gpt-image-2",
    "gpt-image-1",
    "chatgpt-image-latest",
}


def _names(model_id: str | None) -> set[str]:
    if not model_id:
        return set()
    return {model_id, model_id.rsplit("/", 1)[-1]}


def test_deprecated_image_ids_stay_out_of_catalog_and_defaults() -> None:
    seen: set[str] = set()
    for model_id in (*AVAILABLE_TEXT_MODELS, *AVAILABLE_IMAGE_MODELS):
        seen |= _names(model_id)
    for preset in ASSISTANT_PRESETS:
        seen |= _names(preset.text_model)
        seen |= _names(preset.image_model)
    for field_name in (
        "default_text_model",
        "default_image_model",
        "default_image_edit_model",
    ):
        seen |= _names(AssistAIConfig.model_fields[field_name].default)
    assert _DEPRECATED.isdisjoint(seen)


def test_route_defaults_do_not_reference_deprecated_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.llm.providers import (
        load_audio_fallback,
        load_image_fallback,
        load_providers,
        load_routes,
        load_text_fallback,
        load_video_fallback,
    )

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
    monkeypatch.setenv("PROXYAPI_API_KEY", "test-proxy-key")
    monkeypatch.setenv("POLZA_API_KEY", "test-polza-key")
    routes = load_routes()
    # Без *_PROVIDER: primary = polza, fallback = proxyapi.
    assert set(routes.values()) == {"polza"}
    assert load_image_fallback() == "proxyapi"
    assert load_text_fallback() == "proxyapi"
    assert load_audio_fallback() == "proxyapi"
    assert load_video_fallback() == "proxyapi"
    seen: set[str] = set()
    for value in (
        *routes.values(),
        load_image_fallback(),
        load_text_fallback(),
        load_audio_fallback(),
        load_video_fallback(),
        *load_providers(),
    ):
        seen |= _names(value)
    assert _DEPRECATED.isdisjoint(seen)
    # Пустые маршруты в конфиге: подставляются ProxyAPI в клиенте, модели не задаются.
    assert AssistAIConfig.model_fields["routes"].default_factory is dict
    assert AssistAIConfig.model_fields["image_provider_fallback"].default == ""
    assert AssistAIConfig.model_fields["text_provider_fallback"].default == ""
    assert AssistAIConfig.model_fields["audio_provider_fallback"].default == ""
    assert AssistAIConfig.model_fields["video_provider_fallback"].default == ""

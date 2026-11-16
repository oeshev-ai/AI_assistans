"""Снятые image-id не возвращаются в каталог и дефолты."""

from __future__ import annotations

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
